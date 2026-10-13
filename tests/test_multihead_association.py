import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from training.association_dataset import (
    AnchorGroupBatchSampler,
    PairwiseAssociationDataset,
    PrecomputedPairwiseAssociationDataset,
    pairwise_association_collate_fn,
    precomputed_pairwise_association_collate_fn,
)
from training.association_model import ClassConditionedAssociationHeadBank
from training.association_metrics import _binary_auc, compute_association_metrics
from tools.train_pairwise_association import (
    _association_bce_weights,
    _association_listwise_loss_from_logits,
    _parse_class_ids,
    _run_precomputed_epoch,
    _select_precomputed_class,
)


def _sample(anchor_id, class_id, label, logit_hint=0.0, negative_type="positive"):
    return {
        "category": "bicycle" if class_id == 2 else "car",
        "class_id": class_id,
        "state_bucket": "matched",
        "anchor_instance_token": anchor_id,
        "current_sample_token": "cur",
        "future_sample_token": "fut",
        "label": label,
        "negative_type": negative_type,
        "anchor_history_12": [[float(logit_hint)] * 12 for _ in range(3)],
        "candidate_obs_feature_12": [float(logit_hint)] * 12,
        "center_distance": 0.5,
        "yaw_diff": 0.1,
        "size_l1": 0.2,
        "anchor_det_score": 0.8,
        "candidate_det_score": 0.7,
    }


class MultiHeadAssociationTest(unittest.TestCase):
    def test_old_pairwise_feature_schema_is_rejected_for_stage_b(self):
        sample = _sample("a", 2, 1)
        with self.assertRaisesRegex(ValueError, "outdated feature schema"):
            PairwiseAssociationDataset(
                [sample], history_len=3, required_feature_schema_version=2
            )
        sample["feature_schema_version"] = 2
        self.assertEqual(
            len(PairwiseAssociationDataset(
                [sample], history_len=3, required_feature_schema_version=2
            )),
            1,
        )

    def test_balanced_bce_weights_preserve_hard_negative_ratio_and_neutral_prior(self):
        labels = torch.tensor([1.0, 0.0, 0.0])

        weights = _association_bce_weights(
            labels,
            ["positive", "hard", "easy"],
            hard_negative_weight=2.0,
            balance_positives=True,
        )

        positive_mass = weights[labels > 0.5].sum()
        negative_mass = weights[labels <= 0.5].sum()
        self.assertTrue(torch.allclose(positive_mass, negative_mass))
        self.assertAlmostEqual(float(weights.mean()), 1.0, places=6)
        self.assertAlmostEqual(float(weights[1] / weights[2]), 2.0, places=6)

    def test_anchor_group_batch_sampler_never_splits_candidate_set(self):
        keys = ["a", "a", "a", "b", "b", "c"]
        sampler = AnchorGroupBatchSampler(keys, batch_size=4, shuffle=False)

        batches = list(sampler)

        self.assertEqual(batches, [[0, 1, 2], [3, 4, 5]])
        for key in set(keys):
            containing = [batch for batch in batches if any(keys[index] == key for index in batch)]
            self.assertEqual(len(containing), 1)

    def test_listwise_loss_prefers_correct_candidate_ranking(self):
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0])
        anchor_keys = ["a", "a", "b", "b"]

        good = _association_listwise_loss_from_logits(
            torch.tensor([3.0, -1.0, 2.0, -2.0]), labels, anchor_keys
        )
        bad = _association_listwise_loss_from_logits(
            torch.tensor([-1.0, 3.0, -2.0, 2.0]), labels, anchor_keys
        )

        self.assertLess(float(good), float(bad))

    def test_precomputed_samples_can_be_partitioned_without_reencoding(self):
        samples = {
            "pair_vector": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]),
            "class_id": torch.tensor([0, 2, 0]),
            "label": torch.tensor([1.0, 0.0, 0.0]),
            "anchor_keys": ["a", "b", "c"],
            "negative_types": ["positive", "hard", "hard"],
            "categories": ["car", "bicycle", "car"],
        }

        selected = _select_precomputed_class(samples, 0)

        self.assertEqual(selected["class_id"].tolist(), [0, 0])
        self.assertEqual(selected["anchor_keys"], ["a", "c"])
        self.assertTrue(torch.equal(selected["pair_vector"], torch.tensor([[1.0, 2.0], [5.0, 6.0]])))

    def test_class_id_parser_rejects_out_of_range_heads(self):
        self.assertEqual(_parse_class_ids("2,0,2", 3), [2, 0])
        with self.assertRaises(ValueError):
            _parse_class_ids("3", 3)

    def test_single_class_optimizer_cannot_update_other_heads(self):
        bank = ClassConditionedAssociationHeadBank(
            input_dim=4,
            num_classes=3,
            hidden_dims=(8,),
            dropout=0.0,
        )
        samples = [
            {
                "pair_vector": [1.0, 0.0, 0.0, 0.0],
                "class_id": 1,
                "label": 1,
                "anchor_key": "a",
                "negative_type": "positive",
                "category": "pedestrian",
            },
            {
                "pair_vector": [0.0, 1.0, 0.0, 0.0],
                "class_id": 1,
                "label": 0,
                "anchor_key": "a",
                "negative_type": "hard",
                "category": "pedestrian",
            },
        ]
        loader = DataLoader(
            PrecomputedPairwiseAssociationDataset(samples),
            batch_size=2,
            shuffle=False,
            collate_fn=precomputed_pairwise_association_collate_fn,
        )
        untouched_before = {
            key: value.detach().clone() for key, value in bank.heads[0].state_dict().items()
        }
        target_before = {
            key: value.detach().clone() for key, value in bank.heads[1].state_dict().items()
        }
        optimizer = torch.optim.AdamW(bank.heads[1].parameters(), lr=1.0e-2)

        _run_precomputed_epoch(
            bank,
            loader,
            optimizer,
            torch.device("cpu"),
            train=True,
            hard_negative_weight=1.0,
            ranking_margin=0.2,
            ranking_weight=0.1,
        )

        self.assertTrue(all(
            torch.equal(value, untouched_before[key])
            for key, value in bank.heads[0].state_dict().items()
        ))
        self.assertTrue(any(
            not torch.equal(value, target_before[key])
            for key, value in bank.heads[1].state_dict().items()
        ))

    def test_pairwise_dataset_returns_model_ready_tensors(self):
        samples = [_sample("a", 2, 1), _sample("a", 2, 0, negative_type="hard")]
        ds = PairwiseAssociationDataset(samples, history_len=3)

        item = ds[0]

        self.assertEqual(tuple(item["anchor_history"].shape), (3, 12))
        self.assertEqual(tuple(item["candidate_history"].shape), (3, 12))
        self.assertEqual(tuple(item["pair_features"].shape), (7,))
        self.assertEqual(int(item["class_id"].item()), 2)
        self.assertEqual(float(item["label"].item()), 1.0)

    def test_pairwise_collate_batches_string_metadata(self):
        batch = pairwise_association_collate_fn([
            PairwiseAssociationDataset([_sample("a", 2, 1)], history_len=3)[0],
            PairwiseAssociationDataset([_sample("b", 3, 0, negative_type="hard")], history_len=3)[0],
        ])

        self.assertEqual(tuple(batch["anchor_history"].shape), (2, 3, 12))
        self.assertEqual(batch["anchor_keys"], ["a::cur", "b::cur"])
        self.assertEqual(batch["negative_types"], ["positive", "hard"])

    def test_class_conditioned_head_routes_by_class(self):
        bank = ClassConditionedAssociationHeadBank(
            input_dim=4,
            num_classes=3,
            hidden_dims=(8,),
            dropout=0.0,
        )
        with torch.no_grad():
            for head in bank.heads:
                for module in head.modules():
                    if isinstance(module, torch.nn.Linear):
                        module.weight.zero_()
                        module.bias.zero_()
            bank.heads[1][-1].bias.fill_(2.0)
            bank.heads[2][-1].bias.fill_(-3.0)

        logits = bank(torch.zeros(3, 4), torch.tensor([1, 2, 1]))

        self.assertTrue(torch.allclose(logits, torch.tensor([2.0, -3.0, 2.0])))

    def test_precomputed_pairwise_dataset_returns_pair_vectors(self):
        samples = [
            {
                "pair_vector": [0.1, 0.2, 0.3, 0.4],
                "class_id": 1,
                "label": 1,
                "anchor_key": "a::cur",
                "negative_type": "positive",
                "category": "car",
            },
            {
                "pair_vector": [0.5, 0.6, 0.7, 0.8],
                "class_id": 2,
                "label": 0,
                "anchor_key": "a::cur",
                "negative_type": "hard",
                "category": "bicycle",
            },
        ]

        batch = precomputed_pairwise_association_collate_fn([
            PrecomputedPairwiseAssociationDataset(samples)[0],
            PrecomputedPairwiseAssociationDataset(samples)[1],
        ])

        self.assertEqual(tuple(batch["pair_vector"].shape), (2, 4))
        self.assertEqual(batch["anchor_keys"], ["a::cur", "a::cur"])
        self.assertEqual(batch["negative_types"], ["positive", "hard"])
        head = ClassConditionedAssociationHeadBank(
            input_dim=4,
            num_classes=3,
            hidden_dims=(8,),
            dropout=0.0,
        )
        logits = head(batch["pair_vector"], batch["class_id"])
        self.assertEqual(tuple(logits.shape), (2,))

    def test_dropout_must_match_training_state_dict_keys(self):
        trained_bank = ClassConditionedAssociationHeadBank(
            input_dim=4,
            num_classes=2,
            hidden_dims=(8, 4),
            dropout=0.1,
        )
        eval_bank = ClassConditionedAssociationHeadBank(
            input_dim=4,
            num_classes=2,
            hidden_dims=(8, 4),
            dropout=0.1,
        )

        missing, unexpected = eval_bank.load_state_dict(trained_bank.state_dict(), strict=False)

        self.assertEqual(missing, [])
        self.assertEqual(unexpected, [])
        self.assertIsInstance(eval_bank.heads[0][3], torch.nn.Dropout)
        self.assertIsInstance(eval_bank.heads[0][4], torch.nn.Linear)

    def test_metrics_compute_per_class_auc_and_topk(self):
        records = [
            {"class_id": 2, "anchor_key": "a", "label": 1, "score": 0.9, "negative_type": "positive"},
            {"class_id": 2, "anchor_key": "a", "label": 0, "score": 0.1, "negative_type": "hard"},
            {"class_id": 2, "anchor_key": "b", "label": 1, "score": 0.4, "negative_type": "positive"},
            {"class_id": 2, "anchor_key": "b", "label": 0, "score": 0.6, "negative_type": "hard"},
        ]

        metrics = compute_association_metrics(records, topk=(1, 3))

        self.assertAlmostEqual(metrics["overall"]["top1"], 0.5)
        self.assertAlmostEqual(metrics["overall"]["top3"], 1.0)
        self.assertAlmostEqual(metrics["per_class"]["2"]["hard_negative_accuracy"], 0.5)
        self.assertGreater(metrics["per_class"]["2"]["auc"], 0.0)

    def test_inference_margin_negatives_count_as_hard_negative_accuracy(self):
        records = [
            {"class_id": 2, "anchor_key": "a", "label": 1, "score": 0.9, "negative_type": "positive"},
            {"class_id": 2, "anchor_key": "a", "label": 0, "score": 0.1, "negative_type": "inference_margin"},
        ]

        metrics = compute_association_metrics(records)

        self.assertAlmostEqual(metrics["overall"]["hard_negative_accuracy"], 1.0)

    def test_binary_auc_handles_ties_with_rank_formula(self):
        labels = [1, 1, 0, 0]
        scores = [0.5, 0.8, 0.5, 0.2]

        auc = _binary_auc(labels, scores)

        self.assertAlmostEqual(auc, 0.875)


if __name__ == "__main__":
    unittest.main()
