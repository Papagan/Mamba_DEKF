import tempfile
import unittest
from pathlib import Path

import torch

from training.association_dataset import PairwiseAssociationDataset, pairwise_association_collate_fn
from training.association_model import ClassConditionedAssociationHeadBank
from training.association_metrics import compute_association_metrics


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


if __name__ == "__main__":
    unittest.main()
