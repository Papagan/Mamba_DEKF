import copy
import unittest

import numpy as np

from tracker.inference_pair_cache import build_candidate_groups, InferencePairCacheAccumulator
from tracker.inference_pair_cache import validate_candidate_group, validate_accumulator_state
from training.replay_association_samples import replay_groups_to_samples


class ReplayAssociationSamplesTest(unittest.TestCase):
    def test_partial_groups_survive_require_gt_export_and_summary(self):
        payload = self._payload(det_tokens=[None, "gt-a"])
        accumulator = InferencePairCacheAccumulator(
            runtime_manifest=payload["runtime_manifest"], require_audit_gt=True,
        )
        accumulator.add_group(payload["groups"][0])
        state = accumulator.export_state()
        self.assertEqual(len(state["groups"]), 1)
        self.assertEqual(state["groups"][0]["gt_target_status"], "partial")
        restored = InferencePairCacheAccumulator(runtime_manifest=payload["runtime_manifest"], require_audit_gt=True)
        restored.merge_state(state)
        self.assertEqual(len(restored.export_state()["groups"]), 1)

    def _payload(self, *, target="gt-a", det_tokens=None):
        det_tokens = ["gt-b", "gt-a"] if det_tokens is None else det_tokens
        groups = build_candidate_groups(
            payload={
                "scene_id": "scene-1", "sample_token": "sample-2", "frame_id": 2,
                "track_ids": [8], "class_ids": [0], "class_names": ["car"],
                "state_buckets": ["unmatched"], "track_gt_tokens": [target],
                "det_gt_tokens": det_tokens, "pair_indices": [(0, 0), (0, 1)],
                "pair_vectors": np.arange(30, dtype=np.float32).reshape(2, 15),
                "track_embeddings": np.ones((1, 2), dtype=np.float32),
                "det_embeddings": np.ones((2, 2), dtype=np.float32),
            },
            base_cost_matrix=np.array([[0.2, 0.21]], dtype=np.float32),
            production_matches=np.array([[0, 0]], dtype=np.int64),
            production_costs=np.array([0.2], dtype=np.float32), thresholds={0: 1.1},
        )
        return {"runtime_manifest": {
            "mamba_checkpoint_sha256": "a" * 64,
            "embedding_dim": 2, "pair_vector_dim": 15,
        }, "groups": groups}

    def test_uses_real_vectors_and_gt_label_not_production_assignment(self):
        samples, manifest = replay_groups_to_samples(self._payload())
        self.assertEqual([sample["label"] for sample in samples], [0, 1])
        self.assertEqual(samples[0]["pair_vector"], list(range(15)))
        self.assertTrue(all(sample["state_bucket"] == "unmatched" for sample in samples))
        self.assertEqual(manifest["mamba_checkpoint_sha256"], "a" * 64)

    def test_known_no_match_is_kept_as_all_negative(self):
        samples, _ = replay_groups_to_samples(self._payload(target="gt-not-detected"))
        self.assertEqual([sample["label"] for sample in samples], [0, 0])

    def test_unknown_identity_is_not_converted_to_negative(self):
        payload = self._payload(det_tokens=[None, "gt-a"])
        samples, _ = replay_groups_to_samples(payload)
        self.assertEqual([sample["label"] for sample in samples], [1])
        self.assertEqual(samples[0]["pair_vector"], list(range(15, 30)))
        samples, _ = replay_groups_to_samples(self._payload(det_tokens=[None, "gt-b"]))
        self.assertEqual([sample["label"] for sample in samples], [0])
        payload = self._payload(det_tokens=[None, None])
        with self.assertRaisesRegex(ValueError, "no finite candidates with trusted GT labels"):
            replay_groups_to_samples(payload)

    def test_replay_requires_backbone_identity_and_correct_vector_width(self):
        payload = self._payload()
        payload["runtime_manifest"]["mamba_checkpoint_sha256"] = None
        with self.assertRaisesRegex(ValueError, "checkpoint SHA-256"):
            replay_groups_to_samples(payload)
        payload = self._payload()
        payload["runtime_manifest"]["pair_vector_dim"] = 10
        with self.assertRaisesRegex(ValueError, "seven geometry"):
            replay_groups_to_samples(payload)


class BoundedReplayTest(unittest.TestCase):
    def _bounded_group(self, tokens, production=6):
        count = len(tokens)
        costs = np.array([[.1, .11, .12, .13, .6, .7, .8, .9]], dtype=np.float32)
        return build_candidate_groups(
            payload={
                "scene_id": "scene", "sample_token": "sample", "frame_id": 1,
                "track_ids": [1], "class_ids": [0], "class_names": ["car"],
                "state_buckets": ["matched"], "track_gt_tokens": ["gt-a"], "det_gt_tokens": tokens,
                "pair_indices": [(0, index) for index in range(count)],
                "pair_vectors": np.arange(count * 15, dtype=np.float32).reshape(count, 15),
                "track_embeddings": np.ones((1, 2)), "det_embeddings": np.ones((count, 2)),
            },
            base_cost_matrix=costs, production_matches=np.array([[0, production]]),
            production_costs=np.array([costs[0, production]]), thresholds={0: 1.1},
            max_candidates_per_group=4, candidate_cost_margin=.05,
        )

    def test_pruning_keeps_best_production_positive_and_unknown_marker(self):
        group = self._bounded_group(["b", "c", "d", "e", "f", None, "g", "gt-a"])[0]
        validate_candidate_group(group)
        self.assertEqual([item["det_index"] for item in group["candidates"]], [0, 5, 6, 7])
        self.assertEqual([item["gt_label"] for item in group["candidates"]], [0, None, 0, 1])
        self.assertEqual(group["production_det_index"], 6)
        self.assertEqual(group["gt_target_status"], "partial")
        self.assertEqual(group["candidate_count"], 4)

    def test_pruning_does_not_turn_duplicate_positive_into_unique_label(self):
        group = self._bounded_group(["b", "c", "d", "e", "f", None, "gt-a", "gt-a"])[0]
        validate_candidate_group(group)
        labels = {item["det_index"]: item["gt_label"] for item in group["candidates"]}
        self.assertIsNone(labels[6])
        self.assertIsNone(labels[7])
        self.assertEqual(self._bounded_group(["gt-a"] * 8), [])

    def test_hash_priority_caps_are_order_independent_and_merge_exactly(self):
        payload = ReplayAssociationSamplesTest()._payload()
        manifest = dict(payload["runtime_manifest"], export_options={"sampling_seed": 19, "require_audit_gt": True})
        groups = []
        for index in range(30):
            group = copy.deepcopy(payload["groups"][0])
            group["scene_id"] = f"scene-{index:03d}"
            groups.append(group)
        def accumulator():
            return InferencePairCacheAccumulator(runtime_manifest=manifest, require_audit_gt=True, max_groups_per_class_state=5)
        direct = accumulator()
        for group in groups:
            direct.add_group(group)
        merged = accumulator()
        for offset in (2, 1, 0):
            worker = accumulator()
            for group in reversed(groups[offset::3]):
                worker.add_group(group)
            validate_accumulator_state(worker.export_state())
            merged.merge_state(worker.export_state())
        self.assertEqual(direct.export_state(), merged.export_state())
        retained = direct.export_state()["groups"]
        self.assertEqual(len(retained), 5)
        self.assertNotEqual([group["scene_id"] for group in retained], [group["scene_id"] for group in groups[:5]])


if __name__ == "__main__":
    unittest.main()
