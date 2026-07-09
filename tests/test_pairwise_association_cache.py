import unittest
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from training.pairwise_association_cache import (
    build_pairwise_association_samples,
    summarize_pairwise_association_samples,
)


def _frame(sample_token, frame_id, x, y, *, matched=True, score=0.8):
    return {
        "sample_token": sample_token,
        "timestamp": float(frame_id),
        "frame_id": frame_id,
        "scene_id": "scene-a",
        "is_matched": matched,
        "det_score": score if matched else 0.0,
        "obs_feature_12": [x, y, 0.0, 0.0, 0.0, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, score if matched else 0.0],
        "det_global_xyz": [x, y, 0.0] if matched else None,
        "det_lwh": [4.0, 2.0, 1.5] if matched else None,
        "det_yaw": 0.0 if matched else None,
        "det_velocity": [0.0, 0.0] if matched else None,
    }


class PairwiseAssociationCacheTest(unittest.TestCase):
    def test_builds_positive_and_hard_negative_from_same_frame_same_class(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 1.0, 0.0)],
            },
            {
                "instance_token": "car-b",
                "category": "car",
                "frames": [_frame("s0", 0, 2.0, 0.0), _frame("s1", 1, 1.4, 0.0)],
            },
        ]

        samples, summary = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            hard_negative_distance=1.0,
            max_hard_negatives=2,
            max_easy_negatives=0,
        )

        labels = [sample["label"] for sample in samples if sample["anchor_instance_token"] == "car-a"]
        negative_types = [
            sample["negative_type"]
            for sample in samples
            if sample["anchor_instance_token"] == "car-a" and sample["label"] == 0
        ]
        self.assertEqual(labels.count(1), 1)
        self.assertEqual(labels.count(0), 1)
        self.assertEqual(negative_types, ["hard"])
        self.assertEqual(summary["overall"]["positive_pairs"], 2)
        self.assertEqual(summary["overall"]["negative_pairs"], 2)

    def test_ignores_cross_class_candidates(self):
        tracklets = [
            {
                "instance_token": "bike-a",
                "category": "bicycle",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 1.0, 0.0)],
            },
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 1.1, 0.0)],
            },
        ]

        samples, summary = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            max_easy_negatives=4,
        )

        self.assertEqual(len(samples), 2)
        self.assertTrue(all(sample["label"] == 1 for sample in samples))
        self.assertEqual(summary["overall"]["negative_pairs"], 0)

    def test_uses_class_specific_hard_negative_distance(self):
        tracklets = [
            {
                "instance_token": "bus-a",
                "category": "bus",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 0.0, 0.0)],
            },
            {
                "instance_token": "bus-b",
                "category": "bus",
                "frames": [_frame("s0", 0, 20.0, 0.0), _frame("s1", 1, 8.0, 0.0)],
            },
        ]

        samples, summary = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            hard_negative_distance=4.0,
            hard_negative_distance_by_class={"bus": 10.0},
            max_hard_negatives=2,
            max_easy_negatives=0,
        )

        bus_negatives = [
            sample for sample in samples
            if sample["category"] == "bus" and sample["label"] == 0
        ]
        self.assertEqual([sample["negative_type"] for sample in bus_negatives], ["hard", "hard"])
        self.assertEqual(summary["per_class"]["bus"]["hard_negative_pairs"], 2)

    def test_limits_pairs_per_class_when_requested(self):
        tracklets = [
            {
                "instance_token": f"car-{idx}",
                "category": "car",
                "frames": [_frame("s0", 0, float(idx), 0.0), _frame("s1", 1, float(idx), 0.0)],
            }
            for idx in range(4)
        ]

        samples, summary = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            hard_negative_distance=10.0,
            max_hard_negatives=1,
            max_easy_negatives=0,
            max_pairs_per_class={"car": 5},
        )

        self.assertEqual(len(samples), 5)
        self.assertEqual(summary["per_class"]["car"]["pairs"], 5)

    def test_summary_reports_per_class_pair_counts(self):
        samples = [
            {"category": "bicycle", "label": 1, "negative_type": "positive"},
            {"category": "bicycle", "label": 0, "negative_type": "hard"},
            {"category": "trailer", "label": 1, "negative_type": "positive"},
        ]

        summary = summarize_pairwise_association_samples(samples)

        self.assertEqual(summary["per_class"]["bicycle"]["positive_pairs"], 1)
        self.assertEqual(summary["per_class"]["bicycle"]["hard_negative_pairs"], 1)
        self.assertEqual(summary["per_class"]["trailer"]["positive_pairs"], 1)
        self.assertEqual(summary["per_class"]["bicycle"]["candidate_pairs"], 2)
        self.assertAlmostEqual(summary["per_class"]["bicycle"]["positive_ratio"], 0.5)
        self.assertAlmostEqual(summary["per_class"]["bicycle"]["negative_ratio"], 0.5)
        self.assertAlmostEqual(summary["per_class"]["bicycle"]["hard_negative_ratio"], 0.5)

    def test_cli_runs_from_project_root(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 1.0, 0.0)],
            },
            {
                "instance_token": "car-b",
                "category": "car",
                "frames": [_frame("s0", 0, 2.0, 0.0), _frame("s1", 1, 1.4, 0.0)],
            },
        ]
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            input_path = tmp_path / "tracklets.pkl"
            output_path = tmp_path / "pairwise.pkl"
            summary_path = tmp_path / "summary.json"
            with open(input_path, "wb") as f:
                pickle.dump(tracklets, f)

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/build_pairwise_association_cache.py",
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                    "--summary-output",
                    str(summary_path),
                    "--history-len",
                    "2",
                    "--max-easy-negatives",
                    "0",
                ],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(output_path.exists())
            self.assertTrue(summary_path.exists())


if __name__ == "__main__":
    unittest.main()
