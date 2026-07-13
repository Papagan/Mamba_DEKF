import unittest
import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

from training.pairwise_association_cache import (
    build_pairwise_association_samples,
    summarize_pairwise_association_samples,
)


def _frame(sample_token, frame_id, x, y, *, matched=True, score=0.8, fusion_valid=False, vx=0.0, vy=0.0):
    obs = [x, y, 0.0, vx, vy, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, score if matched else 0.0]
    fusion = [x + 10.0, y + 10.0, 0.0, vx, vy, 0.0, 4.0, 2.0, 1.5, 0.0, 0.0, score if matched else 0.0]
    return {
        "sample_token": sample_token,
        "timestamp": float(frame_id),
        "frame_id": frame_id,
        "scene_id": "scene-a",
        "is_matched": matched,
        "det_score": score if matched else 0.0,
        "obs_feature_12": obs,
        "fusion_feature_12": fusion,
        "fusion_valid": bool(fusion_valid),
        "fusion_is_fake": False,
        "det_global_xyz": [x, y, 0.0] if matched else None,
        "det_lwh": [4.0, 2.0, 1.5] if matched else None,
        "det_yaw": 0.0 if matched else None,
        "det_velocity": [vx, vy] if matched else None,
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

    def test_fusion_history_source_uses_only_valid_fusion_frames(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 1.0, 0.0, fusion_valid=False),
                    _frame("s1", 1, 2.0, 0.0, fusion_valid=True),
                    _frame("s2", 2, 3.0, 0.0, fusion_valid=True),
                ],
            },
            {
                "instance_token": "car-b",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 5.0, 0.0, fusion_valid=True),
                    _frame("s1", 1, 6.0, 0.0, fusion_valid=True),
                    _frame("s2", 2, 7.0, 0.0, fusion_valid=True),
                ],
            },
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            history_source="fusion",
            max_hard_negatives=0,
            max_easy_negatives=0,
        )

        anchor = next(
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a"
            and sample["current_sample_token"] == "s1"
        )
        self.assertEqual(anchor["anchor_history_12"][0], [0.0] * 12)
        self.assertEqual(anchor["anchor_history_12"][1][0], 12.0)
        self.assertEqual(anchor["history_source"], "fusion")

    def test_track_candidate_pair_geometry_uses_current_anchor_state(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 10.0, 0.0)],
            },
            {
                "instance_token": "car-b",
                "category": "car",
                "frames": [_frame("s0", 0, 20.0, 0.0), _frame("s1", 1, 13.0, 0.0)],
            },
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            pair_geometry_source="track_candidate",
            hard_negative_distance=20.0,
            max_hard_negatives=1,
            max_easy_negatives=0,
        )

        positive = next(
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a" and sample["label"] == 1
        )
        negative = next(
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a" and sample["label"] == 0
        )
        self.assertAlmostEqual(positive["center_distance"], 10.0)
        self.assertAlmostEqual(negative["center_distance"], 13.0)
        self.assertEqual(positive["pair_geometry_source"], "track_candidate")

    def test_predicted_track_candidate_geometry_uses_velocity_extrapolation(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 0.0, 0.0, vx=10.0),
                    _frame("s1", 1, 10.0, 0.0, vx=10.0),
                ],
            },
            {
                "instance_token": "car-b",
                "category": "car",
                "frames": [_frame("s0", 0, 20.0, 0.0), _frame("s1", 1, 13.0, 0.0)],
            },
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            pair_geometry_source="predicted_track_candidate",
            hard_negative_distance=20.0,
            max_hard_negatives=1,
            max_easy_negatives=0,
        )

        positive = next(
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a" and sample["label"] == 1
        )
        negative = next(
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a" and sample["label"] == 0
        )
        self.assertAlmostEqual(positive["center_distance"], 0.0)
        self.assertAlmostEqual(negative["center_distance"], 3.0)
        self.assertEqual(positive["pair_geometry_source"], "predicted_track_candidate")

    def test_candidate_history_uses_inference_detection_token(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [_frame("s0", 0, 0.0, 0.0), _frame("s1", 1, 10.0, 5.0)],
            }
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=3,
            future_step=1,
            max_hard_negatives=0,
            max_easy_negatives=0,
        )

        positive = samples[0]
        self.assertEqual(len(positive["candidate_history_12"]), 3)
        self.assertEqual(positive["candidate_history_12"][0], [0.0] * 12)
        self.assertEqual(positive["candidate_history_12"][1], [0.0] * 12)
        self.assertEqual(positive["candidate_history_12"][2][0:2], [0.0, 0.0])
        self.assertEqual(positive["candidate_history_12"][2][2], 0.0)
        self.assertEqual(positive["candidate_history_12"][2][11], 0.8)
        self.assertEqual(positive["candidate_obs_feature_12"][0:2], [10.0, 5.0])

    def test_inference_margin_negative_mining_keeps_only_near_best_predicted_candidates(self):
        tracklets = [
            {
                "instance_token": "car-a",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 0.0, 0.0, vx=10.0),
                    _frame("s1", 1, 10.0, 0.0, vx=10.0),
                ],
            },
            {
                "instance_token": "car-close",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 20.0, 0.0),
                    _frame("s1", 1, 10.03, 0.0),
                ],
            },
            {
                "instance_token": "car-far",
                "category": "car",
                "frames": [
                    _frame("s0", 0, 20.0, 0.0),
                    _frame("s1", 1, 12.0, 0.0),
                ],
            },
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            pair_geometry_source="predicted_track_candidate",
            negative_mining_mode="inference_margin",
            cost_margin_eps=0.05,
            max_hard_negatives=4,
            max_easy_negatives=4,
        )

        car_a_samples = [
            sample for sample in samples
            if sample["anchor_instance_token"] == "car-a"
        ]
        candidate_ids = {sample["candidate_instance_token"] for sample in car_a_samples}
        negative_ids = {
            sample["candidate_instance_token"]
            for sample in car_a_samples
            if sample["label"] == 0
        }
        self.assertIn("car-a", candidate_ids)
        self.assertEqual(negative_ids, {"car-close"})
        close = next(sample for sample in car_a_samples if sample["candidate_instance_token"] == "car-close")
        self.assertEqual(close["negative_type"], "inference_margin")
        self.assertAlmostEqual(close["center_distance"], 0.03, places=6)
        self.assertEqual(close["negative_mining_mode"], "inference_margin")

    def test_inference_margin_negative_mining_falls_back_to_nearest_topk_when_margin_empty(self):
        tracklets = [
            {
                "instance_token": "bus-a",
                "category": "bus",
                "frames": [
                    _frame("s0", 0, 0.0, 0.0, vx=0.0),
                    _frame("s1", 1, 0.0, 0.0, vx=0.0),
                ],
            },
            {
                "instance_token": "bus-near",
                "category": "bus",
                "frames": [
                    _frame("s0", 0, 20.0, 0.0),
                    _frame("s1", 1, 3.0, 0.0),
                ],
            },
            {
                "instance_token": "bus-far",
                "category": "bus",
                "frames": [
                    _frame("s0", 0, 20.0, 0.0),
                    _frame("s1", 1, 8.0, 0.0),
                ],
            },
        ]

        samples, _ = build_pairwise_association_samples(
            tracklets,
            history_len=2,
            future_step=1,
            pair_geometry_source="predicted_track_candidate",
            negative_mining_mode="inference_margin",
            cost_margin_eps=0.05,
            min_hard_negatives=1,
            max_hard_negatives=2,
            max_easy_negatives=0,
        )

        bus_a_negatives = [
            sample for sample in samples
            if sample["anchor_instance_token"] == "bus-a" and sample["label"] == 0
        ]
        self.assertEqual(len(bus_a_negatives), 1)
        self.assertEqual(bus_a_negatives[0]["candidate_instance_token"], "bus-near")
        self.assertEqual(bus_a_negatives[0]["negative_type"], "inference_topk")
        self.assertAlmostEqual(bus_a_negatives[0]["center_distance"], 3.0)

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
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["settings"]["history_source"], "fusion")


if __name__ == "__main__":
    unittest.main()
