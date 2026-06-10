import unittest

from tools.calibrate_track_score import prepare_matrix
from tools.export_track_quality_features import compute_track_features


class TrackScoreToolsTest(unittest.TestCase):
    def test_compute_track_features_basic(self):
        track_key = "scene-1|bicycle|42"
        entries = [
            {
                "timestamp": 0.0,
                "tracking_score": 0.6,
                "translation": [0.0, 0.0, 0.0],
                "velocity": [1.0, 0.0],
                "matched_gt": "gt-1",
                "match_dist": 0.5,
            },
            {
                "timestamp": 0.5,
                "tracking_score": 0.7,
                "translation": [1.0, 0.0, 0.0],
                "velocity": [1.0, 0.0],
                "matched_gt": "gt-1",
                "match_dist": 0.4,
            },
            {
                "timestamp": 1.0,
                "tracking_score": 0.4,
                "translation": [2.0, 0.0, 0.0],
                "velocity": [1.0, 0.0],
                "matched_gt": None,
                "match_dist": None,
            },
        ]
        gt_track_counts = {("scene-1", "bicycle", "gt-1"): 4}

        rec = compute_track_features(track_key, entries, gt_track_counts)

        self.assertEqual(rec["class_name"], "bicycle")
        self.assertEqual(rec["num_frames"], 3)
        self.assertAlmostEqual(rec["tp_ratio"], 2 / 3, places=6)
        self.assertAlmostEqual(rec["purity"], 1.0, places=6)
        self.assertAlmostEqual(rec["dominant_recall"], 0.5, places=6)
        self.assertGreater(rec["score_mean"], 0.5)
        self.assertEqual(rec["is_good_track"], 1)

    def test_prepare_matrix_standardizes_and_tracks(self):
        records = [
            {
                "track_key": "a",
                "score_mean": 0.8,
                "score_last": 0.7,
                "score_std": 0.1,
                "num_frames": 5,
                "duration_sec": 2.0,
                "tp_ratio": 0.9,
                "purity": 1.0,
                "dominant_recall": 0.7,
                "straightness": 0.8,
                "mean_match_dist": 0.3,
                "gap_count": 0,
                "is_good_track": 1,
                "quality_target": 0.8,
            },
            {
                "track_key": "b",
                "score_mean": 0.3,
                "score_last": 0.2,
                "score_std": 0.2,
                "num_frames": 2,
                "duration_sec": 0.5,
                "tp_ratio": 0.2,
                "purity": 0.5,
                "dominant_recall": 0.1,
                "straightness": 0.3,
                "mean_match_dist": 2.5,
                "gap_count": 2,
                "is_good_track": 0,
                "quality_target": 0.1,
            },
        ]
        feature_names = [
            "score_mean",
            "score_last",
            "score_std",
            "num_frames",
            "duration_sec",
            "tp_ratio",
            "purity",
            "dominant_recall",
            "straightness",
            "mean_match_dist",
            "gap_count",
        ]

        Z, y, weights, orig_scores, track_keys, mean, std = prepare_matrix(records, feature_names)

        self.assertEqual(Z.shape, (2, len(feature_names)))
        self.assertListEqual(track_keys, ["a", "b"])
        self.assertListEqual(y.tolist(), [1.0, 0.0])
        self.assertEqual(orig_scores.tolist(), [0.8, 0.3])
        self.assertEqual(len(mean), len(feature_names))
        self.assertEqual(len(std), len(feature_names))
        self.assertTrue((std > 0).all())


if __name__ == "__main__":
    unittest.main()
