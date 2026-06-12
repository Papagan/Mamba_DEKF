import json
import os
import tempfile
import unittest

from tools.compare_nuscenes_results import build_comparison_payload, maybe_load_or_eval_orig_summary


class CompareNuScenesResultsTest(unittest.TestCase):
    def test_build_comparison_payload(self):
        orig = {
            "amota": 0.7,
            "recall": 0.8,
            "mota": 0.6,
            "label_metrics": {
                "amota": {"car": 0.8, "truck": 0.5},
                "recall": {"car": 0.9, "truck": 0.6},
                "mota": {"car": 0.7, "truck": 0.4},
            },
        }
        cal = {
            "amota": 0.72,
            "recall": 0.82,
            "mota": 0.61,
            "label_metrics": {
                "amota": {"car": 0.79, "truck": 0.58},
                "recall": {"car": 0.91, "truck": 0.63},
                "mota": {"car": 0.71, "truck": 0.45},
            },
        }

        payload = build_comparison_payload(orig, cal, class_names=["car", "truck"])

        self.assertAlmostEqual(payload["aggregate"]["amota"]["delta"], 0.02, places=6)
        self.assertAlmostEqual(payload["aggregate"]["recall"]["delta"], 0.02, places=6)
        self.assertAlmostEqual(payload["aggregate"]["mota"]["delta"], 0.01, places=6)
        self.assertAlmostEqual(payload["per_class"]["truck"]["amota"]["delta"], 0.08, places=6)
        self.assertAlmostEqual(payload["per_class"]["car"]["amota"]["delta"], -0.01, places=6)

    def test_maybe_load_or_eval_orig_summary_prefers_existing_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            summary_path = os.path.join(tmpdir, "metrics_summary.json")
            payload = {"amota": 0.7, "label_metrics": {"amota": {}}}
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            loaded = maybe_load_or_eval_orig_summary(
                result_path="/tmp/results.json",
                output_dir=os.path.join(tmpdir, "unused"),
                existing_summary_path=summary_path,
                reuse_existing=False,
                nusc_dataroot="/tmp/nusc",
                version="v1.0-trainval",
                eval_set="val",
            )

            self.assertEqual(loaded["amota"], 0.7)


if __name__ == "__main__":
    unittest.main()
