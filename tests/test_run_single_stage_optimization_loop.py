import json
import os
import tempfile
import unittest

from tools.run_single_stage_optimization_loop import (
    build_feedback_comparison,
    derive_result_root,
    pick_new_run_dir,
    prune_old_run_dirs,
)


class RunSingleStageOptimizationLoopTest(unittest.TestCase):
    def test_derive_result_root(self):
        cfg = {
            "SAVE_PATH": "/root/autodl-tmp/results/nuscenes/",
            "DATASET": "nuscenes",
        }
        self.assertEqual(derive_result_root(cfg), "/root/autodl-tmp/results/nuscenes/nuscenes")

    def test_pick_new_run_dir_prefers_newest_new_directory(self):
        before = {"/tmp/a", "/tmp/b"}
        after = {"/tmp/a", "/tmp/b", "/tmp/c", "/tmp/d"}
        mtimes = {
            "/tmp/c": 10.0,
            "/tmp/d": 20.0,
        }
        picked = pick_new_run_dir(before, after, mtime_lookup=mtimes.get)
        self.assertEqual(picked, "/tmp/d")

    def test_build_feedback_comparison(self):
        prev_summary = {
            "amota": 0.68,
            "recall": 0.70,
            "mota": 0.60,
            "label_metrics": {
                "amota": {"bicycle": 0.40, "car": 0.80},
                "recall": {"bicycle": 0.50, "car": 0.85},
                "mota": {"bicycle": 0.30, "car": 0.70},
            },
        }
        curr_summary = {
            "amota": 0.71,
            "recall": 0.74,
            "mota": 0.66,
            "label_metrics": {
                "amota": {"bicycle": 0.55, "car": 0.82},
                "recall": {"bicycle": 0.66, "car": 0.88},
                "mota": {"bicycle": 0.48, "car": 0.75},
            },
        }
        payload = build_feedback_comparison(prev_summary, curr_summary, ["bicycle", "car"], ["amota", "recall", "mota"])
        self.assertAlmostEqual(payload["aggregate"]["amota"]["delta"], 0.03)
        self.assertAlmostEqual(payload["per_class"]["bicycle"]["amota"]["delta"], 0.15)

    def test_prune_old_run_dirs_keeps_latest_two(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dirs = []
            for name in ["r1", "r2", "r3"]:
                path = os.path.join(tmpdir, name)
                os.makedirs(path, exist_ok=True)
                with open(os.path.join(path, "results.json"), "w", encoding="utf-8") as f:
                    json.dump({}, f)
                run_dirs.append(path)

            kept = prune_old_run_dirs(run_dirs, keep_last=2)

            self.assertEqual(len(kept), 2)
            self.assertFalse(os.path.exists(run_dirs[0]))
            self.assertTrue(os.path.exists(run_dirs[1]))
            self.assertTrue(os.path.exists(run_dirs[2]))


if __name__ == "__main__":
    unittest.main()
