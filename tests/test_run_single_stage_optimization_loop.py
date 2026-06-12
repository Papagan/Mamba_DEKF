import json
import os
import subprocess
import sys
import tempfile
import unittest

from tools.run_single_stage_optimization_loop import (
    accept_suggested_changes_by_class,
    build_feedback_comparison,
    build_next_base_config_path,
    derive_result_root,
    find_latest_loop_root,
    pick_new_run_dir,
    prune_old_run_dirs,
    stage_complete,
)


class RunSingleStageOptimizationLoopTest(unittest.TestCase):
    def test_script_help_runs_without_module_import_error(self):
        proc = subprocess.run(
            [sys.executable, "tools/run_single_stage_optimization_loop.py", "--help"],
            cwd="/home/alvin/demo/Mamba-DEKF",
            capture_output=True,
            text=True,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr)
        self.assertIn("Run eval -> export -> calibrate -> compare -> suggest", proc.stdout)

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

    def test_find_latest_loop_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = os.path.join(tmpdir, "loop_20260611_100000")
            b = os.path.join(tmpdir, "loop_20260611_100500")
            os.makedirs(a, exist_ok=True)
            os.makedirs(b, exist_ok=True)
            self.assertEqual(find_latest_loop_root(tmpdir), b)

    def test_stage_complete_requires_all_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            a = os.path.join(tmpdir, "a.json")
            b = os.path.join(tmpdir, "b.json")
            with open(a, "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertFalse(stage_complete([a, b]))
            with open(b, "w", encoding="utf-8") as f:
                f.write("{}")
            self.assertTrue(stage_complete([a, b]))

    def test_accept_suggested_changes_by_class_accepts_only_improved_classes(self):
        base_cfg = {
            "CATEGORY_MAP_TO_NUMBER": {"car": 0, "bicycle": 2},
            "THRESHOLD": {
                "INPUT_SCORE": {
                    "ONLINE": {0: 0.20, 2: 0.18},
                    "OFFLINE": {0: 0.20, 2: 0.18},
                },
                "TRAJECTORY_THRE": {
                    "OUTPUT_SCORE": {0: 0.45, 2: 0.40},
                },
            },
            "TRACK_SCORE": {
                "W_DET": {0: 0.35, 2: 0.45},
            },
        }
        candidate_cfg = {
            "CATEGORY_MAP_TO_NUMBER": {"car": 0, "bicycle": 2},
            "THRESHOLD": {
                "INPUT_SCORE": {
                    "ONLINE": {0: 0.17, 2: 0.14},
                    "OFFLINE": {0: 0.17, 2: 0.14},
                },
                "TRAJECTORY_THRE": {
                    "OUTPUT_SCORE": {0: 0.43, 2: 0.31},
                },
            },
            "TRACK_SCORE": {
                "W_DET": {0: 0.37, 2: 0.52},
            },
        }
        suggestion_report = {
            "changes": [
                {"path": "THRESHOLD.INPUT_SCORE", "class_name": "car", "old": 0.20, "new": 0.17},
                {"path": "THRESHOLD.INPUT_SCORE", "class_name": "bicycle", "old": 0.18, "new": 0.14},
                {"path": "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", "class_name": "bicycle", "old": 0.40, "new": 0.31},
                {"path": "TRACK_SCORE.W_DET", "class_name": "bicycle", "old": 0.45, "new": 0.52},
            ]
        }
        feedback = {
            "aggregate": {"amota": {"delta": -0.01}},
            "per_class": {
                "car": {"amota": {"delta": -0.02}},
                "bicycle": {"amota": {"delta": 0.10}},
            },
        }

        merged_cfg, acceptance = accept_suggested_changes_by_class(
            base_cfg,
            candidate_cfg,
            suggestion_report,
            feedback,
        )

        self.assertEqual(merged_cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"][0], 0.20)
        self.assertEqual(merged_cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"][0], 0.20)
        self.assertEqual(merged_cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"][2], 0.14)
        self.assertEqual(merged_cfg["THRESHOLD"]["TRAJECTORY_THRE"]["OUTPUT_SCORE"][2], 0.31)
        self.assertEqual(merged_cfg["TRACK_SCORE"]["W_DET"][2], 0.52)
        self.assertEqual(acceptance["accepted_count"], 3)
        self.assertEqual(acceptance["rejected_count"], 1)

    def test_build_next_base_config_path_prefers_accepted_config_after_regression(self):
        self.assertEqual(
            build_next_base_config_path(
                accepted_config_path="/tmp/accepted.yaml",
                current_config_path="/tmp/current.yaml",
                accepted_count=0,
                rejected_count=4,
            ),
            "/tmp/accepted.yaml",
        )
        self.assertEqual(
            build_next_base_config_path(
                accepted_config_path="/tmp/accepted.yaml",
                current_config_path="/tmp/current.yaml",
                accepted_count=2,
                rejected_count=1,
            ),
            "/tmp/accepted.yaml",
        )


if __name__ == "__main__":
    unittest.main()
