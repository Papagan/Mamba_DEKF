import unittest
import ast
import pathlib
from typing import Dict, List, Tuple

import torch

from kalmanfilter.state_residual import apply_bounded_state_residuals


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_base_tracker_methods(method_names):
    module_path = REPO_ROOT / "tracker" / "base_tracker.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = {
        "apply_bounded_state_residuals": apply_bounded_state_residuals,
        "torch": torch,
        "Dict": Dict,
        "List": List,
        "Tuple": Tuple,
    }
    loaded = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != "Base3DTracker":
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name in method_names:
                exec(ast.get_source_segment(source, child), namespace)
                loaded[child.name] = namespace[child.name]
        break
    missing = [name for name in method_names if name not in loaded]
    if missing:
        raise AssertionError(f"Missing Base3DTracker methods: {missing}")
    return loaded


class StateResidualInferTest(unittest.TestCase):
    def test_tracker_extracts_state_residual_config(self):
        methods = _load_base_tracker_methods(["_state_residual_cfg"])
        tracker = type("Tracker", (), {})()
        tracker.cfg = {"DEKF_BASE_NOISE": {"MAMBA_STATE_RESIDUAL": {"ENABLED": True}}}

        self.assertEqual(methods["_state_residual_cfg"](tracker), {"ENABLED": True})

    def test_apply_state_residual_to_prediction_changes_only_active_track(self):
        methods = _load_base_tracker_methods(
            ["_state_residual_cfg", "_apply_state_residual_to_prediction"]
        )
        tracker = type("Tracker", (), {})()
        tracker.cfg = {
            "DEKF_BASE_NOISE": {
                "MAMBA_STATE_RESIDUAL": {
                    "ENABLED": True,
                    "ACTIVE_CLASS_STATES": {2: ["unmatched"]},
                    "DEFAULT_BOUNDS": {
                        "POS_XY": 0.5,
                        "POS_Z": 0.2,
                        "VEL_XY": 0.4,
                        "YAW": 0.1,
                    },
                }
            }
        }
        tracker._state_residual_cfg = methods["_state_residual_cfg"].__get__(tracker)

        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        mamba_out = {
            "delta_pos": torch.tensor(
                [
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                    [1.0, 1.0, 1.0, 1.0, 1.0, 0.5],
                ]
            )
        }
        class_ids = torch.tensor([2, 3])
        state_buckets = ["unmatched", "unmatched"]

        out_pos, out_siz, out_ori, mask = methods["_apply_state_residual_to_prediction"](
            tracker,
            pos, siz, ori, mamba_out, class_ids, state_buckets
        )

        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.1)
        self.assertTrue(torch.equal(out_pos[1], pos[1]))
        self.assertTrue(torch.equal(out_siz, siz))


if __name__ == "__main__":
    unittest.main()
