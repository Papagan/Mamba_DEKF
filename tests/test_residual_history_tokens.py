import ast
import pathlib
import sys
import types
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.modules.setdefault("pyquaternion", types.SimpleNamespace(Quaternion=object))

from tracker.trajectory import Trajectory


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class _ResidualTorch:
    Tensor = np.ndarray
    bool = np.bool_

    @staticmethod
    def zeros(*shape, dtype=None, device=None):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        np_dtype = np.bool_ if dtype is _ResidualTorch.bool else np.float32
        return np.zeros(shape, dtype=np_dtype)

    @staticmethod
    def tensor(values, dtype=None, device=None):
        np_dtype = np.bool_ if dtype is _ResidualTorch.bool else np.float32
        return np.asarray(values, dtype=np_dtype)


def _load_class_methods(module_path, class_name, method_names, extra_namespace=None):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = dict(extra_namespace or {})
    loaded = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name in method_names:
                fn_source = ast.get_source_segment(source, child)
                exec(fn_source, namespace)
                loaded[child.name] = namespace[child.name]
        break

    missing = [name for name in method_names if name not in loaded]
    if missing:
        raise AssertionError(f"Missing methods in {module_path}:{class_name}: {missing}")
    return loaded


class DummyBBox:
    def __init__(self, *, frame_id=0, timestamp=0.0, det_score=0.8):
        self.category = "bicycle"
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.det_score = det_score
        self.raw_det_score = det_score
        self.is_fake = False
        self.global_xyz = [1.0, 2.0, 0.5]
        self.lwh = [4.0, 1.8, 1.5]
        self.global_yaw = 0.1
        self.global_velocity = [0.5, 0.0]
        self.global_velocity_fusion = [0.5, 0.0]
        self.global_yaw_fusion = 0.1
        self.lwh_fusion = [4.0, 1.8, 1.5]
        self.global_xyz_lwh_yaw = [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1]
        self.global_xyz_lwh_yaw_fusion = [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1]
        self.global_xyz_lwh_yaw_predict = [1.2, 2.1, 0.5, 4.0, 1.8, 1.5, 0.1]
        self.x1y1x2y2 = [0.0, 0.0, 1.0, 1.0]
        self.x1y1x2y2_fusion = [0.0, 0.0, 1.0, 1.0]
        self.x1y1x2y2_predict = [0.0, 0.0, 1.0, 1.0]
        self.unmatch_length = 0


def _make_cfg():
    return {
        "FRAME_RATE": 2,
        "TRACKER_COMPAT_MODE": "default",
        "CATEGORY_MAP_TO_NUMBER": {
            "car": 0,
            "pedestrian": 1,
            "bicycle": 2,
            "motorcycle": 3,
            "bus": 4,
            "trailer": 5,
            "truck": 6,
        },
        "MATCHING": {"BEV": {"COST_MODE": {2: "RO_GDIOU_3D"}}},
        "THRESHOLD": {
            "TRAJECTORY_THRE": {
                "CACHE_BBOX_LENGTH": {2: 30},
                "PREDICT_BBOX_LENGTH": {2: 13},
                "MAX_UNMATCH_LENGTH": {2: 1},
                "CONFIRMED_TRACK_LENGTH": {2: 2},
                "DELET_OUT_VIEW_LENGTH": {2: 0},
                "CONFIRMED_DET_SCORE": {2: 0.7},
                "CONFIRMED_MATCHED_SCORE": {2: 0.3},
                "OUTPUT_SCORE": {2: 0.45},
                "IS_FILTER_PREDICT_BOX": {2: -1},
            }
        },
    }


def _make_traj_with_residual_history(
    *,
    class_name="bicycle",
    unmatch_length=0,
    matched_flags=None,
):
    if matched_flags is None:
        matched_flags = [True, True, False, True]
    residual_history = []
    for idx, is_matched in enumerate(matched_flags):
        residual_history.append(
            {
                "is_matched": bool(is_matched),
                "pos_residual": [0.1 * (idx + 1), -0.05 * idx, 0.0, 0.2, -0.1],
                "siz_residual": [0.01 * (idx + 1), 0.0, -0.01 * idx],
                "ori_residual": 0.02 * idx,
                "det_score": 0.7 if is_matched else 0.0,
                "timestamp": float(idx),
            }
        )

    bbox = types.SimpleNamespace(
        category=class_name,
        global_xyz=[3.0, 4.0, 0.0],
        global_velocity=[2.0, 0.0],
        global_yaw=0.2,
    )
    return types.SimpleNamespace(
        track_id=17,
        bboxes=[bbox],
        residual_history=residual_history,
        unmatch_length=unmatch_length,
    )


def _build_tracker_stub(history_len=8):
    methods = _load_class_methods(
        REPO_ROOT / "tracker" / "base_tracker.py",
        "Base3DTracker",
        [
            "_class_window_cfg",
            "_resolve_effective_history_len",
            "_encode_residual_token",
            "_extract_residual_token_history",
        ],
        extra_namespace={
            "Trajectory": Trajectory,
            "Tuple": Tuple,
            "List": List,
            "Dict": Dict,
            "Optional": Optional,
            "torch": _ResidualTorch,
            "np": np,
        },
    )
    tracker = types.SimpleNamespace(
        history_len=history_len,
        mamba_input_dim=12,
        device="cpu",
        default_window_cfg={"MIN_HISTORY_LEN": 4, "MAX_HISTORY_LEN": history_len},
        runtime_window_cfg={
            "bicycle": {"MIN_HISTORY_LEN": 3, "MAX_HISTORY_LEN": min(history_len, 6)},
            "motorcycle": {"MIN_HISTORY_LEN": 3, "MAX_HISTORY_LEN": min(history_len, 6)},
        },
    )
    for name, fn in methods.items():
        setattr(tracker, name, types.MethodType(fn, tracker))
    return tracker


class ResidualHistoryTest(unittest.TestCase):
    def test_record_matched_residual_appends_real_residual_entry(self):
        traj = Trajectory(track_id=7, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.record_matched_residual(
            pos_residual=[0.2, -0.1, 0.0, 0.3, -0.2],
            siz_residual=[0.1, 0.0, -0.1],
            ori_residual=0.05,
            det_score=0.61,
            timestamp=1000,
        )

        self.assertEqual(len(traj.residual_history), 1)
        self.assertTrue(traj.residual_history[-1]["is_matched"])
        self.assertEqual(
            traj.residual_history[-1]["pos_residual"],
            [0.2, -0.1, 0.0, 0.3, -0.2],
        )
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.1, 0.0, -0.1])
        self.assertAlmostEqual(traj.residual_history[-1]["ori_residual"], 0.05)
        self.assertAlmostEqual(traj.residual_history[-1]["det_score"], 0.61)
        self.assertEqual(traj.residual_history[-1]["timestamp"], 1000)

    def test_record_coast_residual_appends_masked_placeholder(self):
        traj = Trajectory(track_id=8, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.record_coast_residual(timestamp=2000)

        self.assertEqual(len(traj.residual_history), 1)
        self.assertFalse(traj.residual_history[-1]["is_matched"])
        self.assertEqual(traj.residual_history[-1]["pos_residual"], [0.0] * 5)
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.0] * 3)
        self.assertEqual(traj.residual_history[-1]["ori_residual"], 0.0)
        self.assertEqual(traj.residual_history[-1]["det_score"], 0.0)
        self.assertEqual(traj.residual_history[-1]["timestamp"], 2000)

    def test_update_records_matched_residual_when_provided(self):
        traj = Trajectory(track_id=9, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.update(
            DummyBBox(frame_id=1, timestamp=0.5, det_score=0.9),
            0.2,
            matched_residual={
                "pos": [0.3, -0.2, 0.1, 0.0, 0.4],
                "siz": [0.1, 0.2, -0.3],
                "ori": -0.05,
            },
        )

        self.assertEqual(len(traj.residual_history), 1)
        self.assertTrue(traj.residual_history[-1]["is_matched"])
        self.assertEqual(
            traj.residual_history[-1]["pos_residual"],
            [0.3, -0.2, 0.1, 0.0, 0.4],
        )
        self.assertAlmostEqual(traj.residual_history[-1]["det_score"], 0.9)
        self.assertAlmostEqual(traj.residual_history[-1]["timestamp"], 0.5)

    def test_update_without_matched_residual_keeps_residual_history_unchanged(self):
        traj = Trajectory(track_id=11, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.update(DummyBBox(frame_id=1, timestamp=0.5, det_score=0.9), 0.2)

        self.assertEqual(traj.track_length, 2)
        self.assertEqual(len(traj.bboxes), 2)
        self.assertEqual(traj.matched_scores, [0.2])
        self.assertEqual(traj.residual_history, [])

    def test_update_rejects_malformed_matched_residual_before_mutating_trajectory(self):
        traj = Trajectory(track_id=12, init_bbox=DummyBBox(), cfg=_make_cfg())
        bbox = DummyBBox(frame_id=1, timestamp=0.5, det_score=0.9)

        with self.assertRaises(ValueError):
            traj.update(
                bbox,
                0.2,
                matched_residual={
                    "pos": [0.3, -0.2, 0.1, 0.0],
                    "siz": [0.1, 0.2, -0.3],
                    "ori": -0.05,
                },
            )

        self.assertEqual(traj.track_length, 1)
        self.assertEqual(len(traj.bboxes), 1)
        self.assertEqual(traj.matched_scores, [])
        self.assertEqual(traj.residual_history, [])
        self.assertEqual(traj.last_updated_frame, 0)
        self.assertFalse(hasattr(bbox, "track_id"))

    def test_unmatch_update_records_coast_placeholder(self):
        traj = Trajectory(track_id=10, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.unmatch_update(frame_id=1, timestamp=0.5)

        self.assertEqual(len(traj.residual_history), 1)
        self.assertFalse(traj.residual_history[-1]["is_matched"])
        self.assertEqual(traj.residual_history[-1]["pos_residual"], [0.0] * 5)
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.0] * 3)
        self.assertEqual(traj.residual_history[-1]["ori_residual"], 0.0)
        self.assertEqual(traj.residual_history[-1]["det_score"], 0.0)
        self.assertAlmostEqual(traj.residual_history[-1]["timestamp"], 0.5)

    def test_extract_residual_token_history_uses_recent_residuals_and_masks(self):
        tracker = _build_tracker_stub(history_len=8)
        traj = _make_traj_with_residual_history(
            class_name="bicycle",
            matched_flags=[True, True, False, True],
        )

        tokens, valid_mask, match_mask = tracker._extract_residual_token_history([traj])

        self.assertEqual(tuple(tokens.shape), (1, 8, tracker.mamba_input_dim))
        self.assertEqual(int(valid_mask.sum().item()), 4)
        self.assertEqual(int(match_mask.sum().item()), 3)
        self.assertEqual(valid_mask[0, -4:].tolist(), [True, True, True, True])
        self.assertEqual(match_mask[0, -4:].tolist(), [True, True, False, True])
        self.assertTrue(
            np.allclose(
                tokens[0, -1, :9],
                np.asarray([0.4, -0.15, 0.0, 0.2, -0.1, 0.04, 0.0, -0.03, 0.06], dtype=np.float32),
            )
        )

    def test_effective_history_is_shortened_for_unmatched_bicycle(self):
        tracker = _build_tracker_stub(history_len=8)
        traj = _make_traj_with_residual_history(
            class_name="bicycle",
            unmatch_length=2,
            matched_flags=[True] * 8,
        )

        effective = tracker._resolve_effective_history_len(traj, valid_history_len=8)

        self.assertLessEqual(effective, 4)


if __name__ == "__main__":
    unittest.main()
