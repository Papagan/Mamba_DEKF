import unittest
from pathlib import Path

import yaml
import torch

from tracker.mctrack_motion import MCTrackOriginalPoseMotion, original_mctrack_motion_enabled


REPO_ROOT = Path(__file__).resolve().parents[1]


class _BBox:
    category = "bicycle"
    global_xyz = [1.0, 2.0, 0.5]
    global_velocity = [3.0, 0.0]
    global_acceleration = [0.0, 0.0]
    global_yaw = 0.0


def _cfg(enabled=True):
    return {
        "MCTRACK_ORIGINAL_MOTION": {
            "ENABLED": enabled,
            "APPLY_IN_FILTER_MODES": ["pure_dekf"],
            "ACTIVE_CLASSES": [2, 4],
            "MOTION_MODE": {2: "CTRA", 4: "CA"},
            "CTRA_YAW_RATE_EPS": 1e-3,
        },
        "KALMAN_FILTER_POSE": {
            "MOTION_MODE": {0: "CV", 1: "CV", 2: "CTRA", 3: "CV", 4: "CA", 5: "CV", 6: "CA"},
            "CV": {
                "N": 4,
                "M": 4,
                "NOISE": {2: {"P": [1, 1, 1, 1], "Q": [1, 1, 1, 1], "R": [1, 1, 1, 1]}},
            },
            "CA": {
                "N": 6,
                "M": 2,
                "NOISE": {4: {"P": [1, 1, 1, 1, 1, 1], "Q": [1, 1, 1, 1, 1, 1], "R": [1, 1]}},
            },
            "CTRA": {
                "N": 6,
                "M": 2,
                "NOISE": {2: {"P": [1, 1, 1, 1, 1, 1], "Q": [1, 1, 1, 1, 1, 1], "R": [1, 1]}},
            },
        },
    }


class MCTrackOriginalMotionTest(unittest.TestCase):
    def test_enabled_is_gated_by_filter_mode(self):
        self.assertTrue(original_mctrack_motion_enabled(_cfg(True), "pure_dekf"))
        self.assertFalse(original_mctrack_motion_enabled(_cfg(True), "mamba"))
        self.assertFalse(original_mctrack_motion_enabled(_cfg(False), "pure_dekf"))

    def test_mode_for_class_uses_override(self):
        backend = MCTrackOriginalPoseMotion(_cfg(True), frame_rate=2.0, filter_mode="pure_dekf")
        self.assertEqual(backend.mode_for_class(2), "CTRA")
        self.assertEqual(backend.mode_for_class(4), "CA")
        self.assertEqual(backend.mode_for_class(3), "CV")

    def test_active_classes_gate_filter_creation(self):
        backend = MCTrackOriginalPoseMotion(_cfg(True), frame_rate=2.0, filter_mode="pure_dekf")
        backend.init_track(7, _BBox(), 3)
        self.assertNotIn(7, backend.filters)
        backend.init_track(8, _BBox(), 2)
        self.assertIn(8, backend.filters)

    def test_ctra_predict_is_finite_with_near_zero_yaw_rate(self):
        backend = MCTrackOriginalPoseMotion(_cfg(True), frame_rate=2.0, filter_mode="pure_dekf")
        backend.init_track(7, _BBox(), 2)
        state = backend.predict(7, 0.5)
        self.assertIsNotNone(state)
        self.assertTrue(all(abs(float(value)) < 1e6 for value in state[:2]))

    def test_apply_state_to_pos_tensor_updates_xy_and_velocity_only(self):
        backend = MCTrackOriginalPoseMotion(_cfg(True), frame_rate=2.0, filter_mode="pure_dekf")
        pos_x = torch.zeros(1, 6, 1)
        out = backend.apply_vector_to_pos_tensor([1.0, 2.0, 0.0, 3.0, 0.1, 0.0], "CTRA", pos_x)
        self.assertAlmostEqual(float(out[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1, 0]), 2.0)
        self.assertAlmostEqual(float(out[0, 3, 0]), 3.0)
        self.assertAlmostEqual(float(out[0, 4, 0]), 0.0)
        self.assertAlmostEqual(float(out[0, 2, 0]), 0.0)
        self.assertAlmostEqual(float(out[0, 5, 0]), 0.0)

    def test_nuscenes_motion_configs_use_7_class_mamba_checkpoint_shape(self):
        for name in [
            "nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml",
            "nuscenes_single_stage_mctrack_original_motion_ablation.yaml",
        ]:
            cfg = yaml.safe_load((REPO_ROOT / "config" / name).read_text(encoding="utf-8"))
            self.assertEqual(cfg["MAMBA"]["NUM_CLASSES"], 7)

    def test_original_motion_ablation_only_activates_positive_class(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_original_motion_ablation.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["MCTRACK_ORIGINAL_MOTION"]["ACTIVE_CLASSES"], [3])

    def test_motion_residual_combo_keeps_only_known_positive_gates(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_motion_residual_combo.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["FILTER_MODE"], "mamba_multihead_closure")
        self.assertEqual(cfg["MAMBA"]["NUM_CLASSES"], 7)
        self.assertEqual(cfg["MCTRACK_ORIGINAL_MOTION"]["ACTIVE_CLASSES"], [3])
        self.assertEqual(
            cfg["MCTRACK_ORIGINAL_MOTION"]["APPLY_IN_FILTER_MODES"],
            ["mamba_multihead_closure"],
        )
        self.assertEqual(
            cfg["DEKF_BASE_NOISE"]["MAMBA_STATE_RESIDUAL"]["ACTIVE_CLASS_STATES"],
            {3: ["unmatched"], 5: ["unmatched"]},
        )


if __name__ == "__main__":
    unittest.main()
