import pathlib
import unittest

import yaml


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


class MambaMultiheadClosureConfigTest(unittest.TestCase):
    def test_train_config_contains_class_windows_for_all_7_classes(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        self.assertEqual(cfg["FILTER_MODE"], "mamba_multihead_closure")
        class_window = cfg["DATA"]["CLASS_WINDOW"]
        for key in ["car", "pedestrian", "bicycle", "motorcycle", "bus", "trailer", "truck"]:
            self.assertIn(key, class_window)

    def test_train_config_contains_closure_orientation_curriculum_knobs(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        closure_cfg = cfg["BASE_NOISE"]["MAMBA_CLOSURE"]
        for key in [
            "ORI_WARMUP_EPOCHS",
            "ORI_TRANSITION_EPOCHS",
            "ORI_STATE_WEIGHT",
            "ORI_WRAPPED_NLL_WEIGHT",
            "ORI_SATURATION_REG_WEIGHT",
            "ORI_MAX_EFFECTIVE_KAPPA",
        ]:
            self.assertIn(key, closure_cfg)

    def test_train_config_uses_static_closure_prior_for_training(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        closure_cfg = cfg["BASE_NOISE"]["MAMBA_CLOSURE"]
        self.assertFalse(closure_cfg["USE_CONDITIONAL_PRIOR"])
        self.assertTrue(closure_cfg["TRAIN_ALL_CLASS_STATES"])
        self.assertEqual(closure_cfg["FORCE_PRIOR_STATES"], ["matched"])
        self.assertEqual(closure_cfg["ACTIVE_CLASS_STATES"], {})

    def test_train_config_uses_recursive_motion_rollout_instead_of_one_step_auxiliary(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        residual_loss = cfg["LOSS"]["RESIDUAL_SUPERVISION"]
        residual_runtime = cfg["BASE_NOISE"]["MAMBA_STATE_RESIDUAL"]
        motion_rollout = cfg["BASE_NOISE"]["MOTION_ROLLOUT"]
        self.assertFalse(residual_loss["ENABLED"])
        self.assertEqual(residual_loss["WEIGHT"], 0.0)
        self.assertTrue(residual_runtime["ENABLED"])
        self.assertEqual(residual_runtime["BOUNDING_MODE"], "tanh")
        self.assertTrue(motion_rollout["ENABLED"])
        self.assertGreater(motion_rollout["TRAIN_OPEN_LOOP_PROB"], 0.5)
        self.assertTrue(motion_rollout["VALIDATION_OPEN_LOOP"])

    def test_branch_config_keeps_frozen_baseline_path_separate(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["FILTER_MODE"], "mamba_multihead_closure")
        self.assertEqual(cfg["TRACKER_COMPAT_MODE"], "mctrack")
        self.assertEqual(cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]["MODE"], "prior_conditioned_multihead")
        self.assertTrue(cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]["FORCE_COAST_PRIOR_ONLY"])

    def test_audit_config_enables_noise_audit_for_closure_branch(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml")
            .read_text(encoding="utf-8")
        )
        self.assertEqual(cfg["FILTER_MODE"], "mamba_multihead_closure")
        self.assertTrue(cfg["AUDIT"]["NOISE_AUDIT"]["ENABLED"])
        self.assertEqual(cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]["MODE"], "prior_conditioned_multihead")

    def test_eval_closure_config_is_all_prior_equivalence_guard(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml")
            .read_text(encoding="utf-8")
        )
        closure_cfg = cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]
        self.assertFalse(closure_cfg["USE_CONDITIONAL_PRIOR"])
        self.assertEqual(closure_cfg["FORCE_PRIOR_STATES"], ["matched", "unmatched"])
        self.assertEqual(closure_cfg["ACTIVE_CLASS_STATES"], {})

    def test_frozen_guard_disables_residual_but_motion_branch_enables_it(self):
        eval_cfg = yaml.safe_load(
            (
                REPO_ROOT
                / "config"
                / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml"
            ).read_text(encoding="utf-8")
        )
        motion_cfg = yaml.safe_load(
            (
                REPO_ROOT
                / "config"
                / "nuscenes_single_stage_mctrack_motion_residual_combo.yaml"
            ).read_text(encoding="utf-8")
        )

        eval_residual = eval_cfg["DEKF_BASE_NOISE"]["MAMBA_STATE_RESIDUAL"]
        motion_residual = motion_cfg["DEKF_BASE_NOISE"]["MAMBA_STATE_RESIDUAL"]
        self.assertFalse(eval_residual["ENABLED"])
        self.assertEqual(eval_residual["ACTIVE_CLASS_STATES"], {})
        self.assertTrue(motion_residual["ENABLED"])
        self.assertEqual(motion_cfg["MAMBA"]["INPUT_HISTORY_SOURCE"], "track")
        self.assertFalse(motion_cfg["MCTRACK_ORIGINAL_MOTION"]["ENABLED"])
        for class_id in range(7):
            self.assertEqual(
                motion_residual["ACTIVE_CLASS_STATES"][class_id],
                ["matched", "unmatched"],
            )

    def test_audit_keeps_agile_unmatched_gate_for_future_retraining(self):
        cfg = yaml.safe_load(
            (
                REPO_ROOT
                / "config"
                / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml"
            ).read_text(encoding="utf-8")
        )
        active = cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]["ACTIVE_CLASS_STATES"]
        self.assertEqual(active[2], ["unmatched"])
        self.assertEqual(active[3], ["unmatched"])
        self.assertEqual(set(active.keys()), {2, 3})
