import ast
import pathlib
import unittest

import torch
import torch.nn.functional as F

from training.losses import wrap_to_pi_torch


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_train_function(name):
    module_path = REPO_ROOT / "training" / "train.py"
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = {
        "torch": torch,
        "F": F,
        "wrap_to_pi_torch": wrap_to_pi_torch,
    }
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            exec(ast.get_source_segment(source, node), namespace)
            return namespace[name]
    raise AssertionError(f"Missing function: {name}")


class ResidualSupervisionLossTest(unittest.TestCase):
    def test_initial_pos_state_does_not_treat_yaw_residual_as_vz(self):
        fn = _load_train_function("_build_initial_pos_state_with_residual")
        obs_pos = torch.zeros(1, 6)
        delta = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [9.0]]])

        out = fn(obs_pos, delta, apply_residual=True)

        self.assertEqual(out[0, :, 0].tolist(), [1.0, 2.0, 3.0, 4.0, 5.0, 0.0])

    def test_initial_pos_state_skips_delta_when_direct_residual_supervision_is_active(self):
        fn = _load_train_function("_build_initial_pos_state_with_residual")
        obs_pos = torch.zeros(1, 6)
        delta = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0], [9.0]]])

        out = fn(obs_pos, delta, apply_residual=False)

        self.assertEqual(out[0, :, 0].tolist(), [0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_cv_prior_state_uses_deterministic_no_residual_prediction(self):
        fn = _load_train_function("_build_cv_prior_states_for_residual_target")
        obs_pos = torch.tensor([[1.0, 2.0, 0.5, 0.4, -0.2, 0.1]])
        obs_ori = torch.tensor([[0.2, 0.05]])
        dt = torch.tensor([0.5])

        pos_prior, ori_prior = fn(obs_pos, obs_ori, dt)

        expected_pos = torch.tensor([1.2, 1.9, 0.55, 0.4, -0.2, 0.1])
        self.assertTrue(torch.allclose(pos_prior[0, :, 0], expected_pos))
        self.assertAlmostEqual(float(ori_prior[0, 0, 0]), 0.225)
        self.assertAlmostEqual(float(ori_prior[0, 1, 0]), 0.05)

    def test_disabled_residual_supervision_returns_zero(self):
        fn = _load_train_function("_compute_residual_supervision_loss")
        delta = torch.ones(2, 6)
        pos_pred = torch.zeros(2, 6, 1)
        ori_pred = torch.zeros(2, 2, 1)
        gt_pos = torch.zeros(2, 3)
        gt_vel = torch.zeros(2, 2)
        gt_ori = torch.zeros(2, 1)

        loss, detail = fn(
            delta,
            pos_pred,
            ori_pred,
            gt_pos,
            gt_vel,
            gt_ori,
            torch.tensor([True, True]),
            {"ENABLED": False},
        )

        self.assertEqual(float(loss.item()), 0.0)
        self.assertEqual(detail["loss_residual"], 0.0)
        self.assertEqual(detail["_loss_residual_per_sample"].tolist(), [0.0, 0.0])

    def test_matching_target_residual_has_lower_loss(self):
        fn = _load_train_function("_compute_residual_supervision_loss")
        pos_pred = torch.zeros(1, 6, 1)
        ori_pred = torch.zeros(1, 2, 1)
        gt_pos = torch.tensor([[0.2, -0.1, 0.05]])
        gt_vel = torch.tensor([[0.3, -0.2]])
        gt_ori = torch.tensor([[0.1]])
        target_delta = torch.tensor([[0.2, -0.1, 0.05, 0.3, -0.2, 0.1]])
        bad_delta = torch.zeros(1, 6)
        cfg = {
            "ENABLED": True,
            "WEIGHT": 1.0,
            "W_POS_XY": 1.0,
            "W_POS_Z": 1.0,
            "W_VEL_XY": 1.0,
            "W_YAW": 1.0,
        }

        good_loss, _ = fn(
            target_delta,
            pos_pred,
            ori_pred,
            gt_pos,
            gt_vel,
            gt_ori,
            torch.tensor([True]),
            cfg,
        )
        bad_loss, _ = fn(
            bad_delta,
            pos_pred,
            ori_pred,
            gt_pos,
            gt_vel,
            gt_ori,
            torch.tensor([True]),
            cfg,
        )

        self.assertLess(float(good_loss.item()), float(bad_loss.item()))
        self.assertAlmostEqual(float(good_loss.item()), 0.0, places=6)

    def test_yaw_target_uses_wrapped_angle_difference(self):
        fn = _load_train_function("_compute_residual_supervision_loss")
        pos_pred = torch.zeros(1, 6, 1)
        ori_pred = torch.tensor([[[3.13], [0.0]]])
        gt_pos = torch.zeros(1, 3)
        gt_vel = torch.zeros(1, 2)
        gt_ori = torch.tensor([[-3.13]])
        wrapped_yaw_delta = wrap_to_pi_torch(gt_ori[:, 0] - ori_pred[:, 0, 0])
        delta = torch.zeros(1, 6)
        delta[:, 5] = wrapped_yaw_delta

        loss, _ = fn(
            delta,
            pos_pred,
            ori_pred,
            gt_pos,
            gt_vel,
            gt_ori,
            torch.tensor([True]),
            {
                "ENABLED": True,
                "WEIGHT": 1.0,
                "W_POS_XY": 0.0,
                "W_POS_Z": 0.0,
                "W_VEL_XY": 0.0,
                "W_YAW": 1.0,
            },
        )

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)


if __name__ == "__main__":
    unittest.main()
