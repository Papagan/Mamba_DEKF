import unittest

import torch

from training.losses import JointLoss


class LossesPerSampleTest(unittest.TestCase):
    def test_joint_loss_stores_residual_supervision_config(self):
        loss_fn = JointLoss(residual_supervision={"ENABLED": True, "WEIGHT": 0.2})
        self.assertEqual(loss_fn.residual_supervision["ENABLED"], True)
        self.assertAlmostEqual(loss_fn.residual_supervision["WEIGHT"], 0.2)

    def test_joint_loss_per_sample_mean_matches_scalar_without_weights_or_contrastive(self):
        torch.manual_seed(0)
        dtype = torch.float32
        batch = 2

        pos_x_pred = torch.zeros(batch, 6, 1, dtype=dtype)
        pos_x_pred[:, 0, 0] = torch.tensor([0.2, -0.1], dtype=dtype)
        pos_x_pred[:, 1, 0] = torch.tensor([0.0, 0.3], dtype=dtype)
        pos_P_pred = torch.eye(6, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1) * 0.2

        siz_x_pred = torch.zeros(batch, 3, 1, dtype=dtype)
        siz_x_pred[:, :, 0] = torch.tensor(
            [[1.8, 4.0, 1.5], [0.7, 0.8, 1.6]], dtype=dtype
        )
        siz_P_pred = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1) * 0.15

        ori_x_pred = torch.zeros(batch, 2, 1, dtype=dtype)
        ori_x_pred[:, 0, 0] = torch.tensor([0.1, -0.2], dtype=dtype)
        ori_P_pred = torch.eye(2, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1) * 0.1

        gt_next_pos = torch.tensor([[0.0, 0.1, 0.0], [0.2, 0.1, -0.1]], dtype=dtype)
        gt_next_siz = torch.tensor([[1.7, 4.1, 1.4], [0.8, 0.7, 1.5]], dtype=dtype)
        gt_next_ori = torch.tensor([[0.0], [0.05]], dtype=dtype)
        gt_next_vel = torch.tensor([[0.1, -0.1], [0.0, 0.2]], dtype=dtype)

        R_pos = torch.eye(5, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1) * 0.35
        R_siz = torch.eye(3, dtype=dtype).unsqueeze(0).repeat(batch, 1, 1) * 0.25
        R_ori = torch.ones(batch, 1, 1, dtype=dtype) * 0.4
        kappa_ori = torch.ones(batch, 1, dtype=dtype) * 1.7

        loss_fn = JointLoss(
            w_pos=1.0,
            w_siz=0.5,
            w_ori=0.8,
            w_vel=0.3,
            w_nis=0.0,
            lambda_contrast=0.1,
            physics_scale=7.0,
        )

        scalar_loss, detail = loss_fn(
            pos_x_pred,
            pos_P_pred,
            siz_x_pred,
            siz_P_pred,
            ori_x_pred,
            ori_P_pred,
            gt_next_pos,
            gt_next_siz,
            gt_next_ori,
            None,
            None,
            R_pos=R_pos,
            R_siz=R_siz,
            R_ori=R_ori,
            kappa_ori=kappa_ori,
            gt_next_vel=gt_next_vel,
            use_wrapped_orientation_nll=True,
            return_per_sample=True,
        )

        self.assertIn("loss_state_per_sample", detail)
        self.assertIn("loss_total_per_sample", detail)
        self.assertEqual(tuple(detail["loss_state_per_sample"].shape), (batch,))
        self.assertEqual(tuple(detail["loss_total_per_sample"].shape), (batch,))
        self.assertTrue(
            torch.allclose(
                detail["loss_state_per_sample"].mean(),
                torch.as_tensor(detail["loss_state"], dtype=dtype),
                atol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(detail["loss_total_per_sample"].mean(), scalar_loss, atol=1e-6)
        )


if __name__ == "__main__":
    unittest.main()
