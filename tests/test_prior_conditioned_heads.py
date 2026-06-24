import unittest

import torch

from kalmanfilter.mamba_adaptive_kf import TemporalMamba
from kalmanfilter.prior_conditioned_heads import PriorConditionedHeadBank


class PriorConditionedHeadBankTest(unittest.TestCase):
    def test_selects_outputs_for_per_class_heads(self):
        bank = PriorConditionedHeadBank(d_model=4, num_classes=7)
        with torch.no_grad():
            for family_heads in bank.family_heads.values():
                for head in family_heads:
                    head.proj.weight.zero_()
                    head.proj.bias.zero_()

            bank.family_heads["r_pos_xyz"][4].proj.bias.fill_(1.25)
            bank.family_heads["r_pos_xyz"][6].proj.bias.fill_(-1.25)

        h = torch.zeros(2, 4)
        class_ids = torch.tensor([4, 6], dtype=torch.long)
        out = bank(h, class_ids)

        self.assertIn("r_pos_xyz", out)
        self.assertEqual(out["r_pos_xyz"].shape, (2, 1))
        self.assertTrue(torch.allclose(out["r_pos_xyz"][0:1], bank.family_heads["r_pos_xyz"][4](h[0:1])))
        self.assertTrue(torch.allclose(out["r_pos_xyz"][1:2], bank.family_heads["r_pos_xyz"][6](h[1:2])))

    def test_bounded_ratio_stays_within_limits(self):
        bank = PriorConditionedHeadBank(d_model=4, num_classes=7)
        class_ids = torch.tensor([2], dtype=torch.long)

        low = bank(torch.full((1, 4), -1e6), class_ids)["r_ori"]
        high = bank(torch.full((1, 4), 1e6), class_ids)["r_ori"]
        limits = bank.ratio_limits["r_ori"]

        self.assertGreaterEqual(float(low.min()), limits[0] - 1e-5)
        self.assertLessEqual(float(high.max()), limits[1] + 1e-5)


class TemporalMambaPriorConditionedBranchTest(unittest.TestCase):
    def test_multihead_closure_reconstructs_covariances_from_priors(self):
        model = TemporalMamba(
            d_model=8,
            d_state=4,
            d_conv=2,
            expand=2,
            n_mamba_layers=1,
            embed_dim=4,
            num_classes=7,
            force_gru=True,
            base_noise_cfg={
                "Q": {
                    "POS": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                    "SIZ": [0.05, 0.06, 0.07],
                    "ORI": [0.1],
                },
                "R": {
                    "MEAS_MULTIPLIER": 1.0,
                    "POS_STD_XY": [[2.0, 3.0]],
                    "SIZ_STD_LW": [[6.0, 8.0]],
                    "VEL_STD_XY": [[4.0, 5.0]],
                    "ORI_STD": [3.0],
                },
            },
        )

        class _StubHeadBank(torch.nn.Module):
            def forward(self, h, class_ids):
                return {
                    "q_pos_xyz": h.new_full((h.shape[0], 1), 2.0),
                    "q_pos_vxyz": h.new_full((h.shape[0], 1), 0.5),
                    "r_pos_xyz": h.new_full((h.shape[0], 1), 1.5),
                    "r_pos_vxy": h.new_full((h.shape[0], 1), 0.25),
                    "r_siz_lw": h.new_full((h.shape[0], 1), 1.25),
                    "r_siz_h": h.new_full((h.shape[0], 1), 0.5),
                    "r_ori": h.new_full((h.shape[0], 1), 4.0),
                }

        model.head_bank = _StubHeadBank()

        out = model(
            torch.zeros(1, 3, 12),
            class_ids=torch.tensor([0], dtype=torch.long),
            current_range=torch.tensor([0.0]),
            detection_driven_mask=torch.tensor([True]),
            history_mask=torch.ones(1, 3, dtype=torch.bool),
            history_match_mask=torch.ones(1, 3, dtype=torch.bool),
            mode="mamba_multihead_closure",
        )

        self.assertIn("ratios", out)
        self.assertTrue(
            torch.allclose(
                torch.diagonal(out["Q_pos"], dim1=-2, dim2=-1),
                torch.tensor([[2.0, 4.0, 6.0, 2.0, 2.5, 3.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.diagonal(out["R_pos"], dim1=-2, dim2=-1),
                torch.tensor([[6.0, 13.5, 9.375, 4.0, 6.25]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.diagonal(out["R_siz"], dim1=-2, dim2=-1),
                torch.tensor([[45.0, 80.0, 24.5]]),
            )
        )
        self.assertTrue(torch.allclose(out["R_ori"], torch.tensor([[[36.0]]])))
        self.assertTrue(torch.allclose(out["kappa_ori"], torch.tensor([[1.0 / 36.0]])))
