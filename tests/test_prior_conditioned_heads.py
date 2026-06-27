import unittest
from unittest import mock

import torch

import kalmanfilter.mamba_adaptive_kf as mamba_adaptive_kf_module
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
    def test_multihead_closure_force_prior_states_defaults_to_matched_tracks(self):
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
                "MAMBA_CLOSURE": {
                    "FORCE_COAST_PRIOR_ONLY": True,
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
            torch.zeros(2, 3, 12),
            class_ids=torch.tensor([0, 0], dtype=torch.long),
            current_range=torch.tensor([0.0, 0.0]),
            detection_driven_mask=torch.tensor([True, True]),
            history_mask=torch.ones(2, 3, dtype=torch.bool),
            history_match_mask=torch.ones(2, 3, dtype=torch.bool),
            state_buckets=["matched", "unmatched"],
            mode="mamba_multihead_closure",
        )

        self.assertTrue(
            torch.allclose(
                torch.diagonal(out["Q_pos"][0:1], dim1=-2, dim2=-1),
                torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]),
            )
        )
        self.assertTrue(
            torch.allclose(
                torch.diagonal(out["Q_pos"][1:2], dim1=-2, dim2=-1),
                torch.tensor([[2.0, 4.0, 6.0, 2.0, 2.5, 3.0]]),
            )
        )

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

    def test_multihead_closure_forwards_conditioning_inputs_into_prior_build(self):
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
                "Q": {"POS": [1.0] * 6, "SIZ": [0.05, 0.06, 0.07], "ORI": [0.1]},
                "R": {
                    "MEAS_MULTIPLIER": 1.0,
                    "POS_STD_XY": [[2.0, 3.0]],
                    "SIZ_STD_LW": [[6.0, 8.0]],
                    "VEL_STD_XY": [[4.0, 5.0]],
                    "ORI_STD": [3.0],
                },
                "CONDITIONAL_NOISE": {"ENABLED": True},
            },
        )

        class _IdentityHeadBank(torch.nn.Module):
            def forward(self, h, class_ids):
                return {
                    "q_pos_xyz": h.new_ones((h.shape[0], 1)),
                    "q_pos_vxyz": h.new_ones((h.shape[0], 1)),
                    "r_pos_xyz": h.new_ones((h.shape[0], 1)),
                    "r_pos_vxy": h.new_ones((h.shape[0], 1)),
                    "r_siz_lw": h.new_ones((h.shape[0], 1)),
                    "r_siz_h": h.new_ones((h.shape[0], 1)),
                    "r_ori": h.new_ones((h.shape[0], 1)),
                }

        model.head_bank = _IdentityHeadBank()
        base_cov = {
            "Q_pos_base": torch.diag_embed(torch.tensor([[1.0] * 6])),
            "Q_siz_base": torch.diag_embed(torch.tensor([[0.05, 0.06, 0.07]])),
            "Q_ori_base": torch.diag_embed(torch.tensor([[0.1, 0.5]])),
            "R_pos_base": torch.diag_embed(torch.tensor([[4.0, 9.0, 6.25, 16.0, 25.0]])),
            "R_siz_base": torch.diag_embed(torch.tensor([[36.0, 64.0, 49.0]])),
            "R_ori_base": torch.diag_embed(torch.tensor([[9.0]])),
        }
        captured = {}

        def _stub_build_base_covariances(**kwargs):
            captured.update(kwargs)
            return {
                key: value.to(dtype=kwargs["dtype"], device=kwargs["device"])
                for key, value in base_cov.items()
            }

        residual_token_history = torch.tensor(
            [[[101.0, -5.0, 2.0, 7.0, -3.0, 1.0, 0.5, 0.2, -0.1, 0.0, 4.0, 0.9]]]
        )

        with mock.patch.object(
            mamba_adaptive_kf_module,
            "build_base_covariances",
            side_effect=_stub_build_base_covariances,
        ):
            out = model(
                residual_token_history,
                class_ids=torch.tensor([0], dtype=torch.long),
                current_range=torch.tensor([42.0]),
                detection_driven_mask=torch.tensor([True]),
                history_mask=torch.ones(1, 1, dtype=torch.bool),
                history_match_mask=torch.ones(1, 1, dtype=torch.bool),
                prior_track_history=torch.ones(1, 1, 12, dtype=torch.float32),
                prior_history_mask=torch.ones(1, 1, dtype=torch.bool),
                prior_history_match_mask=torch.zeros(1, 1, dtype=torch.bool),
                mode="mamba_multihead_closure",
            )

        self.assertIn("track_history", captured)
        self.assertIn("current_range", captured)
        self.assertIn("detection_driven_mask", captured)
        self.assertIn("history_mask", captured)
        self.assertIn("history_match_mask", captured)
        self.assertTrue(torch.allclose(out["prior_covariances"]["Q_pos"], base_cov["Q_pos_base"]))
        self.assertTrue(torch.allclose(out["prior_covariances"]["R_ori"], base_cov["R_ori_base"]))

    def test_multihead_closure_keeps_kappa_outputs_and_legacy_clamp_ceiling(self):
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
                    "ORI_STD": [0.1],
                },
            },
        )

        class _StubHeadBank(torch.nn.Module):
            def forward(self, h, class_ids):
                return {
                    "q_pos_xyz": h.new_ones((h.shape[0], 1)),
                    "q_pos_vxyz": h.new_ones((h.shape[0], 1)),
                    "r_pos_xyz": h.new_ones((h.shape[0], 1)),
                    "r_pos_vxy": h.new_ones((h.shape[0], 1)),
                    "r_siz_lw": h.new_ones((h.shape[0], 1)),
                    "r_siz_h": h.new_ones((h.shape[0], 1)),
                    "r_ori": h.new_full((h.shape[0], 1), 0.01),
                }

        model.head_bank = _StubHeadBank()

        out = model(
            torch.zeros(1, 3, 12),
            class_ids=torch.tensor([0], dtype=torch.long),
            mode="mamba_multihead_closure",
        )

        self.assertIn("kappa_ori", out)
        self.assertIn("kappa_ori_unc", out)
        self.assertEqual(out["kappa_ori"].shape, (1, 1))
        self.assertEqual(out["kappa_ori_unc"].shape, (1, 1))
        self.assertLessEqual(float(out["kappa_ori"].max()), 5.0)
        self.assertGreater(float(out["kappa_ori_unc"].item()), 5.0)
        self.assertTrue(torch.allclose(out["kappa_ori"], torch.tensor([[5.0]])))
        self.assertTrue(torch.allclose(out["R_ori"], torch.tensor([[[0.2]]])))
