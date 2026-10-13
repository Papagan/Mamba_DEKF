import unittest

import torch

from kalmanfilter.mamba_adaptive_kf import TemporalMamba


class TemporalMotionHeadTest(unittest.TestCase):
    def test_returns_transition_residual_consumed_by_online_tracker(self):
        model = TemporalMamba(
            d_model=16,
            d_state=4,
            n_mamba_layers=1,
            embed_dim=8,
            num_classes=7,
            force_gru=True,
        )
        history = torch.randn(3, 4, 12)

        output = model(history, class_ids=torch.tensor([0, 1, 0]))

        self.assertEqual(tuple(output["delta_pos"].shape), (3, 6))

    def test_motion_loss_reaches_online_head_and_backbone(self):
        model = TemporalMamba(
            d_model=16,
            d_state=4,
            n_mamba_layers=1,
            embed_dim=8,
            num_classes=7,
            force_gru=True,
        )
        history = torch.randn(2, 4, 12)

        output = model(history, class_ids=torch.tensor([0, 1]))
        output["delta_pos"].pow(2).mean().backward()

        self.assertIsNotNone(model.head_delta_pos[-1].weight.grad)
        self.assertGreater(float(model.head_delta_pos[-1].weight.grad.abs().sum()), 0.0)
        self.assertIsNotNone(model.input_proj.weight.grad)
        self.assertGreater(float(model.input_proj.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
