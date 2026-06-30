import math
import unittest

import torch

from kalmanfilter.state_residual import apply_bounded_state_residuals


class StateResidualTest(unittest.TestCase):
    def test_disabled_returns_original_tensors(self):
        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        residual = torch.ones(2, 6)

        out_pos, out_siz, out_ori, mask = apply_bounded_state_residuals(
            pos,
            siz,
            ori,
            residual,
            class_ids=torch.tensor([2, 3]),
            state_buckets=["unmatched", "unmatched"],
            cfg={"ENABLED": False},
        )

        self.assertTrue(torch.equal(out_pos, pos))
        self.assertTrue(torch.equal(out_siz, siz))
        self.assertTrue(torch.equal(out_ori, ori))
        self.assertFalse(bool(mask.any().item()))

    def test_only_active_class_state_is_modified(self):
        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        residual = torch.tensor(
            [
                [2.0, -2.0, 1.0, 5.0, -5.0, 0.4],
                [2.0, 2.0, 1.0, 5.0, 5.0, 0.4],
            ]
        )

        out_pos, out_siz, out_ori, mask = apply_bounded_state_residuals(
            pos,
            siz,
            ori,
            residual,
            class_ids=torch.tensor([2, 5]),
            state_buckets=["unmatched", "unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {2: ["unmatched"]},
                "DEFAULT_BOUNDS": {
                    "POS_XY": 0.3,
                    "POS_Z": 0.1,
                    "VEL_XY": 0.5,
                    "YAW": 0.2,
                },
            },
        )

        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.3)
        self.assertAlmostEqual(float(out_pos[0, 1, 0]), -0.3)
        self.assertAlmostEqual(float(out_pos[0, 2, 0]), 0.1)
        self.assertAlmostEqual(float(out_pos[0, 3, 0]), 0.5)
        self.assertAlmostEqual(float(out_pos[0, 4, 0]), -0.5)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.2)
        self.assertTrue(torch.equal(out_pos[1], pos[1]))
        self.assertTrue(torch.equal(out_ori[1], ori[1]))
        self.assertTrue(torch.equal(out_siz, siz))

    def test_class_bounds_override_default_bounds(self):
        pos = torch.zeros(1, 6, 1)
        siz = torch.zeros(1, 3, 1)
        ori = torch.zeros(1, 2, 1)
        residual = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        out_pos, _, out_ori, _ = apply_bounded_state_residuals(
            pos,
            siz,
            ori,
            residual,
            class_ids=torch.tensor([5]),
            state_buckets=["unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                "DEFAULT_BOUNDS": {
                    "POS_XY": 0.3,
                    "POS_Z": 0.2,
                    "VEL_XY": 0.4,
                    "YAW": 0.2,
                },
                "CLASS_BOUNDS": {
                    5: {
                        "unmatched": {
                            "POS_XY": 0.8,
                            "POS_Z": 0.4,
                            "VEL_XY": 0.2,
                            "YAW": 0.05,
                        }
                    }
                },
            },
        )

        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.8)
        self.assertAlmostEqual(float(out_pos[0, 2, 0]), 0.4)
        self.assertAlmostEqual(float(out_pos[0, 3, 0]), 0.2)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.05)

    def test_yaw_is_wrapped_after_residual(self):
        pos = torch.zeros(1, 6, 1)
        siz = torch.zeros(1, 3, 1)
        ori = torch.tensor([[[math.pi - 0.01], [0.0]]])
        residual = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])

        _, _, out_ori, _ = apply_bounded_state_residuals(
            pos,
            siz,
            ori,
            residual,
            class_ids=torch.tensor([3]),
            state_buckets=["unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {3: ["unmatched"]},
                "DEFAULT_BOUNDS": {
                    "POS_XY": 0.3,
                    "POS_Z": 0.2,
                    "VEL_XY": 0.4,
                    "YAW": 0.2,
                },
            },
        )

        self.assertAlmostEqual(float(out_ori[0, 0, 0]), -2.9515927, places=5)


if __name__ == "__main__":
    unittest.main()
