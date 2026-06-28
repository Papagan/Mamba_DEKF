import unittest

import numpy as np

from kalmanfilter.checkpoint_compat import (
    adapt_num_class_state_dict,
    filter_heads_only_state_dict,
)


class CheckpointCompatTest(unittest.TestCase):
    def test_adapt_num_class_state_dict_truncates_checkpoint_rows_to_model_rows(self):
        state_dict = {
            "raw_q_siz.weight": np.arange(30, dtype=np.float32).reshape(10, 3),
            "raw_r_siz.weight": (np.arange(30, dtype=np.float32).reshape(10, 3) + 100.0),
        }
        model_state_dict = {
            "raw_q_siz.weight": np.full((7, 3), -1.0, dtype=np.float32),
            "raw_r_siz.weight": np.full((7, 3), -2.0, dtype=np.float32),
        }

        adapted, adapted_keys = adapt_num_class_state_dict(state_dict, model_state_dict)

        self.assertEqual(set(adapted_keys), {"raw_q_siz.weight", "raw_r_siz.weight"})
        self.assertEqual(adapted["raw_q_siz.weight"].shape, (7, 3))
        self.assertEqual(adapted["raw_r_siz.weight"].shape, (7, 3))
        np.testing.assert_array_equal(
            adapted["raw_q_siz.weight"],
            state_dict["raw_q_siz.weight"][:7],
        )
        np.testing.assert_array_equal(
            adapted["raw_r_siz.weight"],
            state_dict["raw_r_siz.weight"][:7],
        )

    def test_adapt_num_class_state_dict_preserves_model_init_for_extra_rows(self):
        state_dict = {
            "raw_q_siz.weight": np.arange(15, dtype=np.float32).reshape(5, 3),
        }
        model_state_dict = {
            "raw_q_siz.weight": np.full((7, 3), -1.0, dtype=np.float32),
        }

        adapted, adapted_keys = adapt_num_class_state_dict(state_dict, model_state_dict)

        self.assertEqual(adapted_keys, ["raw_q_siz.weight"])
        np.testing.assert_array_equal(
            adapted["raw_q_siz.weight"][:5],
            state_dict["raw_q_siz.weight"],
        )
        np.testing.assert_array_equal(
            adapted["raw_q_siz.weight"][5:],
            model_state_dict["raw_q_siz.weight"][5:],
        )

    def test_filter_heads_only_state_dict_skips_backbones_and_shape_mismatches(self):
        state_dict = {
            "fallback_gru.weight_ih_l0": np.ones((3, 3), dtype=np.float32),
            "mamba_layers.0.in_proj.weight": np.ones((3, 3), dtype=np.float32),
            "head_bank.family_heads.r_pos_xyz.0.proj.weight": np.ones((1, 4), dtype=np.float32),
            "input_proj.weight": np.ones((4, 12), dtype=np.float32),
            "raw_q_siz.weight": np.ones((7, 3), dtype=np.float32),
            "bad_shape.weight": np.ones((9, 9), dtype=np.float32),
        }
        model_state_dict = {
            "head_bank.family_heads.r_pos_xyz.0.proj.weight": np.zeros((1, 4), dtype=np.float32),
            "input_proj.weight": np.zeros((4, 12), dtype=np.float32),
            "raw_q_siz.weight": np.zeros((7, 3), dtype=np.float32),
            "bad_shape.weight": np.zeros((8, 8), dtype=np.float32),
        }

        filtered, skipped = filter_heads_only_state_dict(state_dict, model_state_dict)

        self.assertEqual(
            set(filtered),
            {
                "head_bank.family_heads.r_pos_xyz.0.proj.weight",
                "input_proj.weight",
                "raw_q_siz.weight",
            },
        )
        self.assertIn("fallback_gru.weight_ih_l0", skipped["backbone"])
        self.assertIn("mamba_layers.0.in_proj.weight", skipped["backbone"])
        self.assertIn("bad_shape.weight", skipped["shape_mismatch"])


if __name__ == "__main__":
    unittest.main()
