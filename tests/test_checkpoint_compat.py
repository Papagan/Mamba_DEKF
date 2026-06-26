import unittest

import numpy as np

from kalmanfilter.checkpoint_compat import adapt_num_class_state_dict


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


if __name__ == "__main__":
    unittest.main()
