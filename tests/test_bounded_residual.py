import unittest

from kalmanfilter.bounded_residual import (
    STATE_MATCHED,
    STATE_UNMATCHED,
    PROFILE_STABLE_LARGE,
    infer_state_bucket,
    map_class_to_profile,
    clamp_ratio_value,
)


class BoundedResidualHelperTest(unittest.TestCase):
    def test_map_class_to_profile(self):
        self.assertEqual(map_class_to_profile(0), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(4), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(2), "agile_weak")
        self.assertEqual(map_class_to_profile(3), "agile_weak")
        self.assertEqual(map_class_to_profile(5), "heavy_long")
        self.assertEqual(map_class_to_profile(6), "heavy_long")
        self.assertEqual(map_class_to_profile(1), "human")

    def test_infer_state_bucket(self):
        self.assertEqual(infer_state_bucket(0), STATE_MATCHED)
        self.assertEqual(infer_state_bucket(2), STATE_UNMATCHED)

    def test_clamp_ratio_value(self):
        self.assertEqual(clamp_ratio_value(0.5, min_ratio=0.8, max_ratio=1.6), 0.8)
        self.assertEqual(clamp_ratio_value(1.2, min_ratio=0.8, max_ratio=1.6), 1.2)
        self.assertEqual(clamp_ratio_value(2.4, min_ratio=0.8, max_ratio=1.6), 1.6)
