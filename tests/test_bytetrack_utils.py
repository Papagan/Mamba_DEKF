import unittest
from types import SimpleNamespace

from tracker.bytetrack_utils import (
    classify_bytetrack_score,
    split_bytetrack_detections,
)


class ByteTrackUtilsTest(unittest.TestCase):
    def test_classify_score_supports_tentative_bucket(self):
        self.assertEqual(classify_bytetrack_score(0.62, 0.7, 0.5), "tentative")
        self.assertEqual(classify_bytetrack_score(0.75, 0.7, 0.5), "high")
        self.assertEqual(classify_bytetrack_score(0.25, 0.7, 0.5), "low")
        self.assertIsNone(classify_bytetrack_score(0.05, 0.7, 0.5))

    def test_tentative_bucket_is_disabled_when_equal_to_birth(self):
        self.assertEqual(classify_bytetrack_score(0.62, 0.7, 0.7), "low")
        self.assertEqual(classify_bytetrack_score(0.62, 0.7, None), "low")

    def test_split_respects_per_class_thresholds(self):
        dets = [
            SimpleNamespace(category="car", det_score=0.45),
            SimpleNamespace(category="car", det_score=0.30),
            SimpleNamespace(category="car", det_score=0.08),
            SimpleNamespace(category="bicycle", det_score=0.43),
            SimpleNamespace(category="bicycle", det_score=0.18),
        ]
        category_map = {"car": 0, "bicycle": 2}
        birth_cfg = {0: 0.4, 2: 0.45}
        tentative_cfg = {0: 0.25, 2: 0.35}

        high, tentative, low = split_bytetrack_detections(
            dets,
            category_map=category_map,
            birth_cfg=birth_cfg,
            tentative_birth_cfg=tentative_cfg,
            low_score_floor=0.1,
        )

        self.assertEqual(high, [0])
        self.assertEqual(tentative, [1, 3])
        self.assertEqual(low, [4])


if __name__ == "__main__":
    unittest.main()
