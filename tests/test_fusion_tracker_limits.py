import tempfile
import unittest
from pathlib import Path

import yaml

from tools.augment_tracklet_cache_with_fusion import (
    _lookup_class_limit,
    build_tracker_limits,
)


class FusionTrackerLimitsTest(unittest.TestCase):
    def test_uses_per_class_runtime_predict_lengths(self):
        with tempfile.TemporaryDirectory() as tmp:
            runtime_path = Path(tmp) / "runtime.yaml"
            runtime_path.write_text(
                yaml.safe_dump({
                    "THRESHOLD": {
                        "TRAJECTORY_THRE": {
                            "PREDICT_BBOX_LENGTH": {0: 17, 1: 7, 5: 9},
                        }
                    }
                }),
                encoding="utf-8",
            )
            limits = build_tracker_limits({
                "DATA": {"FUSION_MAX_PREDICT_LEN": 3},
                "TRAINING": {"INFERENCE_CONFIG": str(runtime_path)},
            })

        self.assertEqual(limits["PREDICT_BBOX_LENGTH"], {0: 17, 1: 7, 5: 9})
        self.assertEqual(_lookup_class_limit(limits["PREDICT_BBOX_LENGTH"], "car", 0, 3), 17)
        self.assertEqual(_lookup_class_limit(limits["PREDICT_BBOX_LENGTH"], "trailer", 5, 3), 9)

    def test_explicit_training_override_wins(self):
        limits = build_tracker_limits({
            "DATA": {
                "FUSION_MAX_PREDICT_LEN": 3,
                "FUSION_MAX_PREDICT_LEN_BY_CLASS": {"car": 4, "pedestrian": 2},
            },
            "TRAINING": {"INFERENCE_CONFIG": "/does/not/need/to/exist.yaml"},
        })

        self.assertEqual(_lookup_class_limit(limits["PREDICT_BBOX_LENGTH"], "car", 0, 3), 4)
        self.assertEqual(_lookup_class_limit(limits["PREDICT_BBOX_LENGTH"], "pedestrian", 1, 3), 2)


if __name__ == "__main__":
    unittest.main()
