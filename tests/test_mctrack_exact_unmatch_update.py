import unittest
import types
import sys

sys.modules.setdefault("pyquaternion", types.SimpleNamespace(Quaternion=object))

from tracker.compat_utils import use_mctrack_exact_unmatch_update
from tracker.trajectory import Trajectory


class MCTrackExactUnmatchUpdateTest(unittest.TestCase):
    def test_exact_unmatch_update_only_enabled_for_configured_mctrack_classes(self):
        cfg = {
            "TRACKER_COMPAT_MODE": "mctrack",
            "MCTRACK_EXACT_UNMATCH_UPDATE": True,
            "MCTRACK_EXACT_UNMATCH_UPDATE_CLASSES": [2, 3, 5, 6],
        }
        self.assertTrue(use_mctrack_exact_unmatch_update(cfg, 2))
        self.assertTrue(use_mctrack_exact_unmatch_update(cfg, 6))
        self.assertFalse(use_mctrack_exact_unmatch_update(cfg, 0))

        cfg["MCTRACK_EXACT_UNMATCH_UPDATE"] = False
        self.assertFalse(use_mctrack_exact_unmatch_update(cfg, 2))

    def test_unmatch_update_prefers_fake_update_state_when_enabled(self):
        class DummyBBox:
            def __init__(self):
                self.category = "bicycle"
                self.frame_id = 0
                self.timestamp = 0.0
                self.det_score = 0.8
                self.raw_det_score = 0.8
                self.is_fake = False
                self.global_xyz = [1.0, 2.0, 0.5]
                self.lwh = [4.0, 1.8, 1.5]
                self.global_yaw = 0.1
                self.global_velocity = [0.5, 0.0]
                self.global_velocity_fusion = [0.5, 0.0]
                self.global_yaw_fusion = 0.1
                self.lwh_fusion = [4.0, 1.8, 1.5]
                self.global_xyz_lwh_yaw = [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1]
                self.global_xyz_lwh_yaw_fusion = [1.0, 2.0, 0.5, 4.0, 1.8, 1.5, 0.1]
                self.global_xyz_lwh_yaw_predict = [1.5, 2.5, 0.5, 4.0, 1.8, 1.5, 0.1]
                self.global_xyz_lwh_yaw_fake_update = [1.8, 2.8, 0.5, 4.0, 1.8, 1.5, 0.1]

        cfg = {
            "FRAME_RATE": 2,
            "TRACKER_COMPAT_MODE": "mctrack",
            "MCTRACK_EXACT_UNMATCH_UPDATE": True,
            "MCTRACK_EXACT_UNMATCH_UPDATE_CLASSES": [2],
            "CATEGORY_MAP_TO_NUMBER": {
                "car": 0,
                "pedestrian": 1,
                "bicycle": 2,
                "motorcycle": 3,
                "bus": 4,
                "trailer": 5,
                "truck": 6,
            },
            "MATCHING": {"BEV": {"COST_MODE": {2: "RO_GDIOU_3D"}}},
            "THRESHOLD": {
                "TRAJECTORY_THRE": {
                    "CACHE_BBOX_LENGTH": {2: 30},
                    "PREDICT_BBOX_LENGTH": {2: 13},
                    "MAX_UNMATCH_LENGTH": {2: 1},
                    "CONFIRMED_TRACK_LENGTH": {2: 1},
                    "DELET_OUT_VIEW_LENGTH": {2: 0},
                    "CONFIRMED_DET_SCORE": {2: 0.7},
                    "CONFIRMED_MATCHED_SCORE": {2: 0.3},
                    "OUTPUT_SCORE": {2: 0.45},
                    "IS_FILTER_PREDICT_BOX": {2: -1},
                }
            },
        }

        traj = Trajectory(track_id=1, init_bbox=DummyBBox(), cfg=cfg)
        traj.unmatch_update(frame_id=1, timestamp=0.5)

        fake_bbox = traj.bboxes[-1]
        self.assertTrue(fake_bbox.is_fake)
        self.assertEqual(fake_bbox.global_xyz, [1.8, 2.8, 0.5])
        self.assertEqual(fake_bbox.global_xyz_lwh_yaw_predict, [1.8, 2.8, 0.5, 4.0, 1.8, 1.5, 0.1])


if __name__ == "__main__":
    unittest.main()
