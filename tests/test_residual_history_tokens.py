import sys
import types
import unittest

sys.modules.setdefault("pyquaternion", types.SimpleNamespace(Quaternion=object))

from tracker.trajectory import Trajectory


class DummyBBox:
    def __init__(self, *, frame_id=0, timestamp=0.0, det_score=0.8):
        self.category = "bicycle"
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.det_score = det_score
        self.raw_det_score = det_score
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
        self.global_xyz_lwh_yaw_predict = [1.2, 2.1, 0.5, 4.0, 1.8, 1.5, 0.1]
        self.x1y1x2y2 = [0.0, 0.0, 1.0, 1.0]
        self.x1y1x2y2_fusion = [0.0, 0.0, 1.0, 1.0]
        self.x1y1x2y2_predict = [0.0, 0.0, 1.0, 1.0]
        self.unmatch_length = 0


def _make_cfg():
    return {
        "FRAME_RATE": 2,
        "TRACKER_COMPAT_MODE": "default",
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
                "CONFIRMED_TRACK_LENGTH": {2: 2},
                "DELET_OUT_VIEW_LENGTH": {2: 0},
                "CONFIRMED_DET_SCORE": {2: 0.7},
                "CONFIRMED_MATCHED_SCORE": {2: 0.3},
                "OUTPUT_SCORE": {2: 0.45},
                "IS_FILTER_PREDICT_BOX": {2: -1},
            }
        },
    }


class ResidualHistoryTest(unittest.TestCase):
    def test_record_matched_residual_appends_real_residual_entry(self):
        traj = Trajectory(track_id=7, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.record_matched_residual(
            pos_residual=[0.2, -0.1, 0.0, 0.3, -0.2],
            siz_residual=[0.1, 0.0, -0.1],
            ori_residual=0.05,
            det_score=0.61,
            timestamp=1000,
        )

        self.assertEqual(len(traj.residual_history), 1)
        self.assertTrue(traj.residual_history[-1]["is_matched"])
        self.assertEqual(
            traj.residual_history[-1]["pos_residual"],
            [0.2, -0.1, 0.0, 0.3, -0.2],
        )
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.1, 0.0, -0.1])
        self.assertAlmostEqual(traj.residual_history[-1]["ori_residual"], 0.05)
        self.assertAlmostEqual(traj.residual_history[-1]["det_score"], 0.61)
        self.assertEqual(traj.residual_history[-1]["timestamp"], 1000)

    def test_record_coast_residual_appends_masked_placeholder(self):
        traj = Trajectory(track_id=8, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.record_coast_residual(timestamp=2000)

        self.assertEqual(len(traj.residual_history), 1)
        self.assertFalse(traj.residual_history[-1]["is_matched"])
        self.assertEqual(traj.residual_history[-1]["pos_residual"], [0.0] * 5)
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.0] * 3)
        self.assertEqual(traj.residual_history[-1]["ori_residual"], 0.0)
        self.assertEqual(traj.residual_history[-1]["det_score"], 0.0)
        self.assertEqual(traj.residual_history[-1]["timestamp"], 2000)

    def test_update_records_matched_residual_when_provided(self):
        traj = Trajectory(track_id=9, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.update(
            DummyBBox(frame_id=1, timestamp=0.5, det_score=0.9),
            0.2,
            matched_residual={
                "pos": [0.3, -0.2, 0.1, 0.0, 0.4],
                "siz": [0.1, 0.2, -0.3],
                "ori": -0.05,
            },
        )

        self.assertEqual(len(traj.residual_history), 1)
        self.assertTrue(traj.residual_history[-1]["is_matched"])
        self.assertEqual(
            traj.residual_history[-1]["pos_residual"],
            [0.3, -0.2, 0.1, 0.0, 0.4],
        )
        self.assertAlmostEqual(traj.residual_history[-1]["det_score"], 0.9)
        self.assertAlmostEqual(traj.residual_history[-1]["timestamp"], 0.5)

    def test_unmatch_update_records_coast_placeholder(self):
        traj = Trajectory(track_id=10, init_bbox=DummyBBox(), cfg=_make_cfg())

        traj.unmatch_update(frame_id=1, timestamp=0.5)

        self.assertEqual(len(traj.residual_history), 1)
        self.assertFalse(traj.residual_history[-1]["is_matched"])
        self.assertEqual(traj.residual_history[-1]["pos_residual"], [0.0] * 5)
        self.assertEqual(traj.residual_history[-1]["siz_residual"], [0.0] * 3)
        self.assertEqual(traj.residual_history[-1]["ori_residual"], 0.0)
        self.assertEqual(traj.residual_history[-1]["det_score"], 0.0)
        self.assertAlmostEqual(traj.residual_history[-1]["timestamp"], 0.5)


if __name__ == "__main__":
    unittest.main()
