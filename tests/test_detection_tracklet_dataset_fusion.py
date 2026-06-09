import os
import pickle
import tempfile
import unittest

from training.det_tracklet_dataset import DetectionTrackletDataset


def _frame(
    frame_id,
    ts,
    *,
    is_matched,
    det_xyz,
    det_lwh,
    det_yaw,
    det_vel,
    det_score,
    fusion_valid,
    fusion_is_fake,
    fusion_xyz,
    fusion_lwh,
    fusion_yaw,
    fusion_vel,
):
    obs_feature = [
        float(det_xyz[0]) if det_xyz is not None else 0.0,
        float(det_xyz[1]) if det_xyz is not None else 0.0,
        float(det_xyz[2]) if det_xyz is not None else 0.0,
        float(det_vel[0]) if det_vel is not None else 0.0,
        float(det_vel[1]) if det_vel is not None else 0.0,
        0.0,
        float(det_lwh[0]) if det_lwh is not None else 0.0,
        float(det_lwh[1]) if det_lwh is not None else 0.0,
        float(det_lwh[2]) if det_lwh is not None else 0.0,
        float(det_yaw) if det_yaw is not None else 0.0,
        0.0,
        float(det_score),
    ]
    fusion_feature = [
        float(fusion_xyz[0]) if fusion_xyz is not None else 0.0,
        float(fusion_xyz[1]) if fusion_xyz is not None else 0.0,
        float(fusion_xyz[2]) if fusion_xyz is not None else 0.0,
        float(fusion_vel[0]) if fusion_vel is not None else 0.0,
        float(fusion_vel[1]) if fusion_vel is not None else 0.0,
        0.0,
        float(fusion_lwh[0]) if fusion_lwh is not None else 0.0,
        float(fusion_lwh[1]) if fusion_lwh is not None else 0.0,
        float(fusion_lwh[2]) if fusion_lwh is not None else 0.0,
        float(fusion_yaw) if fusion_yaw is not None else 0.0,
        0.0,
        0.0 if fusion_is_fake else float(det_score),
    ]
    return {
        "sample_token": f"sample_{frame_id}",
        "timestamp": float(ts),
        "frame_id": int(frame_id),
        "scene_id": "scene_x",
        "is_matched": bool(is_matched),
        "match_distance": 0.0 if is_matched else None,
        "det_score": float(det_score),
        "gt_feature_12": [1.0, 2.0, 1.5, 0.0, 0.0, 0.0, 4.0, 1.8, 1.6, 0.1, 0.0, 1.0],
        "obs_feature_12": obs_feature,
        "gt_global_xyz": [1.0, 2.0, 1.5],
        "gt_lwh": [4.0, 1.8, 1.6],
        "gt_yaw": 0.1,
        "gt_velocity": [0.0, 0.0],
        "det_global_xyz": det_xyz,
        "det_lwh": det_lwh,
        "det_yaw": det_yaw,
        "det_velocity": det_vel,
        "fusion_valid": bool(fusion_valid),
        "fusion_is_fake": bool(fusion_is_fake),
        "fusion_global_xyz": fusion_xyz,
        "fusion_lwh": fusion_lwh,
        "fusion_yaw": fusion_yaw,
        "fusion_velocity": fusion_vel,
        "fusion_feature_12": fusion_feature,
    }


class DetectionTrackletDatasetFusionTest(unittest.TestCase):
    def test_requires_fusion_fields_when_fusion_sources_enabled(self):
        tracklets = [{
            "instance_token": "inst_plain",
            "category": "car",
            "frames": [{
                "sample_token": "sample_0",
                "timestamp": 1.0,
                "frame_id": 0,
                "scene_id": "scene_x",
                "is_matched": True,
                "match_distance": 0.0,
                "det_score": 0.9,
                "gt_feature_12": [0.0] * 12,
                "obs_feature_12": [0.0] * 12,
                "gt_global_xyz": [0.0, 0.0, 0.0],
                "gt_lwh": [4.0, 1.8, 1.6],
                "gt_yaw": 0.0,
                "gt_velocity": [0.0, 0.0],
                "det_global_xyz": [0.0, 0.0, 0.0],
                "det_lwh": [4.0, 1.8, 1.6],
                "det_yaw": 0.0,
                "det_velocity": [0.0, 0.0],
            }],
        }]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp:
            pickle.dump(tracklets, fp)
            pkl_path = fp.name

        try:
            with self.assertRaises(ValueError):
                DetectionTrackletDataset(
                    pkl_path,
                    history_len=1,
                    rollout_steps=1,
                    require_current_match=True,
                    min_history_match_ratio=0.0,
                    adaptive_windows=False,
                    history_source="fusion",
                    init_state_source="fusion",
                )
        finally:
            os.remove(pkl_path)

    def test_fusion_history_and_init_state_sources(self):
        tracklets = [{
            "instance_token": "inst_1",
            "category": "car",
            "frames": [
                _frame(0, 1.0, is_matched=False, det_xyz=None, det_lwh=None, det_yaw=None, det_vel=None, det_score=0.0,
                       fusion_valid=False, fusion_is_fake=False, fusion_xyz=None, fusion_lwh=None, fusion_yaw=None, fusion_vel=None),
                _frame(1, 2.0, is_matched=True, det_xyz=[10.0, 20.0, 1.0], det_lwh=[4.5, 1.9, 1.6], det_yaw=0.2, det_vel=[3.0, 4.0], det_score=0.8,
                       fusion_valid=True, fusion_is_fake=False, fusion_xyz=[11.0, 21.0, 1.2], fusion_lwh=[4.2, 1.8, 1.5], fusion_yaw=0.25, fusion_vel=[1.5, 2.5]),
                _frame(2, 3.0, is_matched=False, det_xyz=None, det_lwh=None, det_yaw=None, det_vel=None, det_score=0.0,
                       fusion_valid=True, fusion_is_fake=True, fusion_xyz=[11.5, 21.5, 1.2], fusion_lwh=[4.2, 1.8, 1.5], fusion_yaw=0.25, fusion_vel=[1.4, 2.4]),
                _frame(3, 4.0, is_matched=True, det_xyz=[12.0, 22.0, 1.0], det_lwh=[4.6, 2.0, 1.6], det_yaw=0.3, det_vel=[3.1, 4.1], det_score=0.82,
                       fusion_valid=True, fusion_is_fake=False, fusion_xyz=[12.1, 22.1, 1.1], fusion_lwh=[4.3, 1.85, 1.55], fusion_yaw=0.28, fusion_vel=[1.6, 2.6]),
                _frame(4, 5.0, is_matched=True, det_xyz=[13.0, 23.0, 1.0], det_lwh=[4.7, 2.1, 1.6], det_yaw=0.35, det_vel=[3.2, 4.2], det_score=0.83,
                       fusion_valid=True, fusion_is_fake=False, fusion_xyz=[13.1, 23.1, 1.1], fusion_lwh=[4.35, 1.9, 1.55], fusion_yaw=0.32, fusion_vel=[1.7, 2.7]),
            ],
        }]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as fp:
            pickle.dump(tracklets, fp)
            pkl_path = fp.name

        try:
            ds = DetectionTrackletDataset(
                pkl_path,
                history_len=3,
                rollout_steps=1,
                require_current_match=True,
                min_history_match_ratio=0.0,
                adaptive_windows=False,
                history_source="fusion",
                init_state_source="fusion",
            )
            sample = ds[0]
            self.assertEqual(sample["track_history"].shape[0], 3)
            self.assertEqual(sample["history_mask"].tolist(), [True, True, True])
            self.assertEqual(sample["history_match_mask"].tolist(), [True, False, True])
            self.assertAlmostEqual(float(sample["track_history"][-1, 0].item()), 0.0, places=5)
            self.assertAlmostEqual(float(sample["track_history"][-1, 1].item()), 0.0, places=5)
            self.assertAlmostEqual(float(sample["obs_current_state_pos"][0].item()), 12.1, places=5)
            self.assertAlmostEqual(float(sample["obs_current_state_pos"][3].item()), 1.6, places=5)
            self.assertAlmostEqual(float(sample["obs_current_state_siz"][0].item()), 4.3, places=5)
            self.assertAlmostEqual(float(sample["obs_current_state_ori"][0].item()), 0.28, places=5)
        finally:
            os.unlink(pkl_path)


if __name__ == "__main__":
    unittest.main()
