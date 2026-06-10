import unittest

from tracker.compat_utils import (
    allow_single_stage_birth_under_mode,
    compute_track_quality_score,
    extract_bbox_history_fields,
    initial_status_flag_for_mode,
    sync_bbox_fields_from_state,
    select_filtered_tracking_score,
    score_for_unmatched_fake_bbox,
    select_output_tracking_score,
    use_mctrack_single_stage_flow,
)


class MCTrackCompatUtilsTest(unittest.TestCase):
    def test_initial_status_flag_matches_mode(self):
        self.assertEqual(initial_status_flag_for_mode("default"), 0)
        self.assertEqual(initial_status_flag_for_mode("mctrack"), 1)

    def test_unmatched_fake_bbox_score_matches_mode(self):
        self.assertAlmostEqual(
            score_for_unmatched_fake_bbox(0.8, 2, "default"),
            0.8 * (0.8 ** 2),
        )
        self.assertEqual(score_for_unmatched_fake_bbox(0.8, 2, "mctrack"), 0.0)

    def test_output_score_selection_matches_mode(self):
        self.assertEqual(
            select_output_tracking_score(
                current_score=0.41,
                real_scores=[0.41, 0.52],
                quality_scores=[0.52],
                compat_mode="mctrack",
            ),
            0.41,
        )
        self.assertEqual(
            select_output_tracking_score(
                current_score=0.41,
                real_scores=[0.41, 0.52],
                quality_scores=[0.52],
                compat_mode="default",
            ),
            0.52,
        )

    def test_mctrack_mode_bypasses_single_stage_birth_gate(self):
        self.assertFalse(
            allow_single_stage_birth_under_mode(
                compat_mode="default",
                gate_allowed=False,
            )
        )
        self.assertTrue(
            allow_single_stage_birth_under_mode(
                compat_mode="mctrack",
                gate_allowed=False,
            )
        )

    def test_use_mctrack_single_stage_flow_only_for_single_stage_mctrack(self):
        self.assertTrue(use_mctrack_single_stage_flow("mctrack", use_bytetrack=False))
        self.assertFalse(use_mctrack_single_stage_flow("mctrack", use_bytetrack=True))
        self.assertFalse(use_mctrack_single_stage_flow("default", use_bytetrack=False))

    def test_filtered_score_selection_matches_mode(self):
        transformed_scores = [-0.4, -0.1]
        quality_scores = [-0.1]
        self.assertAlmostEqual(
            select_filtered_tracking_score(
                compat_mode="mctrack",
                original_scores=[0.4, 0.52],
                transformed_scores=transformed_scores,
                quality_scores=quality_scores,
                fallback_score=0.52,
            ),
            sum(transformed_scores) / (len(transformed_scores) + 1e-5),
        )
        self.assertEqual(
            select_filtered_tracking_score(
                compat_mode="default",
                original_scores=[0.4, 0.52],
                transformed_scores=transformed_scores,
                quality_scores=quality_scores,
                fallback_score=0.52,
            ),
            -0.1,
        )

    def test_sync_bbox_fields_from_state_updates_split_fields(self):
        class DummyBBox:
            pass

        bbox = DummyBBox()
        bbox.global_xyz = [0.0, 0.0, 0.0]
        bbox.lwh = [1.0, 1.0, 1.0]
        bbox.global_yaw = 0.0
        bbox.global_xyz_lwh_yaw = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]
        bbox.global_xyz_lwh_yaw_fusion = [0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0]

        state = [2.0, 3.0, 1.5, 4.0, 1.8, 1.6, 0.4]
        sync_bbox_fields_from_state(bbox, state, update_fusion=True, update_predict=False)

        self.assertEqual(bbox.global_xyz, [2.0, 3.0, 1.5])
        self.assertEqual(bbox.lwh, [4.0, 1.8, 1.6])
        self.assertEqual(bbox.global_yaw, 0.4)
        self.assertEqual(list(bbox.global_xyz_lwh_yaw_fusion), state)

    def test_extract_bbox_history_fields_prefers_fusion_in_mctrack_mode(self):
        class DummyBBox:
            pass

        bbox = DummyBBox()
        bbox.global_xyz = [10.0, 20.0, 1.0]
        bbox.global_velocity = [3.0, 4.0]
        bbox.lwh = [4.5, 1.9, 1.6]
        bbox.global_yaw = 0.8
        bbox.global_xyz_lwh_yaw_fusion = [11.0, 21.0, 1.2, 4.2, 1.8, 1.5, 0.6]
        bbox.global_velocity_fusion = [1.5, 2.5]
        bbox.lwh_fusion = [4.2, 1.8, 1.5]
        bbox.global_yaw_fusion = 0.6

        xyz, vel, lwh, yaw = extract_bbox_history_fields(bbox, "mctrack")
        self.assertEqual(xyz, [11.0, 21.0, 1.2])
        self.assertEqual(vel, [1.5, 2.5])
        self.assertEqual(lwh, [4.2, 1.8, 1.5])
        self.assertEqual(yaw, 0.6)

    def test_extract_bbox_history_fields_keeps_raw_fields_in_default_mode(self):
        class DummyBBox:
            pass

        bbox = DummyBBox()
        bbox.global_xyz = [10.0, 20.0, 1.0]
        bbox.global_velocity = [3.0, 4.0]
        bbox.lwh = [4.5, 1.9, 1.6]
        bbox.global_yaw = 0.8
        bbox.global_xyz_lwh_yaw_fusion = [11.0, 21.0, 1.2, 4.2, 1.8, 1.5, 0.6]
        bbox.global_velocity_fusion = [1.5, 2.5]
        bbox.lwh_fusion = [4.2, 1.8, 1.5]
        bbox.global_yaw_fusion = 0.6

        xyz, vel, lwh, yaw = extract_bbox_history_fields(bbox, "default")
        self.assertEqual(xyz, [10.0, 20.0, 1.0])
        self.assertEqual(vel, [3.0, 4.0])
        self.assertEqual(lwh, [4.5, 1.9, 1.6])
        self.assertEqual(yaw, 0.8)

    def test_track_quality_score_prefers_clean_stable_tracks(self):
        class DummyBBox:
            def __init__(self, score, *, is_fake=False, is_low_score_match=False, matched_score=None):
                self.det_score = score
                self.is_fake = is_fake
                self.is_low_score_match = is_low_score_match
                if matched_score is not None:
                    self.matched_score = matched_score

        class DummyTraj:
            pass

        score_cfg = {
            "ENABLED": True,
            "MODE": "quality_v1",
            "REAL_SCORE_TOPK": 3,
            "RECENT_WINDOW": 3,
            "MIN_SCORE": 0.01,
            "MAX_SCORE": 0.995,
            "DEFAULT_CURRENT_FAKE_SCALE": 0.75,
            "MATURE_LEN": {2: 4},
            "CURRENT_FAKE_SCALE": {2: 0.70},
            "W_DET": {2: 0.28},
            "W_ASSOC": {2: 0.28},
            "W_CONT": {2: 0.26},
            "W_MATURE": {2: 0.18},
        }

        good = DummyTraj()
        good.cfg = {"TRACK_SCORE": score_cfg}
        good.category_num = 2
        good._output_score = 0.45
        good._confirmed_match_score = 0.38
        good._confirmed_track_length = 3
        good.bboxes = [
            DummyBBox(0.62, matched_score=0.18),
            DummyBBox(0.68, matched_score=0.15),
            DummyBBox(0.71, matched_score=0.12),
            DummyBBox(0.66, matched_score=0.14),
        ]

        bad = DummyTraj()
        bad.cfg = {"TRACK_SCORE": score_cfg}
        bad.category_num = 2
        bad._output_score = 0.45
        bad._confirmed_match_score = 0.38
        bad._confirmed_track_length = 3
        bad.bboxes = [
            DummyBBox(0.28, matched_score=0.62, is_low_score_match=True),
            DummyBBox(0.31, matched_score=0.71, is_low_score_match=True),
            DummyBBox(0.00, is_fake=True),
            DummyBBox(0.00, is_fake=True),
        ]

        self.assertGreater(
            compute_track_quality_score(good),
            compute_track_quality_score(bad),
        )

    def test_track_quality_score_falls_back_to_current_score_without_real_hits(self):
        class DummyBBox:
            def __init__(self, score, *, is_fake=False):
                self.det_score = score
                self.is_fake = is_fake
                self.is_low_score_match = False

        class DummyTraj:
            pass

        traj = DummyTraj()
        traj.cfg = {"TRACK_SCORE": {"ENABLED": True, "MODE": "quality_v1"}}
        traj.category_num = 0
        traj._output_score = 0.4
        traj._confirmed_match_score = 0.35
        traj._confirmed_track_length = 1
        traj.bboxes = [DummyBBox(0.0, is_fake=True), DummyBBox(0.0, is_fake=True)]

        self.assertEqual(
            compute_track_quality_score(traj, current_score=0.23),
            0.23,
        )


if __name__ == "__main__":
    unittest.main()
