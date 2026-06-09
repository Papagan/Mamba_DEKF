import unittest

from tracker.compat_utils import (
    allow_single_stage_birth_under_mode,
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


if __name__ == "__main__":
    unittest.main()
