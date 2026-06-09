import unittest

from tracker.compat_utils import (
    allow_single_stage_birth_under_mode,
    initial_status_flag_for_mode,
    select_filtered_tracking_score,
    score_for_unmatched_fake_bbox,
    select_output_tracking_score,
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


if __name__ == "__main__":
    unittest.main()
