import unittest

from tracker.compat_utils import (
    apply_dirty_track_suppressor_to_output,
    allow_single_stage_birth_under_mode,
    collect_dirty_track_features,
    compute_track_quality_score,
    dirty_track_suppressor,
    extract_bbox_history_fields,
    get_dirty_track_profile_cfg,
    initial_status_flag_for_mode,
    map_class_to_dirty_profile,
    sync_bbox_fields_from_state,
    select_filtered_tracking_score,
    score_for_unmatched_fake_bbox,
    select_output_tracking_score,
    use_mctrack_single_stage_flow,
)


def make_traj_stub(
    *,
    fake_history=None,
    low_score_history=None,
    recent_match_costs=None,
    pos_trace=0.0,
):
    class DummyBBox:
        pass

    class DummyTraj:
        pass

    fake_history = list(fake_history or [])
    low_score_history = list(low_score_history or [])
    recent_match_costs = list(recent_match_costs or [])

    real_idx = 0
    bboxes = []
    for is_fake in fake_history:
        bbox = DummyBBox()
        bbox.is_fake = bool(is_fake)
        if bbox.is_fake:
            bbox.is_low_score_match = False
            bbox.matched_score = 0.0
        else:
            bbox.is_low_score_match = (
                bool(low_score_history[real_idx]) if real_idx < len(low_score_history) else False
            )
            bbox.matched_score = (
                float(recent_match_costs[real_idx]) if real_idx < len(recent_match_costs) else 0.0
            )
            real_idx += 1
        bboxes.append(bbox)

    traj = DummyTraj()
    traj.bboxes = bboxes
    traj.debug_pos_trace = float(pos_trace)
    return traj


def make_clean_traj_stub():
    return make_traj_stub(
        fake_history=[False, False, False],
        low_score_history=[False, False, False],
        recent_match_costs=[0.2, 0.3, 0.25],
        pos_trace=1.5,
    )


def make_dirty_traj_stub():
    return make_traj_stub(
        fake_history=[False, True, True],
        low_score_history=[True, False, True, False, True],
        recent_match_costs=[0.95, 1.05],
        pos_trace=4.2,
    )


class MCTrackCompatUtilsTest(unittest.TestCase):
    def test_dirty_track_profile_mapping(self):
        self.assertEqual(map_class_to_dirty_profile(0), "stable_large")
        self.assertEqual(map_class_to_dirty_profile(2), "agile_weak")
        self.assertEqual(map_class_to_dirty_profile(5), "heavy_long")
        self.assertEqual(map_class_to_dirty_profile(1), "human")
        self.assertIsNone(map_class_to_dirty_profile(99))
        self.assertIsNone(map_class_to_dirty_profile(None))
        self.assertIsNone(map_class_to_dirty_profile("not-a-class"))

    def test_dirty_track_profile_cfg_allows_class_override(self):
        cfg = {
            "CLASS_PROFILE_OVERRIDES": {5: "trailer_only"},
            "PROFILES": {
                "trailer_only": {"soft_fake_len": 1},
                "heavy_long": {"soft_fake_len": 2},
            },
        }
        self.assertEqual(
            get_dirty_track_profile_cfg(5, cfg).get("soft_fake_len"),
            1,
        )
        self.assertEqual(
            get_dirty_track_profile_cfg(6, cfg).get("soft_fake_len"),
            2,
        )

    def test_dirty_track_profile_cfg_merges_per_class_overrides(self):
        cfg = {
            "PROFILES": {
                "heavy_long": {
                    "soft_fake_len": 2,
                    "soft_pos_trace_ratio": 1.7,
                    "cost_penalty_start": 0.9,
                },
            },
            "CLASS_PROFILES": {
                5: {
                    "MODE": "conjunctive_v1",
                    "MIN_SOFT_SIGNALS": 2,
                    "soft_pos_trace_ratio": 35.0,
                },
            },
        }
        trailer_cfg = get_dirty_track_profile_cfg(5, cfg)
        truck_cfg = get_dirty_track_profile_cfg(6, cfg)
        self.assertEqual(trailer_cfg.get("soft_fake_len"), 2)
        self.assertEqual(trailer_cfg.get("soft_pos_trace_ratio"), 35.0)
        self.assertEqual(trailer_cfg.get("MODE"), "conjunctive_v1")
        self.assertEqual(trailer_cfg.get("MIN_SOFT_SIGNALS"), 2)
        self.assertEqual(truck_cfg.get("soft_pos_trace_ratio"), 1.7)
        self.assertIsNone(truck_cfg.get("MODE"))

    def test_dirty_suppressor_returns_identity_for_clean_track(self):
        suppress = dirty_track_suppressor(
            features={
                "recent_fake_len": 0,
                "fake_ratio": 0.0,
                "recent_low_score_match_count": 0,
                "low_score_ratio": 0.0,
                "recent_match_cost_mean": 0.3,
                "current_det_score": 0.8,
                "pos_trace_ratio": 1.0,
            },
            profile_cfg={
                "soft_fake_len": 2,
                "hard_fake_len": 4,
                "soft_low_score_ratio": 0.35,
                "hard_low_score_ratio": 0.60,
                "soft_pos_trace_ratio": 1.8,
                "hard_pos_trace_ratio": 2.6,
                "cost_penalty_start": 0.9,
            },
        )
        self.assertAlmostEqual(suppress["penalty"], 1.0)
        self.assertFalse(suppress["hard_reject"])

    def test_dirty_suppressor_degrades_safely_for_missing_inputs(self):
        suppress = dirty_track_suppressor(features=None, profile_cfg=None)
        self.assertAlmostEqual(suppress["penalty"], 1.0)
        self.assertFalse(suppress["hard_reject"])

    def test_dirty_suppressor_soft_penalizes_but_does_not_reject_moderate_dirty_track(self):
        suppress = dirty_track_suppressor(
            features={
                "recent_fake_len": 2,
                "fake_ratio": 0.4,
                "recent_low_score_match_count": 1,
                "low_score_ratio": 0.45,
                "recent_match_cost_mean": 1.0,
                "current_det_score": 0.35,
                "pos_trace_ratio": 2.0,
            },
            profile_cfg={
                "soft_fake_len": 2,
                "hard_fake_len": 4,
                "soft_low_score_ratio": 0.35,
                "hard_low_score_ratio": 0.60,
                "soft_pos_trace_ratio": 1.8,
                "hard_pos_trace_ratio": 2.6,
                "cost_penalty_start": 0.9,
            },
        )
        self.assertAlmostEqual(suppress["penalty"], 0.8)
        self.assertFalse(suppress["hard_reject"])

    def test_dirty_suppressor_conjunctive_mode_ignores_pos_trace_only_hit(self):
        suppress = dirty_track_suppressor(
            features={
                "recent_fake_len": 0,
                "fake_ratio": 0.0,
                "recent_low_score_match_count": 0,
                "low_score_ratio": 0.0,
                "recent_match_cost_mean": 0.2,
                "current_det_score": 0.8,
                "pos_trace_ratio": 40.0,
            },
            profile_cfg={
                "MODE": "conjunctive_v1",
                "MIN_SOFT_SIGNALS": 2,
                "soft_fake_len": 3,
                "hard_fake_len": 5,
                "soft_low_score_ratio": 0.50,
                "hard_low_score_ratio": 0.75,
                "soft_pos_trace_ratio": 35.0,
                "hard_pos_trace_ratio": 60.0,
                "cost_penalty_start": 1.05,
            },
        )
        self.assertAlmostEqual(suppress["penalty"], 1.0)
        self.assertFalse(suppress["hard_reject"])
        self.assertEqual(suppress["triggered_reasons"], ["pos_trace_ratio"])

    def test_dirty_suppressor_conjunctive_mode_penalizes_multi_signal_dirty_track(self):
        suppress = dirty_track_suppressor(
            features={
                "recent_fake_len": 3,
                "fake_ratio": 0.5,
                "recent_low_score_match_count": 0,
                "low_score_ratio": 0.0,
                "recent_match_cost_mean": 1.2,
                "current_det_score": 0.8,
                "pos_trace_ratio": 40.0,
            },
            profile_cfg={
                "MODE": "conjunctive_v1",
                "MIN_SOFT_SIGNALS": 2,
                "soft_fake_len": 3,
                "hard_fake_len": 5,
                "soft_low_score_ratio": 0.50,
                "hard_low_score_ratio": 0.75,
                "soft_pos_trace_ratio": 35.0,
                "hard_pos_trace_ratio": 60.0,
                "cost_penalty_start": 1.05,
            },
        )
        self.assertLess(suppress["penalty"], 1.0)
        self.assertFalse(suppress["hard_reject"])
        self.assertEqual(
            suppress["triggered_reasons"],
            ["recent_fake_len", "pos_trace_ratio", "recent_match_cost_mean"],
        )

    def test_dirty_suppressor_hard_rejects_extreme_dirty_track(self):
        suppress = dirty_track_suppressor(
            features={
                "recent_fake_len": 5,
                "fake_ratio": 0.8,
                "recent_low_score_match_count": 4,
                "low_score_ratio": 0.75,
                "recent_match_cost_mean": 1.5,
                "current_det_score": 0.05,
                "pos_trace_ratio": 3.2,
            },
            profile_cfg={
                "soft_fake_len": 2,
                "hard_fake_len": 4,
                "soft_low_score_ratio": 0.35,
                "hard_low_score_ratio": 0.60,
                "soft_pos_trace_ratio": 1.8,
                "hard_pos_trace_ratio": 2.6,
                "cost_penalty_start": 0.9,
            },
        )
        self.assertTrue(suppress["hard_reject"])

    def test_collect_dirty_track_features_uses_recent_fake_and_trace_ratio(self):
        traj = make_traj_stub(
            fake_history=[False, False, True],
            low_score_history=[False, True],
            recent_match_costs=[0.4, 1.1],
            pos_trace=4.0,
        )

        features = collect_dirty_track_features(
            traj,
            base_score=0.3,
            pos_trace=4.0,
            pos_trace_prior=2.0,
        )

        self.assertEqual(features["recent_fake_len"], 1)
        self.assertAlmostEqual(features["low_score_ratio"], 1 / 3)
        self.assertAlmostEqual(features["recent_match_cost_mean"], 0.75)
        self.assertAlmostEqual(features["current_det_score"], 0.3)
        self.assertAlmostEqual(features["pos_trace_ratio"], 2.0)

    def test_apply_dirty_track_suppressor_is_identity_when_disabled(self):
        result = apply_dirty_track_suppressor_to_output(
            base_score=0.8,
            class_id=0,
            traj=make_clean_traj_stub(),
            suppressor_cfg={"ENABLED": False},
            pos_trace=1.5,
            pos_trace_prior=2.0,
        )

        self.assertAlmostEqual(result["final_score"], 0.8)
        self.assertFalse(result["hard_reject"])
        self.assertEqual(result["penalty"], 1.0)

    def test_apply_dirty_track_suppressor_softly_downweights_dirty_track(self):
        result = apply_dirty_track_suppressor_to_output(
            base_score=0.4,
            class_id=5,
            traj=make_dirty_traj_stub(),
            suppressor_cfg={
                "ENABLED": True,
                "PROFILES": {
                    "heavy_long": {
                        "soft_fake_len": 2,
                        "hard_fake_len": 4,
                        "soft_low_score_ratio": 0.35,
                        "hard_low_score_ratio": 0.60,
                        "soft_pos_trace_ratio": 1.8,
                        "hard_pos_trace_ratio": 2.6,
                        "cost_penalty_start": 0.9,
                    }
                },
            },
            pos_trace=4.2,
            pos_trace_prior=2.0,
        )

        self.assertLess(result["final_score"], 0.4)
        self.assertFalse(result["hard_reject"])
        self.assertEqual(result["profile_name"], "heavy_long")

    def test_apply_dirty_track_suppressor_reports_override_profile_name(self):
        result = apply_dirty_track_suppressor_to_output(
            base_score=0.4,
            class_id=5,
            traj=make_dirty_traj_stub(),
            suppressor_cfg={
                "ENABLED": True,
                "CLASS_PROFILE_OVERRIDES": {5: "trailer_only"},
                "PROFILES": {
                    "trailer_only": {
                        "soft_fake_len": 1,
                        "hard_fake_len": 3,
                        "soft_low_score_ratio": 0.35,
                        "hard_low_score_ratio": 0.60,
                        "soft_pos_trace_ratio": 1.5,
                        "hard_pos_trace_ratio": 2.1,
                        "cost_penalty_start": 0.8,
                    }
                },
            },
            pos_trace=4.2,
            pos_trace_prior=2.0,
        )

        self.assertEqual(result["profile_name"], "trailer_only")

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

    def test_track_quality_score_can_be_enabled_for_trailer_only(self):
        class DummyBBox:
            def __init__(self, score, *, matched_score=None, is_fake=False, is_low_score_match=False):
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
            "ENABLED_CLASS_IDS": [5],
            "REAL_SCORE_TOPK": 3,
            "RECENT_WINDOW": 3,
            "MIN_SCORE": 0.01,
            "MAX_SCORE": 0.995,
            "DEFAULT_CURRENT_FAKE_SCALE": 0.75,
            "MATURE_LEN": {5: 3},
            "CURRENT_FAKE_SCALE": {5: 0.78},
            "W_DET": {5: 0.42},
            "W_ASSOC": {5: 0.14},
            "W_CONT": {5: 0.12},
            "W_MATURE": {5: 0.32},
        }

        trailer = DummyTraj()
        trailer.cfg = {"TRACK_SCORE": score_cfg}
        trailer.category_num = 5
        trailer._output_score = 0.47
        trailer._confirmed_match_score = 0.35
        trailer._confirmed_track_length = 1
        trailer.bboxes = [
            DummyBBox(0.62, matched_score=0.18),
            DummyBBox(0.66, matched_score=0.20),
            DummyBBox(0.71, matched_score=0.16),
        ]

        score = compute_track_quality_score(trailer, current_score=0.71)
        self.assertIsNotNone(score)
        self.assertNotEqual(score, 0.71)

    def test_track_quality_score_returns_none_for_non_enabled_class(self):
        class DummyBBox:
            def __init__(self, score, *, matched_score=None):
                self.det_score = score
                self.is_fake = False
                self.is_low_score_match = False
                if matched_score is not None:
                    self.matched_score = matched_score

        class DummyTraj:
            pass

        traj = DummyTraj()
        traj.cfg = {
            "TRACK_SCORE": {
                "ENABLED": True,
                "MODE": "quality_v1",
                "ENABLED_CLASS_IDS": [5],
            }
        }
        traj.category_num = 3
        traj._output_score = 0.46
        traj._confirmed_match_score = 0.38
        traj._confirmed_track_length = 3
        traj.bboxes = [
            DummyBBox(0.60, matched_score=0.20),
            DummyBBox(0.58, matched_score=0.18),
        ]

        self.assertIsNone(compute_track_quality_score(traj, current_score=0.58))


if __name__ == "__main__":
    unittest.main()
