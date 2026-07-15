import unittest
import sys
import types
from unittest.mock import patch

import numpy as np

if "cv2" not in sys.modules:
    sys.modules["cv2"] = types.ModuleType("cv2")

if "shapely" not in sys.modules:
    shapely_stub = types.ModuleType("shapely")
    shapely_geometry_stub = types.ModuleType("shapely.geometry")
    shapely_geometry_stub.Polygon = object
    shapely_stub.geometry = shapely_geometry_stub
    sys.modules["shapely"] = shapely_stub
    sys.modules["shapely.geometry"] = shapely_geometry_stub

if "lap" not in sys.modules:
    lap_stub = types.ModuleType("lap")

    def _unused_lapjv(*args, **kwargs):
        raise AssertionError("lapjv should not be used by association-prior unit tests")

    lap_stub.lapjv = _unused_lapjv
    sys.modules["lap"] = lap_stub

if "pyquaternion" not in sys.modules:
    pyquaternion_stub = types.ModuleType("pyquaternion")

    class _UnusedQuaternion:
        def __init__(self, *args, **kwargs):
            raise AssertionError("Quaternion should not be used by association-prior unit tests")

    pyquaternion_stub.Quaternion = _UnusedQuaternion
    sys.modules["pyquaternion"] = pyquaternion_stub

import tracker.matching as matching
from tracker.matching import (
    apply_mamba_association_prior_to_cost_matrix,
    apply_pairwise_association_head_to_cost_matrix,
)


class _Box:
    def __init__(self, category, det_score=0.8):
        self.category = category
        self.det_score = det_score


class _Traj:
    def __init__(self, category, unmatch_length=0, track_id=None):
        self.bboxes = [_Box(category)]
        self.unmatch_length = unmatch_length
        if track_id is not None:
            self.track_id = track_id


def _matching_cfg(*, shadow_mode=False, head_enabled=True, audit_enabled=True):
    return {
        "CATEGORY_LIST": ["trailer"],
        "CATEGORY_MAP_TO_NUMBER": {"trailer": 0},
        "MATCHING": {
            "BEV": {"MATCHING_MODE": "Greedy"},
            "RV": {"MATCHING_MODE": "Greedy"},
        },
        "THRESHOLD": {
            "BEV": {"COST_THRE": {0: 1.0}},
            "RV": {"COST_THRE": {0: 1.0}},
        },
        "MAMBA_ASSOCIATION_HEAD": {
            "ENABLED": head_enabled,
            "MIN_SCORE": 0.5,
            "ALPHA": 1.0,
            "MAX_DELTA": 0.5,
            "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
        },
        "ASSOCIATION_HEAD_AUDIT": {
            "ENABLED": audit_enabled,
            "SHADOW_MODE": shadow_mode,
        },
    }


def _flippable_matching_inputs():
    return (
        np.array([[0.10, 0.11], [0.11, 0.10]], dtype=np.float32),
        [_Traj("trailer", unmatch_length=1, track_id=10),
         _Traj("trailer", unmatch_length=1, track_id=11)],
        [_Box("trailer"), _Box("trailer")],
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32),
    )


class MambaAssociationPriorTest(unittest.TestCase):
    def test_solve_matching_cost_matrix_preserves_inputs_and_category_mask(self):
        cost = np.array([[0.1, 0.0], [0.0, 0.1]], dtype=np.float32)
        original = np.array(cost, copy=True)
        cfg = {
            "CATEGORY_LIST": ["car", "pedestrian"],
            "CATEGORY_MAP_TO_NUMBER": {"car": 0, "pedestrian": 1},
            "MATCHING": {"BEV": {"MATCHING_MODE": "Greedy"}},
            "THRESHOLD": {"BEV": {"COST_THRE": {0: 1.0, 1: 1.0}}},
        }

        matches, costs = matching.solve_matching_cost_matrix(
            cost,
            np.array(["car", "pedestrian"]),
            np.array(["car", "pedestrian"]),
            cfg,
            "BEV",
        )

        np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1]]))
        np.testing.assert_allclose(costs, np.array([0.1, 0.1]))
        np.testing.assert_array_equal(cost, original)

    def test_solve_matching_cost_matrix_handles_empty_axes_without_mutation(self):
        cfg = _matching_cfg()
        for cost, trajs_category, dets_category in (
            (np.empty((0, 2)), np.array([]), np.array(["trailer", "trailer"])),
            (np.empty((2, 0)), np.array(["trailer", "trailer"]), np.array([])),
        ):
            with self.subTest(shape=cost.shape):
                original = np.array(cost, copy=True)
                matches, costs = matching.solve_matching_cost_matrix(
                    cost, trajs_category, dets_category, cfg, "BEV"
                )
                self.assertEqual(matches.shape, (0, 2))
                self.assertTrue(np.issubdtype(matches.dtype, np.integer))
                self.assertEqual(costs.shape, (0,))
                np.testing.assert_array_equal(cost, original)

    def test_shadow_mode_keeps_production_identical_to_disabled_head(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        base_original = np.array(base, copy=True)
        scores_original = np.array(scores, copy=True)
        records = []

        def fake_cost(*args, **kwargs):
            return base, np.array(["trailer", "trailer"]), np.array(["trailer", "trailer"])

        with patch.object(matching, "cost_calculate_general", side_effect=fake_cost):
            production_matches, production_costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=True),
                association_scores=scores,
                association_shadow_callback=records.extend,
            )
            base_matches, base_costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=False, head_enabled=False),
                association_scores=scores,
            )

        np.testing.assert_array_equal(production_matches, np.array([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(production_matches, base_matches)
        np.testing.assert_array_equal(production_costs, base_costs)
        self.assertTrue(any(record["assignment_changed"] for record in records))
        np.testing.assert_array_equal(base, base_original)
        np.testing.assert_array_equal(scores, scores_original)

    def test_shadow_without_real_track_ids_skips_callback_and_preserves_production(self):
        base, _, dets, scores = _flippable_matching_inputs()
        trajs = [
            _Traj("trailer", unmatch_length=1),
            _Traj("trailer", unmatch_length=1),
        ]
        callback_calls = []

        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(
                base,
                np.array(["trailer", "trailer"]),
                np.array(["trailer", "trailer"]),
            ),
        ):
            production_matches, production_costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=True),
                association_scores=scores,
                association_shadow_callback=callback_calls.append,
            )
            base_matches, base_costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=False, head_enabled=False),
                association_scores=scores,
            )

        self.assertEqual(callback_calls, [])
        np.testing.assert_array_equal(production_matches, base_matches)
        np.testing.assert_array_equal(production_costs, base_costs)

    def test_shadow_disabled_preserves_explicit_corrected_intervention(self):
        base, trajs, dets, scores = _flippable_matching_inputs()

        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(
                base,
                np.array(["trailer", "trailer"]),
                np.array(["trailer", "trailer"]),
            ),
        ):
            matches, costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=False),
                association_scores=scores,
                association_shadow_callback=lambda records: self.fail(
                    "shadow callback must not run during explicit intervention"
                ),
            )

        np.testing.assert_array_equal(matches, np.array([[1, 0], [0, 1]]))
        np.testing.assert_allclose(costs, np.array([0.11, 0.11]), rtol=1e-6)

    def test_rv_matching_never_runs_shadow_assignment_or_callback(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        callback_calls = []
        error_calls = []

        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(
                base,
                np.array(["trailer", "trailer"]),
                np.array(["trailer", "trailer"]),
            ),
        ), patch.object(
            matching,
            "solve_matching_cost_matrix",
            wraps=matching.solve_matching_cost_matrix,
        ) as solve:
            matches, _ = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=True),
                is_rv=True,
                association_scores=scores,
                association_shadow_callback=callback_calls.append,
                association_shadow_error_callback=error_calls.append,
                association_shadow_strict=True,
            )

        np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1]]))
        self.assertEqual(solve.call_count, 1)
        self.assertEqual(callback_calls, [])
        self.assertEqual(error_calls, [])

    def test_uncertainty_aware_shadow_keeps_production_assignment(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        cfg = _matching_cfg(shadow_mode=True)
        cfg["THRESHOLD"]["BEV"]["COST_MODE"] = "full"
        records = []

        with patch.object(
            matching,
            "cost_calculate_uncertainty_aware",
            return_value=(
                base,
                np.array(["trailer", "trailer"]),
                np.array(["trailer", "trailer"]),
            ),
        ):
            matches, costs = matching.match_trajs_and_dets_uncertainty_aware(
                trajs,
                dets,
                cfg,
                trk_embeddings=np.eye(2, dtype=np.float32),
                det_embeddings=np.eye(2, dtype=np.float32),
                association_scores=scores,
                state_buckets=["unmatched", "unmatched"],
                association_shadow_callback=records.extend,
            )

        np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1]]))
        np.testing.assert_allclose(costs, np.array([0.10, 0.10]), rtol=1e-6)
        self.assertTrue(any(record["assignment_changed"] for record in records))

    def test_shadow_missing_or_mismatched_scores_records_unchanged(self):
        base, trajs, dets, _ = _flippable_matching_inputs()
        for scores in (None, np.ones((1, 2), dtype=np.float32)):
            with self.subTest(scores_shape=None if scores is None else scores.shape):
                records = []
                with patch.object(
                    matching,
                    "cost_calculate_general",
                    return_value=(
                        base,
                        np.array(["trailer", "trailer"]),
                        np.array(["trailer", "trailer"]),
                    ),
                ):
                    matches, _ = matching.match_trajs_and_dets(
                        trajs,
                        dets,
                        _matching_cfg(shadow_mode=True),
                        association_scores=scores,
                        association_shadow_callback=records.extend,
                    )

                np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1]]))
                self.assertEqual(len(records), 2)
                self.assertFalse(any(record["assignment_changed"] for record in records))

    def test_shadow_failures_cannot_change_or_abort_production(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        categories = np.array(["trailer", "trailer"])
        expected_matches = np.array([[0, 0], [1, 1]])
        expected_costs = np.array([0.10, 0.10], dtype=np.float32)

        with self.subTest(failure="correction"):
            with patch.object(
                matching,
                "cost_calculate_general",
                return_value=(base, categories, categories),
            ), patch.object(
                matching,
                "apply_pairwise_association_head_to_cost_matrix",
                side_effect=RuntimeError("correction failed"),
            ):
                matches, costs = matching.match_trajs_and_dets(
                    trajs, dets, _matching_cfg(shadow_mode=True), association_scores=scores
                )
            np.testing.assert_array_equal(matches, expected_matches)
            np.testing.assert_array_equal(costs, expected_costs)

        real_solver = matching.solve_matching_cost_matrix
        solve_calls = 0

        def fail_second_solve(*args, **kwargs):
            nonlocal solve_calls
            solve_calls += 1
            if solve_calls == 2:
                raise RuntimeError("shadow solve failed")
            return real_solver(*args, **kwargs)

        with self.subTest(failure="solver"):
            with patch.object(
                matching,
                "cost_calculate_general",
                return_value=(base, categories, categories),
            ), patch.object(
                matching, "solve_matching_cost_matrix", side_effect=fail_second_solve
            ):
                matches, costs = matching.match_trajs_and_dets(
                    trajs, dets, _matching_cfg(shadow_mode=True), association_scores=scores
                )
            np.testing.assert_array_equal(matches, expected_matches)
            np.testing.assert_array_equal(costs, expected_costs)

        with self.subTest(failure="callback"):
            with patch.object(
                matching,
                "cost_calculate_general",
                return_value=(base, categories, categories),
            ):
                matches, costs = matching.match_trajs_and_dets(
                    trajs,
                    dets,
                    _matching_cfg(shadow_mode=True),
                    association_scores=scores,
                    association_shadow_callback=lambda records: (_ for _ in ()).throw(
                        RuntimeError("callback failed")
                    ),
                )
            np.testing.assert_array_equal(matches, expected_matches)
            np.testing.assert_array_equal(costs, expected_costs)

    def test_shadow_failure_records_error_and_strict_controls_reraise(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        categories = np.array(["trailer", "trailer"])
        expected_matches = np.array([[0, 0], [1, 1]])
        errors = []

        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(base, categories, categories),
        ), patch.object(
            matching,
            "apply_pairwise_association_head_to_cost_matrix",
            side_effect=RuntimeError("shadow correction failed"),
        ):
            matches, costs = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=True),
                association_scores=scores,
                association_shadow_error_callback=errors.append,
                association_shadow_metadata={
                    "context": {"scene_id": "scene-real", "sample_token": "sample-real"}
                },
            )

        np.testing.assert_array_equal(matches, expected_matches)
        np.testing.assert_array_equal(costs, np.array([0.10, 0.10], dtype=np.float32))
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]["stage"], "shadow_correction")
        self.assertEqual(errors[0]["type"], "RuntimeError")
        self.assertEqual(errors[0]["context"]["scene_id"], "scene-real")

        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(base, categories, categories),
        ), patch.object(
            matching,
            "apply_pairwise_association_head_to_cost_matrix",
            side_effect=RuntimeError("shadow correction failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "shadow correction failed"):
                matching.match_trajs_and_dets(
                    trajs,
                    dets,
                    _matching_cfg(shadow_mode=True),
                    association_scores=scores,
                    association_shadow_strict=True,
                    association_shadow_error_callback=errors.append,
                )

    def test_production_solver_failure_is_never_treated_as_shadow_error(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        errors = []
        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(base, np.array(["trailer"] * 2), np.array(["trailer"] * 2)),
        ), patch.object(
            matching,
            "solve_matching_cost_matrix",
            side_effect=RuntimeError("production solver failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "production solver failed"):
                matching.match_trajs_and_dets(
                    trajs,
                    dets,
                    _matching_cfg(shadow_mode=True),
                    association_scores=scores,
                    association_shadow_error_callback=errors.append,
                )
        self.assertEqual(errors, [])

    def test_legacy_shadow_mode_without_enabled_keeps_production_unmodified(self):
        base, trajs, dets, scores = _flippable_matching_inputs()
        callback_calls = []
        with patch.object(
            matching,
            "cost_calculate_general",
            return_value=(base, np.array(["trailer"] * 2), np.array(["trailer"] * 2)),
        ):
            matches, _ = matching.match_trajs_and_dets(
                trajs,
                dets,
                _matching_cfg(shadow_mode=True, audit_enabled=False),
                association_scores=scores,
                association_shadow_callback=callback_calls.append,
            )

        np.testing.assert_array_equal(matches, np.array([[0, 0], [1, 1]]))
        self.assertTrue(any(record["assignment_changed"] for record in callback_calls[0]))

    def test_uncertainty_shadow_uses_actual_adaptive_thresholds(self):
        base = np.array([[0.15]], dtype=np.float32)
        trajs = [_Traj("trailer", track_id=23)]
        dets = [_Box("trailer")]
        cfg = _matching_cfg(shadow_mode=True)
        cfg["THRESHOLD"]["BEV"]["COST_MODE"] = "full"
        cfg["THRESHOLD"]["BEV"]["COST_THRE"] = {0: 0.10}
        records = []
        errors = []

        with patch.object(
            matching,
            "cost_calculate_uncertainty_aware",
            return_value=(
                base,
                np.array(["trailer"]),
                np.array(["trailer"]),
            ),
        ), patch.object(
            matching,
            "compute_uncertainty_penalty",
            return_value=np.array([2.0], dtype=np.float32),
        ):
            matches, costs = matching.match_trajs_and_dets_uncertainty_aware(
                trajs,
                dets,
                cfg,
                trk_embeddings=np.ones((1, 2), dtype=np.float32),
                det_embeddings=np.ones((1, 2), dtype=np.float32),
                trk_pos_P=[np.eye(6, dtype=np.float32)],
                trk_ori_P=[np.eye(2, dtype=np.float32)],
                association_scores=np.ones((1, 1), dtype=np.float32),
                state_buckets=["unmatched"],
                association_shadow_callback=records.extend,
                association_shadow_error_callback=errors.append,
                association_shadow_strict=True,
            )

        np.testing.assert_array_equal(matches, np.array([[0, 0]]))
        np.testing.assert_allclose(costs, np.array([0.15]), rtol=1e-6)
        self.assertEqual(errors, [])
        self.assertEqual(len(records), 1)
        self.assertFalse(records[0]["assignment_changed"])

    def test_disabled_config_leaves_cost_matrix_unchanged(self):
        cost = np.array([[1.0, 1.5], [2.0, 0.5]], dtype=np.float32)
        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("car"), _Traj("trailer")],
            [_Box("car"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"car": 0, "trailer": 5},
                "MAMBA_ASSOCIATION_PRIOR": {"ENABLED": False},
            },
            trk_embeddings=np.eye(2, dtype=np.float32),
            det_embeddings=np.eye(2, dtype=np.float32),
        )

        np.testing.assert_array_equal(out, cost)
        self.assertIsNot(out, cost)

    def test_enabled_config_applies_only_active_class_state_with_bounds(self):
        cost = np.ones((2, 2), dtype=np.float32)
        trk_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        det_embeddings = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        cfg = {
            "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 3, "trailer": 5},
            "MAMBA_ASSOCIATION_PRIOR": {
                "ENABLED": True,
                "MAX_DELTA": 0.05,
                "ACTIVE_CLASS_STATES": {
                    3: ["unmatched"],
                    5: ["matched"],
                },
                "ALPHA": {
                    3: {"unmatched": 0.20},
                    5: {"matched": 0.20},
                },
            },
        }

        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("motorcycle", unmatch_length=1), _Traj("trailer", unmatch_length=1)],
            [_Box("motorcycle"), _Box("motorcycle")],
            cfg,
            trk_embeddings=trk_embeddings,
            det_embeddings=det_embeddings,
        )

        # Motorcycle/unmatched is active: good embedding pair unchanged,
        # bad embedding pair is penalized but clipped to MAX_DELTA.
        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        # Trailer/unmatched is inactive because only matched is configured.
        self.assertAlmostEqual(float(out[1, 0]), 1.0)
        self.assertAlmostEqual(float(out[1, 1]), 1.0)

    def test_cross_class_pairs_are_not_rescored(self):
        cost = np.ones((1, 2), dtype=np.float32)
        out = apply_mamba_association_prior_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("car"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"car": 0, "trailer": 5},
                "MAMBA_ASSOCIATION_PRIOR": {
                    "ENABLED": True,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                    "ALPHA": {5: {"unmatched": 0.20}},
                },
            },
            trk_embeddings=np.array([[0.0, 1.0]], dtype=np.float32),
            det_embeddings=np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)

    def test_match_trajs_applies_mamba_prior_before_assignment(self):
        seen = {}
        original_cost = matching.cost_calculate_general
        original_hungarian = matching.Hungarian

        def fake_cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
            return (
                np.array([[1.0, 1.0]], dtype=np.float32),
                np.array(["motorcycle"]),
                np.array(["motorcycle", "motorcycle"]),
            )

        def fake_hungarian(trans_cost_matrix, thresholds):
            seen["trans_cost_matrix"] = np.array(trans_cost_matrix, copy=True)
            return [0], [0], np.array([1]), np.array([], dtype=int), np.array([0.0])

        matching.cost_calculate_general = fake_cost_calculate_general
        matching.Hungarian = fake_hungarian
        try:
            matched, _ = matching.match_trajs_and_dets(
                [_Traj("motorcycle", unmatch_length=1)],
                [_Box("motorcycle"), _Box("motorcycle")],
                {
                    "CATEGORY_LIST": ["motorcycle"],
                    "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 0},
                    "MATCHING": {
                        "BEV": {"MATCHING_MODE": "Hungarian"},
                        "RV": {"MATCHING_MODE": "Hungarian"},
                    },
                    "THRESHOLD": {
                        "BEV": {"COST_THRE": {0: 2.0}},
                        "RV": {"COST_THRE": {0: 2.0}},
                    },
                    "MAMBA_ASSOCIATION_PRIOR": {
                        "ENABLED": True,
                        "MAX_DELTA": 0.05,
                        "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
                        "ALPHA": {0: {"unmatched": 0.20}},
                    },
                },
                trk_embeddings=np.array([[1.0, 0.0]], dtype=np.float32),
                det_embeddings=np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32),
            )
        finally:
            matching.cost_calculate_general = original_cost
            matching.Hungarian = original_hungarian

        np.testing.assert_array_equal(matched, np.array([[0, 0]]))
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 1, 0]), 1.05, places=6)

    def test_pairwise_head_disabled_leaves_cost_unchanged(self):
        cost = np.array([[1.0, 1.5]], dtype=np.float32)
        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {"ENABLED": False},
            },
            association_scores=np.array([[0.1, 0.9]], dtype=np.float32),
        )

        np.testing.assert_array_equal(out, cost)
        self.assertIsNot(out, cost)

    def test_pairwise_head_only_increases_low_score_active_pairs(self):
        cost = np.ones((2, 2), dtype=np.float32)
        scores = np.array([[0.9, 0.2], [0.1, 0.1]], dtype=np.float32)
        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("motorcycle", unmatch_length=1), _Traj("trailer", unmatch_length=1)],
            [_Box("motorcycle"), _Box("motorcycle")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"motorcycle": 3, "trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {3: ["unmatched"]},
                },
            },
            association_scores=scores,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        self.assertAlmostEqual(float(out[1, 0]), 1.0)
        self.assertAlmostEqual(float(out[1, 1]), 1.0)

    def test_pairwise_head_reports_audit_records(self):
        records = []
        cost = np.ones((1, 2), dtype=np.float32)

        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                },
            },
            association_scores=np.array([[0.9, 0.1]], dtype=np.float32),
            audit_callback=records.append,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.0)
        self.assertAlmostEqual(float(out[0, 1]), 1.05, places=6)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["class_name"], "trailer")
        self.assertEqual(records[0]["state_bucket"], "unmatched")
        self.assertAlmostEqual(records[0]["score"], 0.9, places=6)
        self.assertAlmostEqual(records[0]["delta"], 0.0, places=6)
        self.assertAlmostEqual(records[1]["score"], 0.1, places=6)
        self.assertAlmostEqual(records[1]["delta"], 0.05, places=6)

    def test_pairwise_head_margin_tiebreak_only_penalizes_near_best_candidates(self):
        cost = np.array([[1.00, 1.03, 1.30]], dtype=np.float32)
        scores = np.array([[0.9, 0.1, 0.1]], dtype=np.float32)
        records = []

        out = apply_pairwise_association_head_to_cost_matrix(
            cost,
            [_Traj("trailer", unmatch_length=1)],
            [_Box("trailer"), _Box("trailer"), _Box("trailer")],
            {
                "CATEGORY_MAP_TO_NUMBER": {"trailer": 5},
                "MAMBA_ASSOCIATION_HEAD": {
                    "ENABLED": True,
                    "APPLY_MODE": "margin_tiebreak",
                    "COST_MARGIN_EPS": 0.05,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.2,
                    "MAX_DELTA": 0.05,
                    "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                },
            },
            association_scores=scores,
            audit_callback=records.append,
        )

        self.assertAlmostEqual(float(out[0, 0]), 1.00, places=6)
        self.assertAlmostEqual(float(out[0, 1]), 1.08, places=6)
        self.assertAlmostEqual(float(out[0, 2]), 1.30, places=6)
        self.assertEqual(len(records), 3)
        self.assertTrue(records[0]["active"])
        self.assertTrue(records[1]["active"])
        self.assertFalse(records[2]["active"])
        self.assertAlmostEqual(records[2]["delta"], 0.0, places=6)

    def test_match_trajs_applies_pairwise_head_scores_before_assignment(self):
        seen = {}
        original_cost = matching.cost_calculate_general
        original_hungarian = matching.Hungarian

        def fake_cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
            return (
                np.array([[1.0, 1.0]], dtype=np.float32),
                np.array(["trailer"]),
                np.array(["trailer", "trailer"]),
            )

        def fake_hungarian(trans_cost_matrix, thresholds):
            seen["trans_cost_matrix"] = np.array(trans_cost_matrix, copy=True)
            return [0], [0], np.array([1]), np.array([], dtype=int), np.array([0.0])

        matching.cost_calculate_general = fake_cost_calculate_general
        matching.Hungarian = fake_hungarian
        try:
            matched, _ = matching.match_trajs_and_dets(
                [_Traj("trailer", unmatch_length=1)],
                [_Box("trailer"), _Box("trailer")],
                {
                    "CATEGORY_LIST": ["trailer"],
                    "CATEGORY_MAP_TO_NUMBER": {"trailer": 0},
                    "MATCHING": {
                        "BEV": {"MATCHING_MODE": "Hungarian"},
                        "RV": {"MATCHING_MODE": "Hungarian"},
                    },
                    "THRESHOLD": {
                        "BEV": {"COST_THRE": {0: 2.0}},
                        "RV": {"COST_THRE": {0: 2.0}},
                    },
                    "MAMBA_ASSOCIATION_HEAD": {
                        "ENABLED": True,
                        "MIN_SCORE": 0.6,
                        "ALPHA": 0.2,
                        "MAX_DELTA": 0.05,
                        "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
                    },
                },
                association_scores=np.array([[0.9, 0.1]], dtype=np.float32),
            )
        finally:
            matching.cost_calculate_general = original_cost
            matching.Hungarian = original_hungarian

        np.testing.assert_array_equal(matched, np.array([[0, 0]]))
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(seen["trans_cost_matrix"][0, 1, 0]), 1.05, places=6)


if __name__ == "__main__":
    unittest.main()
