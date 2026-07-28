import copy
import sys
import types
import unittest
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
    lap_stub.lapjv = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("lapjv must not run in inference pair cache tests")
    )
    sys.modules["lap"] = lap_stub

if "pyquaternion" not in sys.modules:
    pyquaternion_stub = types.ModuleType("pyquaternion")
    pyquaternion_stub.Quaternion = object
    sys.modules["pyquaternion"] = pyquaternion_stub

from tracker import matching


class _Box:
    def __init__(self, category):
        self.category = category
        self.det_score = 0.8


class _Traj:
    def __init__(self, category, track_id, unmatch_length=0):
        self.bboxes = [_Box(category)]
        self.track_id = track_id
        self.unmatch_length = unmatch_length


def _cfg(*, shadow=False, head=False, cost_mode="geometric"):
    return {
        "CATEGORY_LIST": ["trailer", "car"],
        "CATEGORY_MAP_TO_NUMBER": {"trailer": 0, "car": 1},
        "MATCHING": {
            "BEV": {"MATCHING_MODE": "Greedy"},
            "RV": {"MATCHING_MODE": "Greedy"},
        },
        "THRESHOLD": {
            "BEV": {"COST_THRE": {0: 1.0, 1: 1.0}, "COST_MODE": cost_mode},
            "RV": {"COST_THRE": {0: 1.0, 1: 1.0}},
        },
        "MAMBA_ASSOCIATION_HEAD": {
            "ENABLED": head,
            "MIN_SCORE": 0.5,
            "ALPHA": 1.0,
            "MAX_DELTA": 0.5,
            "ACTIVE_CLASS_STATES": {0: ["unmatched"]},
        },
        "ASSOCIATION_HEAD_AUDIT": {"SHADOW_MODE": shadow},
        "INFERENCE_PAIR_CACHE": {
            "INCLUDE_EMBEDDINGS": True,
            "INCLUDE_PAIR_VECTORS": True,
        },
    }


def _payload(track_count=2, det_count=2, pair_indices=None):
    if pair_indices is None:
        pair_indices = [
            (track_index, det_index)
            for track_index in range(track_count)
            for det_index in range(det_count)
        ]
    return {
        "scene_id": "scene-1",
        "sample_token": "sample-1",
        "frame_id": 4,
        "track_ids": list(range(10, 10 + track_count)),
        "class_ids": [0] * track_count,
        "class_names": ["trailer"] * track_count,
        "state_buckets": ["unmatched"] * track_count,
        "pair_indices": np.asarray(pair_indices, dtype=np.int64).reshape(-1, 2),
        "pair_vectors": np.arange(
            max(1, len(pair_indices) * 2), dtype=np.float32
        )[: len(pair_indices) * 2].reshape(len(pair_indices), 2),
        "track_embeddings": np.arange(
            max(1, track_count * 2), dtype=np.float32
        )[: track_count * 2].reshape(track_count, 2),
        "det_embeddings": np.arange(
            max(1, det_count * 2), dtype=np.float32
        )[: det_count * 2].reshape(det_count, 2),
        "association_scores": np.full(
            (track_count, det_count), 0.9, dtype=np.float32
        ),
    }


def _gt(track_count=2, det_count=2):
    return {
        "track_gt_tokens": [f"gt-{index}" for index in range(track_count)],
        "det_gt_tokens": [f"gt-{index}" for index in range(det_count)],
    }


class InferencePairCacheMatchingTest(unittest.TestCase):
    def setUp(self):
        self.trajs = [
            _Traj("trailer", 10, unmatch_length=1),
            _Traj("trailer", 11, unmatch_length=1),
        ]
        self.dets = [_Box("trailer"), _Box("trailer")]
        self.base = np.asarray([[0.10, 0.30], [0.40, 0.20]], dtype=np.float32)
        self.categories = np.asarray(["trailer", "trailer"])

    def _cost_patch(self, base=None, traj_categories=None, det_categories=None):
        return patch.object(
            matching,
            "cost_calculate_general",
            return_value=(
                self.base if base is None else base,
                self.categories if traj_categories is None else traj_categories,
                self.categories if det_categories is None else det_categories,
            ),
        )

    def test_disabled_and_enabled_outputs_and_inputs_are_identical(self):
        payload = _payload()
        payload_original = copy.deepcopy(payload)
        base_bytes = self.base.tobytes()
        observed = []

        with self._cost_patch():
            disabled = matching.match_trajs_and_dets(self.trajs, self.dets, _cfg())
            enabled = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(enabled[0], disabled[0])
        np.testing.assert_array_equal(enabled[1], disabled[1])
        self.assertEqual(self.base.tobytes(), base_bytes)
        self.assertEqual(len(observed), 2)
        for key, value in payload.items():
            original = payload_original[key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original)
            else:
                self.assertEqual(value, original)

    def test_callback_mutation_cannot_reach_production_or_input_objects(self):
        payload = _payload()
        original_payload = copy.deepcopy(payload)
        original_base = self.base.copy()

        def mutate(groups):
            groups[0]["track_embedding"][0] = -999.0
            groups[0]["candidates"][0]["pair_vector"][0] = -999.0
            groups.clear()

        with self._cost_patch():
            matches, costs = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=mutate,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.20], dtype=np.float32))
        np.testing.assert_array_equal(self.base, original_base)
        for key in ("pair_indices", "pair_vectors", "track_embeddings", "det_embeddings"):
            np.testing.assert_array_equal(payload[key], original_payload[key])

    def test_provider_and_callback_failures_record_one_error_non_strict(self):
        for failing_stage in ("gt_provider", "callback"):
            errors = []
            callbacks = []

            def provider():
                if failing_stage == "gt_provider":
                    raise RuntimeError("provider failed")
                return _gt()

            def callback(groups):
                callbacks.append(groups)
                if failing_stage == "callback":
                    raise RuntimeError("callback failed")

            with self.subTest(stage=failing_stage), self._cost_patch():
                matches, costs = matching.match_trajs_and_dets(
                    self.trajs,
                    self.dets,
                    _cfg(),
                    inference_pair_cache_payload=_payload(),
                    inference_pair_cache_callback=callback,
                    inference_pair_cache_error_callback=errors.append,
                    inference_pair_cache_gt_provider=provider,
                )

            np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
            np.testing.assert_array_equal(
                costs, np.asarray([0.10, 0.20], dtype=np.float32)
            )
            self.assertEqual(len(errors), 1)
            self.assertEqual(errors[0]["stage"], failing_stage)
            self.assertEqual(errors[0]["error_type"], "RuntimeError")
            self.assertEqual(errors[0]["scene_id"], "scene-1")
            self.assertEqual(errors[0]["frame_id"], 4)

    def test_error_callback_failure_never_affects_production(self):
        def fail_error_callback(record):
            raise RuntimeError("error callback failed")

        with self._cost_patch():
            matches, costs = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=_payload(),
                inference_pair_cache_callback=lambda groups: (_ for _ in ()).throw(
                    ValueError("callback failed")
                ),
                inference_pair_cache_error_callback=fail_error_callback,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.20], dtype=np.float32))

    def test_production_solver_failure_is_not_an_export_error(self):
        errors = []
        with self._cost_patch(), patch.object(
            matching,
            "solve_matching_cost_matrix",
            side_effect=RuntimeError("production solver failed"),
        ):
            with self.assertRaisesRegex(RuntimeError, "production solver failed"):
                matching.match_trajs_and_dets(
                    self.trajs,
                    self.dets,
                    _cfg(),
                    inference_pair_cache_payload=_payload(),
                    inference_pair_cache_callback=lambda groups: None,
                    inference_pair_cache_error_callback=errors.append,
                    inference_pair_cache_gt_provider=lambda: _gt(),
                )

        self.assertEqual(errors, [])

    def test_malformed_context_emits_task3_valid_safe_error(self):
        from tracker import inference_pair_cache

        payload = _payload()
        payload["scene_id"] = " invalid "
        payload["sample_token"] = object()
        payload["frame_id"] = -7
        errors = []
        long_message = "x" * 2048

        with self._cost_patch(), patch.object(
            inference_pair_cache,
            "build_candidate_groups",
            side_effect=RuntimeError(long_message),
        ):
            matches, costs = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=lambda groups: None,
                inference_pair_cache_error_callback=errors.append,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.20], dtype=np.float32))
        self.assertEqual(len(errors), 1)
        self.assertEqual(
            inference_pair_cache.validate_error_record(errors[0]), errors[0]
        )
        self.assertEqual(errors[0]["scene_id"], "unknown")
        self.assertEqual(errors[0]["sample_token"], "unknown")
        self.assertEqual(errors[0]["frame_id"], 0)
        self.assertEqual(errors[0]["message"], "x" * 1024)
        self.assertEqual(len(errors[0]["message"]), 1024)

    def test_error_record_validation_failure_is_nonfatal_non_strict(self):
        from tracker import inference_pair_cache

        errors = []
        with self._cost_patch(), patch.object(
            inference_pair_cache,
            "validate_error_record",
            side_effect=RuntimeError("normalization failed"),
        ):
            matches, costs = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=_payload(),
                inference_pair_cache_callback=lambda groups: (_ for _ in ()).throw(
                    ValueError("callback failed")
                ),
                inference_pair_cache_error_callback=errors.append,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.20], dtype=np.float32))
        self.assertEqual(errors, [])

    def test_strict_reraises_original_failure_after_solver_completed(self):
        solver_completed = {"value": False}
        real_solver = matching.solve_matching_cost_matrix

        def solve(*args, **kwargs):
            result = real_solver(*args, **kwargs)
            solver_completed["value"] = True
            return result

        def provider():
            self.assertTrue(solver_completed["value"])
            raise RuntimeError("strict provider failure")

        with self._cost_patch(), patch.object(
            matching, "solve_matching_cost_matrix", side_effect=solve
        ):
            with self.assertRaisesRegex(RuntimeError, "strict provider failure"):
                matching.match_trajs_and_dets(
                    self.trajs,
                    self.dets,
                    _cfg(),
                    inference_pair_cache_payload=_payload(),
                    inference_pair_cache_callback=lambda groups: None,
                    inference_pair_cache_gt_provider=provider,
                    inference_pair_cache_strict=True,
                )

        self.assertTrue(solver_completed["value"])

    def test_gt_provider_is_post_solver_and_never_runs_when_disabled(self):
        state = {"solver_completed": False, "provider_calls": 0}
        real_solver = matching.solve_matching_cost_matrix

        def solve(*args, **kwargs):
            result = real_solver(*args, **kwargs)
            state["solver_completed"] = True
            return result

        def provider():
            self.assertTrue(state["solver_completed"])
            state["provider_calls"] += 1
            return _gt()

        with self._cost_patch(), patch.object(
            matching, "solve_matching_cost_matrix", side_effect=solve
        ):
            matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_gt_provider=provider,
            )
            self.assertEqual(state["provider_calls"], 0)
            state["solver_completed"] = False
            matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=_payload(),
                inference_pair_cache_callback=lambda groups: None,
                inference_pair_cache_gt_provider=provider,
            )

        self.assertEqual(state["provider_calls"], 1)

    def test_empty_tracks_preserve_immediate_return_without_export(self):
        provider = unittest.mock.Mock()
        callback = unittest.mock.Mock()
        matches, costs = matching.match_trajs_and_dets(
            [],
            self.dets,
            _cfg(),
            inference_pair_cache_payload=_payload(track_count=0),
            inference_pair_cache_callback=callback,
            inference_pair_cache_gt_provider=provider,
        )
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(costs.shape, (0,))
        provider.assert_not_called()
        callback.assert_not_called()

    def test_empty_detections_export_zero_candidate_groups_without_solver(self):
        observed = []
        payload = _payload(track_count=2, det_count=0, pair_indices=[])
        cfg = _cfg()
        cfg["MATCHING"]["BEV"]["MATCHING_MODE"] = "Hungarian"
        with patch.object(
            matching,
            "Hungarian",
            side_effect=AssertionError("Hungarian must not run"),
        ), patch.object(
            matching,
            "cost_calculate_general",
            side_effect=AssertionError("cost calculation must not run"),
        ):
            matches, costs = matching.match_trajs_and_dets(
                self.trajs,
                [],
                cfg,
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(det_count=0),
            )

        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(costs.shape, (0,))
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(group["candidate_count"] == 0 for group in observed))

    def test_no_same_class_pair_still_exports_each_track(self):
        observed = []
        dets = [_Box("car")]
        payload = _payload(track_count=2, det_count=1, pair_indices=[])
        with self._cost_patch(
            base=np.asarray([[0.1], [0.2]], dtype=np.float32),
            det_categories=np.asarray(["car"]),
        ):
            matches, _ = matching.match_trajs_and_dets(
                self.trajs,
                dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(det_count=1),
            )

        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(len(observed), 2)
        self.assertTrue(all(group["candidates"] == [] for group in observed))

    def test_nonempty_hungarian_export_preserves_assignment_in_both_shadow_modes(self):
        def fake_hungarian(cost_matrix, thresholds):
            cls_cost = cost_matrix[0]
            diagonal = cls_cost[0, 0] + cls_cost[1, 1]
            off_diagonal = cls_cost[0, 1] + cls_cost[1, 0]
            track_indices = [0, 1] if diagonal <= off_diagonal else [1, 0]
            return (
                [0, 1],
                track_indices,
                np.empty((0,), dtype=int),
                np.empty((0,), dtype=int),
                np.asarray(
                    [
                        cls_cost[0, track_indices[0]],
                        cls_cost[1, track_indices[1]],
                    ]
                ),
            )

        for shadow in (False, True):
            cfg = _cfg(shadow=shadow)
            cfg["MATCHING"]["BEV"]["MATCHING_MODE"] = "Hungarian"
            observed = []
            with self.subTest(shadow=shadow), self._cost_patch(), patch.object(
                matching, "Hungarian", side_effect=fake_hungarian
            ):
                disabled = matching.match_trajs_and_dets(
                    self.trajs, self.dets, cfg
                )
                enabled = matching.match_trajs_and_dets(
                    self.trajs,
                    self.dets,
                    cfg,
                    inference_pair_cache_payload=_payload(),
                    inference_pair_cache_callback=observed.extend,
                    inference_pair_cache_gt_provider=lambda: _gt(),
                )

            np.testing.assert_array_equal(enabled[0], disabled[0])
            np.testing.assert_array_equal(enabled[1], disabled[1])
            np.testing.assert_array_equal(enabled[0], np.asarray([[0, 0], [1, 1]]))
            self.assertEqual(len(observed), 2)

    def test_non_shadow_exports_actual_assignment_with_untouched_base_costs(self):
        observed = []
        payload = _payload()
        base = np.asarray([[0.10, 0.11], [0.11, 0.10]], dtype=np.float32)
        scores = np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)
        with self._cost_patch(base=base):
            matches, _ = matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(head=True),
                association_scores=scores,
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, np.asarray([[1, 0], [0, 1]]))
        self.assertEqual(observed[0]["production_det_index"], 1)
        self.assertAlmostEqual(observed[0]["candidates"][0]["geometric_cost"], 0.10)
        self.assertAlmostEqual(observed[0]["candidates"][1]["geometric_cost"], 0.11)

    def test_shadow_exports_baseline_before_shadow_correction(self):
        observed = []
        payload = _payload(track_count=1, det_count=2)
        base = np.asarray([[0.10, 0.11]], dtype=np.float32)
        scores = np.asarray([[0.0, 1.0]], dtype=np.float32)
        real_correction = matching.apply_pairwise_association_head_to_cost_matrix

        def correction(*args, **kwargs):
            self.assertEqual(len(observed), 1)
            return real_correction(*args, **kwargs)

        with self._cost_patch(
            base=base,
            traj_categories=np.asarray(["trailer"]),
        ), patch.object(
            matching,
            "apply_pairwise_association_head_to_cost_matrix",
            side_effect=correction,
        ):
            matches, _ = matching.match_trajs_and_dets(
                self.trajs[:1],
                self.dets,
                _cfg(shadow=True, head=True),
                association_scores=scores,
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(track_count=1),
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0]]))
        self.assertEqual(observed[0]["production_det_index"], 0)

    def test_uncertainty_aware_normal_and_fallback_forward_export(self):
        for cost_mode in ("full", "geometric"):
            observed = []
            cfg = _cfg(cost_mode=cost_mode)
            cost_patch = (
                patch.object(
                    matching,
                    "cost_calculate_uncertainty_aware",
                    return_value=(self.base, self.categories, self.categories),
                )
                if cost_mode == "full"
                else self._cost_patch()
            )
            with self.subTest(cost_mode=cost_mode), cost_patch:
                matches, costs = matching.match_trajs_and_dets_uncertainty_aware(
                    self.trajs,
                    self.dets,
                    cfg,
                    trk_embeddings=np.eye(2, dtype=np.float32),
                    det_embeddings=np.eye(2, dtype=np.float32),
                    inference_pair_cache_payload=_payload(),
                    inference_pair_cache_callback=observed.extend,
                    inference_pair_cache_gt_provider=lambda: _gt(),
                )

            np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
            np.testing.assert_array_equal(
                costs, np.asarray([0.10, 0.20], dtype=np.float32)
            )
            self.assertEqual(len(observed), 2)

    def test_full_uncertainty_exports_pure_geometry_not_composite_costs(self):
        cfg = _cfg(cost_mode="full")
        cfg["THRESHOLD"]["BEV"]["COST_THRE"] = [0.30, 0.70]
        composite = np.asarray([[0.10, 0.90], [0.90, 0.10]], dtype=np.float32)
        geometric = np.asarray([[0.80, 0.20], [0.20, 0.80]], dtype=np.float32)
        observed = []
        modes = []

        def calculate(*args, **kwargs):
            call_cfg = args[2]
            mode = call_cfg["THRESHOLD"]["BEV"]["COST_MODE"]
            modes.append(mode)
            matrix = composite if mode == "full" else geometric
            return matrix, self.categories, self.categories

        with patch.object(
            matching,
            "cost_calculate_uncertainty_aware",
            side_effect=calculate,
        ):
            disabled = matching.match_trajs_and_dets_uncertainty_aware(
                self.trajs,
                self.dets,
                cfg,
                trk_embeddings=np.eye(2, dtype=np.float32),
                det_embeddings=np.eye(2, dtype=np.float32),
            )
            self.assertEqual(modes, ["full"])
            modes.clear()
            matches, costs = matching.match_trajs_and_dets_uncertainty_aware(
                self.trajs,
                self.dets,
                cfg,
                trk_embeddings=np.eye(2, dtype=np.float32),
                det_embeddings=np.eye(2, dtype=np.float32),
                inference_pair_cache_payload=_payload(),
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: _gt(),
            )

        np.testing.assert_array_equal(matches, disabled[0])
        np.testing.assert_array_equal(costs, disabled[1])
        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.10], dtype=np.float32))
        self.assertEqual(modes, ["full", "geometric"])
        self.assertEqual(cfg["THRESHOLD"]["BEV"]["COST_MODE"], "full")
        self.assertEqual(cfg["THRESHOLD"]["BEV"]["COST_THRE"], [0.30, 0.70])
        first = observed[0]
        self.assertEqual(first["production_det_index"], 0)
        self.assertAlmostEqual(first["production_match_cost"], 0.10)
        self.assertEqual(
            [candidate["candidate_rank"] for candidate in first["candidates"]],
            [2, 1],
        )
        self.assertEqual(
            [candidate["inside_geometric_gate"] for candidate in first["candidates"]],
            [False, True],
        )
        self.assertEqual(
            [candidate["geometric_threshold"] for candidate in first["candidates"]],
            [0.30, 0.30],
        )
        self.assertEqual(
            [candidate["geometric_cost"] for candidate in first["candidates"]],
            [float(geometric[0, 0]), float(geometric[0, 1])],
        )

    def test_full_uncertainty_safety_fallback_exports_general_geometry_once(self):
        cfg = _cfg(cost_mode="full")
        cfg["THRESHOLD"]["BEV"]["COST_THRE"] = [0.30, 0.70]
        composite = np.full((2, 2), 10.0, dtype=np.float32)
        observed = []
        modes = []
        provider_calls = []

        def calculate(*args, **kwargs):
            modes.append(args[2]["THRESHOLD"]["BEV"]["COST_MODE"])
            return composite, self.categories, self.categories

        def provider():
            provider_calls.append(True)
            return _gt()

        with patch.object(
            matching,
            "cost_calculate_uncertainty_aware",
            side_effect=calculate,
        ), self._cost_patch():
            matches, costs = matching.match_trajs_and_dets_uncertainty_aware(
                self.trajs,
                self.dets,
                cfg,
                trk_embeddings=np.eye(2, dtype=np.float32),
                det_embeddings=np.eye(2, dtype=np.float32),
                inference_pair_cache_payload=_payload(),
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=provider,
            )

        np.testing.assert_array_equal(matches, np.asarray([[0, 0], [1, 1]]))
        np.testing.assert_array_equal(costs, np.asarray([0.10, 0.20], dtype=np.float32))
        self.assertEqual(modes, ["full"])
        self.assertEqual(len(provider_calls), 1)
        self.assertEqual(len(observed), 2)
        self.assertAlmostEqual(observed[0]["candidates"][0]["geometric_cost"], 0.10)

    def test_export_uses_shallow_payload_overlay_for_large_arrays(self):
        from tracker import inference_pair_cache

        payload = _payload()
        large_pair_vectors = np.empty((4, 250_000), dtype=np.float32)
        payload["pair_vectors"] = large_pair_vectors
        gt_payload = _gt()
        observed = []

        with self._cost_patch(), patch.object(
            inference_pair_cache,
            "build_candidate_groups",
            return_value=[],
        ) as build:
            matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=observed.extend,
                inference_pair_cache_gt_provider=lambda: gt_payload,
            )

        export_payload = build.call_args.kwargs["payload"]
        self.assertIs(export_payload["pair_vectors"], large_pair_vectors)
        self.assertIs(export_payload["track_embeddings"], payload["track_embeddings"])
        self.assertIs(export_payload["det_embeddings"], payload["det_embeddings"])
        self.assertIsNot(
            export_payload["track_gt_tokens"], gt_payload["track_gt_tokens"]
        )
        self.assertIsNot(export_payload["det_gt_tokens"], gt_payload["det_gt_tokens"])
        self.assertEqual(observed, [])

    def test_provider_and_payload_arrays_are_not_mutated(self):
        payload = _payload()
        gt_payload = {
            "track_gt_tokens": np.asarray(["gt-0", "gt-1"], dtype=object),
            "det_gt_tokens": np.asarray(["gt-0", "gt-1"], dtype=object),
        }
        original_payload = copy.deepcopy(payload)
        original_gt = copy.deepcopy(gt_payload)

        with self._cost_patch():
            matching.match_trajs_and_dets(
                self.trajs,
                self.dets,
                _cfg(),
                inference_pair_cache_payload=payload,
                inference_pair_cache_callback=lambda groups: groups[0].clear(),
                inference_pair_cache_gt_provider=lambda: gt_payload,
            )

        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original_payload[key])
            else:
                self.assertEqual(value, original_payload[key])
        np.testing.assert_array_equal(
            gt_payload["track_gt_tokens"], original_gt["track_gt_tokens"]
        )
        np.testing.assert_array_equal(
            gt_payload["det_gt_tokens"], original_gt["det_gt_tokens"]
        )


if __name__ == "__main__":
    unittest.main()
