import ast
import types
import typing
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch

from training.association_features import compose_pair_vectors
from tracker.inference_pair_cache import validate_error_record


ROOT = Path(__file__).resolve().parents[1]
BASE_TRACKER_PATH = ROOT / "tracker/base_tracker.py"


def _tracker_tree():
    tree = ast.parse(BASE_TRACKER_PATH.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
    )


def _load_methods(*names, extra_globals=None):
    tracker_class = _tracker_tree()
    selected = [
        node
        for node in tracker_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name in names
    ]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"Missing Base3DTracker methods: {sorted(missing)}")
    namespace = {
        "Dict": typing.Dict,
        "List": typing.List,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "Optional": typing.Optional,
        "Tuple": typing.Tuple,
        "Trajectory": object,
        "BBox": object,
        "np": np,
        "torch": torch,
        "compose_pair_vectors": compose_pair_vectors,
        "infer_state_bucket": lambda value: (
            "unmatched" if int(value or 0) > 0 else "matched"
        ),
        "norm_realative_radian": lambda value: (
            (float(value) + np.pi) % (2.0 * np.pi) - np.pi
        ),
    }
    namespace.update(extra_globals or {})
    exec(
        compile(
            ast.Module(body=selected, type_ignores=[]),
            str(BASE_TRACKER_PATH),
            "exec",
        ),
        namespace,
    )
    return {name: namespace[name] for name in names}


def _bbox(category, xyz, *, yaw=0.0, lwh=(1.0, 1.0, 1.0), score=0.5):
    return types.SimpleNamespace(
        category=category,
        global_xyz=list(xyz),
        global_yaw=float(yaw),
        lwh=list(lwh),
        det_score=float(score),
    )


def _traj(track_id, category, xyz, *, unmatch_length=0, **bbox_kwargs):
    return types.SimpleNamespace(
        track_id=track_id,
        unmatch_length=unmatch_length,
        bboxes=[_bbox(category, xyz, **bbox_kwargs)],
    )


class _Head:
    def __init__(self):
        self.pairs = None
        self.classes = None

    def __call__(self, pairs, classes):
        self.pairs = pairs.detach().cpu().clone()
        self.classes = classes.detach().cpu().clone()
        return torch.arange(
            pairs.shape[0], dtype=torch.float32, device=pairs.device
        )


class ExactPayloadTest(unittest.TestCase):
    method_names = (
        "_bbox_center_distance_xy",
        "_bbox_size_l1",
        "_inference_pair_cache_enabled",
        "_compute_pairwise_association_payload",
    )

    def _tracker(self, *, head=None, cache=True):
        methods = _load_methods(*self.method_names)
        tracker = types.SimpleNamespace(
            cfg={
                "CATEGORY_MAP_TO_NUMBER": {
                    "bicycle": 0,
                    "trailer": 5,
                }
            },
            device=torch.device("cpu"),
            pairwise_association_head=head,
            inference_pair_cache=(object() if cache else None),
        )
        for name, method in methods.items():
            if isinstance(method, staticmethod):
                setattr(tracker, name, method.__func__)
            else:
                setattr(tracker, name, types.MethodType(method, tracker))
        return tracker

    def test_cache_enabled_without_head_builds_exact_shared_pair_payload(self):
        tracker = self._tracker(head=None, cache=True)
        trajs = [
            _traj(
                17,
                "trailer",
                (1.0, 2.0, 0.0),
                yaw=0.25,
                lwh=(4.0, 2.0, 1.5),
                score=0.4,
                unmatch_length=2,
            )
        ]
        dets = [
            _bbox(
                "trailer",
                (4.0, 6.0, 0.0),
                yaw=-0.25,
                lwh=(5.0, 2.5, 1.0),
                score=0.9,
            )
        ]
        trk_embeddings = np.asarray([[1.0, 2.0]], dtype=np.float32)
        det_embeddings = np.asarray([[3.0, 5.0]], dtype=np.float32)
        frame = types.SimpleNamespace(
            scene_id="scene-1", cur_sample_token="sample-2", frame_id=3
        )

        scores, payload = tracker._compute_pairwise_association_payload(
            trajs,
            dets,
            trk_embeddings,
            det_embeddings,
            ["unmatched"],
            frame,
        )

        self.assertIsNone(scores)
        self.assertEqual(payload["scene_id"], "scene-1")
        self.assertEqual(payload["sample_token"], "sample-2")
        self.assertEqual(payload["frame_id"], 3)
        self.assertEqual(payload["track_ids"], [17])
        self.assertEqual(payload["class_ids"], [5])
        self.assertEqual(payload["class_names"], ["trailer"])
        self.assertEqual(payload["state_buckets"], ["unmatched"])
        np.testing.assert_array_equal(payload["pair_indices"], [[0, 0]])
        geometry = torch.tensor(
            [[5.0, 0.5, 2.0, 0.4, 0.9, 1.0, 5.0]],
            dtype=torch.float32,
        )
        expected = compose_pair_vectors(
            torch.tensor(trk_embeddings),
            torch.tensor(det_embeddings),
            geometry,
        ).numpy()
        np.testing.assert_allclose(payload["pair_vectors"], expected)

        trk_embeddings[:] = -100.0
        det_embeddings[:] = -200.0
        self.assertEqual(payload["track_embeddings"].dtype, np.float32)
        self.assertEqual(payload["det_embeddings"].dtype, np.float32)
        self.assertEqual(payload["pair_vectors"].dtype, np.float32)
        np.testing.assert_array_equal(payload["track_embeddings"], [[1.0, 2.0]])
        np.testing.assert_array_equal(payload["det_embeddings"], [[3.0, 5.0]])

    def test_loaded_head_scores_the_same_pair_tensor_stored_in_payload(self):
        head = _Head()
        tracker = self._tracker(head=head, cache=True)
        trajs = [_traj(1, "bicycle", (0.0, 0.0, 0.0))]
        dets = [
            _bbox("bicycle", (1.0, 0.0, 0.0)),
            _bbox("trailer", (1.0, 0.0, 0.0)),
        ]
        scores, payload = tracker._compute_pairwise_association_payload(
            trajs,
            dets,
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0], [8.0, 9.0]], dtype=np.float32),
            ["matched"],
            types.SimpleNamespace(
                scene_id="scene", cur_sample_token="sample", frame_id=4
            ),
        )
        np.testing.assert_array_equal(payload["pair_indices"], [[0, 0]])
        np.testing.assert_allclose(head.pairs.numpy(), payload["pair_vectors"])
        self.assertAlmostEqual(float(scores[0, 0]), 0.5)
        self.assertTrue(np.isnan(scores[0, 1]))
        np.testing.assert_allclose(
            payload["association_scores"], scores, equal_nan=True
        )

    def test_include_flags_do_not_disable_pair_composition_or_head_scoring(self):
        head = _Head()
        tracker = self._tracker(head=head, cache=True)
        tracker.inference_pair_cache_cfg = {
            "INCLUDE_EMBEDDINGS": False,
            "INCLUDE_PAIR_VECTORS": False,
        }
        scores, payload = tracker._compute_pairwise_association_payload(
            [_traj(1, "bicycle", (0.0, 0.0, 0.0))],
            [_bbox("bicycle", (1.0, 0.0, 0.0))],
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.asarray([[3.0, 4.0]], dtype=np.float32),
            ["matched"],
            types.SimpleNamespace(
                scene_id="scene", cur_sample_token="sample", frame_id=4
            ),
        )
        self.assertIsNotNone(scores)
        self.assertEqual(head.pairs.shape[0], 1)
        self.assertIn("pair_vectors", payload)
        self.assertIn("track_embeddings", payload)
        self.assertIn("det_embeddings", payload)

    def test_empty_detections_produce_complete_zero_candidate_payload(self):
        tracker = self._tracker(head=None, cache=True)
        scores, payload = tracker._compute_pairwise_association_payload(
            [_traj(17, "trailer", (1.0, 2.0, 0.0), unmatch_length=1)],
            [],
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
            ["unmatched"],
            types.SimpleNamespace(
                scene_id="scene", cur_sample_token="sample", frame_id=5
            ),
        )
        self.assertIsNone(scores)
        self.assertEqual(payload["det_embeddings"].shape, (0, 2))
        self.assertEqual(payload["det_embeddings"].dtype, np.float32)
        self.assertEqual(payload["pair_indices"].shape, (0, 2))
        self.assertEqual(payload["pair_vectors"].shape, (0, 15))
        self.assertEqual(payload["pair_vectors"].dtype, np.float32)
        np.testing.assert_array_equal(payload["track_embeddings"], [[1.0, 2.0]])
        self.assertEqual(payload["track_ids"], [17])

    def test_shape_errors_record_without_gt_and_strict_controls_raise(self):
        methods = _load_methods(
            "_record_inference_pair_payload_error",
            "_inference_pair_cache_options",
        )
        frame = types.SimpleNamespace(
            scene_id="scene", cur_sample_token="sample", frame_id=6
        )
        trajs = [_traj(1, "bicycle", (0.0, 0.0, 0.0))]
        for strict in (False, True):
            tracker = self._tracker(head=None, cache=True)
            tracker.inference_pair_cache_cfg = {"STRICT": strict}
            errors = []
            gt_calls = []
            tracker._record_inference_pair_cache_error = errors.append
            tracker._record_inference_pair_payload_error = types.MethodType(
                methods["_record_inference_pair_payload_error"], tracker
            )
            tracker._association_det_gt_tokens = lambda *_: gt_calls.append(1)
            tracker._record_inference_pair_groups = lambda groups: None
            tracker._inference_pair_cache_options = types.MethodType(
                methods["_inference_pair_cache_options"], tracker
            )

            call = lambda: tracker._compute_pairwise_association_payload(
                trajs,
                [],
                np.asarray([[1.0, 2.0]], dtype=np.float32),
                np.empty((0, 3), dtype=np.float32),
                ["matched"],
                frame,
            )
            if strict:
                with self.assertRaisesRegex(ValueError, "shape mismatch"):
                    call()
            else:
                self.assertEqual(call(), (None, None))
            self.assertEqual(len(errors), 1)
            self.assertEqual(validate_error_record(errors[0]), errors[0])
            self.assertEqual(errors[0]["stage"], "payload")
            self.assertEqual(errors[0]["error_type"], "ValueError")
            self.assertEqual(
                tracker._inference_pair_cache_options(
                    frame, trajs, [], None
                ),
                {},
            )
            self.assertEqual(gt_calls, [])

    def test_head_only_shape_error_preserves_legacy_silent_none(self):
        tracker = self._tracker(head=_Head(), cache=False)
        scores, payload = tracker._compute_pairwise_association_payload(
            [_traj(1, "bicycle", (0.0, 0.0, 0.0))],
            [],
            np.asarray([[1.0, 2.0]], dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            ["matched"],
            types.SimpleNamespace(
                scene_id="scene", cur_sample_token="sample", frame_id=6
            ),
        )
        self.assertIsNone(scores)
        self.assertIsNone(payload)

    def test_head_and_cache_disabled_keep_fast_return(self):
        tracker = self._tracker(head=None, cache=False)
        scores, payload = tracker._compute_pairwise_association_payload(
            [_traj(1, "bicycle", (0.0, 0.0, 0.0))],
            [_bbox("bicycle", (1.0, 0.0, 0.0))],
            None,
            None,
            ["matched"],
            None,
        )
        self.assertIsNone(scores)
        self.assertIsNone(payload)


class CacheConfigurationTest(unittest.TestCase):
    def test_defaults_are_safe_and_accumulator_is_only_constructed_when_enabled(self):
        calls = []

        class Accumulator:
            def __init__(self, **kwargs):
                calls.append(kwargs)

        methods = _load_methods(
            "_inference_pair_cache_cfg",
            "_initialize_inference_pair_cache",
            extra_globals={"InferencePairCacheAccumulator": Accumulator},
        )
        tracker = types.SimpleNamespace(cfg={})
        tracker._inference_pair_cache_cfg = types.MethodType(
            methods["_inference_pair_cache_cfg"], tracker
        )
        methods["_initialize_inference_pair_cache"](tracker)
        self.assertIsNone(tracker.inference_pair_cache)
        self.assertEqual(calls, [])
        self.assertEqual(
            tracker.inference_pair_cache_cfg,
            {
                "ENABLED": False,
                "MAX_GROUPS_PER_CLASS_STATE": 0,
                "INCLUDE_EMBEDDINGS": True,
                "INCLUDE_PAIR_VECTORS": True,
                "REQUIRE_AUDIT_GT": False,
                "STRICT": False,
            },
        )

        manifest = {"checkpoint_sha256": "abc", "worker": {"rank": 0}}
        tracker = types.SimpleNamespace(
            cfg={
                "INFERENCE_PAIR_CACHE": {
                    "ENABLED": True,
                    "MAX_GROUPS_PER_CLASS_STATE": 9,
                    "REQUIRE_AUDIT_GT": True,
                },
                "_INFERENCE_PAIR_CACHE_RUNTIME_MANIFEST": manifest,
            }
        )
        tracker._inference_pair_cache_cfg = types.MethodType(
            methods["_inference_pair_cache_cfg"], tracker
        )
        methods["_initialize_inference_pair_cache"](tracker)
        self.assertIsNotNone(tracker.inference_pair_cache)
        self.assertIs(calls[-1]["runtime_manifest"], manifest)
        self.assertEqual(calls[-1]["runtime_manifest"], manifest)
        self.assertEqual(calls[-1]["max_groups_per_class_state"], 9)
        self.assertTrue(calls[-1]["require_audit_gt"])

    def test_record_and_export_helpers_are_noops_when_disabled(self):
        methods = _load_methods(
            "_record_inference_pair_groups",
            "_record_inference_pair_cache_error",
            "export_inference_pair_cache_state",
        )
        tracker = types.SimpleNamespace(inference_pair_cache=None)
        methods["_record_inference_pair_groups"](tracker, [{"x": 1}])
        methods["_record_inference_pair_cache_error"](tracker, {"x": 1})
        self.assertIsNone(methods["export_inference_pair_cache_state"](tracker))


class LazyGroundTruthTest(unittest.TestCase):
    def test_provider_is_lazy_and_returns_copied_aligned_tokens(self):
        methods = _load_methods(
            "_inference_pair_cache_enabled",
            "_association_det_gt_tokens",
            "_inference_pair_cache_options",
        )
        calls = {"det": 0}
        tracker = types.SimpleNamespace(
            inference_pair_cache=object(),
            inference_pair_cache_cfg={"STRICT": False},
            _association_track_gt_tokens={17: "track-gt"},
        )
        tracker._inference_pair_cache_enabled = types.MethodType(
            methods["_inference_pair_cache_enabled"], tracker
        )

        def det_tokens(self, frame_info, dets):
            calls["det"] += 1
            return ["det-gt"]

        tracker._association_det_gt_tokens = types.MethodType(det_tokens, tracker)
        tracker._record_inference_pair_groups = lambda groups: None
        tracker._record_inference_pair_cache_error = lambda record: None
        trajs = [_traj(17, "trailer", (0.0, 0.0, 0.0))]
        dets = [_bbox("trailer", (1.0, 0.0, 0.0))]
        options = methods["_inference_pair_cache_options"](
            tracker,
            types.SimpleNamespace(),
            trajs,
            dets,
            {"pair_indices": np.asarray([[0, 0]])},
        )
        self.assertEqual(calls["det"], 0)
        gt = options["inference_pair_cache_gt_provider"]()
        self.assertEqual(calls["det"], 1)
        self.assertEqual(gt["track_gt_tokens"], ["track-gt"])
        self.assertEqual(gt["det_gt_tokens"], ["det-gt"])
        gt["track_gt_tokens"][0] = "mutated"
        self.assertEqual(tracker._association_track_gt_tokens[17], "track-gt")

    def test_sidecar_is_not_blocked_when_only_cache_is_enabled(self):
        methods = _load_methods(
            "_inference_pair_cache_enabled",
            "_association_frame_gt_tokens",
            "_update_association_gt_sidecar",
        )
        tracker = types.SimpleNamespace(
            association_head_audit=None,
            inference_pair_cache=object(),
            inference_pair_cache_cfg={"STRICT": False},
            _association_track_gt_tokens={},
        )
        for name, method in methods.items():
            setattr(tracker, name, types.MethodType(method, tracker))
        frame = types.SimpleNamespace(
            bboxes=[object()], audit_gt_tokens=["instance-1"]
        )
        tracker._update_association_gt_sidecar(12, frame, 0)
        self.assertEqual(tracker._association_track_gt_tokens[12], "instance-1")


class MainPathWiringTest(unittest.TestCase):
    def test_cache_enabled_participates_in_detection_embedding_gate(self):
        method = next(
            node
            for node in _tracker_tree().body
            if isinstance(node, ast.FunctionDef) and node.name == "track_single_frame"
        )
        calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_inference_pair_cache_enabled"
        ]
        self.assertGreaterEqual(len(calls), 1)

    def test_cache_only_does_not_embed_detections_without_tracks(self):
        method = _load_methods("_should_compute_detection_embeddings")[
            "_should_compute_detection_embeddings"
        ]
        should_compute = method.__func__ if isinstance(method, staticmethod) else method
        self.assertFalse(should_compute(0, 3, False, True))
        self.assertTrue(should_compute(0, 3, True, True))
        self.assertTrue(should_compute(1, 3, False, True))
        self.assertFalse(should_compute(1, 0, False, True))

    def test_cache_only_embeddings_are_not_fed_back_into_production_matching(self):
        method = next(
            node
            for node in _tracker_tree().body
            if isinstance(node, ast.FunctionDef) and node.name == "track_single_frame"
        )
        payload_calls = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_compute_pairwise_association_payload"
        ]
        self.assertEqual(len(payload_calls), 1)
        self.assertIsInstance(payload_calls[0].args[3], ast.Name)
        self.assertEqual(payload_calls[0].args[3].id, "det_embeddings_all")

        production_kwargs = [
            node
            for node in ast.walk(method)
            if isinstance(node, ast.Dict)
            and any(
                isinstance(key, ast.Constant)
                and key.value == "det_embeddings"
                for key in node.keys
            )
        ]
        self.assertEqual(len(production_kwargs), 1)
        det_value = next(
            value
            for key, value in zip(
                production_kwargs[0].keys, production_kwargs[0].values
            )
            if isinstance(key, ast.Constant)
            and key.value == "det_embeddings"
        )
        self.assertIsInstance(det_value, ast.Name)
        self.assertEqual(det_value.id, "det_embeddings_for_matching")

    def test_default_and_compat_main_match_calls_receive_cache_options(self):
        tracker_class = _tracker_tree()
        methods = {
            node.name: node
            for node in tracker_class.body
            if isinstance(node, ast.FunctionDef)
        }
        checks = {
            "_track_single_stage_mctrack_compat": "match_trajs_and_dets",
            "track_single_frame": "match_trajs_and_dets_uncertainty_aware",
        }
        for method_name, matcher_name in checks.items():
            calls = [
                node
                for node in ast.walk(methods[method_name])
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_run_primary_match"
            ]
            self.assertEqual(len(calls), 1)
            self.assertIsInstance(calls[0].args[0], ast.Name)
            self.assertEqual(calls[0].args[0].id, matcher_name)
            self.assertIsInstance(calls[0].args[4], ast.Name)
            self.assertEqual(calls[0].args[4].id, "cache_options")

    def test_empty_detection_dynamic_export_for_compat_and_default_matchers(self):
        from tests import test_inference_pair_cache_matching as fixture

        methods = _load_methods(
            "_should_run_primary_match",
            "_run_primary_match",
        )
        tracker = types.SimpleNamespace(cfg=fixture._cfg())
        for name, method in methods.items():
            if isinstance(method, staticmethod):
                setattr(tracker, name, method.__func__)
            else:
                setattr(tracker, name, types.MethodType(method, tracker))

        payload = fixture._payload(
            track_count=1, det_count=0, pair_indices=[]
        )
        observed = []
        gt_calls = []
        options = {
            "inference_pair_cache_payload": payload,
            "inference_pair_cache_callback": observed.extend,
            "inference_pair_cache_error_callback": lambda record: self.fail(
                f"unexpected export error: {record}"
            ),
            "inference_pair_cache_gt_provider": lambda: (
                gt_calls.append(1)
                or {"track_gt_tokens": ["gt-0"], "det_gt_tokens": []}
            ),
            "inference_pair_cache_strict": True,
        }
        trajs = [fixture._Traj("trailer", 10, unmatch_length=1)]
        for matcher in (
            fixture.matching.match_trajs_and_dets,
            fixture.matching.match_trajs_and_dets_uncertainty_aware,
        ):
            observed.clear()
            before_gt = len(gt_calls)
            matches, costs = tracker._run_primary_match(
                matcher,
                trajs,
                [],
                lambda: {},
                options,
            )
            self.assertEqual(matches.shape, (0, 2))
            self.assertEqual(costs.shape, (0,))
            self.assertEqual(len(gt_calls), before_gt + 1)
            self.assertEqual(len(observed), 1)
            self.assertEqual(observed[0]["candidates"], [])
            self.assertEqual(observed[0]["gt_target_status"], "no_match")

    def test_empty_detection_cache_disabled_does_not_call_matcher_or_kwargs(self):
        methods = _load_methods(
            "_should_run_primary_match",
            "_run_primary_match",
        )
        tracker = types.SimpleNamespace(cfg={})
        for name, method in methods.items():
            if isinstance(method, staticmethod):
                setattr(tracker, name, method.__func__)
            else:
                setattr(tracker, name, types.MethodType(method, tracker))
        calls = []

        def fail_matcher(*args, **kwargs):
            calls.append("matcher")
            raise AssertionError("matcher must not run")

        def fail_kwargs():
            calls.append("kwargs")
            raise AssertionError("kwargs provider must not run")

        matches, costs = tracker._run_primary_match(
            fail_matcher,
            [object()],
            [],
            fail_kwargs,
            {},
        )
        self.assertEqual(matches.shape, (0, 2))
        self.assertEqual(costs.shape, (0,))
        self.assertEqual(calls, [])

    def test_cache_options_are_only_built_for_the_primary_bev_match(self):
        source = BASE_TRACKER_PATH.read_text(encoding="utf-8")
        self.assertEqual(source.count("_inference_pair_cache_options("), 2)


if __name__ == "__main__":
    unittest.main()
