import ast
import copy
import importlib
import json
import os
import stat
import sys
import tempfile
import types
import unittest
from collections.abc import Mapping, Sequence
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tracker.association_head_audit import AssociationHeadAuditAccumulator


ROOT = Path(__file__).resolve().parents[1]


def _load_main_audit_functions():
    tree = ast.parse((ROOT / "main.py").read_text(encoding="utf-8"))
    wanted = {
        "_build_association_head_audit_cfg",
        "_write_merged_association_head_audit",
    }
    functions = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in wanted
    ]
    namespace = {"AssociationHeadAuditAccumulator": AssociationHeadAuditAccumulator}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "main.py", "exec"), namespace)
    return namespace["_write_merged_association_head_audit"]


def _load_base_tracker_methods(*names):
    tree = ast.parse((ROOT / "tracker/base_tracker.py").read_text(encoding="utf-8"))
    tracker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
    )
    selected = [
        node
        for node in tracker_class.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names
    ]
    missing = set(names) - {node.name for node in selected}
    if missing:
        raise AssertionError(f"Missing Base3DTracker methods: {sorted(missing)}")
    namespace = {
        "AssociationHeadAuditAccumulator": AssociationHeadAuditAccumulator,
        "Dict": dict,
        "List": list,
        "Mapping": Mapping,
        "Sequence": Sequence,
        "infer_state_bucket": lambda unmatch_length: (
            "unmatched" if int(unmatch_length or 0) > 0 else "matched"
        ),
        "np": np,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "tracker/base_tracker.py", "exec"),
        namespace,
    )
    return {name: namespace[name] for name in names}


class SceneContextWiringTest(unittest.TestCase):
    def test_dataset_copies_scene_and_sample_context_to_frame(self):
        nusc_stub = types.ModuleType("utils.nusc_utils")
        nusc_stub.obtain_box_bottom_corners = lambda *args, **kwargs: None
        nusc_stub.filter_bboxes_with_nms = lambda boxes, cfg: boxes
        pyquaternion_stub = types.ModuleType("pyquaternion")
        pyquaternion_stub.Quaternion = object
        with patch.dict(
            sys.modules,
            {
                "utils.nusc_utils": nusc_stub,
                "pyquaternion": pyquaternion_stub,
                "cv2": types.ModuleType("cv2"),
            },
        ):
            module = importlib.import_module("dataset.baseversion_dataset")
            dataset = module.BaseVersionTrackingDataset(
                "scene-real",
                [
                    {
                        "frame_id": "3",
                        "timestamp": 12.5,
                        "cur_sample_token": "sample-real",
                        "transform_matrix": np.eye(4).tolist(),
                        "bboxes": [
                            {
                                "category": "trailer",
                                "detection_score": 0.9,
                                "instance_token": "det-instance-real",
                                "lwh": [4.0, 2.0, 1.5],
                                "global_xyz": [1.0, 2.0, 0.0],
                                "global_orientation": [1.0, 0.0, 0.0, 0.0],
                                "global_yaw": 0.0,
                                "global_velocity": [0.0, 0.0],
                                "global_acceleration": [0.0, 0.0],
                                "bbox_image": {},
                            }
                        ],
                    }
                ],
                cfg={
                    "CATEGORY_MAP_TO_NUMBER": {"trailer": 0},
                    "TRACKING_MODE": "ONLINE",
                    "THRESHOLD": {"INPUT_SCORE": {"ONLINE": {0: 0.0}}},
                    "DATASET": "kitti",
                },
            )

            frame = dataset[0]
        self.assertEqual(frame.scene_id, "scene-real")
        self.assertEqual(frame.cur_sample_token, "sample-real")
        self.assertEqual(frame.bboxes[0].instance_token, "det-instance-real")

    def test_tracker_accumulator_constructor_receives_runtime_contract(self):
        tree = ast.parse((ROOT / "tracker/base_tracker.py").read_text(encoding="utf-8"))
        tracker_class = next(
            node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
        )
        init = next(
            node for node in tracker_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        calls = [
            node
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AssociationHeadAuditAccumulator"
        ]
        self.assertEqual(len(calls), 1)
        self.assertEqual(
            {keyword.arg for keyword in calls[0].keywords},
            {"max_samples_per_bucket", "runtime_manifest", "strict_manifest"},
        )

    def test_tracker_decision_callback_requires_dict_sequence_and_adds_each(self):
        method = _load_base_tracker_methods("_record_association_shadow_decisions")[
            "_record_association_shadow_decisions"
        ]
        tracker = types.SimpleNamespace(
            association_head_audit=AssociationHeadAuditAccumulator()
        )
        records = [
            AssociationHeadAuditAccumulatorTest._decision(production_det_index=0),
            AssociationHeadAuditAccumulatorTest._decision(production_det_index=2),
        ]

        method(tracker, records)
        self.assertEqual(
            tracker.association_head_audit.to_summary()["decision_buckets"][0]["total_decisions"],
            2,
        )
        with self.assertRaises(TypeError):
            method(tracker, {"not": "a sequence of records"})
        with self.assertRaises(TypeError):
            method(tracker, [records[0], ("not", "a dict")])

    def test_tracker_shadow_metadata_uses_real_lifecycle_and_context(self):
        methods = _load_base_tracker_methods(
            "_real_association_gt_token",
            "_association_shadow_options",
            "_record_association_shadow_decisions",
            "_record_association_shadow_error",
        )
        tracker = types.SimpleNamespace(
            association_head_audit=AssociationHeadAuditAccumulator(),
            association_head_audit_cfg={"ENABLED": True, "SHADOW_MODE": True, "STRICT": False},
            cfg={"CATEGORY_MAP_TO_NUMBER": {"trailer": 5}},
        )
        traj_box = types.SimpleNamespace(
            category="trailer", is_fake=False, instance_token="track-gt"
        )
        traj = types.SimpleNamespace(
            track_id=17,
            unmatch_length=2,
            status_flag=1,
            bboxes=[traj_box],
        )
        det = types.SimpleNamespace(
            category="trailer", is_fake=False, instance_token="det-gt"
        )
        frame = types.SimpleNamespace(
            scene_id="scene-real",
            cur_sample_token="sample-real",
            frame_id=9,
        )

        tracker._real_association_gt_token = types.MethodType(
            methods["_real_association_gt_token"], tracker
        )
        tracker._record_association_shadow_decisions = types.MethodType(
            methods["_record_association_shadow_decisions"], tracker
        )
        tracker._record_association_shadow_error = types.MethodType(
            methods["_record_association_shadow_error"], tracker
        )
        options = methods["_association_shadow_options"](tracker, frame, [traj], [det])
        metadata = options["association_shadow_metadata"]

        self.assertEqual(metadata["track_ids"], [17])
        self.assertEqual(metadata["class_ids"], [5])
        self.assertEqual(metadata["class_names"], ["trailer"])
        self.assertEqual(metadata["state_buckets"], ["unmatched"])
        self.assertEqual(metadata["track_gt_tokens"], ["track-gt"])
        self.assertEqual(metadata["det_gt_tokens"], ["det-gt"])
        self.assertEqual(
            metadata["context"],
            {"scene_id": "scene-real", "sample_token": "sample-real", "frame_id": 9},
        )
        self.assertIs(
            options["association_shadow_callback"].__self__, tracker
        )

        no_gt_traj = types.SimpleNamespace(
            track_id=18,
            unmatch_length=0,
            status_flag=0,
            bboxes=[types.SimpleNamespace(category="trailer", is_fake=True)],
        )
        no_gt_det = types.SimpleNamespace(category="trailer", is_fake=False)
        no_gt = methods["_association_shadow_options"](
            tracker, frame, [no_gt_traj], [no_gt_det]
        )["association_shadow_metadata"]
        self.assertEqual(no_gt["track_gt_tokens"], [None])
        self.assertEqual(no_gt["det_gt_tokens"], [None])

        tracker.association_head_audit_cfg["SHADOW_MODE"] = False
        self.assertEqual(
            methods["_association_shadow_options"](tracker, frame, [traj], [det]),
            {},
        )


class AssociationHeadAuditAccumulatorTest(unittest.TestCase):
    @staticmethod
    def _decision(**overrides):
        record = {
            "class_id": 5,
            "class_name": "trailer",
            "track_state_bucket": "unmatched",
            "assignment_changed": True,
            "production_det_index": 0,
            "shadow_det_index": 1,
            "threshold_crossing": False,
            "outcome": "improved",
            "production_selected_association_score": 0.1,
            "shadow_selected_association_score": 0.9,
            "original_cost_gap": 0.01,
        }
        record.update(overrides)
        return record

    @staticmethod
    def _pair(**overrides):
        record = {
            "class_id": 5,
            "class_name": "trailer",
            "state_bucket": "unmatched",
            "score": 0.8,
            "delta": 0.02,
            "cost_before": 1.0,
            "cost_after": 1.02,
            "active": True,
            "finite": True,
        }
        record.update(overrides)
        return record

    @staticmethod
    def _legacy_pair_state():
        return {
            "schema_version": 1,
            "buckets": [
                {
                    "class_id": 2,
                    "class_name": "bicycle",
                    "state_bucket": "matched",
                    "pair_count": 2,
                    "finite_pair_count": 2,
                    "active_pair_count": 1,
                    "penalized_pair_count": 1,
                    "avg_score": 0.4,
                    "avg_delta": 0.01,
                    "avg_cost_before": 1.0,
                    "avg_cost_after": 1.01,
                }
            ],
        }

    def test_summarizes_penalties_by_class_and_state(self):
        audit = AssociationHeadAuditAccumulator()

        audit.add_pair(
            class_id=5,
            class_name="trailer",
            state_bucket="unmatched",
            score=0.4,
            delta=0.01,
            cost_before=1.0,
            cost_after=1.01,
            active=True,
            finite=True,
        )
        audit.add_pair(
            class_id=5,
            class_name="trailer",
            state_bucket="unmatched",
            score=0.8,
            delta=0.0,
            cost_before=0.9,
            cost_after=0.9,
            active=True,
            finite=True,
        )

        summary = audit.to_summary()
        bucket = summary["buckets"][0]

        self.assertEqual(bucket["class_id"], 5)
        self.assertEqual(bucket["class_name"], "trailer")
        self.assertEqual(bucket["state_bucket"], "unmatched")
        self.assertEqual(bucket["pair_count"], 2)
        self.assertEqual(bucket["active_pair_count"], 2)
        self.assertEqual(bucket["penalized_pair_count"], 1)
        self.assertAlmostEqual(bucket["penalized_ratio"], 0.5)
        self.assertAlmostEqual(bucket["avg_score"], 0.6)
        self.assertAlmostEqual(bucket["avg_delta"], 0.005)
        self.assertAlmostEqual(bucket["score_quantiles"]["p50"], 0.6)
        self.assertAlmostEqual(bucket["score_quantiles"]["p90"], 0.76)
        self.assertAlmostEqual(bucket["delta_quantiles"]["p50"], 0.005)

    def test_keeps_bounded_sample_records_for_replay_debugging(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)

        for idx in range(3):
            audit.add_pair(
                class_id=2,
                class_name="bicycle",
                state_bucket="matched",
                score=0.1 * idx,
                delta=0.01 * idx,
                cost_before=1.0 + idx,
                cost_after=1.0 + idx + 0.01 * idx,
                active=idx != 2,
                finite=True,
                sample={
                    "scene_id": "scene",
                    "track_index": 0,
                    "det_index": idx,
                    "center_distance": float(idx),
                },
            )

        bucket = audit.to_summary()["buckets"][0]
        self.assertEqual(bucket["sample_count"], 2)
        self.assertEqual(len(bucket["samples"]), 2)
        self.assertEqual(bucket["samples"][0]["det_index"], 0)
        self.assertEqual(bucket["samples"][1]["det_index"], 1)

    def test_pair_statistics_capacity_is_fixed_after_one_hundred_thousand_pairs(self):
        audit = AssociationHeadAuditAccumulator()
        for index in range(100_000):
            value = float((index % 2001) - 1000) / 37.0
            audit.add_pair(
                **self._pair(
                    score=value,
                    delta=abs(value),
                    cost_before=value + 30.0,
                    cost_after=value + 31.0,
                )
            )

        bucket = next(iter(audit._buckets.values()))
        self.assertFalse(hasattr(bucket, "scores"))
        self.assertEqual(
            set(bucket.histograms),
            {"score", "delta", "cost_before", "cost_after"},
        )
        self.assertTrue(
            all(
                len(histogram.counts) == 4096
                for histogram in bucket.histograms.values()
            )
        )
        state_bucket = audit.export_state()["pair_buckets"][0]
        self.assertEqual(state_bucket["pair_count"], 100_000)
        self.assertTrue(
            all(
                len(sketch["bins"]) <= 4096
                for sketch in state_bucket["quantile_sketches"].values()
            )
        )

    def test_pair_quantiles_and_counts_are_identical_across_merge_trees(self):
        workers = []
        for worker_index, values in enumerate(
            (
                (-10.0, -0.25, 0.0, 0.5),
                (0.125, 0.25, 0.75, 8.0, 64.0),
                (-4.0, -1.0, 0.0625, 1.0, 16.0, 128.0),
            )
        ):
            worker = AssociationHeadAuditAccumulator()
            for value in values:
                worker.add_pair(
                    **self._pair(
                        score=value,
                        delta=abs(value) / 10.0,
                        cost_before=value + 20.0,
                        cost_after=value + 21.0,
                        active=worker_index != 1,
                    )
                )
            workers.append(worker.export_state())

        def merge(states):
            result = AssociationHeadAuditAccumulator()
            for state in states:
                result.merge_state(copy.deepcopy(state))
            return result

        direct = merge(workers)
        left_tree = merge((merge(workers[:2]).export_state(), workers[2]))
        right_tree = merge((workers[0], merge(workers[1:]).export_state()))

        expected = direct.export_state()["pair_buckets"][0]
        for result in (left_tree, right_tree):
            actual = result.export_state()["pair_buckets"][0]
            self.assertEqual(actual["pair_count"], expected["pair_count"])
            self.assertEqual(
                actual["finite_pair_count"], expected["finite_pair_count"]
            )
            self.assertEqual(
                actual["active_pair_count"], expected["active_pair_count"]
            )
            self.assertEqual(
                actual["penalized_pair_count"], expected["penalized_pair_count"]
            )
            self.assertEqual(actual["quantile_sketches"], expected["quantile_sketches"])
            for field_name in (
                "score_quantiles",
                "delta_quantiles",
                "cost_before_quantiles",
                "cost_after_quantiles",
            ):
                self.assertEqual(actual[field_name], expected[field_name])

    def test_add_pair_rejects_nonfinite_numeric_fields_without_mutation(self):
        for field_name in ("score", "delta", "cost_before", "cost_after"):
            for value in (float("nan"), float("inf"), float("-inf")):
                with self.subTest(field=field_name, value=value):
                    audit = AssociationHeadAuditAccumulator()
                    with self.assertRaisesRegex(ValueError, field_name):
                        audit.add_pair(**self._pair(**{field_name: value}))
                    self.assertEqual(audit.to_summary()["pair_buckets"], [])
                    json.dumps(audit.to_summary(), allow_nan=False)

    def test_add_pair_normalizes_or_rejects_samples_before_mutation(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
        audit.add_pair(
            **self._pair(),
            sample={"det_index": np.int64(3), "costs": (np.float32(0.5),)},
        )
        normalized = audit.to_summary()["pair_buckets"][0]["samples"][0]
        self.assertEqual(normalized, {"det_index": 3, "costs": [0.5]})
        json.dumps(audit.to_summary(), allow_nan=False)

        before = copy.deepcopy(audit.to_summary())
        for sample in ({"bad": object()}, {"bad": float("nan")}, {"bad": float("inf")}):
            with self.subTest(sample=sample):
                with self.assertRaisesRegex((TypeError, ValueError), "sample.bad"):
                    audit.add_pair(**self._pair(), sample=sample)
                self.assertEqual(audit.to_summary(), before)

    def test_pair_exact_sums_and_averages_are_identical_across_merge_trees(self):
        max_float = np.finfo(np.float64).max

        def state_for(value):
            worker = AssociationHeadAuditAccumulator()
            worker.add_pair(**self._pair(score=value))
            return worker.export_state()

        def merge(states):
            result = AssociationHeadAuditAccumulator()
            for state in states:
                result.merge_state(copy.deepcopy(state))
            return result

        cases = (
            ((1e16, -1e16, 1.0), 1.0 / 3.0),
            ((max_float, max_float, -max_float), max_float / 3.0),
        )
        for values, expected_average in cases:
            with self.subTest(values=values):
                states = [state_for(value) for value in values]
                direct_add = AssociationHeadAuditAccumulator()
                for value in values:
                    direct_add.add_pair(**self._pair(score=value))

                direct_merge = merge(states)
                left_merge = merge((merge(states[:2]).export_state(), states[2]))
                right_merge = merge((states[0], merge(states[1:]).export_state()))
                buckets = [
                    audit.export_state()["pair_buckets"][0]
                    for audit in (direct_add, direct_merge, left_merge, right_merge)
                ]

                self.assertTrue(
                    all(
                        isinstance(bucket["aggregate_sums"]["score"], int)
                        for bucket in buckets
                    )
                )
                self.assertTrue(
                    all(
                        bucket["aggregate_sums"]["score"]
                        == buckets[0]["aggregate_sums"]["score"]
                        for bucket in buckets[1:]
                    )
                )
                self.assertTrue(
                    all(
                        bucket["avg_score"] == buckets[0]["avg_score"]
                        for bucket in buckets[1:]
                    )
                )
                self.assertEqual(buckets[0]["avg_score"], expected_average)

    def test_extreme_quantiles_and_derived_summary_are_strict_json_safe(self):
        max_float = np.finfo(np.float64).max
        audit = AssociationHeadAuditAccumulator()
        audit.add_pair(**self._pair(score=-max_float))
        audit.add_pair(**self._pair(score=max_float))

        summary = audit.to_summary()
        quantiles = summary["pair_buckets"][0]["score_quantiles"]

        self.assertTrue(all(np.isfinite(value) for value in quantiles.values()))
        self.assertEqual(quantiles["min"], -max_float)
        self.assertEqual(quantiles["max"], max_float)
        self.assertEqual(quantiles["p50"], 0.0)
        json.dumps(summary, allow_nan=False, sort_keys=True)

    def test_merge_rejects_nonrepresentable_pair_average_atomically(self):
        max_float = np.finfo(np.float64).max
        for field_name in ("score", "delta", "cost_before", "cost_after"):
            with self.subTest(field=field_name):
                target = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
                worker = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
                worker.add_pair(
                    **self._pair(**{field_name: max_float}),
                    sample={"source": "worker"},
                )
                malformed = worker.export_state()
                malformed["pair_buckets"][0]["aggregate_sums"][field_name] *= 2
                before = copy.deepcopy(target.to_summary())

                with self.assertRaisesRegex(ValueError, field_name):
                    target.merge_state(malformed)

                self.assertEqual(target.to_summary(), before)

    def test_merges_exported_states_and_writes_json(self):
        first = AssociationHeadAuditAccumulator()
        second = AssociationHeadAuditAccumulator()
        first.add_pair(
            class_id=3,
            class_name="motorcycle",
            state_bucket="matched",
            score=0.2,
            delta=0.02,
            cost_before=1.0,
            cost_after=1.02,
            active=True,
            finite=True,
        )
        second.add_pair(
            class_id=3,
            class_name="motorcycle",
            state_bucket="matched",
            score=0.6,
            delta=0.0,
            cost_before=1.0,
            cost_after=1.0,
            active=False,
            finite=True,
        )

        merged = AssociationHeadAuditAccumulator()
        merged.merge_state(first.export_state())
        merged.merge_state(second.export_state())

        bucket = merged.to_summary()["buckets"][0]
        self.assertEqual(bucket["pair_count"], 2)
        self.assertEqual(bucket["active_pair_count"], 1)
        self.assertEqual(bucket["penalized_pair_count"], 1)
        self.assertAlmostEqual(bucket["avg_score"], 0.4)

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            merged.write_json(path)
            self.assertTrue(path.exists())

    def test_schema_v2_exports_verifiable_quantile_sketch_state(self):
        worker = AssociationHeadAuditAccumulator()
        for score in (-2.0, 0.0, 0.5, 4.0):
            worker.add_pair(**self._pair(score=score))
        state = worker.export_state()
        bucket = state["pair_buckets"][0]

        self.assertEqual(bucket["quantile_method"], "fixed_float_histogram_v1")
        self.assertTrue(bucket["quantile_sketch_complete"])
        self.assertEqual(
            set(bucket["quantile_sketches"]),
            {"score", "delta", "cost_before", "cost_after"},
        )
        for sketch in bucket["quantile_sketches"].values():
            self.assertEqual(sketch["type"], "fixed_float_histogram_v1")
            self.assertEqual(sketch["bin_count"], 4096)
            self.assertEqual(sketch["count"], bucket["pair_count"])
            self.assertEqual(
                sum(count for _, count in sketch["bins"]), sketch["count"]
            )

        target = AssociationHeadAuditAccumulator()
        before = copy.deepcopy(target.to_summary())
        corruptions = (
            lambda item: item["quantile_sketches"]["score"].__setitem__("count", 3),
            lambda item: item["quantile_sketches"]["score"]["bins"][0].__setitem__(1, 0),
            lambda item: item["score_quantiles"].__setitem__("p50", 123.0),
        )
        for corrupt in corruptions:
            with self.subTest(corrupt=corrupt):
                malformed = copy.deepcopy(state)
                corrupt(malformed["pair_buckets"][0])
                with self.assertRaises((TypeError, ValueError)):
                    target.merge_state(malformed)
                self.assertEqual(target.to_summary(), before)

        non_json_state = copy.deepcopy(state)
        non_json_state["ignored_extension"] = object()
        with self.assertRaisesRegex(TypeError, "audit_state.ignored_extension"):
            target.merge_state(non_json_state)
        self.assertEqual(target.to_summary(), before)

    def test_schema_v2_rejects_inconsistent_pair_aggregates_atomically(self):
        worker = AssociationHeadAuditAccumulator()
        worker.add_pair(**self._pair(score=1.0))
        worker.add_pair(**self._pair(score=3.0))
        baseline = worker.export_state()

        def changed(callback):
            state = copy.deepcopy(baseline)
            callback(state["pair_buckets"][0])
            return state

        def empty_histograms(bucket):
            bucket["quantile_sketch_complete"] = False
            bucket["quantile_semantics"] = "unavailable_incomplete_sketch"
            for metric, sketch in bucket["quantile_sketches"].items():
                sketch.update({"count": 0, "min": None, "max": None, "bins": []})
                bucket[f"{metric}_quantiles"] = {}

        def average_above_maximum(bucket):
            four_units = bucket["aggregate_sums"]["score"]
            bucket["aggregate_sums"]["score"] = four_units * 2
            bucket["avg_score"] = 4.0

        def zero_count_with_nonzero_sum(bucket):
            bucket["pair_count"] = 0
            bucket["finite_pair_count"] = 0
            bucket["active_pair_count"] = 0
            bucket["penalized_pair_count"] = 0
            empty_histograms(bucket)

        malformed_states = (
            ("avg_score", changed(lambda bucket: bucket.__setitem__("avg_score", 2.5))),
            ("score", changed(average_above_maximum)),
            ("count", changed(empty_histograms)),
            ("zero", changed(zero_count_with_nonzero_sum)),
        )
        target = AssociationHeadAuditAccumulator()
        before = copy.deepcopy(target.to_summary())
        for expected, state in malformed_states:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    target.merge_state(state)
                self.assertEqual(target.to_summary(), before)

    def test_signed_zero_histogram_state_round_trips(self):
        worker = AssociationHeadAuditAccumulator()
        worker.add_pair(**self._pair(score=-0.0))
        worker.add_pair(**self._pair(score=0.0))

        merged = AssociationHeadAuditAccumulator()
        merged.merge_state(worker.export_state())

        bucket = merged.to_summary()["pair_buckets"][0]
        self.assertEqual(bucket["pair_count"], 2)
        self.assertEqual(bucket["score_quantiles"]["min"], 0.0)
        self.assertEqual(bucket["score_quantiles"]["max"], 0.0)

    def test_imported_signed_zero_bounds_are_canonical_across_merge_orders(self):
        worker = AssociationHeadAuditAccumulator()
        worker.add_pair(**self._pair(score=0.0))
        positive = worker.export_state()
        negative = copy.deepcopy(positive)
        negative_bucket = negative["pair_buckets"][0]
        negative_bucket["avg_score"] = -0.0
        negative_bucket["quantile_sketches"]["score"]["min"] = -0.0
        negative_bucket["quantile_sketches"]["score"]["max"] = -0.0
        negative_bucket["quantile_sketches"]["score"]["bins"][0][0] = 2047
        negative_bucket["score_quantiles"]["min"] = -0.0
        negative_bucket["score_quantiles"]["p10"] = -0.0
        negative_bucket["score_quantiles"]["p50"] = -0.0
        negative_bucket["score_quantiles"]["p90"] = -0.0
        negative_bucket["score_quantiles"]["max"] = -0.0

        for states in ((negative, positive), (positive, negative)):
            merged = AssociationHeadAuditAccumulator()
            for state in states:
                merged.merge_state(copy.deepcopy(state))
            bucket = merged.export_state()["pair_buckets"][0]

            self.assertFalse(np.signbit(bucket["avg_score"]))
            self.assertFalse(np.signbit(bucket["score_quantiles"]["min"]))
            self.assertFalse(np.signbit(bucket["score_quantiles"]["max"]))
            self.assertFalse(
                np.signbit(bucket["quantile_sketches"]["score"]["min"])
            )
            self.assertFalse(
                np.signbit(bucket["quantile_sketches"]["score"]["max"])
            )

    def test_schema_v1_merge_does_not_fabricate_quantiles(self):
        merged = AssociationHeadAuditAccumulator()
        merged.merge_state(self._legacy_pair_state())

        bucket = merged.to_summary()["pair_buckets"][0]
        self.assertFalse(bucket["quantile_sketch_complete"])
        self.assertEqual(bucket["score_quantiles"], {})
        self.assertEqual(bucket["delta_quantiles"], {})
        self.assertEqual(bucket["cost_before_quantiles"], {})
        self.assertEqual(bucket["cost_after_quantiles"], {})
        self.assertEqual(bucket["quantile_semantics"], "unavailable_incomplete_sketch")

    def test_write_json_atomically_replaces_complete_file_in_same_directory(self):
        audit = AssociationHeadAuditAccumulator()
        audit.add_pair(**self._pair())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text("old-content", encoding="utf-8")
            real_replace = os.replace

            def observe_replace(source, target):
                self.assertEqual(Path(source).parent, path.parent)
                self.assertEqual(Path(target), path)
                self.assertEqual(path.read_text(encoding="utf-8"), "old-content")
                json.loads(Path(source).read_text(encoding="utf-8"))
                real_replace(source, target)

            with patch(
                "tracker.association_head_audit.os.replace",
                side_effect=observe_replace,
            ) as replace:
                audit.write_json(path)

            replace.assert_called_once()
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8")), audit.to_summary()
            )
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_json_failure_preserves_target_and_removes_temp_file(self):
        audit = AssociationHeadAuditAccumulator()
        audit.add_pair(**self._pair())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text("old-content", encoding="utf-8")
            with patch(
                "tracker.association_head_audit.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    audit.write_json(path)

            self.assertEqual(path.read_text(encoding="utf-8"), "old-content")
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_write_json_preserves_existing_mode_stat_and_xattrs(self):
        audit = AssociationHeadAuditAccumulator()
        audit.add_pair(**self._pair())

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "audit.json"
            path.write_text("old-content", encoding="utf-8")
            path.chmod(0o644)
            timestamp_ns = 1_650_000_000_123_456_789
            os.utime(path, ns=(timestamp_ns, timestamp_ns))
            xattr_name = b"user.association_head_audit_test"
            xattr_value = b"preserve-me"
            xattr_supported = False
            if all(
                hasattr(os, name)
                for name in ("setxattr", "getxattr", "listxattr")
            ):
                try:
                    os.setxattr(path, xattr_name, xattr_value)
                    xattr_supported = True
                except OSError:
                    pass

            audit.write_json(path)

            target_stat = path.stat()
            self.assertEqual(stat.S_IMODE(target_stat.st_mode), 0o644)
            self.assertEqual(target_stat.st_mtime_ns, timestamp_ns)
            if xattr_supported:
                self.assertEqual(os.getxattr(path, xattr_name), xattr_value)
            self.assertEqual(list(path.parent.glob(f".{path.name}.*.tmp")), [])

    def test_schema_v2_includes_manifest_pair_buckets_and_improved_decision(self):
        audit = AssociationHeadAuditAccumulator(
            max_samples_per_bucket=2,
            runtime_manifest={"config_sha256": "abc"},
        )
        audit.add_decision(self._decision())

        summary = audit.to_summary()
        bucket = summary["decision_buckets"][0]

        self.assertEqual(summary["schema_version"], 2)
        self.assertEqual(summary["runtime_manifest"]["config_sha256"], "abc")
        self.assertIn("pair_buckets", summary)
        self.assertTrue(summary["decision_evidence_available"])
        self.assertEqual(bucket["class_id"], 5)
        self.assertEqual(bucket["class_name"], "trailer")
        self.assertEqual(bucket["track_state_bucket"], "unmatched")
        self.assertEqual(bucket["total_decisions"], 1)
        self.assertEqual(bucket["improved_count"], 1)

    def test_counts_assignment_changes_thresholds_and_outcomes(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=4)
        records = [
            self._decision(threshold_crossing=True, outcome="improved"),
            self._decision(
                production_det_index=2,
                shadow_det_index=None,
                threshold_crossing=True,
                outcome="damaged",
            ),
            self._decision(
                production_det_index=None,
                shadow_det_index=3,
                threshold_crossing=False,
                outcome="neutral",
            ),
            self._decision(
                assignment_changed=False,
                production_det_index=None,
                shadow_det_index=None,
                threshold_crossing=None,
                outcome="unknown",
            ),
        ]
        for record in records:
            audit.add_decision(record)

        bucket = audit.to_summary()["decision_buckets"][0]
        expected = {
            "total_decisions": 4,
            "changed_decisions": 3,
            "changed_detection_count": 1,
            "changed_to_match_count": 1,
            "changed_to_unmatched_count": 1,
            "threshold_crossing_count": 2,
            "known_gt_decisions": 3,
            "improved_count": 1,
            "damaged_count": 1,
            "neutral_count": 1,
            "unknown_count": 1,
            "known_changed_count": 3,
            "net_improvement": 0,
        }
        for field, value in expected.items():
            self.assertEqual(bucket[field], value, field)
        self.assertAlmostEqual(bucket["harm_precision"], 1.0 / 3.0)

    def test_decision_diagnostics_are_bounded_and_nonpositive_bound_disables_them(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
        for idx in range(3):
            audit.add_decision(
                self._decision(
                    production_det_index=idx,
                    shadow_det_index=idx + 1,
                    production_selected_association_score=0.1 * idx,
                    shadow_selected_association_score=0.2 * idx,
                    original_cost_gap=0.01 * idx,
                )
            )

        bucket = audit.to_summary()["decision_buckets"][0]
        self.assertEqual(bucket["total_decisions"], 3)
        self.assertEqual(len(bucket["raw_decisions"]), 2)
        self.assertEqual(len(bucket["association_scores"]), 2)
        self.assertEqual(len(bucket["cost_gaps"]), 2)

        for bound in (0, -1):
            with self.subTest(bound=bound):
                disabled = AssociationHeadAuditAccumulator(
                    max_samples_per_bucket=bound
                )
                disabled.add_decision(self._decision())
                disabled_bucket = disabled.to_summary()["decision_buckets"][0]
                self.assertEqual(disabled_bucket["total_decisions"], 1)
                self.assertEqual(disabled_bucket["raw_decisions"], [])
                self.assertEqual(disabled_bucket["association_scores"], [])
                self.assertEqual(disabled_bucket["cost_gaps"], [])

    def test_merge_preserves_exact_counts_and_target_sample_bound(self):
        manifest = {"config_sha256": "same"}
        workers = [
            AssociationHeadAuditAccumulator(
                max_samples_per_bucket=3,
                runtime_manifest=manifest,
            )
            for _ in range(2)
        ]
        for worker_index, worker in enumerate(workers):
            for offset in range(2):
                idx = worker_index * 2 + offset
                worker.add_decision(
                    self._decision(
                        production_det_index=idx,
                        shadow_det_index=idx + 1,
                        outcome="damaged" if idx == 3 else "improved",
                        original_cost_gap=0.1 * idx,
                    )
                )
                worker.add_pair(
                    class_id=5,
                    class_name="trailer",
                    state_bucket="unmatched",
                    score=0.1 * idx,
                    delta=0.0,
                    cost_before=1.0,
                    cost_after=1.0,
                    active=True,
                    finite=True,
                    sample={"worker": worker_index, "index": idx},
                )

        merged = AssociationHeadAuditAccumulator(
            max_samples_per_bucket=2,
            runtime_manifest=manifest,
            strict_manifest=True,
        )
        for worker in workers:
            merged.merge_state(worker.export_state())

        summary = merged.to_summary()
        decision_bucket = summary["decision_buckets"][0]
        pair_bucket = summary["pair_buckets"][0]
        self.assertEqual(decision_bucket["total_decisions"], 4)
        self.assertEqual(decision_bucket["improved_count"], 3)
        self.assertEqual(decision_bucket["damaged_count"], 1)
        self.assertEqual(decision_bucket["net_improvement"], 2)
        self.assertEqual(len(decision_bucket["raw_decisions"]), 2)
        self.assertEqual(len(decision_bucket["association_scores"]), 2)
        self.assertEqual(len(decision_bucket["cost_gaps"]), 2)
        self.assertEqual(pair_bucket["pair_count"], 4)
        self.assertEqual(len(pair_bucket["samples"]), 2)

    def test_audit_errors_are_json_safe_bounded_and_mergeable(self):
        workers = []
        for scene_index in range(2):
            worker = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
            worker.add_error(
                stage="shadow_solver",
                error=RuntimeError(f"scene {scene_index} failed"),
                context={"scene_id": np.str_(f"scene-{scene_index}")},
            )
            workers.append(worker)

        merged = AssociationHeadAuditAccumulator(max_samples_per_bucket=1)
        for worker in workers:
            merged.merge_state(worker.export_state())

        summary = merged.to_summary()
        self.assertEqual(summary["audit_errors"]["count"], 2)
        self.assertEqual(summary["audit_error_count"], 2)
        self.assertEqual(len(summary["audit_errors"]["samples"]), 1)
        sample = summary["audit_errors"]["samples"][0]
        self.assertEqual(sample["stage"], "shadow_solver")
        self.assertEqual(sample["type"], "RuntimeError")
        self.assertEqual(sample["context"]["scene_id"], "scene-0")
        json.dumps(summary, allow_nan=False)

    def test_empty_exception_message_uses_error_type_fallback(self):
        audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=1)
        audit.add_error(stage="worker_merge", error=RuntimeError(), context={})

        sample = audit.to_summary()["audit_errors"]["samples"][0]
        self.assertEqual(sample["type"], "RuntimeError")
        self.assertEqual(sample["message"], "RuntimeError")

    def test_merged_writer_preserves_bound_manifest_and_worker_errors(self):
        manifest = {"config_sha256": "same"}
        workers = {}
        for scene_index in range(2):
            worker = AssociationHeadAuditAccumulator(
                max_samples_per_bucket=2,
                runtime_manifest=manifest,
            )
            for offset in range(2):
                worker.add_decision(
                    self._decision(
                        production_det_index=scene_index * 2 + offset,
                        shadow_det_index=scene_index * 2 + offset + 1,
                    )
                )
            workers[f"scene-{scene_index}"] = worker.export_state()
        workers["scene-bad"] = {"schema_version": 2}

        write_merged = _load_main_audit_functions()
        with tempfile.TemporaryDirectory() as tmp:
            output_path = Path(tmp) / "merged.json"
            cfg = {
                "_ASSOCIATION_AUDIT_RUNTIME_MANIFEST": manifest,
                "ASSOCIATION_HEAD_AUDIT": {
                    "ENABLED": True,
                    "SHADOW_MODE": True,
                    "STRICT": False,
                    "MAX_SAMPLES_PER_BUCKET": 2,
                    "INFER_OUTPUT_PATH": str(output_path),
                },
            }
            write_merged(cfg, workers)
            summary = json.loads(output_path.read_text(encoding="utf-8"))

        self.assertEqual(summary["runtime_manifest"], manifest)
        self.assertEqual(summary["decision_buckets"][0]["total_decisions"], 4)
        self.assertEqual(len(summary["decision_buckets"][0]["raw_decisions"]), 2)
        self.assertEqual(summary["audit_errors"]["count"], 1)
        self.assertEqual(summary["audit_errors"]["samples"][0]["context"]["scene_id"], "scene-bad")
        self.assertFalse(summary["decision_evidence_complete"])
        self.assertIn("audit_errors", summary["decision_evidence_incomplete_reasons"])

    def test_merged_writer_strict_mode_reraises_bad_worker(self):
        write_merged = _load_main_audit_functions()
        cfg = {
            "_ASSOCIATION_AUDIT_RUNTIME_MANIFEST": {"config_sha256": "same"},
            "ASSOCIATION_HEAD_AUDIT": {
                "ENABLED": True,
                "STRICT": True,
                "MAX_SAMPLES_PER_BUCKET": 2,
            },
        }
        with self.assertRaisesRegex(ValueError, "runtime_manifest"):
            write_merged(cfg, {"scene-bad": {"schema_version": 2}})

    def test_strict_merge_rejects_incompatible_manifest_with_difference(self):
        merged = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "left", "alpha": 0.1},
            strict_manifest=True,
        )
        worker = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "right", "alpha": 0.1}
        )

        with self.assertRaisesRegex(ValueError, "config_sha256"):
            merged.merge_state(worker.export_state())
        self.assertEqual(merged.to_summary()["runtime_manifest"]["config_sha256"], "left")

    def test_strict_merge_accepts_compatible_manifest(self):
        manifest = {"config_sha256": "same", "active_class_states": {"5": ["unmatched"]}}
        worker = AssociationHeadAuditAccumulator(runtime_manifest=manifest)
        worker.add_decision(self._decision())
        merged = AssociationHeadAuditAccumulator(
            runtime_manifest=dict(manifest), strict_manifest=True
        )

        merged.merge_state(worker.export_state())

        self.assertEqual(merged.to_summary()["runtime_manifest"], manifest)
        self.assertEqual(
            merged.to_summary()["decision_buckets"][0]["total_decisions"], 1
        )

    def test_decision_evidence_stays_incomplete_for_v1_in_both_merge_orders(self):
        worker = AssociationHeadAuditAccumulator(runtime_manifest=None)
        worker.add_decision(self._decision())
        v2_state = worker.export_state()
        legacy_state = self._legacy_pair_state()

        for states in ((v2_state, legacy_state), (legacy_state, v2_state)):
            with self.subTest(order=[state["schema_version"] for state in states]):
                merged = AssociationHeadAuditAccumulator()
                for state in states:
                    merged.merge_state(copy.deepcopy(state))

                summary = merged.to_summary()
                self.assertEqual(
                    summary["decision_buckets"][0]["total_decisions"], 1
                )
                self.assertFalse(summary["decision_evidence_available"])
                self.assertFalse(summary["decision_evidence_complete"])
                self.assertIn(
                    "schema_v1_state",
                    summary["decision_evidence_incomplete_reasons"],
                )

    def test_pair_only_v2_state_marks_decision_evidence_incomplete(self):
        decisions = AssociationHeadAuditAccumulator(runtime_manifest=None)
        decisions.add_decision(self._decision())
        pair_only = AssociationHeadAuditAccumulator(runtime_manifest=None)
        pair_only.add_pair(**self._pair())
        merged = AssociationHeadAuditAccumulator()

        merged.merge_state(decisions.export_state())
        merged.merge_state(pair_only.export_state())

        summary = merged.to_summary()
        self.assertFalse(summary["decision_evidence_available"])
        self.assertFalse(summary["decision_evidence_complete"])
        self.assertIn(
            "pair_only_state", summary["decision_evidence_incomplete_reasons"]
        )

    def test_strict_manifest_uses_first_worker_null_as_a_real_baseline(self):
        null_state = AssociationHeadAuditAccumulator(
            runtime_manifest=None
        ).export_state()
        manifest_state = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "abc"}
        ).export_state()

        for states in ((null_state, manifest_state), (manifest_state, null_state)):
            with self.subTest(first_manifest=states[0]["runtime_manifest"]):
                merged = AssociationHeadAuditAccumulator(strict_manifest=True)
                merged.merge_state(copy.deepcopy(states[0]))
                before = copy.deepcopy(merged.to_summary())

                with self.assertRaisesRegex(ValueError, "runtime_manifest"):
                    merged.merge_state(copy.deepcopy(states[1]))

                self.assertEqual(merged.to_summary(), before)

        explicitly_null = AssociationHeadAuditAccumulator(
            runtime_manifest=None, strict_manifest=True
        )
        before = copy.deepcopy(explicitly_null.to_summary())
        with self.assertRaisesRegex(ValueError, "runtime_manifest"):
            explicitly_null.merge_state(copy.deepcopy(manifest_state))
        self.assertEqual(explicitly_null.to_summary(), before)

        locally_populated = AssociationHeadAuditAccumulator(strict_manifest=True)
        locally_populated.add_decision(self._decision())
        before = copy.deepcopy(locally_populated.to_summary())
        with self.assertRaisesRegex(ValueError, "runtime_manifest"):
            locally_populated.merge_state(copy.deepcopy(manifest_state))
        self.assertEqual(locally_populated.to_summary(), before)

    def test_non_strict_manifest_mismatch_merges_only_pair_diagnostics(self):
        merged = AssociationHeadAuditAccumulator(
            max_samples_per_bucket=2,
            runtime_manifest={"config_sha256": "left"},
        )
        merged.add_decision(self._decision())
        incompatible = AssociationHeadAuditAccumulator(
            max_samples_per_bucket=2,
            runtime_manifest={"config_sha256": "right"},
        )
        incompatible.add_pair(**self._pair(), sample={"worker": "right"})
        incompatible.add_decision(self._decision(outcome="damaged"))

        merged.merge_state(incompatible.export_state())

        summary = merged.to_summary()
        self.assertEqual(summary["pair_buckets"][0]["pair_count"], 1)
        self.assertEqual(summary["decision_buckets"], [])
        self.assertFalse(summary["decision_evidence_available"])
        self.assertFalse(summary["manifest_compatible"])
        self.assertTrue(summary["manifest_mismatches"])

        compatible_after_mismatch = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "left"}
        )
        compatible_after_mismatch.add_pair(**self._pair(score=0.2))
        compatible_after_mismatch.add_decision(self._decision())
        merged.merge_state(compatible_after_mismatch.export_state())
        summary = merged.to_summary()
        self.assertEqual(summary["pair_buckets"][0]["pair_count"], 2)
        self.assertEqual(summary["decision_buckets"], [])

    def test_manifest_mismatch_decision_result_is_merge_order_independent(self):
        def worker_state(manifest, score):
            worker = AssociationHeadAuditAccumulator(
                max_samples_per_bucket=2,
                runtime_manifest={"config_sha256": manifest},
            )
            worker.add_pair(**self._pair(score=score))
            worker.add_decision(self._decision())
            return worker.export_state()

        def merged_state(states):
            merged = AssociationHeadAuditAccumulator(max_samples_per_bucket=4)
            for state in states:
                merged.merge_state(copy.deepcopy(state))
            return merged

        state_a = worker_state("left", 0.1)
        state_b = worker_state("right", 0.2)
        state_c = worker_state("left", 0.3)

        direct_ab = merged_state((state_a, state_b))
        direct_abc = merged_state((state_a, state_b, state_c))
        left_tree = merged_state((direct_ab.export_state(), state_c))
        right_branch = merged_state((state_b, state_c))
        right_tree = merged_state((state_a, right_branch.export_state()))
        reverse = merged_state((state_c, state_b, state_a))

        self.assertEqual(direct_ab.to_summary()["decision_buckets"], [])
        for merged in (direct_abc, left_tree, right_tree, reverse):
            with self.subTest(manifest=merged.to_summary()["runtime_manifest"]):
                summary = merged.to_summary()
                self.assertEqual(summary["decision_buckets"], [])
                self.assertFalse(summary["decision_evidence_available"])
                self.assertFalse(summary["manifest_compatible"])
                self.assertEqual(summary["pair_buckets"][0]["pair_count"], 3)

    def test_manifest_initialized_protocol_and_atomic_failures(self):
        empty_uninitialized = AssociationHeadAuditAccumulator().export_state()
        target = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "target"},
            strict_manifest=True,
        )
        target.add_decision(self._decision())
        before = copy.deepcopy(target.to_summary())

        target.merge_state(copy.deepcopy(empty_uninitialized))
        self.assertEqual(target.to_summary(), before)

        for invalid in (None, 0, "false"):
            with self.subTest(invalid=invalid):
                state = copy.deepcopy(empty_uninitialized)
                state["manifest_initialized"] = invalid
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "manifest_initialized"
                ):
                    target.merge_state(state)
                self.assertEqual(target.to_summary(), before)

        for data_kind in ("pair", "decision"):
            with self.subTest(data_kind=data_kind):
                worker = AssociationHeadAuditAccumulator(runtime_manifest=None)
                if data_kind == "pair":
                    worker.add_pair(**self._pair())
                else:
                    worker.add_decision(self._decision())
                state = worker.export_state()
                state["manifest_initialized"] = False
                with self.assertRaisesRegex(ValueError, "manifest_initialized"):
                    target.merge_state(state)
                self.assertEqual(target.to_summary(), before)

        legacy_empty = {"schema_version": 1, "buckets": []}
        legacy_target = AssociationHeadAuditAccumulator(strict_manifest=True)
        legacy_target.merge_state(legacy_empty)
        self.assertFalse(legacy_target.to_summary()["manifest_initialized"])
        baseline = AssociationHeadAuditAccumulator(
            runtime_manifest={"config_sha256": "baseline"}
        )
        legacy_target.merge_state(baseline.export_state())
        self.assertEqual(
            legacy_target.to_summary()["runtime_manifest"],
            {"config_sha256": "baseline"},
        )

    def test_manifest_mismatch_diff_is_type_sensitive_and_atomic(self):
        merged = AssociationHeadAuditAccumulator(
            runtime_manifest={"options": {"enabled": True}},
            strict_manifest=True,
        )
        worker = AssociationHeadAuditAccumulator(
            runtime_manifest={"options": {"enabled": 1}}
        )
        before = copy.deepcopy(merged.to_summary())

        with self.assertRaisesRegex(
            ValueError,
            r"options\.enabled.*True.*bool.*1.*int",
        ):
            merged.merge_state(worker.export_state())

        self.assertEqual(merged.to_summary(), before)

    def test_merges_schema_v1_pairs_without_fabricating_decision_evidence(self):
        merged = AssociationHeadAuditAccumulator()

        merged.merge_state(self._legacy_pair_state())

        summary = merged.to_summary()
        self.assertEqual(summary["pair_buckets"][0]["pair_count"], 2)
        self.assertEqual(summary["decision_buckets"], [])
        self.assertFalse(summary["decision_evidence_available"])

    def test_merge_rejects_malformed_state_atomically(self):
        worker = AssociationHeadAuditAccumulator(
            max_samples_per_bucket=1,
            runtime_manifest={"config_sha256": "incoming"},
        )
        worker.add_pair(**self._pair(), sample={"worker": 1})
        worker.add_decision(self._decision())
        baseline = worker.export_state()

        def changed(callback):
            state = copy.deepcopy(baseline)
            callback(state)
            return state

        def threshold_exceeds_changed(state):
            bucket = state["decision_buckets"][0]
            bucket["total_decisions"] = 2
            bucket["unknown_count"] = 1
            bucket["threshold_crossing_count"] = 2

        malformed_states = [
            ("schema_version", changed(lambda state: state.pop("schema_version"))),
            (
                "schema_version",
                changed(lambda state: state.__setitem__("schema_version", 3)),
            ),
            (
                "total_decisions",
                changed(
                    lambda state: state["decision_buckets"][0].pop(
                        "total_decisions"
                    )
                ),
            ),
            (
                "total_decisions",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "total_decisions", True
                    )
                ),
            ),
            (
                "damaged_count",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "damaged_count", -1
                    )
                ),
            ),
            (
                "outcome counts",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "unknown_count", 1
                    )
                ),
            ),
            (
                "known_gt_decisions",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "known_gt_decisions", 0
                    )
                ),
            ),
            (
                "changed_decisions",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "changed_decisions", 2
                    )
                ),
            ),
            (
                "changed_detection_count",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "changed_detection_count", 2
                    )
                ),
            ),
            (
                "changed assignment categories",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "changed_to_match_count", 1
                    )
                ),
            ),
            (
                "changed assignment categories",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "changed_detection_count", 0
                    )
                ),
            ),
            (
                "known_changed_count",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "known_changed_count", 2
                    )
                ),
            ),
            (
                "known_changed_count",
                changed(
                    lambda state: state["decision_buckets"][0].__setitem__(
                        "known_changed_count", 0
                    )
                ),
            ),
            (
                "threshold_crossing_count",
                changed(threshold_exceeds_changed),
            ),
            (
                "raw_decisions",
                changed(
                    lambda state: state["decision_buckets"][0]["raw_decisions"][
                        0
                    ].__setitem__("class_id", 6)
                ),
            ),
            (
                "score_quantiles",
                changed(
                    lambda state: state["pair_buckets"][0].__setitem__(
                        "score_quantiles", {1: 0.5}
                    )
                ),
            ),
        ]

        for expected_field, state in malformed_states:
            with self.subTest(expected_field=expected_field):
                merged = AssociationHeadAuditAccumulator(max_samples_per_bucket=1)
                before = copy.deepcopy(merged.to_summary())
                with self.assertRaisesRegex(
                    (TypeError, ValueError), expected_field
                ):
                    merged.merge_state(state)
                self.assertEqual(merged.to_summary(), before)

    def test_rejects_malformed_decisions_and_summary_is_strict_json(self):
        required_fields = (
            "class_id",
            "class_name",
            "track_state_bucket",
            "assignment_changed",
            "production_det_index",
            "shadow_det_index",
            "threshold_crossing",
            "outcome",
        )
        for field in required_fields:
            with self.subTest(missing=field):
                record = self._decision()
                del record[field]
                with self.assertRaises((TypeError, ValueError)):
                    AssociationHeadAuditAccumulator().add_decision(record)

        malformed = [
            self._decision(outcome="better"),
            self._decision(track_state_bucket="tentative"),
            self._decision(assignment_changed=1),
            self._decision(production_det_index=True),
            self._decision(production_det_index=-1),
            self._decision(assignment_changed=False),
            self._decision(threshold_crossing="yes"),
            self._decision(production_selected_association_score=True),
            self._decision(shadow_selected_association_score="high"),
            self._decision(original_cost_gap=True),
            self._decision(original_cost_gap=float("inf")),
            self._decision(shadow_selected_association_score=float("nan")),
            self._decision(geometric_candidate_rank=0),
            self._decision(shadow_candidate_rank=True),
            self._decision(production_pair_label=True),
            self._decision(shadow_pair_label=2),
        ]
        for record in malformed:
            with self.subTest(record=record):
                with self.assertRaises((TypeError, ValueError)):
                    AssociationHeadAuditAccumulator().add_decision(record)

        valid = AssociationHeadAuditAccumulator(max_samples_per_bucket=1)
        valid.add_decision(self._decision(threshold_crossing=None))
        json.dumps(valid.to_summary(), allow_nan=False)


if __name__ == "__main__":
    unittest.main()
