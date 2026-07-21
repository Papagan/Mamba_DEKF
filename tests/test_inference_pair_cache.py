import copy
import hashlib
import json
import math
import os
import pickle
import subprocess
import tempfile
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from tracker import inference_pair_cache
from tracker.inference_pair_cache import build_candidate_groups


class InferencePairCacheGroupTest(unittest.TestCase):
    def _payload(self):
        return {
            "scene_id": "scene-1",
            "sample_token": "sample-1",
            "frame_id": 3,
            "track_ids": [17],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["unmatched"],
            "track_gt_tokens": ["gt-b"],
            "det_gt_tokens": ["gt-a", "gt-b"],
            "pair_indices": [(0, 0), (0, 1)],
            "pair_vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "track_embeddings": np.asarray([[0.1, 0.2]], dtype=np.float32),
            "det_embeddings": np.asarray([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
            "association_scores": np.asarray([[0.1, 0.9]], dtype=np.float32),
        }

    def _build(self, payload=None, **kwargs):
        base_cost_matrix = kwargs.pop(
            "base_cost_matrix", np.asarray([[0.2, 0.21]], dtype=np.float32)
        )
        production_matches = kwargs.pop(
            "production_matches", np.asarray([[0, 0]], dtype=np.int64)
        )
        production_costs = kwargs.pop(
            "production_costs", np.asarray([0.2], dtype=np.float32)
        )
        thresholds = kwargs.pop("thresholds", {5: 1.26})
        return build_candidate_groups(
            payload=self._payload() if payload is None else payload,
            base_cost_matrix=base_cost_matrix,
            production_matches=production_matches,
            production_costs=production_costs,
            thresholds=thresholds,
            **kwargs,
        )

    def test_group_keeps_production_and_gt_labels_separate(self):
        group = self._build()[0]

        self.assertEqual(group["production_det_index"], 0)
        self.assertEqual(group["gt_target_det_index"], 1)
        self.assertEqual(group["gt_target_status"], "matched")
        self.assertTrue(group["candidates"][0]["production_selected"])
        self.assertEqual(group["candidates"][0]["gt_label"], 0)
        self.assertEqual(group["candidates"][1]["gt_label"], 1)

    def test_known_track_without_gt_candidate_is_no_match(self):
        payload = self._payload()
        payload["track_gt_tokens"] = ["missing-gt"]
        group = self._build(
            payload,
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,), dtype=np.float32),
        )[0]

        self.assertEqual(group["gt_target_status"], "no_match")
        self.assertIsNone(group["gt_target_det_index"])
        self.assertTrue(all(item["gt_label"] == 0 for item in group["candidates"]))

    def test_missing_track_gt_is_unknown_not_negative(self):
        payload = self._payload()
        payload["track_gt_tokens"] = [None]
        group = self._build(
            payload,
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,), dtype=np.float32),
        )[0]

        self.assertEqual(group["gt_target_status"], "unknown")
        self.assertIsNone(group["gt_target_det_index"])
        self.assertTrue(all(item["gt_label"] is None for item in group["candidates"]))

    def test_duplicate_candidate_gt_tokens_are_ambiguous(self):
        payload = self._payload()
        payload["track_gt_tokens"] = ["gt-a"]
        payload["det_gt_tokens"] = ["gt-a", "gt-a"]

        group = self._build(payload)[0]

        self.assertEqual(group["gt_target_status"], "unknown")
        self.assertIsNone(group["gt_target_det_index"])
        self.assertEqual(
            [candidate["gt_label"] for candidate in group["candidates"]],
            [None, None],
        )

    def test_missing_candidate_gt_identity_makes_known_track_unknown(self):
        payload = self._payload()
        payload["det_gt_tokens"] = [None, None]

        group = self._build(payload)[0]

        self.assertEqual(group["gt_target_status"], "unknown")
        self.assertIsNone(group["gt_target_det_index"])
        self.assertEqual(
            [candidate["gt_label"] for candidate in group["candidates"]],
            [None, None],
        )

    def test_unknown_candidate_prevents_unique_positive_inference(self):
        payload = self._payload()
        payload["det_gt_tokens"] = [None, "gt-b"]

        group = self._build(payload)[0]

        self.assertEqual(group["gt_target_status"], "unknown")
        self.assertIsNone(group["gt_target_det_index"])
        self.assertEqual(
            [candidate["gt_label"] for candidate in group["candidates"]],
            [None, None],
        )

    def test_ranks_finite_candidates_and_retains_non_finite_costs(self):
        payload = self._payload()
        payload["det_gt_tokens"] = ["gt-a", "gt-b", "gt-c"]
        payload["det_embeddings"] = np.asarray(
            [[0.3, 0.4], [0.5, 0.6], [0.7, 0.8]], dtype=np.float32
        )
        payload["pair_indices"] = [(0, 2), (0, 0), (0, 1)]
        payload["pair_vectors"] = np.asarray(
            [[2.0, 2.1], [0.0, 0.1], [1.0, 1.1]], dtype=np.float32
        )
        payload["association_scores"] = np.asarray([[0.2, 0.3, 0.4]], dtype=np.float32)

        group = self._build(
            payload,
            base_cost_matrix=np.asarray([[np.inf, 1.5, 0.5]]),
            production_matches=np.asarray([[0, 2]], dtype=np.int64),
            production_costs=np.asarray([0.5]),
            thresholds={5: 1.0},
        )[0]

        self.assertEqual([item["det_index"] for item in group["candidates"]], [0, 1, 2])
        self.assertEqual(group["candidate_count"], 3)
        self.assertEqual(group["finite_candidate_count"], 2)
        by_det = {item["det_index"]: item for item in group["candidates"]}
        self.assertIsNone(by_det[0]["geometric_cost"])
        self.assertFalse(by_det[0]["finite"])
        self.assertFalse(by_det[0]["inside_geometric_gate"])
        self.assertIsNone(by_det[0]["candidate_rank"])
        self.assertIsNone(by_det[0]["cost_gap"])
        self.assertEqual(by_det[1]["candidate_rank"], 2)
        self.assertEqual(by_det[1]["best_cost"], 0.5)
        self.assertEqual(by_det[1]["cost_gap"], 1.0)
        self.assertFalse(by_det[1]["inside_geometric_gate"])
        self.assertEqual(by_det[2]["candidate_rank"], 1)
        self.assertEqual(by_det[2]["cost_gap"], 0.0)
        self.assertTrue(by_det[2]["inside_geometric_gate"])

    def test_candidate_rank_breaks_cost_ties_by_detection_index(self):
        group = self._build(
            base_cost_matrix=np.asarray([[0.2, 0.2]], dtype=np.float32)
        )[0]

        self.assertEqual(
            [candidate["candidate_rank"] for candidate in group["candidates"]],
            [1, 2],
        )

    def test_zero_candidate_tracks_keep_no_match_and_unknown_groups(self):
        payload = {
            "scene_id": "scene-1",
            "sample_token": "sample-2",
            "frame_id": 4,
            "track_ids": [20, 21],
            "class_ids": [5, 5],
            "class_names": ["trailer", "trailer"],
            "state_buckets": ["matched", "unmatched"],
            "track_gt_tokens": ["known", None],
            "det_gt_tokens": [],
            "pair_indices": [],
            "pair_vectors": np.empty((0, 2), dtype=np.float32),
            "track_embeddings": np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "det_embeddings": np.empty((0, 2), dtype=np.float32),
        }

        groups = self._build(
            payload,
            base_cost_matrix=np.empty((2, 0), dtype=np.float32),
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,), dtype=np.float32),
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["gt_target_status"], "no_match")
        self.assertEqual(groups[1]["gt_target_status"], "unknown")
        for group in groups:
            self.assertEqual(group["candidates"], [])
            self.assertEqual(group["candidate_count"], 0)
            self.assertEqual(group["finite_candidate_count"], 0)
            self.assertIsNone(group["production_det_index"])
            self.assertIsNone(group["production_match_cost"])

    def test_missing_or_none_association_scores_serialize_null(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                payload = self._payload()
                if missing:
                    payload.pop("association_scores")
                else:
                    payload["association_scores"] = None
                group = self._build(payload)[0]
                self.assertEqual(
                    [
                        candidate["association_score"]
                        for candidate in group["candidates"]
                    ],
                    [None, None],
                )

    def test_serialization_flags_only_control_vector_fields(self):
        baseline = self._build()[0]
        for include_embeddings in (False, True):
            for include_pair_vectors in (False, True):
                with self.subTest(
                    include_embeddings=include_embeddings,
                    include_pair_vectors=include_pair_vectors,
                ):
                    group = self._build(
                        include_embeddings=include_embeddings,
                        include_pair_vectors=include_pair_vectors,
                    )[0]
                    self.assertEqual("track_embedding" in group, include_embeddings)
                    self.assertTrue(
                        all(
                            ("det_embedding" in candidate) == include_embeddings
                            for candidate in group["candidates"]
                        )
                    )
                    self.assertTrue(
                        all(
                            ("pair_vector" in candidate) == include_pair_vectors
                            for candidate in group["candidates"]
                        )
                    )
                    for field in (
                        "track_id",
                        "production_det_index",
                        "gt_target_det_index",
                        "gt_target_status",
                        "candidate_count",
                    ):
                        self.assertEqual(group[field], baseline[field])
                    for candidate, baseline_candidate in zip(
                        group["candidates"], baseline["candidates"]
                    ):
                        for field in (
                            "det_index",
                            "geometric_cost",
                            "candidate_rank",
                            "inside_geometric_gate",
                            "gt_label",
                            "production_selected",
                        ):
                            self.assertEqual(
                                candidate[field], baseline_candidate[field]
                            )

        self.assertEqual(
            baseline["track_embedding"],
            [float(value) for value in self._payload()["track_embeddings"][0]],
        )
        self.assertEqual(
            baseline["candidates"][0]["det_embedding"],
            [float(value) for value in self._payload()["det_embeddings"][0]],
        )
        self.assertEqual(baseline["candidates"][0]["pair_vector"], [1.0, 2.0])

    def test_result_uses_json_and_pickle_safe_builtins(self):
        payload = self._payload()
        payload["frame_id"] = np.int64(3)

        groups = self._build(payload)

        json.dumps(groups, allow_nan=False)
        self.assertEqual(pickle.loads(pickle.dumps(groups)), groups)
        self.assertIs(type(groups[0]["frame_id"]), int)

    def test_rejects_raw_boolean_pair_and_assignment_indices(self):
        pair_cases = (
            [(False, 0), (0, 1)],
            [(0, False), (0, 1)],
        )
        for pair_indices in pair_cases:
            with self.subTest(pair_indices=pair_indices):
                payload = self._payload()
                payload["pair_indices"] = pair_indices
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "pair_indices.*bool"
                ):
                    self._build(payload)

        match_cases = ([(False, 0)], [(0, False)])
        for production_matches in match_cases:
            with self.subTest(production_matches=production_matches):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), "production_matches.*bool"
                ):
                    self._build(production_matches=production_matches)

    def test_rejects_invalid_context_metadata(self):
        cases = (
            ("scene_id", None),
            ("scene_id", True),
            ("scene_id", ""),
            ("sample_token", None),
            ("sample_token", False),
            ("sample_token", ""),
            ("frame_id", None),
            ("frame_id", True),
            ("frame_id", -1),
            ("frame_id", 3.0),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                payload = self._payload()
                payload[field] = value
                with self.assertRaisesRegex((TypeError, ValueError), field):
                    self._build(payload)

    def test_rejects_noncanonical_context_and_class_name_strings(self):
        class StringSubclass(str):
            pass

        context_cases = (
            ("scene_id", " "),
            ("scene_id", " scene-1"),
            ("scene_id", "scene-1 "),
            ("scene_id", StringSubclass("scene-1")),
            ("sample_token", "\t"),
            ("sample_token", " sample-1"),
            ("sample_token", "sample-1 "),
            ("sample_token", StringSubclass("sample-1")),
        )
        for field, value in context_cases:
            with self.subTest(field=field, value=value):
                payload = self._payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._build(payload)

        for class_name in (
            "",
            " ",
            " trailer",
            "trailer ",
            StringSubclass("trailer"),
        ):
            with self.subTest(class_name=class_name):
                payload = self._payload()
                payload["class_names"] = [class_name]
                with self.assertRaisesRegex(ValueError, "class_names"):
                    self._build(payload)

    def test_rejects_state_and_gt_string_subclasses_and_emits_builtin_strings(self):
        class StringSubclass(str):
            pass

        cases = (
            ("state_buckets", [StringSubclass("matched")]),
            ("track_gt_tokens", [StringSubclass("gt-b")]),
            ("det_gt_tokens", [StringSubclass("gt-a"), "gt-b"]),
        )
        for field, value in cases:
            with self.subTest(field=field):
                payload = self._payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._build(payload)

        groups = self._build()
        group = groups[0]
        for field in (
            "scene_id",
            "sample_token",
            "class_name",
            "state_bucket",
            "track_gt_token",
        ):
            self.assertIs(type(group[field]), str)
        for candidate in group["candidates"]:
            self.assertIs(type(candidate["det_gt_token"]), str)
        pickle.dumps(groups)

    def test_thresholds_require_finite_real_numeric_scalars_without_coercion(self):
        invalid_thresholds = (
            {5: True},
            {5: "1.26"},
            {5: np.nan},
            {5: np.inf},
            {5: np.asarray([1.26])},
        )
        for thresholds in invalid_thresholds:
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex((TypeError, ValueError), "threshold"):
                    self._build(thresholds=thresholds)

    def test_accepts_fraction_threshold_as_finite_python_float(self):
        groups = self._build(thresholds={5: Fraction(5, 4)})

        for candidate in groups[0]["candidates"]:
            self.assertIs(type(candidate["geometric_threshold"]), float)
            self.assertEqual(candidate["geometric_threshold"], 1.25)
        json.dumps(groups, allow_nan=False)

    def test_rejects_finite_numeric_values_that_overflow_python_float(self):
        overflowing = np.longdouble(np.finfo(float).max) * np.longdouble(2.0)
        self.assertTrue(np.isfinite(overflowing))

        cases = []
        payload = self._payload()
        cases.append(
            (
                "base_cost_matrix",
                payload,
                {"base_cost_matrix": np.asarray([[overflowing, 0.21]])},
            )
        )
        payload = self._payload()
        payload["track_embeddings"] = payload["track_embeddings"].astype(np.longdouble)
        payload["track_embeddings"][0, 0] = overflowing
        cases.append(("track_embeddings", payload, {}))
        payload = self._payload()
        payload["det_embeddings"] = payload["det_embeddings"].astype(np.longdouble)
        payload["det_embeddings"][0, 0] = overflowing
        cases.append(("det_embeddings", payload, {}))
        payload = self._payload()
        payload["pair_vectors"] = payload["pair_vectors"].astype(np.longdouble)
        payload["pair_vectors"][0, 0] = overflowing
        cases.append(("pair_vectors", payload, {}))
        payload = self._payload()
        payload["association_scores"] = payload["association_scores"].astype(
            np.longdouble
        )
        payload["association_scores"][0, 0] = overflowing
        cases.append(("association_scores", payload, {}))
        cases.append(
            (
                "production_costs",
                self._payload(),
                {"production_costs": np.asarray([overflowing])},
            )
        )
        cases.append(("threshold", self._payload(), {"thresholds": {5: overflowing}}))

        for expected, payload, kwargs in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), f"{expected}.*Python float"
                ):
                    self._build(payload, **kwargs)

    def test_rejects_nonzero_numeric_values_that_underflow_python_float(self):
        underflowing = np.nextafter(
            np.longdouble(0.0), np.longdouble(1.0), dtype=np.longdouble
        )
        self.assertNotEqual(underflowing, 0)
        self.assertEqual(float(underflowing), 0.0)

        cases = []
        payload = self._payload()
        cases.append(
            (
                "base_cost_matrix",
                payload,
                {"base_cost_matrix": np.asarray([[underflowing, 0.21]])},
            )
        )
        payload = self._payload()
        payload["track_embeddings"] = payload["track_embeddings"].astype(np.longdouble)
        payload["track_embeddings"][0, 0] = underflowing
        cases.append(("track_embeddings", payload, {}))
        payload = self._payload()
        payload["det_embeddings"] = payload["det_embeddings"].astype(np.longdouble)
        payload["det_embeddings"][0, 0] = underflowing
        cases.append(("det_embeddings", payload, {}))
        payload = self._payload()
        payload["pair_vectors"] = payload["pair_vectors"].astype(np.longdouble)
        payload["pair_vectors"][0, 0] = underflowing
        cases.append(("pair_vectors", payload, {}))
        payload = self._payload()
        payload["association_scores"] = payload["association_scores"].astype(
            np.longdouble
        )
        payload["association_scores"][0, 0] = underflowing
        cases.append(("association_scores", payload, {}))
        cases.append(
            (
                "production_costs",
                self._payload(),
                {"production_costs": np.asarray([underflowing])},
            )
        )
        cases.append(("threshold", self._payload(), {"thresholds": {5: underflowing}}))

        for expected, payload, kwargs in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex(
                    (TypeError, ValueError), f"{expected}.*underflow"
                ):
                    self._build(payload, **kwargs)

    def test_rejects_distinct_costs_that_collapse_and_change_ranking(self):
        lower = np.longdouble(1.0)
        higher = np.nextafter(lower, np.longdouble(2.0), dtype=np.longdouble)
        self.assertGreater(higher, lower)
        self.assertEqual(float(higher), float(lower))

        with self.assertRaisesRegex(ValueError, "base_cost_matrix.*same Python float"):
            self._build(base_cost_matrix=np.asarray([[higher, lower]]))

    def test_rejects_cost_gap_that_overflows_python_float(self):
        maximum = np.finfo(float).max
        with self.assertRaisesRegex(ValueError, "cost_gap.*finite"):
            self._build(base_cost_matrix=np.asarray([[maximum, -maximum]]))

    def test_non_finite_association_scores_remain_json_safe_nulls(self):
        for score in (np.nan, np.inf, -np.inf):
            with self.subTest(score=score):
                payload = self._payload()
                payload["association_scores"][0, 0] = score

                groups = self._build(payload)

                self.assertIsNone(groups[0]["candidates"][0]["association_score"])
                json.dumps(groups, allow_nan=False)

    def test_large_arrays_do_not_use_scalar_conversion_per_element(self):
        candidate_count = 128
        embedding_width = 32
        pair_width = 16
        payload = {
            "scene_id": "scene-large",
            "sample_token": "sample-large",
            "frame_id": 8,
            "track_ids": [31],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["matched"],
            "track_gt_tokens": [None],
            "det_gt_tokens": [f"det-{index}" for index in range(candidate_count)],
            "pair_indices": [(0, index) for index in range(candidate_count)],
            "pair_vectors": np.full(
                (candidate_count, pair_width), 0.25, dtype=np.float32
            ),
            "track_embeddings": np.full((1, embedding_width), 0.5, dtype=np.float32),
            "det_embeddings": np.full(
                (candidate_count, embedding_width), 0.75, dtype=np.float32
            ),
            "association_scores": np.full((1, candidate_count), 0.6, dtype=np.float32),
        }
        base_costs = np.linspace(0.1, 1.0, candidate_count, dtype=np.float64).reshape(
            1, -1
        )

        with patch.object(
            inference_pair_cache,
            "_finite_python_float",
            wraps=inference_pair_cache._finite_python_float,
        ) as scalar_conversion:
            groups = self._build(
                payload,
                base_cost_matrix=base_costs,
                production_costs=np.asarray([base_costs[0, 0]]),
            )

        self.assertEqual(groups[0]["candidate_count"], candidate_count)
        self.assertLessEqual(scalar_conversion.call_count, candidate_count + 2)

    def test_disabled_vector_serialization_still_validates_numeric_arrays(self):
        underflowing = np.nextafter(
            np.longdouble(0.0), np.longdouble(1.0), dtype=np.longdouble
        )
        cases = (
            ("track_embeddings", (0, 0)),
            ("pair_vectors", (0, 0)),
        )
        for field, index in cases:
            with self.subTest(field=field):
                payload = self._payload()
                payload[field] = payload[field].astype(np.longdouble)
                payload[field][index] = underflowing
                with self.assertRaisesRegex(ValueError, f"{field}.*underflow"):
                    self._build(
                        payload,
                        include_embeddings=False,
                        include_pair_vectors=False,
                    )

    def test_rejects_malformed_matrix_vector_and_embedding_shapes(self):
        cases = []
        payload = self._payload()
        cases.append(
            ("base_cost_matrix", payload, {"base_cost_matrix": np.zeros((2, 2))})
        )
        payload = self._payload()
        payload["pair_vectors"] = np.zeros((1, 2))
        cases.append(("pair_vectors", payload, {}))
        payload = self._payload()
        payload["pair_vectors"] = np.zeros((2, 2, 1))
        cases.append(("pair_vectors", payload, {}))
        payload = self._payload()
        payload["track_embeddings"] = np.zeros((2, 2))
        cases.append(("track_embeddings", payload, {}))
        payload = self._payload()
        payload["det_embeddings"] = np.zeros((2, 3))
        cases.append(("embeddings", payload, {}))
        payload = self._payload()
        payload["association_scores"] = np.zeros((2, 2))
        cases.append(("association_scores", payload, {}))

        for expected, payload, kwargs in cases:
            with self.subTest(expected=expected):
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    self._build(payload, **kwargs)

    def test_rejects_zero_embedding_and_pair_vector_widths(self):
        payload = self._payload()
        payload["track_embeddings"] = np.empty((1, 0), dtype=np.float32)
        payload["det_embeddings"] = np.empty((2, 0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "embedding width.*positive"):
            self._build(payload)

        payload = self._payload()
        payload["pair_vectors"] = np.empty((2, 0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "pair_vectors.*width.*positive"):
            self._build(payload)

        payload = self._payload()
        payload["pair_indices"] = []
        payload["pair_vectors"] = np.empty((0, 0), dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "pair_vectors.*width.*positive"):
            self._build(
                payload,
                production_matches=np.empty((0, 2), dtype=np.int64),
                production_costs=np.empty((0,), dtype=np.float32),
            )

    def test_rejects_misaligned_track_and_detection_metadata(self):
        cases = (
            ("class_ids", [5, 5]),
            ("class_names", []),
            ("state_buckets", ["matched", "unmatched"]),
            ("track_gt_tokens", []),
            ("det_gt_tokens", ["only-one"]),
        )
        for field, value in cases:
            with self.subTest(field=field):
                payload = self._payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._build(payload)

    def test_rejects_non_finite_pair_vectors_and_embeddings(self):
        cases = (
            ("pair_vectors", (0, 0)),
            ("track_embeddings", (0, 0)),
            ("det_embeddings", (0, 0)),
        )
        for field, index in cases:
            for value in (np.nan, np.inf, -np.inf):
                with self.subTest(field=field, value=value):
                    payload = self._payload()
                    payload[field][index] = value
                    with self.assertRaisesRegex(ValueError, field):
                        self._build(payload)

    def test_rejects_boolean_negative_duplicate_and_non_integer_track_ids(self):
        for track_ids in ([True], [-1], [1.0]):
            with self.subTest(track_ids=track_ids):
                payload = self._payload()
                payload["track_ids"] = track_ids
                with self.assertRaisesRegex(ValueError, "track_ids"):
                    self._build(payload)

        payload = {
            **self._payload(),
            "track_ids": [17, 17],
            "class_ids": [5, 5],
            "class_names": ["trailer", "trailer"],
            "state_buckets": ["matched", "unmatched"],
            "track_gt_tokens": [None, None],
            "track_embeddings": np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            "pair_indices": [(0, 0), (0, 1)],
            "pair_vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "association_scores": np.zeros((2, 2)),
        }
        with self.assertRaisesRegex(ValueError, "track_ids"):
            self._build(payload, base_cost_matrix=np.zeros((2, 2)))

    def test_rejects_invalid_class_ids_names_states_and_flags(self):
        cases = (
            ("class_ids", [True]),
            ("class_ids", [-1]),
            ("class_ids", [5.0]),
            ("class_names", [5]),
            ("state_buckets", ["lost"]),
        )
        for field, value in cases:
            with self.subTest(field=field, value=value):
                payload = self._payload()
                payload[field] = value
                with self.assertRaisesRegex(ValueError, field):
                    self._build(payload)

        for flag in ("include_embeddings", "include_pair_vectors"):
            with self.subTest(flag=flag):
                with self.assertRaisesRegex(TypeError, flag):
                    self._build(**{flag: 1})

    def test_rejects_duplicate_malformed_and_out_of_range_pair_indices(self):
        cases = (
            ([(0, 0), (0, 0)], "duplicate"),
            ([(1, 0), (0, 1)], "track index"),
            ([(0, 0), (0, 2)], "detection index"),
            ([(0, -1), (0, 1)], "detection index"),
            ([(0.0, 0), (0, 1)], "integer"),
            ([(0, 0, 1)], "shape"),
        )
        for pair_indices, expected in cases:
            with self.subTest(pair_indices=pair_indices):
                payload = self._payload()
                payload["pair_indices"] = pair_indices
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    self._build(payload)

    def test_rejects_malformed_duplicate_and_out_of_range_assignments(self):
        cases = (
            ([(0, 0, 1)], [0.2], "shape"),
            ([(0.0, 0)], [0.2], "integer"),
            ([(0, 0), (0, 1)], [0.2, 0.21], "track index"),
            ([(1, 0)], [0.2], "track index"),
            ([(0, 2)], [0.2], "detection index"),
            ([(0, -1)], [0.2], "detection index"),
        )
        for matches, costs, expected in cases:
            with self.subTest(matches=matches):
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    self._build(
                        production_matches=np.asarray(matches),
                        production_costs=np.asarray(costs),
                    )

        payload = {
            **self._payload(),
            "track_ids": [17, 18],
            "class_ids": [5, 5],
            "class_names": ["trailer", "trailer"],
            "state_buckets": ["matched", "unmatched"],
            "track_gt_tokens": [None, None],
            "track_embeddings": np.asarray([[0.1, 0.2], [0.3, 0.4]]),
            "pair_indices": [(0, 0), (1, 0)],
            "pair_vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "association_scores": np.zeros((2, 2)),
        }
        with self.assertRaisesRegex(ValueError, "detection index"):
            self._build(
                payload,
                base_cost_matrix=np.zeros((2, 2)),
                production_matches=np.asarray([(0, 0), (1, 0)]),
                production_costs=np.asarray([0.2, 0.21]),
            )

    def test_rejects_malformed_production_costs_and_non_candidate_match(self):
        cost_cases = (
            (np.asarray([[0.2]]), "shape"),
            (np.asarray([], dtype=np.float32), "shape"),
            (np.asarray([np.inf]), "finite"),
            (np.asarray(["bad"]), "numeric"),
        )
        for costs, expected in cost_cases:
            with self.subTest(costs=costs):
                with self.assertRaisesRegex((TypeError, ValueError), expected):
                    self._build(production_costs=costs)

        payload = self._payload()
        payload["pair_indices"] = [(0, 1)]
        payload["pair_vectors"] = np.asarray([[3.0, 4.0]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "pair_indices"):
            self._build(payload)

    def test_rejects_production_match_with_non_finite_base_cost(self):
        for base_cost in (np.nan, np.inf, -np.inf):
            with self.subTest(base_cost=base_cost):
                with self.assertRaisesRegex(
                    ValueError, "production.*base_cost_matrix.*finite"
                ):
                    self._build(base_cost_matrix=np.asarray([[base_cost, 0.21]]))

    def test_requires_finite_threshold_for_every_track_class(self):
        for thresholds in ({}, {5: np.inf}, {5: "bad"}):
            with self.subTest(thresholds=thresholds):
                with self.assertRaisesRegex((TypeError, ValueError), "threshold"):
                    self._build(thresholds=thresholds)

    def test_does_not_mutate_any_inputs(self):
        payload = self._payload()
        base_cost_matrix = np.asarray([[0.2, 0.21]], dtype=np.float32)
        production_matches = np.asarray([[0, 0]], dtype=np.int64)
        production_costs = np.asarray([0.2], dtype=np.float32)
        thresholds = {5: 1.26}
        originals = copy.deepcopy(
            (
                payload,
                base_cost_matrix,
                production_matches,
                production_costs,
                thresholds,
            )
        )

        self._build(
            payload,
            base_cost_matrix=base_cost_matrix,
            production_matches=production_matches,
            production_costs=production_costs,
            thresholds=thresholds,
        )

        (
            original_payload,
            original_base,
            original_matches,
            original_costs,
            original_thresholds,
        ) = originals
        self.assertEqual(payload.keys(), original_payload.keys())
        for key, value in payload.items():
            original = original_payload[key]
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, original)
            else:
                self.assertEqual(value, original)
        np.testing.assert_array_equal(base_cost_matrix, original_base)
        np.testing.assert_array_equal(production_matches, original_matches)
        np.testing.assert_array_equal(production_costs, original_costs)
        self.assertEqual(thresholds, original_thresholds)


class InferencePairCacheTask3Test(unittest.TestCase):
    def _group(
        self,
        *,
        scene_id="scene-1",
        sample_token="sample-1",
        frame_id=1,
        track_id=10,
        class_id=5,
        class_name="trailer",
        state_bucket="unmatched",
        status="matched",
    ):
        track_gt_token = None if status == "unknown" else "gt-track"
        det_gt_tokens = ["gt-other", "gt-track"]
        if status == "unknown":
            det_gt_tokens = [None, None]
        elif status == "no_match":
            det_gt_tokens = ["gt-other", "gt-another"]
        payload = {
            "scene_id": scene_id,
            "sample_token": sample_token,
            "frame_id": frame_id,
            "track_ids": [track_id],
            "class_ids": [class_id],
            "class_names": [class_name],
            "state_buckets": [state_bucket],
            "track_gt_tokens": [track_gt_token],
            "det_gt_tokens": det_gt_tokens,
            "pair_indices": [(0, 0), (0, 1)],
            "pair_vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
            "track_embeddings": np.asarray([[0.1, 0.2]]),
            "det_embeddings": np.asarray([[0.3, 0.4], [0.5, 0.6]]),
            "association_scores": np.asarray([[0.2, 0.8]]),
        }
        return build_candidate_groups(
            payload=payload,
            base_cost_matrix=np.asarray([[0.2, 0.3]]),
            production_matches=np.asarray([[0, 0]]),
            production_costs=np.asarray([0.2]),
            thresholds={class_id: 0.25},
        )[0]

    def _zero_candidate_group(self, *, known_track_gt):
        payload = {
            "scene_id": "scene-zero",
            "sample_token": "sample-zero",
            "frame_id": 1,
            "track_ids": [10],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["unmatched"],
            "track_gt_tokens": ["gt-track" if known_track_gt else None],
            "det_gt_tokens": [],
            "pair_indices": [],
            "pair_vectors": np.empty((0, 2)),
            "track_embeddings": np.asarray([[0.1, 0.2]]),
            "det_embeddings": np.empty((0, 2)),
        }
        return build_candidate_groups(
            payload=payload,
            base_cost_matrix=np.empty((1, 0)),
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,)),
            thresholds={5: 0.25},
        )[0]

    def _signed_zero_group(self, negative):
        zero = -0.0 if negative else 0.0
        payload = {
            "scene_id": "scene-zero-sign",
            "sample_token": "sample-zero-sign",
            "frame_id": 1,
            "track_ids": [10],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["unmatched"],
            "track_gt_tokens": ["gt-track"],
            "det_gt_tokens": ["gt-track", "gt-other"],
            "pair_indices": [(0, 0), (0, 1)],
            "pair_vectors": np.asarray([[zero, zero], [1.0, zero]]),
            "track_embeddings": np.asarray([[zero, zero]]),
            "det_embeddings": np.asarray([[zero, zero], [1.0, zero]]),
            "association_scores": np.asarray([[zero, 1.0]]),
        }
        return build_candidate_groups(
            payload=payload,
            base_cost_matrix=np.asarray([[zero, 1.0]]),
            production_matches=np.asarray([[0, 0]]),
            production_costs=np.asarray([zero]),
            thresholds={5: zero},
        )[0]

    @staticmethod
    def _error(frame_id, message="message"):
        return {
            "scene_id": "scene-1",
            "sample_token": "sample-1",
            "frame_id": frame_id,
            "stage": "association",
            "error_type": "RuntimeError",
            "message": message,
        }

    def test_candidate_group_validator_returns_independent_builtins(self):
        group = self._group()
        normalized = inference_pair_cache.validate_candidate_group(group)

        group["candidates"][0]["pair_vector"][0] = 99.0
        self.assertEqual(normalized["candidates"][0]["pair_vector"][0], 1.0)
        json.dumps(normalized, allow_nan=False)
        self.assertEqual(pickle.loads(pickle.dumps(normalized)), normalized)

    def test_signed_zero_is_canonical_across_add_and_merge_order(self):
        positive = self._signed_zero_group(False)
        negative = self._signed_zero_group(True)

        def assert_positive_zero(value):
            if isinstance(value, dict):
                for item in value.values():
                    assert_positive_zero(item)
            elif isinstance(value, list):
                for item in value:
                    assert_positive_zero(item)
            elif type(value) is float and value == 0.0:
                self.assertEqual(math.copysign(1.0, value), 1.0)

        for group in (positive, negative):
            assert_positive_zero(group)
            assert_positive_zero(
                inference_pair_cache.validate_candidate_group(group)
            )

        direct_outputs = []
        for ordered in ((positive, negative), (negative, positive)):
            accumulator = inference_pair_cache.InferencePairCacheAccumulator()
            for group in ordered:
                accumulator.add_group(group)
            direct_outputs.append(
                (
                    accumulator.to_payload(),
                    accumulator.to_summary(),
                    accumulator.export_state(),
                )
            )
        self.assertEqual(direct_outputs[0], direct_outputs[1])
        self.assertEqual(
            pickle.dumps(direct_outputs[0], protocol=4),
            pickle.dumps(direct_outputs[1], protocol=4),
        )
        self.assertEqual(
            inference_pair_cache.canonical_json(direct_outputs[0]),
            inference_pair_cache.canonical_json(direct_outputs[1]),
        )

        workers = []
        for group in (positive, negative):
            worker = inference_pair_cache.InferencePairCacheAccumulator()
            worker.add_group(group)
            workers.append(worker.export_state())
        merged_outputs = []
        for states in (workers, list(reversed(workers))):
            accumulator = inference_pair_cache.InferencePairCacheAccumulator()
            for state in states:
                accumulator.merge_state(state)
            merged_outputs.append(
                (
                    accumulator.to_payload(),
                    accumulator.to_summary(),
                    accumulator.export_state(),
                )
            )
        self.assertEqual(
            pickle.dumps(merged_outputs[0], protocol=4),
            pickle.dumps(merged_outputs[1], protocol=4),
        )

    def test_canonical_json_type_tags_mixed_mapping_keys(self):
        first = {1: "integer", "1": "string", 2: {0: -0.0}}
        second = {2: {0: 0.0}, "1": "string", 1: "integer"}

        self.assertEqual(
            inference_pair_cache.canonical_json(first),
            inference_pair_cache.canonical_json(second),
        )
        self.assertNotEqual(
            inference_pair_cache.canonical_json(first),
            inference_pair_cache.canonical_json({"1": "string", 2: {0: 0.0}}),
        )

    def test_candidate_group_validator_rejects_cross_field_inconsistency(self):
        cases = []
        group = self._group()
        group["candidate_count"] = 3
        cases.append(group)
        group = self._group()
        group["candidates"][0]["inside_geometric_gate"] = False
        cases.append(group)
        group = self._group()
        group["candidates"][1]["candidate_rank"] = 1
        cases.append(group)
        group = self._group()
        group["production_match_cost"] = None
        cases.append(group)
        group = self._group()
        group["candidates"][0]["gt_label"] = 1
        cases.append(group)
        group = self._group(status="unknown")
        group["candidates"][0]["gt_label"] = 0
        cases.append(group)
        group = self._group()
        group["candidates"][0]["det_index"] = True
        cases.append(group)
        group = self._group()
        group["candidates"][0]["pair_vector"] = [1.0, np.inf]
        cases.append(group)
        group = self._group()
        group["unexpected"] = "field"
        cases.append(group)
        group = self._group()
        group["schema_version"] = np.bool_(True)
        cases.append(group)

        for group in cases:
            with self.subTest(group=group):
                with self.assertRaises((TypeError, ValueError)):
                    inference_pair_cache.validate_candidate_group(group)

    def test_candidate_group_validator_rederives_gt_semantics(self):
        cases = []

        matched_with_missing_token = self._group()
        matched_with_missing_token["candidates"][0]["det_gt_token"] = None
        cases.append(matched_with_missing_token)

        no_match_with_missing_token = self._group(status="no_match")
        no_match_with_missing_token["candidates"][0]["det_gt_token"] = None
        cases.append(no_match_with_missing_token)

        duplicate_track_tokens = self._group()
        duplicate_track_tokens["candidates"][0]["det_gt_token"] = "gt-track"
        cases.append(duplicate_track_tokens)

        unjustified_matched_unknown = self._group()
        unjustified_matched_unknown["gt_target_status"] = "unknown"
        unjustified_matched_unknown["gt_target_det_index"] = None
        for candidate in unjustified_matched_unknown["candidates"]:
            candidate["gt_label"] = None
        cases.append(unjustified_matched_unknown)

        unjustified_no_match_unknown = self._group(status="no_match")
        unjustified_no_match_unknown["gt_target_status"] = "unknown"
        for candidate in unjustified_no_match_unknown["candidates"]:
            candidate["gt_label"] = None
        cases.append(unjustified_no_match_unknown)

        for group in cases:
            with self.subTest(group=group):
                with self.assertRaisesRegex(ValueError, "GT|gt_target|status|label"):
                    inference_pair_cache.validate_candidate_group(group)

    def test_candidate_group_validator_rederives_zero_candidate_status(self):
        known = self._zero_candidate_group(known_track_gt=True)
        unknown = self._zero_candidate_group(known_track_gt=False)
        self.assertEqual(
            inference_pair_cache.validate_candidate_group(known)[
                "gt_target_status"
            ],
            "no_match",
        )
        self.assertEqual(
            inference_pair_cache.validate_candidate_group(unknown)[
                "gt_target_status"
            ],
            "unknown",
        )

        invalid_known = copy.deepcopy(known)
        invalid_known["gt_target_status"] = "unknown"
        invalid_unknown = copy.deepcopy(unknown)
        invalid_unknown["gt_target_status"] = "no_match"
        for group in (invalid_known, invalid_unknown):
            with self.subTest(group=group):
                with self.assertRaisesRegex(ValueError, "status"):
                    inference_pair_cache.validate_candidate_group(group)

    def test_error_and_state_validators_are_strict_and_defensive(self):
        error = self._error(1, "x" * 1100)
        normalized = inference_pair_cache.validate_error_record(error)
        self.assertEqual(len(normalized["message"]), 1024)
        error["message"] = "changed"
        self.assertEqual(len(normalized["message"]), 1024)

        invalid_errors = [
            {**self._error(1), "frame_id": True},
            {**self._error(1), "stage": " stage"},
            {**self._error(1), "extra": "field"},
        ]
        for record in invalid_errors:
            with self.subTest(record=record):
                with self.assertRaises((TypeError, ValueError)):
                    inference_pair_cache.validate_error_record(record)

        state_source = inference_pair_cache.InferencePairCacheAccumulator(
            runtime_manifest={"run": 1}
        )
        state_source.add_group(self._group())
        state_source.add_error(self._error(1))
        state = state_source.export_state()
        validated = inference_pair_cache.validate_accumulator_state(state)
        state["groups"][0]["scene_id"] = "mutated"
        self.assertEqual(validated["groups"][0]["scene_id"], "scene-1")

        for field, value in (
            ("schema_version", 999),
            ("error_count", True),
            ("error_count", -1),
            ("error_samples", [self._error(1), self._error(2)]),
        ):
            with self.subTest(field=field, value=value):
                invalid = copy.deepcopy(validated)
                invalid[field] = value
                with self.assertRaises((TypeError, ValueError)):
                    inference_pair_cache.validate_accumulator_state(invalid)

    def test_state_validator_enforces_default_and_custom_error_sample_bounds(self):
        samples = [self._error(index) for index in range(33)]
        state = {
            "schema_version": inference_pair_cache.INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "runtime_manifest": {},
            "observed_metadata": [],
            "groups": [],
            "error_count": len(samples),
            "error_samples": samples,
        }
        with self.assertRaisesRegex(ValueError, "max_error_samples"):
            inference_pair_cache.validate_accumulator_state(state)

        validated = inference_pair_cache.validate_accumulator_state(
            state, max_error_samples=33
        )
        self.assertEqual(len(validated["error_samples"]), 33)

        source = inference_pair_cache.InferencePairCacheAccumulator(
            max_error_samples=3
        )
        for sample in samples[:3]:
            source.add_error(sample)
        receiver = inference_pair_cache.InferencePairCacheAccumulator(
            max_error_samples=2
        )
        with self.assertRaisesRegex(ValueError, "max_error_samples"):
            receiver.merge_state(source.export_state())
        self.assertEqual(receiver.export_state()["error_count"], 0)

        with self.assertRaises((TypeError, ValueError)):
            inference_pair_cache.validate_accumulator_state(
                {**state, "error_samples": []}, max_error_samples=True
            )

    def test_accumulator_options_fail_closed(self):
        cls = inference_pair_cache.InferencePairCacheAccumulator
        for kwargs in (
            {"max_groups_per_class_state": True},
            {"max_groups_per_class_state": -1},
            {"max_error_samples": False},
            {"max_error_samples": -1},
            {"require_audit_gt": 1},
        ):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    cls(**kwargs)

    def test_duplicate_group_is_idempotent_but_conflict_raises(self):
        accumulator = inference_pair_cache.InferencePairCacheAccumulator()
        group = self._group()
        self.assertTrue(accumulator.add_group(group))
        self.assertTrue(accumulator.add_group(copy.deepcopy(group)))
        self.assertEqual(len(accumulator.export_state()["groups"]), 1)

        conflict = copy.deepcopy(group)
        conflict["candidates"][0]["association_score"] = 0.9
        with self.assertRaisesRegex(ValueError, "conflicting"):
            accumulator.add_group(conflict)

        payload = accumulator.to_payload()
        payload["groups"][0]["scene_id"] = "mutated"
        self.assertEqual(
            accumulator.to_payload()["groups"][0]["scene_id"], "scene-1"
        )

    def test_manifest_group_contract_rejects_dimensions_exports_and_class(self):
        manifest = {
            "embedding_dim": 2,
            "pair_vector_dim": 2,
            "class_map": {"trailer": 5},
            "export_options": {
                "include_embeddings": True,
                "include_pair_vectors": True,
            },
        }
        cases = []

        wrong_embedding_width = self._group()
        wrong_embedding_width["track_embedding"] = [0.0, 0.0, 0.0]
        for candidate in wrong_embedding_width["candidates"]:
            candidate["det_embedding"] = [0.0, 0.0, 0.0]
        cases.append((manifest, wrong_embedding_width, "embedding_dim"))

        wrong_pair_width = self._group()
        for candidate in wrong_pair_width["candidates"]:
            candidate["pair_vector"] = [0.0, 0.0, 0.0]
        cases.append((manifest, wrong_pair_width, "pair_vector_dim"))

        missing_embeddings = self._group()
        missing_embeddings.pop("track_embedding")
        for candidate in missing_embeddings["candidates"]:
            candidate.pop("det_embedding")
        cases.append((manifest, missing_embeddings, "include_embeddings"))

        missing_pair_vectors = self._group()
        for candidate in missing_pair_vectors["candidates"]:
            candidate.pop("pair_vector")
        cases.append((manifest, missing_pair_vectors, "include_pair_vectors"))

        disabled_manifest = copy.deepcopy(manifest)
        disabled_manifest["export_options"]["include_embeddings"] = False
        cases.append((disabled_manifest, self._group(), "include_embeddings"))

        disabled_pairs = copy.deepcopy(manifest)
        disabled_pairs["export_options"]["include_pair_vectors"] = False
        cases.append((disabled_pairs, self._group(), "include_pair_vectors"))

        wrong_class = copy.deepcopy(manifest)
        wrong_class["class_map"] = {"trailer": 4}
        cases.append((wrong_class, self._group(), "class_map"))

        missing_class = copy.deepcopy(manifest)
        missing_class["class_map"] = {"car": 0}
        cases.append((missing_class, self._group(), "class_map"))

        for runtime_manifest, group, expected in cases:
            with self.subTest(expected=expected):
                accumulator = inference_pair_cache.InferencePairCacheAccumulator(
                    runtime_manifest=runtime_manifest
                )
                with self.assertRaisesRegex(ValueError, expected):
                    accumulator.add_group(group)

    def test_group_ingestion_requires_one_canonical_threshold(self):
        group = self._group()
        group["candidates"][1]["geometric_threshold"] = 0.3
        group["candidates"][1]["inside_geometric_gate"] = True
        accumulator = inference_pair_cache.InferencePairCacheAccumulator()

        with self.assertRaisesRegex(ValueError, "geometric_threshold"):
            accumulator.add_group(group)

    def test_summary_rejects_duplicate_retained_stable_keys(self):
        group = self._group()
        with self.assertRaisesRegex(ValueError, "duplicate.*retained"):
            inference_pair_cache.summarize_candidate_groups(
                observed_groups=[group],
                retained_groups=[group, copy.deepcopy(group)],
                error_count=0,
                error_samples=[],
                runtime_manifest={},
                require_audit_gt=False,
            )

    def test_require_gt_and_per_bucket_cap_are_deterministic(self):
        accumulator = inference_pair_cache.InferencePairCacheAccumulator(
            require_audit_gt=True, max_groups_per_class_state=2
        )
        groups = [
            self._group(frame_id=4, track_id=40),
            self._group(frame_id=2, track_id=20),
            self._group(frame_id=3, track_id=30),
            self._group(frame_id=1, track_id=10, status="unknown"),
            self._group(
                frame_id=5,
                track_id=50,
                class_id=1,
                class_name="pedestrian",
                state_bucket="matched",
            ),
        ]
        for group in reversed(groups):
            accumulator.add_group(group)

        payload = accumulator.to_payload()
        keys = [(group["frame_id"], group["track_id"]) for group in payload["groups"]]
        self.assertEqual(keys, [(2, 20), (3, 30), (5, 50)])
        self.assertEqual(len(accumulator.export_state()["groups"]), 3)
        self.assertEqual(len(accumulator.export_state()["observed_metadata"]), 5)
        overall = accumulator.to_summary()["overall"]
        self.assertEqual(overall["observed_groups"], 5)
        self.assertEqual(overall["accepted_groups"], 3)
        self.assertEqual(overall["rejected_unknown_groups"], 1)
        self.assertEqual(overall["capped_groups"], 1)

    def test_capped_unknown_attribution_and_merge_order_for_both_gt_modes(self):
        known = self._group(frame_id=1, track_id=10)
        unknown = self._group(frame_id=2, track_id=20, status="unknown")

        for require_audit_gt, expected_rejected, expected_capped in (
            (False, 0, 1),
            (True, 1, 0),
        ):
            with self.subTest(require_audit_gt=require_audit_gt):
                worker_states = []
                for group in (known, unknown):
                    worker = inference_pair_cache.InferencePairCacheAccumulator()
                    worker.add_group(group)
                    worker_states.append(worker.export_state())

                outputs = []
                for states in (worker_states, list(reversed(worker_states))):
                    accumulator = inference_pair_cache.InferencePairCacheAccumulator(
                        max_groups_per_class_state=1,
                        require_audit_gt=require_audit_gt,
                    )
                    for state in states:
                        accumulator.merge_state(state)
                    outputs.append(
                        (
                            accumulator.to_payload(),
                            accumulator.to_summary(),
                            accumulator.export_state(),
                        )
                    )

                self.assertEqual(outputs[0], outputs[1])
                overall = outputs[0][1]["overall"]
                self.assertEqual(
                    overall["rejected_unknown_groups"], expected_rejected
                )
                self.assertEqual(overall["capped_groups"], expected_capped)
                self.assertEqual(overall["accepted_groups"], 1)
                for buckets in (
                    outputs[0][1]["per_class"],
                    outputs[0][1]["per_class_state"],
                ):
                    self.assertEqual(
                        sum(
                            bucket["rejected_unknown_groups"]
                            for bucket in buckets.values()
                        ),
                        expected_rejected,
                    )
                    self.assertEqual(
                        sum(
                            bucket["capped_groups"]
                            for bucket in buckets.values()
                        ),
                        expected_capped,
                    )

    def test_merge_order_does_not_change_outputs_or_error_samples(self):
        manifest = {"run": "same"}
        left = inference_pair_cache.InferencePairCacheAccumulator(
            runtime_manifest=manifest, max_error_samples=2
        )
        right = inference_pair_cache.InferencePairCacheAccumulator(
            runtime_manifest=manifest, max_error_samples=2
        )
        left.add_group(self._group(frame_id=2, track_id=20))
        right.add_group(self._group(frame_id=1, track_id=10))
        left.add_error(self._error(3, "z"))
        left.add_error(self._error(1, "a"))
        right.add_error(self._error(2, "b"))
        right.add_error(self._error(1, "a"))

        first = inference_pair_cache.InferencePairCacheAccumulator(
            runtime_manifest=manifest, max_error_samples=2
        )
        second = inference_pair_cache.InferencePairCacheAccumulator(
            runtime_manifest=manifest, max_error_samples=2
        )
        first.merge_state(left.export_state())
        first.merge_state(right.export_state())
        second.merge_state(right.export_state())
        second.merge_state(left.export_state())

        self.assertEqual(first.to_payload(), second.to_payload())
        self.assertEqual(first.to_summary(), second.to_summary())
        self.assertEqual(first.export_state(), second.export_state())
        self.assertEqual(first.export_state()["error_count"], 4)
        self.assertEqual(
            [item["frame_id"] for item in first.export_state()["error_samples"]],
            [1, 2],
        )

    def test_positive_cap_bounds_full_groups_and_preserves_observed_metadata(self):
        accumulator = inference_pair_cache.InferencePairCacheAccumulator(
            max_groups_per_class_state=2,
            require_audit_gt=True,
        )
        observed_groups = []
        for frame_id in range(20):
            status = "unknown" if frame_id % 5 == 0 else "matched"
            group = self._group(
                frame_id=frame_id,
                track_id=100 + frame_id,
                status=status,
            )
            group["track_embedding"] = [float(frame_id)] * 256
            for candidate in group["candidates"]:
                candidate["det_embedding"] = [float(frame_id)] * 256
                candidate["pair_vector"] = [float(frame_id)] * 128
            observed_groups.append(copy.deepcopy(group))
            accumulator.add_group(group)

        state = accumulator.export_state()
        self.assertEqual(len(state["groups"]), 2)
        self.assertEqual(len(state["observed_metadata"]), 20)
        self.assertTrue(
            all(
                "candidates" not in metadata
                and "track_embedding" not in metadata
                for metadata in state["observed_metadata"]
            )
        )
        self.assertEqual(
            [group["frame_id"] for group in state["groups"]], [1, 2]
        )
        overall = accumulator.to_summary()["overall"]
        self.assertEqual(overall["observed_groups"], 20)
        self.assertEqual(overall["rejected_unknown_groups"], 4)
        self.assertEqual(overall["accepted_groups"], 2)
        self.assertEqual(overall["capped_groups"], 14)

        full_observed_bytes = len(pickle.dumps(observed_groups, protocol=4))
        bounded_state_bytes = len(pickle.dumps(state, protocol=4))
        self.assertLess(bounded_state_bytes, full_observed_bytes // 2)

        conflicting_evicted = copy.deepcopy(observed_groups[-1])
        conflicting_evicted["candidates"][0]["association_score"] = 0.9
        with self.assertRaisesRegex(ValueError, "conflicting"):
            accumulator.add_group(conflicting_evicted)

    def test_bounded_state_merge_is_order_invariant_and_detects_metadata_conflict(self):
        groups = [
            self._group(frame_id=frame_id, track_id=100 + frame_id)
            for frame_id in range(10)
        ]
        workers = []
        for partition in (groups[::2], groups[1::2]):
            worker = inference_pair_cache.InferencePairCacheAccumulator(
                max_groups_per_class_state=2
            )
            for group in reversed(partition):
                worker.add_group(group)
            workers.append(worker.export_state())

        outputs = []
        for states in (workers, list(reversed(workers))):
            accumulator = inference_pair_cache.InferencePairCacheAccumulator(
                max_groups_per_class_state=2
            )
            for state in states:
                accumulator.merge_state(state)
            outputs.append(
                (
                    accumulator.to_payload(),
                    accumulator.to_summary(),
                    accumulator.export_state(),
                )
            )
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(len(outputs[0][0]["groups"]), 2)
        self.assertEqual(len(outputs[0][2]["observed_metadata"]), 10)
        self.assertEqual(outputs[0][1]["overall"]["observed_groups"], 10)

        first = inference_pair_cache.InferencePairCacheAccumulator(
            max_groups_per_class_state=1
        )
        first.add_group(groups[-1])
        first.add_group(groups[0])
        conflicting = copy.deepcopy(groups[-1])
        conflicting["candidates"][0]["association_score"] = 0.9
        second = inference_pair_cache.InferencePairCacheAccumulator(
            max_groups_per_class_state=1
        )
        second.add_group(conflicting)
        receiver = inference_pair_cache.InferencePairCacheAccumulator(
            max_groups_per_class_state=1
        )
        receiver.merge_state(first.export_state())
        with self.assertRaisesRegex(ValueError, "conflicting"):
            receiver.merge_state(second.export_state())

    def test_summary_reconciles_exactly_with_retained_payload(self):
        groups = [
            self._group(frame_id=1, track_id=10),
            self._group(frame_id=4, track_id=40, status="no_match"),
            self._group(
                frame_id=2,
                track_id=20,
                state_bucket="matched",
                status="no_match",
            ),
            self._group(
                frame_id=3,
                track_id=30,
                class_id=1,
                class_name="pedestrian",
                state_bucket="matched",
                status="unknown",
            ),
        ]

        def expected_counts(observed_groups, retained_groups, require_audit_gt):
            candidates = [
                candidate
                for group in retained_groups
                for candidate in group["candidates"]
            ]
            eligible_groups = [
                group
                for group in observed_groups
                if not (
                    require_audit_gt
                    and group["gt_target_status"] == "unknown"
                )
            ]
            return {
                "observed_groups": len(observed_groups),
                "accepted_groups": len(retained_groups),
                "rejected_unknown_groups": (
                    len(observed_groups) - len(eligible_groups)
                ),
                "capped_groups": len(eligible_groups) - len(retained_groups),
                "matched_groups": sum(
                    group["gt_target_status"] == "matched"
                    for group in retained_groups
                ),
                "no_match_groups": sum(
                    group["gt_target_status"] == "no_match"
                    for group in retained_groups
                ),
                "unknown_groups": sum(
                    group["gt_target_status"] == "unknown"
                    for group in retained_groups
                ),
                "candidate_count": len(candidates),
                "finite_candidate_count": sum(
                    candidate["finite"] for candidate in candidates
                ),
                "gated_candidate_count": sum(
                    candidate["inside_geometric_gate"]
                    for candidate in candidates
                ),
                "production_selected_count": sum(
                    candidate["production_selected"]
                    for candidate in candidates
                ),
                "positive_count": sum(
                    candidate["gt_label"] == 1 for candidate in candidates
                ),
                "negative_count": sum(
                    candidate["gt_label"] == 0 for candidate in candidates
                ),
                "unknown_label_count": sum(
                    candidate["gt_label"] is None for candidate in candidates
                ),
            }

        for require_audit_gt in (False, True):
            with self.subTest(require_audit_gt=require_audit_gt):
                accumulator = inference_pair_cache.InferencePairCacheAccumulator(
                    max_groups_per_class_state=1,
                    require_audit_gt=require_audit_gt,
                )
                for group in reversed(groups):
                    accumulator.add_group(group)

                observed = accumulator.export_state()["observed_metadata"]
                retained = accumulator.to_payload()["groups"]
                summary = accumulator.to_summary()
                overall_expected = expected_counts(
                    observed, retained, require_audit_gt
                )
                self.assertEqual(
                    summary["overall"], {**overall_expected, "error_count": 0}
                )
                self.assertEqual(
                    summary["overall"]["gated_candidate_count"],
                    sum(
                        candidate["inside_geometric_gate"]
                        for group in retained
                        for candidate in group["candidates"]
                    ),
                )

                class_names = sorted({group["class_name"] for group in observed})
                expected_per_class = {
                    class_name: expected_counts(
                        [
                            group
                            for group in observed
                            if group["class_name"] == class_name
                        ],
                        [
                            group
                            for group in retained
                            if group["class_name"] == class_name
                        ],
                        require_audit_gt,
                    )
                    for class_name in class_names
                }
                self.assertEqual(summary["per_class"], expected_per_class)

                class_states = sorted(
                    {
                        (group["class_id"], group["state_bucket"])
                        for group in observed
                    }
                )
                expected_per_class_state = {}
                for class_id, state_bucket in class_states:
                    key = json.dumps(
                        [class_id, state_bucket], separators=(",", ":")
                    )
                    expected_per_class_state[key] = expected_counts(
                        [
                            group
                            for group in observed
                            if group["class_id"] == class_id
                            and group["state_bucket"] == state_bucket
                        ],
                        [
                            group
                            for group in retained
                            if group["class_id"] == class_id
                            and group["state_bucket"] == state_bucket
                        ],
                        require_audit_gt,
                    )
                self.assertEqual(
                    summary["per_class_state"], expected_per_class_state
                )
                for field, expected_value in overall_expected.items():
                    self.assertEqual(
                        sum(
                            bucket[field]
                            for bucket in summary["per_class"].values()
                        ),
                        expected_value,
                    )
                    self.assertEqual(
                        sum(
                            bucket[field]
                            for bucket in summary["per_class_state"].values()
                        ),
                        expected_value,
                    )

    def test_manifest_hashes_files_and_effective_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            main_checkpoint = root / "main.pt"
            association_checkpoint = root / "association.pt"
            config_path.write_bytes(b"config-file\n")
            main_checkpoint.write_bytes(b"main-checkpoint")
            association_checkpoint.write_bytes(b"association-checkpoint")
            cfg = {
                "FILTER_MODE": "MAMBA",
                "TRACKER_COMPAT_MODE": "MCTrack",
                "CATEGORY_MAP_TO_NUMBER": {"car": 0},
                "MAMBA": {"CHECKPOINT_PATH": "main.pt"},
                "MAMBA_ASSOCIATION_HEAD": {
                    "CHECKPOINT_PATH": "association.pt"
                },
                "THRESHOLD": {"BEV": {"COST_MODE": "geometric"}},
                "MATCHING": {"BEV": {"COST_MODE": {0: "geometric"}}},
                "INFERENCE_PAIR_CACHE": {
                    "INCLUDE_EMBEDDINGS": True,
                    "INCLUDE_PAIR_VECTORS": False,
                    "REQUIRE_AUDIT_GT": True,
                },
            }
            git_results = [
                subprocess.CompletedProcess([], 0, stdout="abc123\n"),
                subprocess.CompletedProcess([], 0, stdout=""),
                subprocess.CompletedProcess([], 0, stdout=b""),
                subprocess.CompletedProcess([], 0, stdout=""),
            ]
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results,
            ):
                manifest = inference_pair_cache.build_inference_pair_cache_manifest(
                    cfg=cfg,
                    config_path=config_path,
                    process_count=2,
                    git_root=root,
                    pair_vector_dim=7,
                    embedding_dim=64,
                )

            self.assertEqual(
                manifest["config_sha256"], hashlib.sha256(b"config-file\n").hexdigest()
            )
            self.assertEqual(
                manifest["effective_config_sha256"],
                hashlib.sha256(
                    inference_pair_cache.canonical_json(cfg).encode("utf-8")
                ).hexdigest(),
            )
            self.assertEqual(
                manifest["mamba_checkpoint_sha256"],
                hashlib.sha256(b"main-checkpoint").hexdigest(),
            )
            self.assertEqual(
                manifest["association_checkpoint_sha256"],
                hashlib.sha256(b"association-checkpoint").hexdigest(),
            )
            self.assertEqual(manifest["git_commit"], "abc123")
            self.assertFalse(manifest["git_dirty"])
            self.assertEqual(len(manifest["git_worktree_sha256"]), 64)
            json.dumps(manifest, allow_nan=False)

            mismatches = {
                "mamba_checkpoint_sha256": "different",
                "association_checkpoint_sha256": "different",
                "effective_config_sha256": "different",
                "git_commit": "different",
                "git_dirty": True,
                "git_worktree_sha256": "different",
            }
            for field, different_value in mismatches.items():
                with self.subTest(field=field):
                    changed = copy.deepcopy(manifest)
                    changed[field] = different_value
                    accumulator = inference_pair_cache.InferencePairCacheAccumulator(
                        runtime_manifest=manifest
                    )
                    state = inference_pair_cache.InferencePairCacheAccumulator(
                        runtime_manifest=changed
                    ).export_state()
                    with self.assertRaisesRegex(ValueError, "manifest mismatch"):
                        accumulator.merge_state(state)

    def test_manifest_git_identity_is_stable_dirty_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            source_path = root / "new_source.py"
            output_path = root / "cache_summary.json"
            config_path.write_text("config\n", encoding="utf-8")
            source_path.write_text("VALUE = 1\n", encoding="utf-8")
            output_path.write_text('{"run": 1}\n', encoding="utf-8")
            cfg = {
                "INFERENCE_PAIR_CACHE": {
                    "OUTPUT_PATH": str(output_path),
                }
            }
            kwargs = {
                "cfg": cfg,
                "config_path": config_path,
                "process_count": 1,
                "git_root": root,
                "pair_vector_dim": 7,
                "embedding_dim": 2,
            }

            def git_results(status, diff, untracked):
                return [
                    subprocess.CompletedProcess([], 0, stdout="commit-1\n"),
                    subprocess.CompletedProcess([], 0, stdout=status),
                    subprocess.CompletedProcess([], 0, stdout=diff),
                    subprocess.CompletedProcess([], 0, stdout=untracked),
                ]

            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results("", b"", ""),
            ):
                clean_first = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results("", b"", ""),
            ):
                clean_second = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            self.assertFalse(clean_first["git_dirty"])
            self.assertEqual(
                clean_first["git_worktree_sha256"],
                clean_second["git_worktree_sha256"],
            )

            dirty_status = " M tracked.py\n?? new_source.py\n?? cache_summary.json\n"
            untracked = "new_source.py\ncache_summary.json\n"
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results(dirty_status, b"tracked-diff", untracked),
            ):
                dirty_first = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            output_path.write_text('{"run": 2}\n', encoding="utf-8")
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results(dirty_status, b"tracked-diff", untracked),
            ):
                output_changed = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            self.assertTrue(dirty_first["git_dirty"])
            self.assertEqual(
                dirty_first["git_worktree_sha256"],
                output_changed["git_worktree_sha256"],
            )
            source_only_status = " M tracked.py\n?? new_source.py\n"
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results(
                    source_only_status, b"tracked-diff", "new_source.py\n"
                ),
            ):
                output_status_removed = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            self.assertEqual(
                dirty_first["git_worktree_sha256"],
                output_status_removed["git_worktree_sha256"],
            )

            source_path.write_text("VALUE = 2\n", encoding="utf-8")
            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=git_results(dirty_status, b"tracked-diff", untracked),
            ):
                source_changed = (
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)
                )
            self.assertNotEqual(
                dirty_first["git_worktree_sha256"],
                source_changed["git_worktree_sha256"],
            )

            with patch(
                "tracker.inference_pair_cache.subprocess.run",
                side_effect=subprocess.CalledProcessError(1, ["git"]),
            ):
                with self.assertRaisesRegex(RuntimeError, "git identity"):
                    inference_pair_cache.build_inference_pair_cache_manifest(**kwargs)

    def test_manifest_rejects_invalid_dimensions_and_missing_configured_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config_path = root / "config.yaml"
            config_path.write_text("config\n", encoding="utf-8")
            base = {
                "cfg": {},
                "config_path": config_path,
                "process_count": 1,
                "git_root": root,
                "pair_vector_dim": 7,
                "embedding_dim": 64,
            }
            for field, value in (
                ("process_count", True),
                ("process_count", 0),
                ("pair_vector_dim", False),
                ("embedding_dim", -1),
            ):
                with self.subTest(field=field, value=value):
                    kwargs = {**base, field: value}
                    with self.assertRaises((TypeError, ValueError)):
                        inference_pair_cache.build_inference_pair_cache_manifest(
                            **kwargs
                        )

            missing = {**base, "cfg": {"MAMBA": {"CHECKPOINT_PATH": "missing.pt"}}}
            with self.assertRaises(FileNotFoundError):
                inference_pair_cache.build_inference_pair_cache_manifest(**missing)

    def test_atomic_writers_replace_successfully_and_leave_no_temp_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pickle_path = root / "nested" / "payload.pkl"
            json_path = root / "nested" / "payload.json"
            pickle_path.parent.mkdir(parents=True)
            pickle_path.write_bytes(b"old-pickle")
            json_path.write_bytes(b"old-json")
            inference_pair_cache.atomic_write_pickle(pickle_path, {"value": 1})
            inference_pair_cache.atomic_write_json(json_path, {"z": 1, "a": "text"})

            with pickle_path.open("rb") as file_obj:
                self.assertEqual(pickle.load(file_obj), {"value": 1})
            self.assertEqual(
                json_path.read_bytes(), b'{\n  "a": "text",\n  "z": 1\n}\n'
            )
            self.assertEqual(list((root / "nested").glob("*.tmp")), [])

    def test_atomic_writers_preserve_destination_on_serialization_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for writer, suffix, patch_target in (
                (
                    inference_pair_cache.atomic_write_pickle,
                    ".pkl",
                    "tracker.inference_pair_cache.pickle.dump",
                ),
                (
                    inference_pair_cache.atomic_write_json,
                    ".json",
                    "tracker.inference_pair_cache.json.dumps",
                ),
            ):
                with self.subTest(suffix=suffix):
                    destination = root / f"existing{suffix}"
                    destination.write_bytes(b"original-bytes")
                    with patch(patch_target, side_effect=RuntimeError("injected")):
                        with self.assertRaisesRegex(RuntimeError, "injected"):
                            writer(destination, {"value": 2})
                    self.assertEqual(destination.read_bytes(), b"original-bytes")
                    self.assertEqual(list(root.glob(f".{destination.name}.*.tmp")), [])

    def test_atomic_writer_preserves_destination_on_fsync_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "existing.pkl"
            destination.write_bytes(b"original-bytes")

            with patch(
                "tracker.inference_pair_cache.os.fsync",
                side_effect=OSError("injected write failure"),
            ):
                with self.assertRaisesRegex(OSError, "injected write failure"):
                    inference_pair_cache.atomic_write_pickle(
                        destination, {"value": 2}
                    )

            self.assertEqual(destination.read_bytes(), b"original-bytes")
            self.assertEqual(list(root.glob(f".{destination.name}.*.tmp")), [])

    def test_atomic_writer_fsyncs_parent_directory_and_closes_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "payload.pkl"
            with patch(
                "tracker.inference_pair_cache.os.open", wraps=os.open
            ) as open_mock, patch(
                "tracker.inference_pair_cache.os.fsync", wraps=os.fsync
            ) as fsync_mock, patch(
                "tracker.inference_pair_cache.os.close", wraps=os.close
            ) as close_mock:
                inference_pair_cache.atomic_write_pickle(destination, {"value": 1})

            self.assertTrue(
                any(
                    call.args
                    == (str(root), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
                    for call in open_mock.call_args_list
                )
            )
            self.assertTrue(
                any(
                    call.args == close_call.args
                    for call in fsync_mock.call_args_list
                    for close_call in close_mock.call_args_list
                )
            )

    def test_atomic_writer_replace_failure_preserves_old_file_and_closes_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "payload.pkl"
            destination.write_bytes(b"original-bytes")
            created_descriptors = []
            real_mkstemp = tempfile.mkstemp

            def capture_mkstemp(*args, **kwargs):
                result = real_mkstemp(*args, **kwargs)
                created_descriptors.append(result[0])
                return result

            with patch(
                "tracker.inference_pair_cache.tempfile.mkstemp",
                side_effect=capture_mkstemp,
            ), patch(
                "tracker.inference_pair_cache.os.replace",
                side_effect=OSError("replace failed"),
            ):
                with self.assertRaisesRegex(OSError, "replace failed"):
                    inference_pair_cache.atomic_write_pickle(
                        destination, {"value": 2}
                    )

            temporary_descriptor = created_descriptors[0]
            with self.assertRaises(OSError):
                os.fstat(temporary_descriptor)
            self.assertEqual(destination.read_bytes(), b"original-bytes")
            self.assertEqual(list(root.glob(f".{destination.name}.*.tmp")), [])

    def test_parent_fsync_failure_reports_after_destination_was_replaced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / "payload.pkl"
            destination.write_bytes(b"original-bytes")
            with patch(
                "tracker.inference_pair_cache.os.fsync",
                side_effect=[None, OSError("parent fsync failed")],
            ), patch(
                "tracker.inference_pair_cache.os.close", wraps=os.close
            ) as close_mock:
                with self.assertRaisesRegex(OSError, "parent fsync failed"):
                    inference_pair_cache.atomic_write_pickle(
                        destination, {"value": 2}
                    )

            with destination.open("rb") as file_obj:
                self.assertEqual(pickle.load(file_obj), {"value": 2})
            self.assertTrue(close_mock.called)
            self.assertEqual(list(root.glob(f".{destination.name}.*.tmp")), [])


if __name__ == "__main__":
    unittest.main()
