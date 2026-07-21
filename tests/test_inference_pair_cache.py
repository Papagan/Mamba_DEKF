import copy
import json
import pickle
import unittest

import numpy as np

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


if __name__ == "__main__":
    unittest.main()
