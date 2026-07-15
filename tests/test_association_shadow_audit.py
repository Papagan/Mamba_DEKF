import unittest

import numpy as np

from tracker.association_shadow_audit import (
    assignment_map,
    classify_assignment_outcome,
    compare_assignment_results,
)


class AssignmentMapTest(unittest.TestCase):
    def test_maps_each_matched_track_to_detection_and_cost(self):
        result = assignment_map(
            np.array([[2, 4], [0, 1]], dtype=int),
            np.array([0.35, 0.12]),
        )

        self.assertEqual(
            result,
            {
                0: {"det_index": 1, "cost": 0.12},
                2: {"det_index": 4, "cost": 0.35},
            },
        )

    def test_rejects_malformed_or_non_unique_assignments(self):
        with self.assertRaisesRegex(ValueError, "shape"):
            assignment_map(np.array([0, 1], dtype=int), np.array([0.2]))
        with self.assertRaisesRegex(TypeError, "integer"):
            assignment_map(np.array([[0.0, 1.0]]), np.array([0.2]))
        with self.assertRaisesRegex(ValueError, "duplicate assignment for track"):
            assignment_map(
                np.array([[0, 0], [0, 1]], dtype=int),
                np.array([0.2, 0.3]),
            )
        with self.assertRaisesRegex(ValueError, "duplicate assignment for detection"):
            assignment_map(
                np.array([[0, 0], [1, 0]], dtype=int),
                np.array([0.2, 0.3]),
            )

    def test_rejects_nonfinite_selected_cost(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            assignment_map(np.array([[0, 0]], dtype=int), np.array([np.inf]))

    def test_accepts_empty_float_match_array_from_assignment_solver(self):
        self.assertEqual(
            assignment_map(np.empty((0, 2)), np.empty((0,), dtype=float)),
            {},
        )

    def test_rejects_complex_cost_arrays_including_empty_arrays(self):
        with self.assertRaisesRegex(TypeError, "costs.*real-valued"):
            assignment_map(
                np.array([[0, 0]], dtype=int),
                np.array([0.2 + 0.0j]),
            )
        with self.assertRaisesRegex(TypeError, "costs.*real-valued"):
            assignment_map(
                np.empty((0, 2), dtype=int),
                np.empty((0,), dtype=complex),
            )


class ClassifyAssignmentOutcomeTest(unittest.TestCase):
    def test_improved_when_shadow_becomes_gt_consistent(self):
        self.assertEqual(
            classify_assignment_outcome("gt-b", "gt-a", "gt-b"),
            "improved",
        )
        self.assertEqual(
            classify_assignment_outcome("gt-b", None, "gt-b"),
            "improved",
        )

    def test_damaged_when_production_gt_consistency_is_lost(self):
        self.assertEqual(
            classify_assignment_outcome("gt-b", "gt-b", "gt-a"),
            "damaged",
        )
        self.assertEqual(
            classify_assignment_outcome("gt-b", "gt-b", None),
            "damaged",
        )

    def test_neutral_when_known_choices_have_equivalent_correctness(self):
        self.assertEqual(
            classify_assignment_outcome("gt-b", "gt-a", "gt-c"),
            "neutral",
        )
        self.assertEqual(
            classify_assignment_outcome("gt-b", "gt-b", "gt-b"),
            "neutral",
        )
        self.assertEqual(
            classify_assignment_outcome("gt-b", None, None),
            "neutral",
        )

    def test_unknown_when_track_gt_identity_is_unavailable(self):
        self.assertEqual(
            classify_assignment_outcome(None, "gt-a", "gt-b"),
            "unknown",
        )

    def test_noncanonical_track_gt_tokens_are_unknown(self):
        invalid_tokens = (True, 1, 1.0, b"gt-b", "", "   ", " gt-b", "gt-b ")
        for token in invalid_tokens:
            with self.subTest(token=token):
                self.assertEqual(
                    classify_assignment_outcome(token, "gt-a", "gt-b"),
                    "unknown",
                )

    def test_noncanonical_detection_tokens_cannot_change_outcome(self):
        invalid_tokens = (True, 1, 1.0, b"gt-b", "", "   ", " gt-b", "gt-b ")
        for token in invalid_tokens:
            with self.subTest(token=token):
                self.assertEqual(
                    classify_assignment_outcome(
                        "gt-b",
                        "gt-a",
                        token,
                        production_matched=True,
                        shadow_matched=True,
                    ),
                    "unknown",
                )

    def test_boolean_integer_alias_cannot_create_improved_or_damaged_outcome(self):
        self.assertEqual(classify_assignment_outcome(1, "wrong", True), "unknown")
        self.assertEqual(classify_assignment_outcome(True, 1, "wrong"), "unknown")

    def test_unknown_when_matched_detection_gt_identity_is_unavailable(self):
        self.assertEqual(
            classify_assignment_outcome(
                "gt-b",
                None,
                "gt-b",
                production_matched=True,
                shadow_matched=True,
            ),
            "unknown",
        )
        self.assertEqual(
            classify_assignment_outcome(
                "gt-b",
                "gt-b",
                None,
                production_matched=True,
                shadow_matched=True,
            ),
            "unknown",
        )


class CompareAssignmentResultsTest(unittest.TestCase):
    @staticmethod
    def _compare(**overrides):
        arguments = {
            "production_matches": np.empty((0, 2), dtype=int),
            "production_costs": np.empty((0,), dtype=float),
            "shadow_matches": np.empty((0, 2), dtype=int),
            "shadow_costs": np.empty((0,), dtype=float),
            "base_cost_matrix": np.array([[0.2, 0.4]]),
            "corrected_cost_matrix": np.array([[0.2, 0.4]]),
            "association_scores": np.array([[0.8, 0.3]]),
            "track_ids": [10],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["unmatched"],
        }
        arguments.update(overrides)
        return compare_assignment_results(**arguments)

    def test_compare_assignments_classifies_improved_gt_decision(self):
        base_cost_matrix = np.array([[0.20, 0.21]])
        corrected_cost_matrix = np.array([[0.23, 0.21]])
        association_scores = np.array([[0.1, 0.9]])
        original_inputs = [
            np.array(base_cost_matrix, copy=True),
            np.array(corrected_cost_matrix, copy=True),
            np.array(association_scores, copy=True),
        ]

        records = compare_assignment_results(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.20]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.21]),
            base_cost_matrix=base_cost_matrix,
            corrected_cost_matrix=corrected_cost_matrix,
            association_scores=association_scores,
            track_ids=[10],
            class_ids=[5],
            class_names=["trailer"],
            state_buckets=["unmatched"],
            track_gt_tokens=["gt-b"],
            det_gt_tokens=["gt-a", "gt-b"],
            context={"scene_id": "scene-1", "sample_token": "sample-1"},
            class_thresholds={5: 1.26},
        )

        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(record["scene_id"], "scene-1")
        self.assertEqual(record["sample_token"], "sample-1")
        self.assertEqual(record["class_id"], 5)
        self.assertEqual(record["class_name"], "trailer")
        self.assertEqual(record["track_id"], 10)
        self.assertEqual(record["track_state_bucket"], "unmatched")
        self.assertEqual(record["production_det_index"], 0)
        self.assertEqual(record["shadow_det_index"], 1)
        self.assertAlmostEqual(record["production_selected_cost"], 0.20)
        self.assertAlmostEqual(record["shadow_selected_cost"], 0.21)
        self.assertAlmostEqual(record["production_selected_association_score"], 0.1)
        self.assertAlmostEqual(record["shadow_selected_association_score"], 0.9)
        self.assertEqual(record["geometric_candidate_rank"], 1)
        self.assertEqual(record["shadow_candidate_rank"], 2)
        self.assertAlmostEqual(record["original_cost_gap"], 0.01)
        self.assertTrue(record["assignment_changed"])
        self.assertFalse(record["threshold_crossing"])
        self.assertEqual(record["track_gt_token"], "gt-b")
        self.assertEqual(record["production_det_gt_token"], "gt-a")
        self.assertEqual(record["shadow_det_gt_token"], "gt-b")
        self.assertEqual(record["production_pair_label"], 0)
        self.assertEqual(record["shadow_pair_label"], 1)
        self.assertEqual(record["outcome"], "improved")
        np.testing.assert_array_equal(base_cost_matrix, original_inputs[0])
        np.testing.assert_array_equal(corrected_cost_matrix, original_inputs[1])
        np.testing.assert_array_equal(association_scores, original_inputs[2])

    def test_returns_records_for_unchanged_matched_and_unmatched_tracks(self):
        records = self._compare(
            production_matches=np.array([[0, 0], [1, 1]], dtype=int),
            production_costs=np.array([0.2, 0.3]),
            shadow_matches=np.array([[0, 0], [1, 1]], dtype=int),
            shadow_costs=np.array([0.2, 0.3]),
            base_cost_matrix=np.array(
                [[0.2, 0.4], [0.5, 0.3], [np.inf, np.nan]]
            ),
            corrected_cost_matrix=np.array(
                [[0.2, 0.4], [0.5, 0.3], [np.inf, np.nan]]
            ),
            association_scores=np.array(
                [[0.8, 0.3], [0.2, 0.7], [0.4, 0.6]]
            ),
            track_ids=[10, 11, 12],
            class_ids=[5, 5, 5],
            class_names=["trailer", "trailer", "trailer"],
            state_buckets=["matched", "unmatched", "matched"],
            track_gt_tokens=["gt-a", "gt-z", "gt-c"],
            det_gt_tokens=["gt-a", "gt-b"],
        )

        self.assertEqual(len(records), 3)
        self.assertEqual([item["track_id"] for item in records], [10, 11, 12])
        self.assertEqual(
            [item["assignment_changed"] for item in records],
            [False, False, False],
        )
        self.assertEqual([item["outcome"] for item in records], ["neutral"] * 3)
        self.assertIsNone(records[2]["production_det_index"])
        self.assertIsNone(records[2]["shadow_selected_cost"])
        self.assertIsNone(records[2]["geometric_candidate_rank"])
        self.assertEqual(records[2]["track_state_bucket"], "matched")

    def test_changed_detection_can_damage_gt_consistency(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.2]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.4]),
            track_gt_tokens=["gt-a"],
            det_gt_tokens=["gt-a", "gt-b"],
        )[0]

        self.assertTrue(record["assignment_changed"])
        self.assertEqual(record["production_pair_label"], 1)
        self.assertEqual(record["shadow_pair_label"], 0)
        self.assertEqual(record["outcome"], "damaged")
        self.assertIsNone(record["threshold_crossing"])

    def test_changed_to_unmatched_is_damaged_and_crosses_match_gate(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.2]),
            corrected_cost_matrix=np.array([[0.8, 0.4]]),
            track_gt_tokens=["gt-a"],
            det_gt_tokens=["gt-a", "gt-b"],
            class_thresholds={"5": 0.5},
        )[0]

        self.assertTrue(record["assignment_changed"])
        self.assertTrue(record["threshold_crossing"])
        self.assertEqual(record["outcome"], "damaged")
        self.assertIsNone(record["shadow_det_index"])
        self.assertIsNone(record["shadow_pair_label"])
        self.assertIsNone(record["original_cost_gap"])

    def test_changed_to_match_without_threshold_reports_none(self):
        record = self._compare(
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.4]),
            track_gt_tokens=["gt-b"],
            det_gt_tokens=["gt-a", "gt-b"],
        )[0]

        self.assertTrue(record["assignment_changed"])
        self.assertIsNone(record["threshold_crossing"])
        self.assertEqual(record["outcome"], "improved")
        self.assertIsNone(record["production_det_index"])
        self.assertEqual(record["shadow_pair_label"], 1)

    def test_unknown_gt_is_never_classified_as_improved_or_damaged(self):
        no_track_identity = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.2]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.4]),
            det_gt_tokens=["gt-a", "gt-b"],
        )[0]
        unknown_selected_detection = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.2]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.4]),
            track_gt_tokens=["gt-b"],
            det_gt_tokens=[None, "gt-b"],
        )[0]

        self.assertEqual(no_track_identity["outcome"], "unknown")
        self.assertIsNone(no_track_identity["production_pair_label"])
        self.assertEqual(unknown_selected_detection["outcome"], "unknown")
        self.assertIsNone(unknown_selected_detection["production_pair_label"])
        self.assertEqual(unknown_selected_detection["shadow_pair_label"], 1)

    def test_ranks_both_choices_from_finite_candidates_in_original_row(self):
        record = self._compare(
            production_matches=np.array([[0, 3]], dtype=int),
            production_costs=np.array([0.3]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.1]),
            base_cost_matrix=np.array([[np.inf, 0.5, np.nan, 0.3]]),
            corrected_cost_matrix=np.array([[np.inf, 0.1, np.nan, 0.4]]),
            association_scores=np.array([[0.0, 0.9, 0.0, 0.2]]),
        )[0]

        self.assertEqual(record["geometric_candidate_rank"], 1)
        self.assertEqual(record["shadow_candidate_rank"], 2)
        self.assertAlmostEqual(record["original_cost_gap"], 0.2)

    def test_candidate_rank_uses_competition_ranking_for_tied_base_costs(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.3]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.2]),
            base_cost_matrix=np.array([[0.3, 0.3, 0.5]]),
            corrected_cost_matrix=np.array([[0.4, 0.2, 0.5]]),
            association_scores=np.array([[0.4, 0.9, 0.2]]),
        )[0]

        self.assertEqual(record["geometric_candidate_rank"], 1)
        self.assertEqual(record["shadow_candidate_rank"], 1)
        self.assertEqual(record["original_cost_gap"], 0.0)

    def test_threshold_crossing_uses_selected_cost_gate_for_string_key(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.5]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.8]),
            base_cost_matrix=np.array([[0.5, 1.2]]),
            corrected_cost_matrix=np.array([[1.2, 0.8]]),
            class_thresholds={"5": 1.0},
        )[0]

        self.assertTrue(record["assignment_changed"])
        self.assertTrue(record["threshold_crossing"])

    def test_integer_threshold_key_supports_string_normalized_class_id(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.5]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([0.8]),
            base_cost_matrix=np.array([[0.5, 1.2]]),
            corrected_cost_matrix=np.array([[1.2, 0.8]]),
            class_ids=["5"],
            class_thresholds={5: 1.0},
        )[0]

        self.assertTrue(record["threshold_crossing"])

    def test_global_assignment_competition_is_not_a_threshold_crossing(self):
        records = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([0.10]),
            shadow_matches=np.array([[1, 0]], dtype=int),
            shadow_costs=np.array([0.10]),
            base_cost_matrix=np.array([[0.10], [0.20]]),
            corrected_cost_matrix=np.array([[0.20], [0.10]]),
            association_scores=np.array([[0.8], [0.9]]),
            track_ids=[10, 11],
            class_ids=[5, 5],
            class_names=["trailer", "trailer"],
            state_buckets=["matched", "unmatched"],
            class_thresholds={5: 1.0},
        )

        self.assertEqual([item["assignment_changed"] for item in records], [True, True])
        self.assertEqual([item["threshold_crossing"] for item in records], [False, False])

    def test_rejects_matched_selected_cost_above_class_threshold(self):
        with self.assertRaisesRegex(ValueError, "production.*threshold"):
            self._compare(
                production_matches=np.array([[0, 0]], dtype=int),
                production_costs=np.array([1.1]),
                base_cost_matrix=np.array([[1.1, 0.4]]),
                class_thresholds={5: 1.0},
            )
        with self.assertRaisesRegex(ValueError, "shadow.*threshold"):
            self._compare(
                shadow_matches=np.array([[0, 1]], dtype=int),
                shadow_costs=np.array([1.1]),
                corrected_cost_matrix=np.array([[0.2, 1.1]]),
                class_thresholds={5: 1.0},
            )

    def test_allows_matched_cost_within_floating_threshold_tolerance(self):
        selected_cost = 1.0 + 5e-9
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([selected_cost]),
            base_cost_matrix=np.array([[selected_cost, 0.4]]),
            corrected_cost_matrix=np.array([[selected_cost, 0.4]]),
            class_thresholds={5: 1.0},
        )[0]

        self.assertAlmostEqual(record["production_selected_cost"], selected_cost)
        self.assertFalse(record["threshold_crossing"])

    def test_rejects_solver_costs_that_disagree_with_source_matrices(self):
        with self.assertRaisesRegex(ValueError, "production_costs"):
            self._compare(
                production_matches=np.array([[0, 0]], dtype=int),
                production_costs=np.array([0.9]),
            )
        with self.assertRaisesRegex(ValueError, "shadow_costs"):
            self._compare(
                shadow_matches=np.array([[0, 1]], dtype=int),
                shadow_costs=np.array([0.9]),
            )

    def test_selected_costs_are_derived_from_base_and_corrected_matrices(self):
        record = self._compare(
            production_matches=np.array([[0, 0]], dtype=int),
            production_costs=np.array([np.float32(0.2)]),
            shadow_matches=np.array([[0, 1]], dtype=int),
            shadow_costs=np.array([np.float32(0.35)]),
            corrected_cost_matrix=np.array([[0.25, 0.35]], dtype=np.float64),
        )[0]

        self.assertEqual(record["production_selected_cost"], 0.2)
        self.assertEqual(record["shadow_selected_cost"], 0.35)

    def test_rejects_shape_metadata_and_index_programmer_errors(self):
        with self.assertRaisesRegex(ValueError, "corrected_cost_matrix"):
            self._compare(corrected_cost_matrix=np.zeros((2, 2)))
        with self.assertRaisesRegex(ValueError, "track_ids"):
            self._compare(track_ids=[])
        with self.assertRaisesRegex(ValueError, "det_gt_tokens"):
            self._compare(det_gt_tokens=["only-one"])
        with self.assertRaisesRegex(IndexError, "track index"):
            self._compare(
                production_matches=np.array([[1, 0]], dtype=int),
                production_costs=np.array([0.2]),
            )
        with self.assertRaisesRegex(IndexError, "detection index"):
            self._compare(
                shadow_matches=np.array([[0, 2]], dtype=int),
                shadow_costs=np.array([0.2]),
            )

    def test_rejects_non_one_dimensional_gt_token_arrays(self):
        with self.assertRaisesRegex(ValueError, "track_gt_tokens"):
            self._compare(track_gt_tokens=np.array([["gt-a"]], dtype=object))
        with self.assertRaisesRegex(ValueError, "det_gt_tokens"):
            self._compare(
                det_gt_tokens=np.array([["gt-a", "gt-b"]], dtype=object)
            )

    def test_rejects_nested_or_non_serializable_gt_tokens(self):
        nested_track_tokens = np.empty((1,), dtype=object)
        nested_track_tokens[0] = np.array(["gt-a"])
        invalid_values = {
            "track_gt_tokens": [
                nested_track_tokens,
                [{"token": "gt-a"}],
                [object()],
            ],
            "det_gt_tokens": [
                [["gt-a"], None],
                [{"token": "gt-a"}, None],
                [object(), None],
            ],
        }
        for name, cases in invalid_values.items():
            for value in cases:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, name
                ):
                    self._compare(**{name: value})

    def test_rejects_noncanonical_gt_tokens(self):
        invalid_tokens = (True, 1, 1.0, b"gt-a", "", "   ", " gt-a", "gt-a ")
        for token in invalid_tokens:
            with self.subTest(field="track_gt_tokens", token=token):
                with self.assertRaisesRegex(ValueError, "track_gt_tokens"):
                    self._compare(track_gt_tokens=[token])
            with self.subTest(field="det_gt_tokens", token=token):
                with self.assertRaisesRegex(ValueError, "det_gt_tokens"):
                    self._compare(det_gt_tokens=[token, None])

    def test_normalizes_valid_track_metadata(self):
        record = self._compare(
            track_ids=np.array([np.int64(10)]),
            class_ids=["5"],
            class_names=[" trailer "],
            state_buckets=[" UNMATCHED "],
        )[0]

        self.assertEqual(record["track_id"], 10)
        self.assertEqual(record["class_id"], 5)
        self.assertEqual(record["class_name"], "trailer")
        self.assertEqual(record["track_state_bucket"], "unmatched")

    def test_rejects_non_one_dimensional_track_metadata(self):
        invalid_values = {
            "track_ids": [[10]],
            "class_ids": [[5]],
            "class_names": [["trailer"]],
            "state_buckets": [["matched"]],
        }
        for name, value in invalid_values.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, name):
                self._compare(**{name: value})

    def test_rejects_empty_duplicate_or_non_json_track_ids(self):
        invalid_track_ids = ([""], [None], [np.nan], [-1], [{"id": 10}])
        for track_ids in invalid_track_ids:
            with self.subTest(track_ids=track_ids), self.assertRaisesRegex(
                ValueError, "track_ids"
            ):
                self._compare(track_ids=track_ids)
        with self.assertRaisesRegex(ValueError, "track_ids"):
            self._compare(
                base_cost_matrix=np.zeros((2, 2)),
                corrected_cost_matrix=np.zeros((2, 2)),
                association_scores=np.zeros((2, 2)),
                track_ids=[10, 10],
                class_ids=[5, 5],
                class_names=["trailer", "trailer"],
                state_buckets=["matched", "unmatched"],
            )

    def test_rejects_invalid_class_and_state_metadata_values(self):
        invalid_values = {
            "class_ids": [["not-an-int"], [1.5]],
            "class_names": [[""], ["   "]],
            "state_buckets": [["coasting"], [""]],
        }
        for name, cases in invalid_values.items():
            for value in cases:
                with self.subTest(name=name, value=value), self.assertRaisesRegex(
                    ValueError, name
                ):
                    self._compare(**{name: value})

    def test_rejects_invalid_threshold_values(self):
        with self.assertRaisesRegex(TypeError, "threshold"):
            self._compare(class_thresholds={5: "not-a-number"})
        with self.assertRaisesRegex(ValueError, "finite"):
            self._compare(class_thresholds={5: np.inf})

    def test_rejects_complex_cost_and_score_matrices(self):
        matrix_names = (
            "base_cost_matrix",
            "corrected_cost_matrix",
            "association_scores",
        )
        for name in matrix_names:
            with self.subTest(name=name), self.assertRaisesRegex(
                TypeError, f"{name}.*real-valued"
            ):
                self._compare(**{name: np.array([[0.2 + 0.0j, 0.4 + 0.0j]])})


if __name__ == "__main__":
    unittest.main()
