import copy
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tracker.association_head_audit import AssociationHeadAuditAccumulator


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

    def test_add_pair_rejects_aggregate_overflow_atomically(self):
        max_float = np.finfo(np.float64).max
        for field_name in ("score", "delta", "cost_before", "cost_after"):
            with self.subTest(field=field_name):
                audit = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
                audit.add_pair(
                    **self._pair(**{field_name: max_float}),
                    sample={"call": 1},
                )
                before = copy.deepcopy(audit.to_summary())

                with self.assertRaisesRegex(ValueError, field_name):
                    audit.add_pair(
                        **self._pair(**{field_name: max_float}),
                        sample={"call": 2},
                    )

                self.assertEqual(audit.to_summary(), before)

    def test_merge_rejects_pair_aggregate_overflow_atomically(self):
        max_float = np.finfo(np.float64).max
        for field_name in ("score", "delta", "cost_before", "cost_after"):
            with self.subTest(field=field_name):
                target = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
                worker = AssociationHeadAuditAccumulator(max_samples_per_bucket=2)
                target.add_pair(
                    **self._pair(**{field_name: max_float}),
                    sample={"source": "target"},
                )
                worker.add_pair(
                    **self._pair(**{field_name: max_float}),
                    sample={"source": "worker"},
                )
                before = copy.deepcopy(target.to_summary())

                with self.assertRaisesRegex(ValueError, field_name):
                    target.merge_state(worker.export_state())

                self.assertEqual(target.to_summary(), before)

        worker = AssociationHeadAuditAccumulator()
        worker.add_pair(**self._pair(score=max_float))
        overflow_product_state = worker.export_state()
        overflow_product_state["pair_buckets"][0]["pair_count"] = 2
        target = AssociationHeadAuditAccumulator()
        before = copy.deepcopy(target.to_summary())
        with self.assertRaisesRegex(ValueError, "score"):
            target.merge_state(overflow_product_state)
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

    def test_merges_schema_v1_pairs_without_fabricating_decision_evidence(self):
        legacy = {
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
        merged = AssociationHeadAuditAccumulator()

        merged.merge_state(legacy)

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
