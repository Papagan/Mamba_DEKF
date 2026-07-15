import copy
import json
import os
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tracker.association_head_audit import AssociationHeadAuditAccumulator
import tools.compare_pairwise_assoc_train_infer as comparator


class ComparePairwiseAssocTrainInferCliTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(__file__).resolve().parents[1]
        self.samples = [
            {
                "class_id": 2,
                "category": "bicycle",
                "label": 1,
                "negative_type": "positive",
                "center_distance": 0.2,
            },
            {
                "class_id": 2,
                "category": "bicycle",
                "label": 0,
                "negative_type": "hard",
                "center_distance": 0.5,
            },
            {
                "class_id": 2,
                "category": "bicycle",
                "label": 0,
                "negative_type": "easy",
                "center_distance": 3.0,
            },
        ]

    def _run_compare(
        self,
        audit,
        *extra_args,
        expect_success=True,
        samples=None,
        audit_text=None,
        existing_output_mode=None,
    ):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.pkl"
            audit_path = tmp_path / "audit.json"
            output_path = tmp_path / "summary.json"
            with open(train_path, "wb") as f:
                pickle.dump(self.samples if samples is None else samples, f)
            audit_path.write_text(
                json.dumps(audit) if audit_text is None else audit_text,
                encoding="utf-8",
            )
            if existing_output_mode is not None:
                output_path.write_text("stale report", encoding="utf-8")
                output_path.chmod(existing_output_mode)

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/compare_pairwise_assoc_train_infer.py",
                    "--train-cache",
                    str(train_path),
                    "--infer-audit",
                    str(audit_path),
                    "--output",
                    str(output_path),
                    *extra_args,
                ],
                cwd=self.root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            if not expect_success:
                return result
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            if existing_output_mode is not None:
                return data, output_path.stat().st_mode & 0o777
            return data

    @staticmethod
    def _pair_bucket():
        return {
            "class_id": 2,
            "class_name": "bicycle",
            "state_bucket": "matched",
            "pair_count": 10,
            "active_pair_count": 4,
            "penalized_pair_count": 3,
            "active_ratio": 0.4,
            "penalized_ratio": 0.75,
            "avg_score": 0.1,
            "avg_delta": 0.01,
            "avg_cost_before": 0.42,
            "avg_cost_after": 0.43,
            "score_quantiles": {
                "min": 0.0,
                "p10": 0.01,
                "p50": 0.08,
                "p90": 0.3,
                "max": 0.5,
            },
            "delta_quantiles": {
                "min": 0.0,
                "p10": 0.0,
                "p50": 0.01,
                "p90": 0.02,
                "max": 0.03,
            },
            "cost_before_quantiles": {
                "min": 0.1,
                "p10": 0.2,
                "p50": 0.4,
                "p90": 0.7,
                "max": 0.8,
            },
            "cost_after_quantiles": {
                "min": 0.11,
                "p10": 0.21,
                "p50": 0.41,
                "p90": 0.71,
                "max": 0.81,
            },
        }

    @staticmethod
    def _decision_bucket(
        known_changed=120,
        improved=100,
        damaged=20,
        state="matched",
        total=200,
    ):
        neutral = known_changed - improved - damaged
        if neutral < 0:
            raise ValueError("known_changed must cover improved and damaged")
        if total < known_changed:
            raise ValueError("total must cover known decisions")
        return {
            "class_id": 2,
            "class_name": "bicycle",
            "track_state_bucket": state,
            "total_decisions": total,
            "changed_decisions": known_changed,
            "changed_detection_count": known_changed,
            "changed_to_match_count": 0,
            "changed_to_unmatched_count": 0,
            "threshold_crossing_count": 0,
            "known_gt_decisions": known_changed,
            "known_changed_count": known_changed,
            "improved_count": improved,
            "damaged_count": damaged,
            "neutral_count": neutral,
            "unknown_count": total - known_changed,
            "net_improvement": improved - damaged,
            "harm_precision": damaged / max(known_changed, 1),
            "sample_count": 0,
            "samples": [],
            "raw_decisions": [],
            "association_scores": [],
            "cost_gaps": [],
        }

    def _complete_audit(self, decision_buckets=None, **overrides):
        audit = {
            "schema_version": 2,
            "runtime_manifest": self._valid_runtime_manifest(),
            "manifest_initialized": True,
            "decision_evidence_available": True,
            "decision_evidence_complete": True,
            "decision_evidence_incomplete_reasons": [],
            "manifest_compatible": True,
            "audit_error_count": 0,
            "audit_errors": {"count": 0, "sample_count": 0, "samples": []},
            "decision_buckets": decision_buckets
            if decision_buckets is not None
            else [self._decision_bucket()],
        }
        audit.update(overrides)
        return audit

    @staticmethod
    def _valid_runtime_manifest(**overrides):
        head_keys = ["heads.2.0.weight", "heads.2.0.bias"]
        runtime_contract = {
            "input_dim": 135,
            "hidden_dims": [256, 128],
            "dropout": 0.1,
            "num_classes": 7,
        }
        manifest = {
            "audit_schema_version": 3,
            "config_sha256": "a" * 64,
            "class_id_to_name": {"2": "bicycle"},
            "active_class_states": {"2": ["matched"]},
            "association_head_enabled": True,
            "association_head_audit_enabled": True,
            "association_head_audit_shadow_mode": True,
            "association_checkpoint_configured": True,
            "association_checkpoint_path": "/tmp/association.pt",
            "association_checkpoint_sha256": "b" * 64,
            "association_checkpoint_loadable": True,
            "association_checkpoint_head_key_count": len(head_keys),
            "association_checkpoint_key_validation_status": "valid",
            "association_checkpoint_validation_error": None,
            "association_checkpoint_runtime_contract_expected": runtime_contract,
            "association_checkpoint_runtime_contract_actual": copy.deepcopy(
                runtime_contract
            ),
            "association_checkpoint_runtime_contract_validation_status": "valid",
            "association_checkpoint_loaded": True,
        }
        manifest.update(overrides)
        return manifest

    def test_schema_v1_track_state_scores_are_observational_and_fail_closed(self):
        data = self._run_compare(
            {"schema_version": 1, "buckets": [self._pair_bucket()]}
        )

        bucket = data["classes"]["2:bicycle"]
        self.assertEqual(data["infer_audit_schema_version"], 1)
        self.assertFalse(data["decision_evidence"]["available"])
        self.assertEqual(bucket["train"]["sample_count"], 3)
        self.assertAlmostEqual(bucket["train"]["positive_ratio"], 1.0 / 3.0)
        self.assertEqual(bucket["train"]["hard_negative_count"], 1)
        self.assertAlmostEqual(
            bucket["train"]["center_distance_quantiles"]["max"], 3.0
        )
        self.assertAlmostEqual(
            bucket["train"]["hard_center_distance_quantiles"]["p50"], 0.5
        )
        self.assertEqual(
            bucket["track_state_scores"]["matched"]["penalized_ratio"], 0.75
        )
        self.assertEqual(
            bucket["track_state_scores"]["matched"]["avg_cost_before"], 0.42
        )
        self.assertEqual(
            bucket["track_state_scores"]["matched"]["avg_cost_after"], 0.43
        )
        self.assertEqual(
            bucket["track_state_scores"]["matched"]["cost_before_quantiles"][
                "p50"
            ],
            0.4,
        )
        self.assertEqual(
            bucket["track_state_scores"]["matched"]["cost_after_quantiles"][
                "p90"
            ],
            0.71,
        )
        self.assertNotIn("infer", bucket)
        self.assertIn("insufficient_decision_evidence", bucket["diagnosis"])
        self.assertNotIn("likely_distribution_mismatch", bucket["diagnosis"])
        self.assertFalse(any("candidate" in item for item in bucket["diagnosis"]))

    def test_schema_v2_safe_decision_counts_are_shadow_ab_candidate(self):
        data = self._run_compare(
            self._complete_audit(pair_buckets=[self._pair_bucket()])
        )

        bucket = data["classes"]["2:bicycle"]
        self.assertEqual(data["infer_audit_schema_version"], 2)
        self.assertTrue(data["decision_evidence"]["available"])
        self.assertEqual(bucket["decision_evidence"]["known_changed_count"], 120)
        self.assertEqual(
            bucket["decision_evidence_by_state"]["matched"][
                "known_changed_count"
            ],
            120,
        )
        self.assertIn(
            "candidate_for_shadow_ab_activation",
            bucket["diagnosis_by_state"]["matched"],
        )
        self.assertIn("candidate_for_shadow_ab_activation", bucket["diagnosis"])

    def test_complete_production_v2_bucket_is_accepted(self):
        runtime_manifest = self._valid_runtime_manifest(
            config_sha256="producer",
            class_id_to_name={"2": "bicycle", "5": "trailer"},
            active_class_states={"5": ["unmatched"]},
        )
        audit = AssociationHeadAuditAccumulator(
            runtime_manifest=runtime_manifest
        )
        audit.add_decision(
            {
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
        )

        data = self._run_compare(audit.to_summary())
        state = data["classes"]["5:trailer"]["decision_evidence_by_state"][
            "unmatched"
        ]
        self.assertTrue(data["decision_evidence"]["usable"])
        self.assertEqual(state["total_decisions"], 1)
        self.assertEqual(state["known_changed_count"], 1)
        self.assertEqual(state["improved_count"], 1)

    def test_default_minimum_requires_100_known_changed_decisions(self):
        audit = self._complete_audit(
            decision_buckets=[
                self._decision_bucket(known_changed=99, improved=90, damaged=9)
            ]
        )

        data = self._run_compare(audit)
        bucket = data["classes"]["2:bicycle"]
        self.assertEqual(data["decision_gates"]["min_known_changed"], 100)
        self.assertIn("shadow_decisions_not_safe", bucket["diagnosis"])
        self.assertFalse(any("candidate" in item for item in bucket["diagnosis"]))

        low_gate = self._run_compare(
            audit,
            "--min-known-changed",
            "50",
            expect_success=False,
        )
        self.assertNotEqual(low_gate.returncode, 0)
        self.assertIn("at least 100", low_gate.stderr)

        raised_gate = self._run_compare(audit, "--min-known-changed", "121")
        self.assertIn(
            "shadow_decisions_not_safe",
            raised_gate["classes"]["2:bicycle"]["diagnosis"],
        )

    def test_zero_known_changed_decisions_are_insufficient(self):
        audit = self._complete_audit(
            decision_buckets=[
                self._decision_bucket(known_changed=0, improved=0, damaged=0)
            ]
        )

        diagnosis = self._run_compare(audit)["classes"]["2:bicycle"]["diagnosis"]
        self.assertIn("insufficient_decision_evidence", diagnosis)
        self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_harm_above_gate_is_not_safe(self):
        audit = self._complete_audit(
            decision_buckets=[
                self._decision_bucket(known_changed=120, improved=90, damaged=30)
            ]
        )

        diagnosis = self._run_compare(audit)["classes"]["2:bicycle"]["diagnosis"]
        self.assertIn("shadow_decisions_not_safe", diagnosis)
        self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_incomplete_or_invalid_audit_metadata_fails_closed(self):
        base = self._complete_audit()
        cases = {
            "evidence_unavailable": {"decision_evidence_available": False},
            "evidence_incomplete": {"decision_evidence_complete": False},
            "manifest_incompatible": {"manifest_compatible": False},
            "audit_errors": {
                "audit_error_count": 1,
                "audit_errors": {"count": 1, "sample_count": 0, "samples": []},
            },
            "missing_buckets": {"decision_buckets": []},
        }

        for name, override in cases.items():
            with self.subTest(name=name):
                audit = {**base, **override}
                data = self._run_compare(audit)
                diagnosis = data["classes"]["2:bicycle"]["diagnosis"]
                self.assertIn("insufficient_decision_evidence", diagnosis)
                self.assertNotIn("likely_distribution_mismatch", diagnosis)
                self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_malformed_audit_errors_fail_closed_with_reason(self):
        base = self._complete_audit()
        malformed_values = {
            "not_mapping": [],
            "boolean_count": {"count": True, "sample_count": 0, "samples": []},
            "negative_count": {"count": -1, "sample_count": 0, "samples": []},
            "float_count": {"count": 1.5, "sample_count": 0, "samples": []},
            "string_count": {"count": "1", "sample_count": 0, "samples": []},
            "missing_sample_count": {"count": 0, "samples": []},
            "sample_count_mismatch": {
                "count": 1,
                "sample_count": 0,
                "samples": [{"message": "error"}],
            },
            "zero_count_with_sample": {
                "count": 0,
                "sample_count": 1,
                "samples": [{"message": "error"}],
            },
        }

        for name, audit_errors in malformed_values.items():
            with self.subTest(name=name):
                data = self._run_compare({**base, "audit_errors": audit_errors})
                evidence = data["decision_evidence"]
                diagnosis = data["classes"]["2:bicycle"]["diagnosis"]
                self.assertFalse(evidence["usable"])
                self.assertIn("invalid_audit_errors", evidence["incomplete_reasons"])
                self.assertIn("insufficient_decision_evidence", diagnosis)
                self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_matched_and_unmatched_decisions_are_gated_independently(self):
        data = self._run_compare(
            self._complete_audit(
                runtime_manifest=self._valid_runtime_manifest(
                    active_class_states={"2": ["matched", "unmatched"]}
                ),
                decision_buckets=[
                    self._decision_bucket(
                        known_changed=100, improved=100, damaged=0, state="matched"
                    ),
                ]
            )
        )

        bicycle = data["classes"]["2:bicycle"]
        self.assertIn(
            "candidate_for_shadow_ab_activation",
            bicycle["diagnosis_by_state"]["matched"],
        )
        self.assertIn(
            "insufficient_decision_evidence",
            bicycle["diagnosis_by_state"]["unmatched"],
        )
        self.assertEqual(
            bicycle["decision_evidence_by_state"]["matched"]["improved_count"],
            100,
        )
        self.assertIn("insufficient_decision_evidence", bicycle["diagnosis"])
        self.assertFalse(any("candidate" in item for item in bicycle["diagnosis"]))

    def test_class_candidate_requires_every_manifest_active_state_to_be_candidate(self):
        data = self._run_compare(
            self._complete_audit(
                runtime_manifest=self._valid_runtime_manifest(
                    active_class_states={2: ["matched", "unmatched"]}
                ),
                decision_buckets=[
                    self._decision_bucket(state="matched"),
                    self._decision_bucket(state="unmatched"),
                ]
            )
        )

        bicycle = data["classes"]["2:bicycle"]
        self.assertIn("candidate_for_shadow_ab_activation", bicycle["diagnosis"])

        unconfigured_observation = self._run_compare(
            self._complete_audit(
                decision_buckets=[
                    self._decision_bucket(state="matched"),
                    self._decision_bucket(
                        known_changed=20,
                        improved=0,
                        damaged=20,
                        state="unmatched",
                    ),
                ]
            )
        )["classes"]["2:bicycle"]
        self.assertIn("unmatched", unconfigured_observation["diagnosis_by_state"])
        self.assertIn(
            "candidate_for_shadow_ab_activation",
            unconfigured_observation["diagnosis"],
        )

    def test_multiple_buckets_for_same_class_and_state_are_aggregated(self):
        data = self._run_compare(
            self._complete_audit(
                decision_buckets=[
                    self._decision_bucket(
                        known_changed=60, improved=50, damaged=10, total=100
                    ),
                    self._decision_bucket(
                        known_changed=60, improved=50, damaged=10, total=100
                    ),
                ]
            )
        )

        state = data["classes"]["2:bicycle"]["decision_evidence_by_state"][
            "matched"
        ]
        self.assertEqual(state["decision_bucket_count"], 2)
        self.assertEqual(state["total_decisions"], 200)
        self.assertEqual(state["known_changed_count"], 120)
        self.assertEqual(state["improved_count"], 100)
        self.assertEqual(state["damaged_count"], 20)
        self.assertIn(
            "candidate_for_shadow_ab_activation",
            data["classes"]["2:bicycle"]["diagnosis_by_state"]["matched"],
        )

    def test_forged_decision_bucket_counters_fail_closed(self):
        base = self._complete_audit()

        def changed(callback):
            audit = copy.deepcopy(base)
            callback(audit["decision_buckets"][0])
            return audit

        malformed = {
            "class_id_string": changed(lambda bucket: bucket.__setitem__("class_id", "2")),
            "class_id_bool": changed(lambda bucket: bucket.__setitem__("class_id", True)),
            "class_id_negative": changed(lambda bucket: bucket.__setitem__("class_id", -1)),
            "empty_class_name": changed(lambda bucket: bucket.__setitem__("class_name", "")),
            "invalid_state": changed(
                lambda bucket: bucket.__setitem__("track_state_bucket", "lost")
            ),
            "missing_counter": changed(lambda bucket: bucket.pop("changed_decisions")),
            "string_counter": changed(
                lambda bucket: bucket.__setitem__("known_changed_count", "120")
            ),
            "boolean_counter": changed(
                lambda bucket: bucket.__setitem__("improved_count", True)
            ),
            "negative_counter": changed(
                lambda bucket: bucket.__setitem__("damaged_count", -1)
            ),
            "outcome_total": changed(
                lambda bucket: bucket.__setitem__("unknown_count", 81)
            ),
            "known_gt_total": changed(
                lambda bucket: bucket.__setitem__("known_gt_decisions", 119)
            ),
            "changed_exceeds_total": changed(
                lambda bucket: bucket.__setitem__("changed_decisions", 201)
            ),
            "change_categories": changed(
                lambda bucket: bucket.__setitem__("changed_to_match_count", 1)
            ),
            "threshold_exceeds_changed": changed(
                lambda bucket: bucket.__setitem__("threshold_crossing_count", 121)
            ),
            "known_changed_too_small": changed(
                lambda bucket: bucket.__setitem__("known_changed_count", 119)
            ),
            "known_changed_too_large": changed(
                lambda bucket: (
                    bucket.__setitem__("changed_decisions", 121),
                    bucket.__setitem__("changed_detection_count", 121),
                    bucket.__setitem__("known_changed_count", 121),
                )
            ),
            "wrong_net_improvement": changed(
                lambda bucket: bucket.__setitem__("net_improvement", 81)
            ),
            "nonfinite_harm_precision": changed(
                lambda bucket: bucket.__setitem__("harm_precision", float("inf"))
            ),
            "wrong_harm_precision": changed(
                lambda bucket: bucket.__setitem__("harm_precision", 0.10)
            ),
        }

        for name, audit in malformed.items():
            with self.subTest(name=name):
                data = self._run_compare(audit)
                evidence = data["decision_evidence"]
                diagnosis = data["classes"]["2:bicycle"]["diagnosis"]
                self.assertFalse(evidence["usable"])
                self.assertIn("invalid_decision_buckets", evidence["incomplete_reasons"])
                self.assertIn("insufficient_decision_evidence", diagnosis)
                self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_top_level_v2_metadata_is_strict_and_fail_closed(self):
        base = self._complete_audit()
        malformed = {}

        string_false = copy.deepcopy(base)
        string_false["decision_evidence_available"] = "false"
        malformed["string_false"] = (string_false, "invalid_decision_evidence_available")

        missing_manifest = copy.deepcopy(base)
        missing_manifest.pop("runtime_manifest")
        malformed["missing_manifest"] = (missing_manifest, "missing_runtime_manifest")

        zero_count_with_sample = copy.deepcopy(base)
        zero_count_with_sample["audit_errors"] = {
            "count": 0,
            "sample_count": 1,
            "samples": [{"message": "forged"}],
        }
        malformed["zero_count_with_sample"] = (
            zero_count_with_sample,
            "invalid_audit_errors",
        )

        for name, (audit, reason) in malformed.items():
            with self.subTest(name=name):
                data = self._run_compare(audit)
                self.assertFalse(data["decision_evidence"]["usable"])
                self.assertIn(reason, data["decision_evidence"]["incomplete_reasons"])
                diagnosis = data["classes"]["2:bicycle"]["diagnosis"]
                self.assertIn("insufficient_decision_evidence", diagnosis)
                self.assertFalse(any("candidate" in item for item in diagnosis))

        activation_cases = {
            "missing_active_class_states": (
                self._valid_runtime_manifest(),
                "missing_active_class_states",
            ),
            "invalid_active_class_states": (
                self._valid_runtime_manifest(
                    active_class_states={"2": ["matched", "lost"]}
                ),
                "invalid_active_class_states_state",
            ),
        }
        activation_cases["missing_active_class_states"][0].pop(
            "active_class_states"
        )
        for name, (runtime_manifest, reason) in activation_cases.items():
            with self.subTest(name=name):
                data = self._run_compare(
                    self._complete_audit(runtime_manifest=runtime_manifest)
                )
                bicycle = data["classes"]["2:bicycle"]
                activation = data["class_state_activation"]
                self.assertFalse(activation["class_level_usable"])
                self.assertIn(reason, activation["incomplete_reasons"])
                self.assertIn(
                    "candidate_for_shadow_ab_activation",
                    bicycle["diagnosis_by_state"]["matched"],
                )
                self.assertIn("insufficient_decision_evidence", bicycle["diagnosis"])
                self.assertFalse(any("candidate" in item for item in bicycle["diagnosis"]))

    def test_canonical_manifest_mapping_validates_every_bucket_source(self):
        cases = {}

        train_samples = copy.deepcopy(self.samples)
        train_samples[0]["category"] = "motorcycle"
        cases["train"] = (
            self._complete_audit(pair_buckets=[self._pair_bucket()]),
            train_samples,
            "train_bucket_class_identity_mismatch",
        )

        pair_audit = self._complete_audit(pair_buckets=[self._pair_bucket()])
        pair_audit["pair_buckets"][0]["class_name"] = "motorcycle"
        cases["pair"] = (
            pair_audit,
            self.samples,
            "pair_bucket_class_identity_mismatch",
        )

        decision_audit = self._complete_audit()
        decision_audit["decision_buckets"][0]["class_name"] = "motorcycle"
        cases["decision"] = (
            decision_audit,
            self.samples,
            "decision_bucket_class_identity_mismatch",
        )

        for source, (audit, samples, reason) in cases.items():
            with self.subTest(source=source):
                data = self._run_compare(audit, samples=samples)
                identity = data["class_identity_validation"]
                self.assertFalse(identity["valid"])
                self.assertIn(reason, identity["incomplete_reasons"])
                bicycle = data["classes"]["2:bicycle"]
                self.assertIn("insufficient_decision_evidence", bicycle["diagnosis"])
                self.assertFalse(any("candidate" in item for item in bicycle["diagnosis"]))

    def test_invalid_or_oversized_manifest_class_keys_fail_closed_without_crash(self):
        invalid_keys = ["02", "+2", "9" * 5000]

        for invalid_key in invalid_keys:
            with self.subTest(key_length=len(invalid_key)):
                audit = self._complete_audit()
                audit["runtime_manifest"]["class_id_to_name"] = {
                    invalid_key: "bicycle"
                }
                audit["runtime_manifest"]["active_class_states"] = {
                    invalid_key: ["matched"]
                }

                data = self._run_compare(audit)
                identity = data["class_identity_validation"]
                self.assertFalse(identity["valid"])
                self.assertIn(
                    "invalid_class_id_to_name_class_id",
                    identity["incomplete_reasons"],
                )
                self.assertFalse(data["decision_evidence"]["usable"])
                self.assertFalse(
                    any(
                        "candidate" in diagnosis
                        for bucket in data["classes"].values()
                        for diagnosis in bucket["diagnosis"]
                    )
                )

    def test_unhashable_pair_state_fails_closed_without_crash(self):
        audit = self._complete_audit(pair_buckets=[self._pair_bucket()])
        audit["pair_buckets"][0]["state_bucket"] = ["matched"]

        data = self._run_compare(audit)

        identity = data["class_identity_validation"]
        self.assertFalse(identity["valid"])
        self.assertIn("invalid_pair_bucket_state", identity["incomplete_reasons"])
        self.assertFalse(data["decision_evidence"]["usable"])
        self.assertIn(
            "insufficient_decision_evidence",
            data["classes"]["2:bicycle"]["diagnosis"],
        )

    def test_oversized_bucket_class_ids_fail_closed_for_every_source(self):
        oversized_class_id = 10**100
        cases = {}

        train_samples = copy.deepcopy(self.samples)
        train_samples[0]["class_id"] = oversized_class_id
        cases["train"] = (
            self._complete_audit(),
            train_samples,
            "train_bucket_invalid_class_identity",
        )

        pair_audit = self._complete_audit(pair_buckets=[self._pair_bucket()])
        pair_audit["pair_buckets"][0]["class_id"] = oversized_class_id
        cases["pair"] = (
            pair_audit,
            self.samples,
            "pair_bucket_invalid_class_identity",
        )

        decision_audit = self._complete_audit()
        decision_audit["decision_buckets"][0]["class_id"] = oversized_class_id
        cases["decision"] = (
            decision_audit,
            self.samples,
            "decision_bucket_invalid_class_identity",
        )

        for source, (audit, samples, reason) in cases.items():
            with self.subTest(source=source):
                data = self._run_compare(audit, samples=samples)
                self.assertFalse(data["class_identity_validation"]["valid"])
                self.assertIn(
                    reason,
                    data["class_identity_validation"]["incomplete_reasons"],
                )
                self.assertFalse(data["decision_evidence"]["usable"])
                self.assertFalse(
                    any(
                        "candidate" in diagnosis
                        for bucket in data["classes"].values()
                        for diagnosis in bucket["diagnosis"]
                    )
                )

    def test_unobserved_manifest_active_classes_and_states_are_insufficient(self):
        data = self._run_compare(
            self._complete_audit(
                runtime_manifest=self._valid_runtime_manifest(
                    class_id_to_name={"2": "bicycle", "5": "trailer"},
                    active_class_states={
                        "2": ["matched"],
                        "5": ["matched", "unmatched"],
                    },
                )
            )
        )

        trailer = data["classes"]["5:trailer"]
        self.assertEqual(
            set(trailer["decision_evidence_by_state"]),
            {"matched", "unmatched"},
        )
        for state in ("matched", "unmatched"):
            self.assertEqual(
                trailer["decision_evidence_by_state"][state][
                    "known_changed_count"
                ],
                0,
            )
            self.assertIn(
                "insufficient_decision_evidence",
                trailer["diagnosis_by_state"][state],
            )
        self.assertIn("insufficient_decision_evidence", trailer["diagnosis"])

    def test_valid_active_scope_remains_visible_when_another_key_is_invalid(self):
        invalid_key = "9" * 5000
        audit = self._complete_audit(
            runtime_manifest=self._valid_runtime_manifest(
                class_id_to_name={"2": "bicycle", "5": "trailer"},
                active_class_states={
                    "2": ["matched"],
                    "5": ["unmatched"],
                    invalid_key: ["matched"],
                },
            )
        )

        data = self._run_compare(audit)

        self.assertFalse(data["class_state_activation"]["class_level_usable"])
        trailer = data["classes"]["5:trailer"]
        self.assertIn("unmatched", trailer["decision_evidence_by_state"])
        self.assertIn(
            "insufficient_decision_evidence",
            trailer["diagnosis_by_state"]["unmatched"],
        )
        self.assertIn("insufficient_decision_evidence", trailer["diagnosis"])

    def test_duplicate_json_keys_fail_closed_before_identity_validation(self):
        audit = self._complete_audit()
        audit_text = json.dumps(audit).replace(
            '"2": "bicycle"',
            '"2": "bicycle", "2": "motorcycle"',
            1,
        )

        data = self._run_compare(audit, audit_text=audit_text)

        self.assertFalse(data["decision_evidence"]["usable"])
        self.assertIn(
            "duplicate_json_key",
            data["class_identity_validation"]["incomplete_reasons"],
        )
        self.assertFalse(
            any(
                "candidate" in diagnosis
                for bucket in data["classes"].values()
                for diagnosis in bucket["diagnosis"]
            )
        )

    def test_valid_mapping_class_remains_visible_with_invalid_mapping_key(self):
        audit = self._complete_audit(
            runtime_manifest=self._valid_runtime_manifest(
                class_id_to_name={
                    "2": "bicycle",
                    "5": "trailer",
                    str(10**100): "invalid",
                },
                active_class_states={"5": ["unmatched"]},
            )
        )

        data = self._run_compare(audit)

        self.assertIn("5:trailer", data["classes"])
        self.assertIn(
            "insufficient_decision_evidence",
            data["classes"]["5:trailer"]["diagnosis_by_state"]["unmatched"],
        )
        self.assertFalse(data["decision_evidence"]["usable"])

    def test_missing_manifest_initialized_fails_closed(self):
        audit = self._complete_audit()
        audit.pop("manifest_initialized")

        data = self._run_compare(audit)

        self.assertFalse(data["decision_evidence"]["usable"])
        self.assertIn(
            "missing_manifest_initialized",
            data["decision_evidence"]["incomplete_reasons"],
        )
        self.assertFalse(
            any(
                "candidate" in diagnosis
                for diagnosis in data["classes"]["2:bicycle"]["diagnosis"]
            )
        )

    def test_runtime_manifest_safety_fields_are_required(self):
        required_fields = (
            "audit_schema_version",
            "config_sha256",
            "association_head_enabled",
            "association_head_audit_enabled",
            "association_head_audit_shadow_mode",
            "association_checkpoint_configured",
            "association_checkpoint_path",
            "association_checkpoint_sha256",
            "association_checkpoint_loadable",
            "association_checkpoint_head_key_count",
            "association_checkpoint_key_validation_status",
            "association_checkpoint_validation_error",
            "association_checkpoint_runtime_contract_expected",
            "association_checkpoint_runtime_contract_actual",
            "association_checkpoint_runtime_contract_validation_status",
            "association_checkpoint_loaded",
        )

        for field in required_fields:
            with self.subTest(field=field):
                manifest = self._valid_runtime_manifest()
                manifest.pop(field)

                data = self._run_compare(
                    self._complete_audit(runtime_manifest=manifest)
                )

                evidence = data["decision_evidence"]
                self.assertFalse(evidence["usable"])
                self.assertIn(
                    f"missing_runtime_manifest_field:{field}",
                    evidence["incomplete_reasons"],
                )
                self.assertFalse(
                    any(
                        "candidate" in diagnosis
                        for diagnosis in data["classes"]["2:bicycle"][
                            "diagnosis"
                        ]
                    )
                )

    def test_checkpoint_hash_validation_and_runtime_probes_fail_closed(self):
        cases = {
            "old_manifest_schema": (
                {"audit_schema_version": 2},
                "invalid_runtime_manifest_field:audit_schema_version",
            ),
            "empty_config_hash": (
                {"config_sha256": ""},
                "invalid_runtime_manifest_field:config_sha256",
            ),
            "null_checkpoint_hash": (
                {"association_checkpoint_sha256": None},
                "invalid_runtime_manifest_field:association_checkpoint_sha256",
            ),
            "not_loadable": (
                {"association_checkpoint_loadable": False},
                "invalid_runtime_manifest_field:association_checkpoint_loadable",
            ),
            "preflight_validation_error": (
                {"association_checkpoint_validation_error": "broken checkpoint"},
                "runtime_manifest_validation_error:association_checkpoint_validation_error",
            ),
            "no_validated_head_keys": (
                {"association_checkpoint_head_key_count": 0},
                "invalid_runtime_manifest_field:association_checkpoint_head_key_count",
            ),
            "future_validation_error": (
                {
                    "association_checkpoint_future_validation_error": (
                        "input_dim mismatch"
                    )
                },
                "runtime_manifest_validation_error:association_checkpoint_future_validation_error",
            ),
            "nested_future_validation_error": (
                {
                    "association_checkpoint_future_status": {
                        "key_validation_error": "missing key"
                    }
                },
                "runtime_manifest_validation_error:association_checkpoint_future_status.key_validation_error",
            ),
            "future_key_validation_invalid": (
                {"association_checkpoint_key_validation_status": "invalid"},
                "invalid_runtime_manifest_field:association_checkpoint_key_validation_status",
            ),
            "configured_but_not_loaded": (
                {"association_checkpoint_loaded": False},
                "configured_association_checkpoint_not_loaded",
            ),
            "runtime_contract_invalid": (
                {"association_checkpoint_runtime_contract_validation_status": "invalid"},
                "invalid_runtime_manifest_field:association_checkpoint_runtime_contract_validation_status",
            ),
            "runtime_contract_mismatch": (
                {
                    "association_checkpoint_runtime_contract_actual": {
                        "input_dim": 999
                    }
                },
                "association_checkpoint_runtime_contract_mismatch",
            ),
        }

        for name, (overrides, reason) in cases.items():
            with self.subTest(name=name):
                manifest = self._valid_runtime_manifest(**overrides)
                data = self._run_compare(
                    self._complete_audit(runtime_manifest=manifest)
                )

                self.assertFalse(data["decision_evidence"]["usable"])
                self.assertIn(
                    reason, data["decision_evidence"]["incomplete_reasons"]
                )
                self.assertIn(
                    "insufficient_decision_evidence",
                    data["classes"]["2:bicycle"]["diagnosis"],
                )

    def test_checkpoint_fields_are_not_required_outside_enabled_shadow_audit(self):
        manifest = self._valid_runtime_manifest(
            association_head_enabled=False,
            association_head_audit_shadow_mode=False,
        )
        for field in list(manifest):
            if field.startswith("association_checkpoint_"):
                manifest.pop(field)

        data = self._run_compare(
            self._complete_audit(runtime_manifest=manifest)
        )

        reasons = data["decision_evidence"]["incomplete_reasons"]
        self.assertFalse(data["decision_evidence"]["usable"])
        self.assertIn("association_shadow_audit_not_enabled", reasons)
        self.assertFalse(
            any(
                reason.startswith(
                    "missing_runtime_manifest_field:association_checkpoint_"
                )
                for reason in reasons
            )
        )

    def test_atomic_json_writer_replaces_from_same_directory_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            output = directory / "report.json"
            output.write_text("stale report", encoding="utf-8")
            output.chmod(0o640)

            with patch.object(
                comparator.os, "replace", wraps=os.replace
            ) as replace_mock:
                comparator._atomic_write_json(output, {"status": "complete"})

            self.assertEqual(replace_mock.call_count, 1)
            temporary_path, replaced_output = replace_mock.call_args.args
            self.assertEqual(Path(temporary_path).parent, output.parent)
            self.assertEqual(Path(replaced_output), output)
            self.assertEqual(output.stat().st_mode & 0o777, 0o640)
            self.assertEqual(
                json.loads(output.read_text(encoding="utf-8")),
                {"status": "complete"},
            )
            self.assertEqual(list(directory.iterdir()), [output])

    def test_cli_atomic_replacement_preserves_existing_output_mode(self):
        data, output_mode = self._run_compare(
            self._complete_audit(), existing_output_mode=0o604
        )

        self.assertTrue(data["decision_evidence"]["usable"])
        self.assertEqual(output_mode, 0o604)


if __name__ == "__main__":
    unittest.main()
