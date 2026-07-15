import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


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

    def _run_compare(self, audit, *extra_args):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.pkl"
            audit_path = tmp_path / "audit.json"
            output_path = tmp_path / "summary.json"
            with open(train_path, "wb") as f:
                pickle.dump(self.samples, f)
            audit_path.write_text(json.dumps(audit), encoding="utf-8")

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

            self.assertEqual(result.returncode, 0, result.stderr)
            return json.loads(output_path.read_text(encoding="utf-8"))

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
        }

    @staticmethod
    def _decision_bucket(known_changed=120, improved=100, damaged=20):
        return {
            "class_id": 2,
            "class_name": "bicycle",
            "track_state_bucket": "matched",
            "total_decisions": 200,
            "known_changed_count": known_changed,
            "improved_count": improved,
            "damaged_count": damaged,
            "harm_precision": damaged / max(known_changed, 1),
        }

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
        self.assertNotIn("infer", bucket)
        self.assertIn("insufficient_decision_evidence", bucket["diagnosis"])
        self.assertNotIn("likely_distribution_mismatch", bucket["diagnosis"])
        self.assertFalse(any("candidate" in item for item in bucket["diagnosis"]))

    def test_schema_v2_safe_decision_counts_are_shadow_ab_candidate(self):
        data = self._run_compare(
            {
                "schema_version": 2,
                "decision_evidence_available": True,
                "decision_evidence_complete": True,
                "manifest_compatible": True,
                "audit_errors": {"count": 0, "samples": []},
                "pair_buckets": [self._pair_bucket()],
                "decision_buckets": [self._decision_bucket()],
            }
        )

        bucket = data["classes"]["2:bicycle"]
        self.assertEqual(data["infer_audit_schema_version"], 2)
        self.assertTrue(data["decision_evidence"]["available"])
        self.assertEqual(bucket["decision_evidence"]["known_changed_count"], 120)
        self.assertIn("candidate_for_shadow_ab_activation", bucket["diagnosis"])

    def test_default_minimum_requires_100_known_changed_decisions(self):
        audit = {
            "schema_version": 2,
            "decision_evidence_available": True,
            "decision_evidence_complete": True,
            "manifest_compatible": True,
            "audit_errors": {"count": 0},
            "decision_buckets": [
                self._decision_bucket(known_changed=99, improved=90, damaged=9)
            ],
        }

        data = self._run_compare(audit)
        bucket = data["classes"]["2:bicycle"]
        self.assertEqual(data["decision_gates"]["min_known_changed"], 100)
        self.assertIn("shadow_decisions_not_safe", bucket["diagnosis"])
        self.assertFalse(any("candidate" in item for item in bucket["diagnosis"]))

        configured = self._run_compare(audit, "--min-known-changed", "50")
        self.assertIn(
            "candidate_for_shadow_ab_activation",
            configured["classes"]["2:bicycle"]["diagnosis"],
        )

    def test_zero_known_changed_decisions_are_insufficient(self):
        audit = {
            "schema_version": 2,
            "decision_evidence_available": True,
            "decision_evidence_complete": True,
            "manifest_compatible": True,
            "audit_errors": {"count": 0},
            "decision_buckets": [
                self._decision_bucket(known_changed=0, improved=0, damaged=0)
            ],
        }

        diagnosis = self._run_compare(audit)["classes"]["2:bicycle"]["diagnosis"]
        self.assertIn("insufficient_decision_evidence", diagnosis)
        self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_harm_above_gate_is_not_safe(self):
        audit = {
            "schema_version": 2,
            "decision_evidence_available": True,
            "decision_evidence_complete": True,
            "manifest_compatible": True,
            "audit_errors": {"count": 0},
            "decision_buckets": [
                self._decision_bucket(known_changed=120, improved=90, damaged=30)
            ],
        }

        diagnosis = self._run_compare(audit)["classes"]["2:bicycle"]["diagnosis"]
        self.assertIn("shadow_decisions_not_safe", diagnosis)
        self.assertFalse(any("candidate" in item for item in diagnosis))

    def test_incomplete_or_invalid_audit_metadata_fails_closed(self):
        base = {
            "schema_version": 2,
            "decision_evidence_available": True,
            "decision_evidence_complete": True,
            "manifest_compatible": True,
            "audit_errors": {"count": 0},
            "decision_buckets": [self._decision_bucket()],
        }
        cases = {
            "evidence_unavailable": {"decision_evidence_available": False},
            "evidence_incomplete": {"decision_evidence_complete": False},
            "manifest_incompatible": {"manifest_compatible": False},
            "audit_errors": {"audit_errors": {"count": 1}},
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


if __name__ == "__main__":
    unittest.main()
