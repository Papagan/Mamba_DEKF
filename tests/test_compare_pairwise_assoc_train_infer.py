import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


class ComparePairwiseAssocTrainInferCliTest(unittest.TestCase):
    def test_writes_class_diagnostics_from_train_cache_and_infer_audit(self):
        root = Path(__file__).resolve().parents[1]
        samples = [
            {"class_id": 2, "category": "bicycle", "label": 1, "negative_type": "positive", "center_distance": 0.2},
            {"class_id": 2, "category": "bicycle", "label": 0, "negative_type": "hard", "center_distance": 0.5},
            {"class_id": 2, "category": "bicycle", "label": 0, "negative_type": "easy", "center_distance": 3.0},
        ]
        audit = {
            "schema_version": 1,
            "buckets": [
                {
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
                    "score_quantiles": {"min": 0.0, "p10": 0.01, "p50": 0.08, "p90": 0.3, "max": 0.5},
                    "delta_quantiles": {"min": 0.0, "p10": 0.0, "p50": 0.01, "p90": 0.02, "max": 0.03},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_path = tmp_path / "train.pkl"
            audit_path = tmp_path / "audit.json"
            output_path = tmp_path / "summary.json"
            with open(train_path, "wb") as f:
                pickle.dump(samples, f)
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
                ],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(output_path.read_text(encoding="utf-8"))
            bucket = data["classes"]["2:bicycle"]
            self.assertEqual(bucket["train"]["sample_count"], 3)
            self.assertAlmostEqual(bucket["train"]["positive_ratio"], 1.0 / 3.0)
            self.assertEqual(bucket["infer"]["matched"]["penalized_ratio"], 0.75)
            self.assertIn("likely_distribution_mismatch", bucket["diagnosis"])


if __name__ == "__main__":
    unittest.main()
