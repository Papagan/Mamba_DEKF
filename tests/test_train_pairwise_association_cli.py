import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


def _sample(anchor_id, class_id, label, value, negative_type="positive"):
    return {
        "category": "bicycle" if class_id == 2 else "car",
        "class_id": class_id,
        "state_bucket": "matched",
        "anchor_instance_token": anchor_id,
        "current_sample_token": "cur",
        "future_sample_token": "fut",
        "label": label,
        "negative_type": negative_type,
        "anchor_history_12": [[value] * 12 for _ in range(4)],
        "candidate_obs_feature_12": [value] * 12,
        "center_distance": 0.3 if label else 1.2,
        "yaw_diff": 0.1,
        "size_l1": 0.2,
        "anchor_det_score": 0.8,
        "candidate_det_score": 0.9 if label else 0.4,
    }


class TrainPairwiseAssociationCliTest(unittest.TestCase):
    def test_dry_run_writes_metrics_json(self):
        root = Path(__file__).resolve().parents[1]
        samples = [
            _sample("a", 2, 1, 1.0, "positive"),
            _sample("a", 2, 0, 0.2, "hard"),
            _sample("b", 2, 1, 0.9, "positive"),
            _sample("b", 2, 0, 0.1, "hard"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            pairwise_path = tmp_path / "pairwise.pkl"
            metrics_path = tmp_path / "metrics.json"
            ckpt_path = tmp_path / "assoc.pt"
            with open(pairwise_path, "wb") as f:
                pickle.dump(samples, f)

            result = subprocess.run(
                [
                    sys.executable,
                    "tools/train_pairwise_association.py",
                    "--train-pkl",
                    str(pairwise_path),
                    "--val-pkl",
                    str(pairwise_path),
                    "--output",
                    str(ckpt_path),
                    "--metrics-output",
                    str(metrics_path),
                    "--epochs",
                    "1",
                    "--batch-size",
                    "2",
                    "--history-len",
                    "4",
                    "--freeze-backbone",
                    "--ranking-margin",
                    "0.2",
                    "--ranking-weight",
                    "0.1",
                    "--dry-run",
                ],
                cwd=root,
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(metrics_path.exists())
            self.assertTrue(ckpt_path.exists())
            self.assertIn("ranking", metrics_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
