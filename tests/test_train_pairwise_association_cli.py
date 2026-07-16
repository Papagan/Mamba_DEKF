import pickle
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools import train_pairwise_association
from training.association_features import compose_pair_vectors


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
    def test_precompute_pair_composition_delegates_to_shared_helper(self):
        track_embeddings = torch.tensor([[1.0, 2.0]])
        detection_embeddings = torch.tensor([[3.0, 5.0]])
        pair_geometry = torch.tensor([[0.4, 0.2, 1.5, 0.8, 0.9, 1.0, 5.0]])

        class StubModel:
            def __init__(self):
                self._embeddings = iter((track_embeddings, detection_embeddings))

            def encode(self, history, class_ids, state_buckets=None):
                return next(self._embeddings)

        batch = {
            "anchor_history": torch.zeros(1, 2, 12),
            "candidate_history": torch.zeros(1, 2, 12),
            "pair_features": pair_geometry,
            "class_id": torch.tensor([2]),
            "state_buckets": ["matched"],
        }
        expected = compose_pair_vectors(
            track_embeddings,
            detection_embeddings,
            pair_geometry,
        )

        with patch.object(
            train_pairwise_association,
            "compose_pair_vectors",
            wraps=compose_pair_vectors,
        ) as shared_compose:
            actual = train_pairwise_association._pair_vectors_from_batch(
                StubModel(), batch, torch.device("cpu")
            )

        shared_compose.assert_called_once()
        self.assertTrue(torch.equal(actual, expected))

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
                    "--precompute-embeddings",
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
            ckpt = torch.load(ckpt_path, map_location="cpu")
            runtime_contract = ckpt["runtime_contract"]
            self.assertTrue(runtime_contract["precompute_embeddings"])
            self.assertEqual(
                runtime_contract["filter_mode"], "mamba_multihead_closure"
            )
            self.assertEqual(runtime_contract["history_source"], "fusion")
            self.assertEqual(runtime_contract["tracker_compat_mode"], "mctrack")
            self.assertEqual(
                runtime_contract["expected_bev_cost_mode"], "geometric"
            )


if __name__ == "__main__":
    unittest.main()
