import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from training.association_stage import run_association_head_stage_if_requested


class TrainAssociationStageTest(unittest.TestCase):
    def _cfg(self, train_pkl, val_pkl, *, run_after_main=True):
        return {
            "MODEL": {"HISTORY_LEN": 8},
            "ASSOCIATION_HEAD_TRAINING": {
                "ENABLED": True,
                "RUN_AFTER_MAIN": run_after_main,
                "TRAIN_PAIRWISE_PKL": str(train_pkl),
                "VAL_PAIRWISE_PKL": str(val_pkl),
                "OUTPUT": "checkpoints/mamba_dekf/best_assoc.pt",
                "METRICS_OUTPUT": "docs/pairwise_assoc_metrics.json",
                "BACKBONE_CHECKPOINT": "checkpoints/mamba_dekf/best.pt",
                "FREEZE_BACKBONE": True,
            },
        }

    def test_disabled_stage_does_not_call_runner(self):
        runner = Mock()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.pkl"
            status = run_association_head_stage_if_requested(
                self._cfg(missing, missing, run_after_main=False),
                config_path="config/train_nuscenes.yaml",
                device="cpu",
                runner=runner,
            )

        self.assertEqual(status, "disabled")
        runner.assert_not_called()

    def test_enabled_stage_requires_existing_pairwise_caches(self):
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "missing.pkl"
            with self.assertRaises(FileNotFoundError):
                run_association_head_stage_if_requested(
                    self._cfg(missing, missing),
                    config_path="config/train_nuscenes.yaml",
                    device="cpu",
                    runner=Mock(),
                )

    def test_enabled_stage_calls_runner_with_frozen_backbone_args(self):
        runner = Mock(return_value={"best_val": 0.1})
        with tempfile.TemporaryDirectory() as tmp:
            train_pkl = Path(tmp) / "train.pkl"
            val_pkl = Path(tmp) / "val.pkl"
            train_pkl.write_bytes(b"\x80\x04]\x94.")
            val_pkl.write_bytes(b"\x80\x04]\x94.")

            status = run_association_head_stage_if_requested(
                self._cfg(train_pkl, val_pkl),
                config_path="config/train_nuscenes.yaml",
                device="cpu",
                runner=runner,
            )

        self.assertEqual(status, "ran")
        args = runner.call_args.args[0]
        self.assertIsInstance(args, SimpleNamespace)
        self.assertTrue(args.freeze_backbone)
        self.assertEqual(args.train_pkl, str(train_pkl))
        self.assertEqual(args.val_pkl, str(val_pkl))
        self.assertEqual(args.device, "cpu")


if __name__ == "__main__":
    unittest.main()
