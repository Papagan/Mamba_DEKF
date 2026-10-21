from pathlib import Path
import unittest

import yaml

from tools.audit_pairwise_association_cache import validate_training_samples


ROOT = Path(__file__).resolve().parents[1]


class RemoteTrainingConfigTest(unittest.TestCase):
    def test_motion_and_association_paths_agree_after_migration(self):
        train = yaml.safe_load((ROOT / "config/train_nuscenes.yaml").read_text())
        runtime = yaml.safe_load((ROOT / "config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml").read_text())
        self.assertEqual(train["DATA"]["NUSC_DATAROOT"], runtime["DATASET_ROOT"])
        self.assertEqual(str(Path(train["TRAINING"]["SAVE_DIR"]) / "best.pt"),
                         runtime["MAMBA"]["CHECKPOINT_PATH"])
        self.assertEqual(train["ASSOCIATION_HEAD_TRAINING"]["BACKBONE_CHECKPOINT"],
                         runtime["MAMBA"]["CHECKPOINT_PATH"])
        self.assertEqual(train["ASSOCIATION_HEAD_TRAINING"]["OUTPUT"],
                         runtime["MAMBA_ASSOCIATION_HEAD"]["CHECKPOINT_PATH"])
        self.assertTrue(runtime["DATASET_ROOT"].startswith("/home/alvin/datasets/track_data/"))
        self.assertEqual(train["TRAINING"]["BATCH_SIZE"] % 2, 0)

    def test_association_preflight_rejects_missing_classes_and_old_schema(self):
        cfg = dict(ASSOCIATION_HEAD_TRAINING={
            "CLASS_IDS": [0, 1], "REQUIRE_ALL_CLASSES": True,
            "REQUIRE_FEATURE_SCHEMA_VERSION": 3,
        })
        samples = [dict(class_id=0, feature_schema_version=3)]
        with self.assertRaisesRegex(ValueError, "classes"):
            validate_training_samples(samples, cfg)
        samples.append(dict(class_id=1, feature_schema_version=2))
        with self.assertRaisesRegex(ValueError, "schema"):
            validate_training_samples(samples, cfg)
        samples[-1]["feature_schema_version"] = 3
        validate_training_samples(samples, cfg)
