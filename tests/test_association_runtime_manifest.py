import hashlib
import tempfile
import unittest
from pathlib import Path

import torch

from tracker.association_runtime_manifest import (
    build_association_runtime_manifest,
    runtime_manifests_compatible,
    sha256_file,
)


class AssociationRuntimeManifestTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.config_path = self.root / "config" / "evaluation.yaml"
        self.config_path.parent.mkdir()
        self.config_bytes = b"SAVE_PATH: original-results\n"
        self.config_path.write_bytes(self.config_bytes)

        self.checkpoint_path = self.root / "checkpoints" / "association.pt"
        self.checkpoint_path.parent.mkdir()
        self.checkpoint_bytes = b"association checkpoint"
        self.checkpoint_path.write_bytes(self.checkpoint_bytes)

    def _cfg(self):
        return {
            "FILTER_MODE": "mamba_multihead_closure",
            "TRACKER_COMPAT_MODE": "mctrack",
            "MAMBA": {"EMBED_DIM": 64, "NUM_CLASSES": 7},
            "MATCHING": {
                "BEV": {
                    "MATCHING_MODE": "Hungarian",
                    "COST_MODE": {2: "RO_GDIOU_3D", 0: "IOU_3D"},
                }
            },
            "MAMBA_ASSOCIATION_HEAD": {
                "CHECKPOINT_PATH": "checkpoints/association.pt",
                "HIDDEN_DIMS": [256, 128],
                "DROPOUT": 0.2,
                "NUM_CLASSES": 7,
                "ACTIVE_CLASS_STATES": {2: ["unmatched", "matched"]},
                "APPLY_MODE": "margin_tiebreak",
                "COST_MARGIN_EPS": 0.05,
                "MIN_SCORE": 0.6,
                "ALPHA": 0.05,
                "MAX_DELTA": 0.03,
            },
        }

    def test_manifest_normalizes_effective_association_contract(self):
        manifest = build_association_runtime_manifest(
            cfg=self._cfg(),
            config_path=str(self.config_path),
            process_count=12,
            git_root=self.root,
        )

        self.assertEqual(manifest["config_path"], str(self.config_path.resolve()))
        self.assertEqual(
            manifest["config_sha256"], hashlib.sha256(self.config_bytes).hexdigest()
        )
        self.assertEqual(
            manifest["association_checkpoint_path"],
            str(self.checkpoint_path.resolve()),
        )
        self.assertEqual(
            manifest["association_checkpoint_sha256"],
            hashlib.sha256(self.checkpoint_bytes).hexdigest(),
        )
        self.assertIn("git_commit", manifest)
        self.assertIn("git_dirty", manifest)
        self.assertEqual(
            manifest["association_head_architecture"],
            {
                "input_dim": 263,
                "hidden_dims": [256, 128],
                "hidden_dims_source": "config",
                "dropout": 0.2,
                "dropout_source": "config",
                "num_classes": 7,
            },
        )
        self.assertEqual(
            manifest["active_class_states"], {"2": ["matched", "unmatched"]}
        )
        self.assertEqual(manifest["apply_mode"], "margin_tiebreak")
        self.assertEqual(manifest["cost_margin_eps"], 0.05)
        self.assertEqual(manifest["min_score"], 0.6)
        self.assertEqual(manifest["alpha"], 0.05)
        self.assertEqual(manifest["max_delta"], 0.03)
        self.assertEqual(manifest["filter_mode"], "mamba_multihead_closure")
        self.assertEqual(manifest["tracker_compat_mode"], "mctrack")
        self.assertEqual(manifest["bev_matching_mode"], "Hungarian")
        self.assertEqual(
            manifest["bev_cost_modes"], {"0": "IOU_3D", "2": "RO_GDIOU_3D"}
        )
        self.assertEqual(manifest["process_count"], 12)
        self.assertEqual(manifest["audit_schema_version"], 2)
        self.assertEqual(len(manifest["config_sha256"]), 64)

    def test_hash_failures_return_none(self):
        self.assertIsNone(sha256_file(self.root / "missing.yaml"))

        cfg = self._cfg()
        cfg["MAMBA_ASSOCIATION_HEAD"]["CHECKPOINT_PATH"] = "missing.pt"
        manifest = build_association_runtime_manifest(
            cfg=cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertEqual(
            manifest["association_checkpoint_path"],
            str((self.root / "missing.pt").resolve()),
        )
        self.assertIsNone(manifest["association_checkpoint_sha256"])

    def test_cyclic_checkpoint_path_does_not_escape_manifest_builder(self):
        (self.root / "loop-a").symlink_to("loop-b")
        (self.root / "loop-b").symlink_to("loop-a")
        cfg = self._cfg()
        cfg["MAMBA_ASSOCIATION_HEAD"]["CHECKPOINT_PATH"] = "loop-a/model.pt"

        manifest = build_association_runtime_manifest(
            cfg=cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertIsNone(manifest["association_checkpoint_path"])
        self.assertIsNone(manifest["association_checkpoint_sha256"])

    def test_architecture_uses_checkpoint_settings_when_config_omits_them(self):
        torch.save(
            {"settings": {"hidden_dims": [96, 48], "dropout": 0.25}},
            self.checkpoint_path,
        )
        cfg = self._cfg()
        del cfg["MAMBA_ASSOCIATION_HEAD"]["HIDDEN_DIMS"]
        del cfg["MAMBA_ASSOCIATION_HEAD"]["DROPOUT"]

        manifest = build_association_runtime_manifest(
            cfg=cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        architecture = manifest["association_head_architecture"]
        self.assertEqual(architecture["hidden_dims"], [96, 48])
        self.assertEqual(architecture["hidden_dims_source"], "checkpoint")
        self.assertEqual(architecture["dropout"], 0.25)
        self.assertEqual(architecture["dropout_source"], "checkpoint")

    def test_manifest_compatibility_rejects_changed_active_classes(self):
        left = {"config_sha256": "a", "active_class_states": {"2": ["matched"]}}
        right = {"config_sha256": "a", "active_class_states": {"3": ["matched"]}}

        self.assertFalse(runtime_manifests_compatible(left, right))

    def test_manifest_compatibility_ignores_mapping_order(self):
        left = {
            "active_class_states": {"2": ["matched", "unmatched"]},
            "config_sha256": "a",
        }
        right = {
            "config_sha256": "a",
            "active_class_states": {"2": ["matched", "unmatched"]},
        }

        self.assertTrue(runtime_manifests_compatible(left, right))
        self.assertTrue(runtime_manifests_compatible(None, None))
        self.assertFalse(runtime_manifests_compatible(left, None))

    def test_active_class_states_match_runtime_normalization_semantics(self):
        mixed_cfg = self._cfg()
        mixed_cfg["MAMBA_ASSOCIATION_HEAD"]["ACTIVE_CLASS_STATES"] = {
            2: [" MATCHED ", "matched"],
            "2": [" UNMATCHED ", "unmatched"],
            3: ["UNMATCHED", " matched "],
        }
        canonical_cfg = self._cfg()
        canonical_cfg["MAMBA_ASSOCIATION_HEAD"]["ACTIVE_CLASS_STATES"] = {
            "3": ["matched", "unmatched"],
            "2": ["unmatched", "matched"],
        }

        mixed = build_association_runtime_manifest(
            cfg=mixed_cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )
        canonical = build_association_runtime_manifest(
            cfg=canonical_cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        expected = {
            "2": ["matched", "unmatched"],
            "3": ["matched", "unmatched"],
        }
        self.assertEqual(mixed["active_class_states"], expected)
        self.assertEqual(canonical["active_class_states"], expected)
        self.assertTrue(runtime_manifests_compatible(mixed, canonical))


if __name__ == "__main__":
    unittest.main()
