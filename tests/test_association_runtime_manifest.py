import hashlib
import json
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
            "CATEGORY_MAP_TO_NUMBER": {"pedestrian": 2, "car": 0},
            "MAMBA": {"EMBED_DIM": 64, "NUM_CLASSES": 7},
            "MATCHING": {
                "BEV": {
                    "MATCHING_MODE": "Hungarian",
                    "COST_MODE": {2: "RO_GDIOU_3D", 0: "IOU_3D"},
                }
            },
            "THRESHOLD": {"BEV": {"COST_MODE": "geometric"}},
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

    def _runtime_contract(self, **overrides):
        contract = {
            "filter_mode": "mamba_multihead_closure",
            "history_source": "fusion",
            "tracker_compat_mode": "mctrack",
            "expected_bev_cost_mode": "geometric",
        }
        contract.update(overrides)
        return contract

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
        self.assertFalse(manifest["association_checkpoint_loadable"])
        self.assertEqual(manifest["association_checkpoint_head_key_count"], 0)
        self.assertIsInstance(
            manifest["association_checkpoint_validation_error"], str
        )
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_expected"],
            self._runtime_contract(),
        )
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_actual"], {}
        )
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_validation_status"],
            "invalid",
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
        self.assertEqual(
            manifest["class_id_to_name"], {"0": "car", "2": "pedestrian"}
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
        self.assertEqual(manifest["audit_schema_version"], 3)
        self.assertEqual(len(manifest["config_sha256"]), 64)
        json.dumps(manifest, allow_nan=False)

    def test_checkpoint_preflight_reports_loadable_head_state(self):
        torch.save(
            {
                "model_state_dict": {
                    "association_heads.heads.0.0.weight": torch.ones(1, 2),
                    "association_heads.heads.0.0.bias": torch.zeros(1),
                    "temporal_encoder.weight": torch.ones(1),
                },
                "runtime_contract": self._runtime_contract(),
            },
            self.checkpoint_path,
        )

        manifest = build_association_runtime_manifest(
            cfg=self._cfg(),
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertTrue(manifest["association_checkpoint_loadable"])
        self.assertEqual(manifest["association_checkpoint_head_key_count"], 2)
        self.assertIsNone(manifest["association_checkpoint_validation_error"])
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_actual"],
            self._runtime_contract(),
        )
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_validation_status"],
            "valid",
        )
        self.assertEqual(
            json.loads(json.dumps(manifest))["association_checkpoint_head_key_count"],
            2,
        )

    def test_checkpoint_preflight_reports_missing_head_keys(self):
        torch.save(
            {
                "model_state_dict": {"encoder.weight": torch.ones(1)},
                "runtime_contract": self._runtime_contract(),
            },
            self.checkpoint_path,
        )

        manifest = build_association_runtime_manifest(
            cfg=self._cfg(),
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertTrue(manifest["association_checkpoint_loadable"])
        self.assertEqual(manifest["association_checkpoint_head_key_count"], 0)
        self.assertIn(
            "association_heads.*",
            manifest["association_checkpoint_validation_error"],
        )

    def test_checkpoint_preflight_rejects_non_mapping_settings(self):
        torch.save(
            {
                "model_state_dict": {
                    "association_heads.heads.0.weight": torch.ones(1)
                },
                "settings": [],
                "runtime_contract": self._runtime_contract(),
            },
            self.checkpoint_path,
        )

        manifest = build_association_runtime_manifest(
            cfg=self._cfg(),
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertIn(
            "settings must be a mapping",
            manifest["association_checkpoint_validation_error"],
        )

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
        self.assertFalse(manifest["association_checkpoint_loadable"])
        self.assertEqual(manifest["association_checkpoint_head_key_count"], 0)
        self.assertIn(
            "does not exist", manifest["association_checkpoint_validation_error"]
        )

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
        self.assertFalse(manifest["association_checkpoint_loadable"])
        self.assertEqual(manifest["association_checkpoint_head_key_count"], 0)
        self.assertIn("path", manifest["association_checkpoint_validation_error"])

    def test_architecture_uses_checkpoint_settings_when_config_omits_them(self):
        torch.save(
            {
                "settings": {"hidden_dims": [96, 48], "dropout": 0.25},
                "runtime_contract": self._runtime_contract(),
            },
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

    def test_malformed_checkpoint_hidden_dims_are_preflight_validation_errors(self):
        torch.save(
            {
                "settings": {"hidden_dims": "bad", "dropout": 0.25},
                "runtime_contract": self._runtime_contract(),
            },
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

        self.assertIn(
            "hidden_dims", manifest["association_checkpoint_validation_error"]
        )

    def test_malformed_checkpoint_dropout_is_a_preflight_validation_error(self):
        torch.save(
            {
                "settings": {"hidden_dims": [96, 48], "dropout": "bad"},
                "runtime_contract": self._runtime_contract(),
            },
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

        self.assertIn(
            "dropout", manifest["association_checkpoint_validation_error"]
        )

    def test_malformed_config_architecture_is_a_preflight_validation_error(self):
        torch.save(
            {
                "model_state_dict": {
                    "association_heads.heads.0.0.weight": torch.ones(1, 2)
                },
                "runtime_contract": self._runtime_contract(),
            },
            self.checkpoint_path,
        )
        cases = [
            ("HIDDEN_DIMS", "bad", "hidden_dims"),
            ("HIDDEN_DIMS", [0], "hidden_dims"),
            ("DROPOUT", "bad", "dropout"),
            ("DROPOUT", 1.5, "dropout"),
        ]

        for key, value, expected in cases:
            with self.subTest(key=key, value=value):
                cfg = self._cfg()
                cfg["MAMBA_ASSOCIATION_HEAD"][key] = value
                manifest = build_association_runtime_manifest(
                    cfg=cfg,
                    config_path=str(self.config_path),
                    process_count=1,
                    git_root=self.root,
                )

                self.assertIn(
                    expected,
                    manifest["association_checkpoint_validation_error"],
                )

    def test_checkpoint_preflight_rejects_incompatible_filter_and_history_contracts(self):
        for field, value in [
            ("filter_mode", "pure_dekf"),
            ("history_source", "det"),
        ]:
            with self.subTest(field=field):
                torch.save(
                    {
                        "model_state_dict": {
                            "association_heads.heads.0.weight": torch.ones(1)
                        },
                        "runtime_contract": self._runtime_contract(**{field: value}),
                    },
                    self.checkpoint_path,
                )

                manifest = build_association_runtime_manifest(
                    cfg=self._cfg(),
                    config_path=str(self.config_path),
                    process_count=1,
                    git_root=self.root,
                )

                self.assertEqual(
                    manifest["association_checkpoint_runtime_contract_expected"],
                    self._runtime_contract(),
                )
                self.assertEqual(
                    manifest["association_checkpoint_runtime_contract_actual"][field],
                    value,
                )
                self.assertEqual(
                    manifest[
                        "association_checkpoint_runtime_contract_validation_status"
                    ],
                    "invalid",
                )
                self.assertIn(
                    field, manifest["association_checkpoint_validation_error"]
                )

    def test_checkpoint_preflight_rejects_missing_runtime_contract(self):
        torch.save(
            {
                "model_state_dict": {
                    "association_heads.heads.0.weight": torch.ones(1)
                }
            },
            self.checkpoint_path,
        )

        manifest = build_association_runtime_manifest(
            cfg=self._cfg(),
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_actual"], {}
        )
        self.assertEqual(
            manifest["association_checkpoint_runtime_contract_validation_status"],
            "invalid",
        )
        self.assertIn(
            "runtime_contract is missing",
            manifest["association_checkpoint_validation_error"],
        )

    def test_category_map_is_strictly_reversed_for_diagnostics(self):
        cfg = self._cfg()
        cfg["CATEGORY_MAP_TO_NUMBER"] = {" pedestrian ": 2, "CAR": 0}

        manifest = build_association_runtime_manifest(
            cfg=cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertEqual(
            manifest["class_id_to_name"], {"0": "car", "2": "pedestrian"}
        )

    def test_conflicting_or_malformed_category_maps_fail_preflight_validation(self):
        cases = [
            ({"car": 0, "truck": 0}, "duplicate class id"),
            ({"car": "zero"}, "integer class id"),
            ({"": 0}, "non-empty class name"),
            (["car", 0], "mapping"),
        ]

        for category_map, expected in cases:
            with self.subTest(category_map=category_map):
                cfg = self._cfg()
                cfg["CATEGORY_MAP_TO_NUMBER"] = category_map
                manifest = build_association_runtime_manifest(
                    cfg=cfg,
                    config_path=str(self.config_path),
                    process_count=1,
                    git_root=self.root,
                )

                self.assertEqual(manifest["class_id_to_name"], {})
                self.assertIn(
                    expected,
                    manifest["association_checkpoint_validation_error"],
                )

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
            2: [" UNMATCHED ", "matched", " MATCHED ", "unmatched"],
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

    def test_active_class_states_prefer_integer_key_over_string_key(self):
        cfg = self._cfg()
        cfg["MAMBA_ASSOCIATION_HEAD"]["ACTIVE_CLASS_STATES"] = {
            2: [" MATCHED ", "matched"],
            "2": ["unmatched"],
        }

        manifest = build_association_runtime_manifest(
            cfg=cfg,
            config_path=str(self.config_path),
            process_count=1,
            git_root=self.root,
        )

        self.assertEqual(manifest["active_class_states"], {"2": ["matched"]})


if __name__ == "__main__":
    unittest.main()
