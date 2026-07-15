import ast
import copy
import os
import tempfile
import types
import unittest
from collections.abc import Mapping
from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import patch

import torch

from tracker.association_head_audit import AssociationHeadAuditAccumulator
from tracker.association_runtime_manifest import (
    preflight_association_checkpoint,
    resolve_association_head_parameters,
    sha256_file,
    validate_association_runtime_contract,
)
from training.association_model import ClassConditionedAssociationHeadBank


ROOT = Path(__file__).resolve().parents[1]


class _StubHead:
    load_result = ([], [])

    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.loaded_state = None
        self.strict = None
        self.eval_called = False

    def to(self, device):
        self.device = device
        return self

    def load_state_dict(self, state, strict):
        self.loaded_state = dict(state)
        self.strict = strict
        return self.load_result

    def eval(self):
        self.eval_called = True
        return self


def _load_tracker_type(head_type=_StubHead):
    tree = ast.parse((ROOT / "tracker/base_tracker.py").read_text(encoding="utf-8"))
    tracker_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
    )
    names = {
        "_pairwise_association_head_cfg",
        "_association_checkpoint_audit_enabled",
        "_association_checkpoint_audit_strict",
        "_record_association_checkpoint_error",
        "_parse_hidden_dims",
        "_load_pairwise_association_head",
    }
    selected = [
        node
        for node in tracker_class.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    missing = names - {node.name for node in selected}
    if missing:
        raise AssertionError(f"Missing Base3DTracker methods: {sorted(missing)}")
    namespace = {
        "AssociationHeadAuditAccumulator": AssociationHeadAuditAccumulator,
        "ClassConditionedAssociationHeadBank": head_type,
        "Dict": dict,
        "Mapping": Mapping,
        "Optional": Optional,
        "Path": Path,
        "Tuple": Tuple,
        "os": os,
        "preflight_association_checkpoint": preflight_association_checkpoint,
        "resolve_association_head_parameters": resolve_association_head_parameters,
        "sha256_file": sha256_file,
        "torch": torch,
        "validate_association_runtime_contract": validate_association_runtime_contract,
    }
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), "tracker/base_tracker.py", "exec"),
        namespace,
    )
    methods = {name: namespace[name] for name in names}
    return type("MinimalTracker", (), methods)


class AssociationCheckpointAuditTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)
        self.checkpoint_path = self.root / "association.pt"
        _StubHead.load_result = ([], [])

    def _tracker(
        self,
        *,
        checkpoint_path=None,
        audit_enabled=True,
        shadow_mode=True,
        strict=False,
        head_enabled=True,
        runtime_manifest=None,
        head_type=_StubHead,
        include_architecture_config=True,
        hidden_dims=None,
        dropout=None,
    ):
        tracker_type = _load_tracker_type(head_type=head_type)
        tracker = tracker_type()
        head_cfg = {
            "ENABLED": head_enabled,
            "CHECKPOINT_PATH": checkpoint_path,
            "NUM_CLASSES": 1,
        }
        if include_architecture_config:
            head_cfg["HIDDEN_DIMS"] = [2] if hidden_dims is None else hidden_dims
            head_cfg["DROPOUT"] = 0.0 if dropout is None else dropout
        tracker.cfg = {
            "FILTER_MODE": "mamba_multihead_closure",
            "TRACKER_COMPAT_MODE": "mctrack",
            "MAMBA": {"NUM_CLASSES": 1},
            "THRESHOLD": {"BEV": {"COST_MODE": "geometric"}},
            "MAMBA_ASSOCIATION_HEAD": head_cfg,
            "ASSOCIATION_HEAD_AUDIT": {
                "ENABLED": audit_enabled,
                "SHADOW_MODE": shadow_mode,
                "STRICT": strict,
                "MAX_SAMPLES_PER_BUCKET": 8,
            },
            "_ASSOCIATION_AUDIT_RUNTIME_MANIFEST": runtime_manifest,
        }
        tracker.association_head_audit_cfg = tracker.cfg["ASSOCIATION_HEAD_AUDIT"]
        tracker.association_head_audit = (
            AssociationHeadAuditAccumulator(
                max_samples_per_bucket=8,
                runtime_manifest=runtime_manifest,
                strict_manifest=strict,
            )
            if audit_enabled
            else None
        )
        tracker.device = torch.device("cpu")
        tracker.embed_dim = 1
        return tracker

    def _runtime_contract(self, **overrides):
        contract = {
            "filter_mode": "mamba_multihead_closure",
            "history_source": "fusion",
            "tracker_compat_mode": "mctrack",
            "expected_bev_cost_mode": "geometric",
        }
        contract.update(overrides)
        return contract

    def _save_checkpoint(self, state, runtime_contract=None, *, omit_contract=False):
        payload = {"model_state_dict": state}
        if not omit_contract:
            payload["runtime_contract"] = (
                self._runtime_contract()
                if runtime_contract is None
                else runtime_contract
            )
        torch.save(payload, self.checkpoint_path)

    def _real_head_state(self):
        head = ClassConditionedAssociationHeadBank(
            input_dim=11,
            num_classes=1,
            hidden_dims=(2,),
            dropout=0.0,
        )
        return {
            f"association_heads.{key}": value.detach().clone()
            for key, value in head.state_dict().items()
        }

    def _runtime_manifest(self, head_cfg=None):
        preflight, _ = preflight_association_checkpoint(
            str(self.checkpoint_path),
            head_cfg or {"HIDDEN_DIMS": [2], "DROPOUT": 0.0},
            runtime_cfg={
                "FILTER_MODE": "mamba_multihead_closure",
                "TRACKER_COMPAT_MODE": "mctrack",
                "THRESHOLD": {"BEV": {"COST_MODE": "geometric"}},
            },
        )
        return {
            "association_checkpoint_path": str(self.checkpoint_path.resolve()),
            "association_checkpoint_sha256": sha256_file(self.checkpoint_path),
            "association_checkpoint_loadable": preflight["loadable"],
            "association_checkpoint_head_key_count": preflight["head_key_count"],
            "association_checkpoint_validation_error": preflight["validation_error"],
            "association_checkpoint_runtime_contract_expected": preflight[
                "runtime_contract_expected"
            ],
            "association_checkpoint_runtime_contract_actual": preflight[
                "runtime_contract_actual"
            ],
            "association_checkpoint_runtime_contract_validation_status": preflight[
                "runtime_contract_validation_status"
            ],
        }

    def _assert_one_load_error(self, tracker):
        summary = tracker.association_head_audit.to_summary()
        self.assertEqual(summary["audit_error_count"], 1)
        self.assertFalse(summary["decision_evidence_complete"])
        self.assertIn("audit_errors", summary["decision_evidence_incomplete_reasons"])
        self.assertEqual(
            summary["audit_errors"]["samples"][0]["stage"],
            "association_checkpoint_load",
        )

    def test_strict_shadow_audit_fails_closed_for_checkpoint_errors(self):
        self._save_checkpoint({"encoder.weight": torch.ones(1)})
        cases = [
            ("empty path", None, None, "CHECKPOINT_PATH is empty"),
            ("missing path", str(self.root / "missing.pt"), None, "not found"),
            ("load failure", str(self.checkpoint_path), RuntimeError("broken bytes"), "failed to load"),
            ("no head keys", str(self.checkpoint_path), None, "no association_heads.* keys"),
        ]
        for label, checkpoint_path, load_error, expected in cases:
            with self.subTest(label=label):
                tracker = self._tracker(checkpoint_path=checkpoint_path, strict=True)
                load_patch = (
                    patch.object(torch, "load", side_effect=load_error)
                    if load_error is not None
                    else patch.object(torch, "load", wraps=torch.load)
                )
                with load_patch:
                    with self.assertRaisesRegex(RuntimeError, expected):
                        tracker._load_pairwise_association_head()

    def test_strict_shadow_audit_rejects_missing_or_unexpected_head_keys(self):
        self._save_checkpoint({"association_heads.heads.0.weight": torch.ones(1)})
        for load_result, expected in [
            ((["heads.0.bias"], []), "missing keys"),
            (([], ["heads.9.weight"]), "unexpected keys"),
        ]:
            with self.subTest(load_result=load_result):
                _StubHead.load_result = load_result
                tracker = self._tracker(
                    checkpoint_path=str(self.checkpoint_path), strict=True
                )
                with self.assertRaisesRegex(RuntimeError, expected):
                    tracker._load_pairwise_association_head()

    def test_non_strict_errors_are_audited_without_blocking_startup(self):
        self._save_checkpoint({"encoder.weight": torch.ones(1)})
        cases = [
            ("empty path", None, None),
            ("missing path", str(self.root / "missing.pt"), None),
            ("load failure", str(self.checkpoint_path), RuntimeError("broken bytes")),
            ("no head keys", str(self.checkpoint_path), None),
        ]
        for label, checkpoint_path, load_error in cases:
            with self.subTest(label=label):
                tracker = self._tracker(checkpoint_path=checkpoint_path)
                load_patch = (
                    patch.object(torch, "load", side_effect=load_error)
                    if load_error is not None
                    else patch.object(torch, "load", wraps=torch.load)
                )
                with load_patch:
                    self.assertIsNone(tracker._load_pairwise_association_head())
                self._assert_one_load_error(tracker)

    def test_non_strict_partial_load_continues_and_records_error(self):
        self._save_checkpoint({"association_heads.heads.0.weight": torch.ones(1)})
        _StubHead.load_result = (["heads.0.bias"], ["heads.9.weight"])
        tracker = self._tracker(checkpoint_path=str(self.checkpoint_path))

        head = tracker._load_pairwise_association_head()

        self.assertIsInstance(head, _StubHead)
        self.assertTrue(head.eval_called)
        self._assert_one_load_error(tracker)

    def test_complete_load_records_no_error_and_does_not_mutate_manifest(self):
        self._save_checkpoint({"association_heads.heads.0.weight": torch.ones(1)})
        manifest = self._runtime_manifest()
        expected_manifest = copy.deepcopy(manifest)
        tracker = self._tracker(
            checkpoint_path="relative/path.pt", runtime_manifest=manifest
        )

        head = tracker._load_pairwise_association_head()

        self.assertIsInstance(head, _StubHead)
        self.assertEqual(manifest, expected_manifest)
        summary = tracker.association_head_audit.to_summary()
        self.assertEqual(summary["audit_error_count"], 0)
        self.assertTrue(summary["decision_evidence_complete"])

    def test_loader_rejects_incompatible_filter_and_history_contracts(self):
        for field, value in [
            ("filter_mode", "pure_dekf"),
            ("history_source", "det"),
        ]:
            with self.subTest(field=field):
                self._save_checkpoint(
                    self._real_head_state(),
                    self._runtime_contract(**{field: value}),
                )

                strict_tracker = self._tracker(
                    checkpoint_path=str(self.checkpoint_path),
                    strict=True,
                    head_type=ClassConditionedAssociationHeadBank,
                )
                with self.assertRaisesRegex(RuntimeError, field):
                    strict_tracker._load_pairwise_association_head()

                non_strict_tracker = self._tracker(
                    checkpoint_path=str(self.checkpoint_path),
                    head_type=ClassConditionedAssociationHeadBank,
                )
                self.assertIsNone(
                    non_strict_tracker._load_pairwise_association_head()
                )
                self._assert_one_load_error(non_strict_tracker)

    def test_loader_rejects_missing_contract_even_when_audit_is_disabled(self):
        self._save_checkpoint(self._real_head_state(), omit_contract=True)
        tracker = self._tracker(
            checkpoint_path=str(self.checkpoint_path),
            audit_enabled=False,
            head_type=ClassConditionedAssociationHeadBank,
        )

        self.assertIsNone(tracker._load_pairwise_association_head())

    def test_runtime_manifest_checkpoint_fields_are_verified_before_load(self):
        self._save_checkpoint(self._real_head_state())
        manifest = self._runtime_manifest()
        cases = [
            ("association_checkpoint_path", f"{self.root}/./association.pt", "path"),
            ("association_checkpoint_sha256", "0" * 64, "sha256"),
            ("association_checkpoint_loadable", False, "loadable"),
            (
                "association_checkpoint_head_key_count",
                manifest["association_checkpoint_head_key_count"] + 1,
                "head_key_count",
            ),
            ("association_checkpoint_validation_error", "stale error", "validation_error"),
            (
                "association_checkpoint_runtime_contract_expected",
                self._runtime_contract(filter_mode="pure_dekf"),
                "runtime_contract_expected",
            ),
            (
                "association_checkpoint_runtime_contract_actual",
                self._runtime_contract(history_source="det"),
                "runtime_contract_actual",
            ),
            (
                "association_checkpoint_runtime_contract_validation_status",
                "invalid",
                "runtime_contract_validation_status",
            ),
        ]

        for field, value, expected in cases:
            with self.subTest(field=field):
                stale_manifest = copy.deepcopy(manifest)
                stale_manifest[field] = value
                tracker = self._tracker(
                    checkpoint_path=str(self.checkpoint_path),
                    strict=True,
                    runtime_manifest=stale_manifest,
                    head_type=ClassConditionedAssociationHeadBank,
                )

                with self.assertRaisesRegex(RuntimeError, expected):
                    tracker._load_pairwise_association_head()

    def test_real_head_checkpoint_complete_missing_and_unexpected_states(self):
        complete_state = self._real_head_state()
        missing_key = next(iter(complete_state))
        cases = [
            ("complete", complete_state, None),
            (
                "missing",
                {key: value for key, value in complete_state.items() if key != missing_key},
                "missing keys",
            ),
            (
                "unexpected",
                {
                    **complete_state,
                    "association_heads.unexpected.weight": torch.ones(1),
                },
                "unexpected keys",
            ),
        ]

        for label, state, expected_error in cases:
            with self.subTest(label=label):
                self._save_checkpoint(state)
                manifest = self._runtime_manifest()
                tracker = self._tracker(
                    checkpoint_path=str(self.checkpoint_path),
                    strict=True,
                    runtime_manifest=manifest,
                    head_type=ClassConditionedAssociationHeadBank,
                )

                if expected_error is None:
                    head = tracker._load_pairwise_association_head()
                    self.assertIsInstance(head, ClassConditionedAssociationHeadBank)
                    self.assertEqual(
                        tracker.association_head_audit.to_summary()["audit_error_count"],
                        0,
                    )
                else:
                    with self.assertRaisesRegex(RuntimeError, expected_error):
                        tracker._load_pairwise_association_head()

    def test_replaced_real_checkpoint_fails_strict_and_marks_non_strict_incomplete(self):
        original_state = self._real_head_state()
        self._save_checkpoint(original_state)
        manifest = self._runtime_manifest()
        expected_manifest = copy.deepcopy(manifest)

        replacement_state = {
            key: value + 1.0 for key, value in original_state.items()
        }
        self._save_checkpoint(replacement_state)

        strict_tracker = self._tracker(
            checkpoint_path=str(self.checkpoint_path),
            strict=True,
            runtime_manifest=manifest,
            head_type=ClassConditionedAssociationHeadBank,
        )
        with patch.object(torch, "load", wraps=torch.load) as load_mock:
            with self.assertRaisesRegex(RuntimeError, "sha256"):
                strict_tracker._load_pairwise_association_head()
        self.assertEqual(load_mock.call_count, 0)

        non_strict_tracker = self._tracker(
            checkpoint_path=str(self.checkpoint_path),
            runtime_manifest=manifest,
            head_type=ClassConditionedAssociationHeadBank,
        )
        head = non_strict_tracker._load_pairwise_association_head()

        self.assertIsInstance(head, ClassConditionedAssociationHeadBank)
        self._assert_one_load_error(non_strict_tracker)
        self.assertEqual(manifest, expected_manifest)

    def test_invalid_checkpoint_architecture_matches_manifest_and_loader_failure(self):
        self._save_checkpoint(self._real_head_state())
        checkpoint = torch.load(self.checkpoint_path, map_location="cpu")
        checkpoint["settings"] = {"hidden_dims": "bad", "dropout": 0.0}
        torch.save(checkpoint, self.checkpoint_path)
        head_cfg = {"NUM_CLASSES": 1}
        manifest = self._runtime_manifest(head_cfg=head_cfg)

        self.assertIn(
            "invalid checkpoint hidden_dims",
            manifest["association_checkpoint_validation_error"],
        )
        tracker = self._tracker(
            checkpoint_path=str(self.checkpoint_path),
            strict=True,
            runtime_manifest=manifest,
            head_type=ClassConditionedAssociationHeadBank,
            include_architecture_config=False,
        )

        with self.assertRaisesRegex(RuntimeError, "invalid checkpoint hidden_dims"):
            tracker._load_pairwise_association_head()

    def test_actual_loader_rejects_falsey_non_mapping_checkpoint_settings(self):
        torch.save(
            {
                "model_state_dict": self._real_head_state(),
                "settings": [],
                "runtime_contract": self._runtime_contract(),
            },
            self.checkpoint_path,
        )
        tracker = self._tracker(
            checkpoint_path=str(self.checkpoint_path),
            audit_enabled=False,
            head_type=ClassConditionedAssociationHeadBank,
        )

        self.assertIsNone(tracker._load_pairwise_association_head())

    def test_disabled_audit_or_non_shadow_mode_keeps_compatibility_behavior(self):
        for audit_enabled, shadow_mode in [(False, True), (True, False)]:
            with self.subTest(audit_enabled=audit_enabled, shadow_mode=shadow_mode):
                tracker = self._tracker(
                    checkpoint_path=None,
                    audit_enabled=audit_enabled,
                    shadow_mode=shadow_mode,
                    strict=True,
                )
                self.assertIsNone(tracker._load_pairwise_association_head())
                if tracker.association_head_audit is not None:
                    self.assertEqual(
                        tracker.association_head_audit.to_summary()["audit_error_count"],
                        0,
                    )

    def test_disabled_head_never_enforces_strict_audit(self):
        tracker = self._tracker(
            checkpoint_path=None, strict=True, head_enabled=False
        )

        self.assertIsNone(tracker._load_pairwise_association_head())
        self.assertEqual(
            tracker.association_head_audit.to_summary()["audit_error_count"], 0
        )

    def test_accumulator_is_created_before_checkpoint_load(self):
        tree = ast.parse((ROOT / "tracker/base_tracker.py").read_text(encoding="utf-8"))
        tracker_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "Base3DTracker"
        )
        init = next(
            node
            for node in tracker_class.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        )
        accumulator_line = min(
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "AssociationHeadAuditAccumulator"
        )
        load_line = min(
            node.lineno
            for node in ast.walk(init)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "_load_pairwise_association_head"
        )

        self.assertLess(accumulator_line, load_line)


if __name__ == "__main__":
    unittest.main()
