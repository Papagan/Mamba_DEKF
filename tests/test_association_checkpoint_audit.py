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


def _load_tracker_type():
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
        "ClassConditionedAssociationHeadBank": _StubHead,
        "Dict": dict,
        "Mapping": Mapping,
        "Optional": Optional,
        "Tuple": Tuple,
        "os": os,
        "torch": torch,
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
    ):
        tracker_type = _load_tracker_type()
        tracker = tracker_type()
        tracker.cfg = {
            "MAMBA": {"NUM_CLASSES": 1},
            "MAMBA_ASSOCIATION_HEAD": {
                "ENABLED": head_enabled,
                "CHECKPOINT_PATH": checkpoint_path,
                "HIDDEN_DIMS": [2],
                "DROPOUT": 0.0,
                "NUM_CLASSES": 1,
            },
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

    def _save_checkpoint(self, state):
        torch.save({"model_state_dict": state}, self.checkpoint_path)

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
        manifest = {
            "association_checkpoint_path": str(self.checkpoint_path.resolve()),
            "association_checkpoint_loadable": True,
            "association_checkpoint_head_key_count": 1,
            "association_checkpoint_validation_error": None,
        }
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
