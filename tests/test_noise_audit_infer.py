import ast
import json
import pathlib
import tempfile
import types
import unittest
from typing import Dict, List, Optional

from kalmanfilter.noise_audit import NoiseAuditAccumulator


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _load_module_functions(module_path, function_names, extra_namespace=None):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = dict(extra_namespace or {})
    loaded = {}

    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name in function_names:
            fn_source = ast.get_source_segment(source, node)
            exec(fn_source, namespace)
            loaded[node.name] = namespace[node.name]

    missing = [name for name in function_names if name not in loaded]
    if missing:
        raise AssertionError(f"Missing functions in {module_path}: {missing}")
    return loaded


def _load_class_methods(module_path, class_name, method_names, extra_namespace=None):
    source = module_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(module_path))
    namespace = dict(extra_namespace or {})
    loaded = {}

    for node in tree.body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for child in node.body:
            if isinstance(child, ast.FunctionDef) and child.name in method_names:
                fn_source = ast.get_source_segment(source, child)
                exec(fn_source, namespace)
                loaded[child.name] = namespace[child.name]
        break

    missing = [name for name in method_names if name not in loaded]
    if missing:
        raise AssertionError(f"Missing methods in {module_path}:{class_name}: {missing}")
    return loaded


class _FakeScalar:
    def __init__(self, value):
        self._value = value

    def item(self):
        return self._value


class _FakeVector:
    def __init__(self, values):
        self._values = list(values)

    def __getitem__(self, index):
        return _FakeScalar(self._values[index])

    def __len__(self):
        return len(self._values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def tolist(self):
        return list(self._values)


class _FakeMatrix:
    def __init__(self, rows):
        self._rows = [list(row) for row in rows]

    def to(self, dtype=None):
        return self

    def sum(self, dim):
        if dim != 1:
            raise AssertionError(f"Unexpected dim: {dim}")
        return _FakeVector(sum(1 for value in row if value) for row in self._rows)


class _FakeTorch:
    Tensor = object
    int64 = "int64"


class _FakeBBox:
    def __init__(self, category, is_fake=False):
        self.category = category
        self.is_fake = is_fake


class _FakeTraj:
    def __init__(self, category, is_fake=False):
        self.bboxes = [_FakeBBox(category=category, is_fake=is_fake)]


class NoiseAuditInferTest(unittest.TestCase):
    def _load_tracker_helpers(self):
        mamba_helpers = _load_module_functions(
            REPO_ROOT / "kalmanfilter" / "mamba_adaptive_kf.py",
            ["build_noise_audit_samples"],
        )
        tracker_helpers = _load_module_functions(
            REPO_ROOT / "tracker" / "base_tracker.py",
            ["_build_noise_audit_cfg", "_noise_audit_enabled"],
        )
        tracker_methods = _load_class_methods(
            REPO_ROOT / "tracker" / "base_tracker.py",
            "Base3DTracker",
            [
                "_record_noise_audit_sample",
                "_stage_noise_audit_samples",
                "_flush_noise_audit_samples",
                "dump_noise_audit_if_needed",
            ],
            extra_namespace={
                "Dict": Dict,
                "List": List,
                "Optional": Optional,
                "Trajectory": _FakeTraj,
                "torch": _FakeTorch,
                "build_noise_audit_samples": mamba_helpers["build_noise_audit_samples"],
            },
        )
        return tracker_helpers, tracker_methods

    def test_disabled_inference_audit_lifecycle_is_noop(self):
        tracker_helpers, tracker_methods = self._load_tracker_helpers()
        build_noise_audit_cfg = tracker_helpers["_build_noise_audit_cfg"]
        noise_audit_enabled = tracker_helpers["_noise_audit_enabled"]

        self.assertEqual(build_noise_audit_cfg(None), {})
        self.assertFalse(noise_audit_enabled({"AUDIT": {"NOISE_AUDIT": {"ENABLED": False}}}))

        tracker = types.SimpleNamespace(
            noise_audit=None,
            noise_audit_cfg={"INFER_OUTPUT_PATH": "unused.json"},
            filter_mode="fusion",
            _noise_audit_pending={"stale": True},
        )

        tracker_methods["_stage_noise_audit_samples"](
            tracker,
            track_ids=[10],
            trajs=[_FakeTraj("car")],
            class_ids=_FakeVector([0]),
            history_mask=_FakeMatrix([[True, False, True]]),
            mamba_out={
                "noise_audit_values": {"q_pos": _FakeVector([1.0]), "r_pos": _FakeVector([2.0]), "r_siz": _FakeVector([3.0]), "r_ori": _FakeVector([4.0])},
                "noise_audit_priors": {"q_pos": _FakeVector([1.0]), "r_pos": _FakeVector([2.0]), "r_siz": _FakeVector([3.0]), "r_ori": _FakeVector([4.0])},
            },
        )
        self.assertIsNone(tracker._noise_audit_pending)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "infer_noise_audit.json"
            tracker.noise_audit_cfg["INFER_OUTPUT_PATH"] = str(out)
            tracker_methods["dump_noise_audit_if_needed"](tracker)
            self.assertFalse(out.exists())

    def test_inference_audit_lifecycle_stages_flushes_and_dumps_summary(self):
        _, tracker_methods = self._load_tracker_helpers()

        tracker = types.SimpleNamespace(
            noise_audit=NoiseAuditAccumulator(),
            noise_audit_cfg={},
            filter_mode="pure_dekf",
            _noise_audit_pending=None,
            all_trajs={
                10: _FakeTraj("car", is_fake=False),
            },
            all_dead_trajs={
                20: _FakeTraj("truck", is_fake=True),
            },
        )
        tracker._record_noise_audit_sample = types.MethodType(
            tracker_methods["_record_noise_audit_sample"], tracker
        )

        tracker_methods["_stage_noise_audit_samples"](
            tracker,
            track_ids=[10, 20],
            trajs=[_FakeTraj("car"), _FakeTraj("truck")],
            class_ids=_FakeVector([0, 6]),
            history_mask=_FakeMatrix(
                [
                    [True, True, False, False],
                    [True, True, True, True],
                ]
            ),
            mamba_out={
                "noise_audit_values": {
                    "q_pos": _FakeVector([1.0, 10.0]),
                    "r_pos": _FakeVector([2.0, 20.0]),
                    "r_siz": _FakeVector([3.0, 30.0]),
                    "r_ori": _FakeVector([4.0, 40.0]),
                },
                "noise_audit_priors": {
                    "q_pos": _FakeVector([0.5, 5.0]),
                    "r_pos": _FakeVector([1.0, 10.0]),
                    "r_siz": _FakeVector([1.5, 15.0]),
                    "r_ori": _FakeVector([2.0, 20.0]),
                },
            },
        )

        self.assertEqual(tracker._noise_audit_pending["track_ids"], [10, 20])
        self.assertEqual(tracker._noise_audit_pending["history_lens"], [2, 4])
        self.assertEqual(tracker._noise_audit_pending["class_names"], ["car", "truck"])

        tracker_methods["_flush_noise_audit_samples"](tracker)
        self.assertIsNone(tracker._noise_audit_pending)

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "infer_noise_audit.json"
            tracker.noise_audit_cfg["INFER_OUTPUT_PATH"] = str(out)
            tracker_methods["dump_noise_audit_if_needed"](tracker)
            payload = json.loads(out.read_text(encoding="utf-8"))

        buckets = payload["buckets"]
        self.assertEqual(len(buckets), 2)

        matched_bucket = next(bucket for bucket in buckets if bucket["state"] == "matched")
        unmatched_bucket = next(bucket for bucket in buckets if bucket["state"] == "unmatched")

        self.assertEqual(matched_bucket["split"], "infer")
        self.assertEqual(matched_bucket["mode"], "pure_dekf")
        self.assertEqual(matched_bucket["class_id"], 0)
        self.assertEqual(matched_bucket["class_name"], "car")
        self.assertEqual(matched_bucket["history_len"], 2)
        self.assertAlmostEqual(matched_bucket["families"]["q_pos"]["median"], 1.0)
        self.assertAlmostEqual(matched_bucket["ratios"]["q_pos"]["median"], 2.0)

        self.assertEqual(unmatched_bucket["split"], "infer")
        self.assertEqual(unmatched_bucket["mode"], "pure_dekf")
        self.assertEqual(unmatched_bucket["class_id"], 6)
        self.assertEqual(unmatched_bucket["class_name"], "truck")
        self.assertEqual(unmatched_bucket["history_len"], 4)
        self.assertIn("families", unmatched_bucket)
        self.assertIn("ratios", unmatched_bucket)
        self.assertAlmostEqual(unmatched_bucket["families"]["r_ori"]["median"], 40.0)
        self.assertAlmostEqual(unmatched_bucket["ratios"]["r_ori"]["median"], 2.0)

    def test_main_helpers_collect_and_merge_run_level_inference_audit(self):
        main_helpers = _load_module_functions(
            REPO_ROOT / "main.py",
            ["_build_noise_audit_cfg", "_collect_scene_inference_audit_state", "_write_merged_infer_noise_audit"],
            extra_namespace={"NoiseAuditAccumulator": NoiseAuditAccumulator},
        )
        build_noise_audit_cfg = main_helpers["_build_noise_audit_cfg"]
        collect_scene_inference_audit_state = main_helpers["_collect_scene_inference_audit_state"]
        write_merged_infer_noise_audit = main_helpers["_write_merged_infer_noise_audit"]

        class _DummyTracker:
            def __init__(self, state):
                self._state = state

            def export_noise_audit_state(self):
                return self._state

        self.assertEqual(build_noise_audit_cfg(None), {})

        scene_states = {}
        first = NoiseAuditAccumulator()
        first.add_sample(
            split="infer",
            mode="fusion",
            class_id=0,
            class_name="car",
            state="matched",
            history_len=2,
            families={"q_pos": 1.0, "r_pos": 2.0, "r_siz": 3.0, "r_ori": 4.0},
            prior_families={"q_pos": 0.5, "r_pos": 1.0, "r_siz": 1.5, "r_ori": 2.0},
        )
        second = NoiseAuditAccumulator()
        second.add_sample(
            split="infer",
            mode="fusion",
            class_id=0,
            class_name="car",
            state="matched",
            history_len=2,
            families={"q_pos": 5.0, "r_pos": 6.0, "r_siz": 7.0, "r_ori": 8.0},
            prior_families={"q_pos": 2.5, "r_pos": 3.0, "r_siz": 3.5, "r_ori": 4.0},
        )

        cfg_disabled = {"AUDIT": {"NOISE_AUDIT": {"ENABLED": False}}}
        collect_scene_inference_audit_state(
            "scene-disabled",
            _DummyTracker(first.export_state()),
            cfg_disabled,
            scene_states,
        )
        self.assertEqual(scene_states, {})

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "infer_noise_audit.json"
            cfg = {"AUDIT": {"NOISE_AUDIT": {"ENABLED": True, "INFER_OUTPUT_PATH": str(out)}}}
            collect_scene_inference_audit_state(
                "scene-1",
                _DummyTracker(first.export_state()),
                cfg,
                scene_states,
            )
            collect_scene_inference_audit_state(
                "scene-2",
                _DummyTracker(second.export_state()),
                cfg,
                scene_states,
            )
            write_merged_infer_noise_audit(cfg, scene_states)
            payload = json.loads(out.read_text(encoding="utf-8"))

        bucket = payload["buckets"][0]
        self.assertEqual(sorted(scene_states.keys()), ["scene-1", "scene-2"])
        self.assertEqual(bucket["count"], 2)
        self.assertAlmostEqual(bucket["families"]["q_pos"]["median"], 3.0)
        self.assertAlmostEqual(bucket["families"]["r_ori"]["max"], 8.0)
        self.assertAlmostEqual(bucket["ratios"]["q_pos"]["median"], 2.0)


if __name__ == "__main__":
    unittest.main()
