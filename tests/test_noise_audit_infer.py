import ast
import builtins
import json
import pathlib
import tempfile
import types
import unittest
from typing import Dict, List, Optional, Tuple

import numpy as np

from kalmanfilter.bounded_residual import infer_state_bucket
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


class _FakeNoGrad:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _PredictTorch:
    long = "long"
    float32 = "float32"
    bool = "bool"

    @staticmethod
    def tensor(values, dtype=None, device=None):
        return list(values)

    @staticmethod
    def ones(size, dtype=None, device=None):
        return [True] * int(size)

    @staticmethod
    def zeros(shape, dtype=None):
        return {"shape": tuple(shape), "dtype": dtype}

    @staticmethod
    def no_grad():
        return _FakeNoGrad()


class _FakeTraceValue:
    def __init__(self, value):
        self.value = float(value)

    def sum(self, dim=None, keepdim=False):
        return self

    def unsqueeze(self, dim):
        return self

    def view(self, *shape):
        return self

    def __truediv__(self, other):
        return _FakeTraceValue(self.value / _coerce_trace_value(other))

    def __rtruediv__(self, other):
        return _FakeTraceValue(_coerce_trace_value(other) / self.value)

    def __mul__(self, other):
        if isinstance(other, _FakeCovariance):
            return other * self
        return _FakeTraceValue(self.value * _coerce_trace_value(other))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return _FakeTraceValue(self.value + _coerce_trace_value(other))

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        return _FakeTraceValue(self.value - _coerce_trace_value(other))

    def __rsub__(self, other):
        return _FakeTraceValue(_coerce_trace_value(other) - self.value)


def _coerce_trace_value(value):
    if isinstance(value, _FakeTraceValue):
        return value.value
    return float(value)


class _FakeCovariance:
    def __init__(self, trace_value, label):
        self.trace_value = float(trace_value)
        self.label = label
        self.device = "cpu"
        self.dtype = "float32"

    def diagonal(self, dim1=-2, dim2=-1):
        return _FakeTraceValue(self.trace_value)

    def __mul__(self, other):
        return _FakeCovariance(self.trace_value * _coerce_trace_value(other), f"{self.label}*")

    def __rmul__(self, other):
        return self.__mul__(other)

    def __add__(self, other):
        return _FakeCovariance(self.trace_value + other.trace_value, f"{self.label}+{other.label}")


class _FusionTorch:
    @staticmethod
    def tensor(value, device=None, dtype=None):
        if isinstance(value, list):
            if len(value) != 1:
                raise AssertionError(f"Unexpected list input: {value}")
            value = value[0]
        return _FakeTraceValue(value)

    @staticmethod
    def clamp(value, min=None, max=None):
        scalar = _coerce_trace_value(value)
        if min is not None:
            scalar = builtins.max(_coerce_trace_value(min), scalar)
        if max is not None:
            scalar = builtins.min(_coerce_trace_value(max), scalar)
        return _FakeTraceValue(scalar)


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

    def test_predict_before_associate_passes_state_buckets_into_predict_with_mamba(self):
        predict_before_associate = _load_class_methods(
            REPO_ROOT / "tracker" / "base_tracker.py",
            "Base3DTracker",
            ["predict_before_associate"],
            extra_namespace={
                "Dict": Dict,
                "List": List,
                "Optional": Optional,
                "Tuple": Tuple,
                "Trajectory": _FakeTraj,
                "np": np,
                "torch": _PredictTorch,
                "infer_state_bucket": infer_state_bucket,
            },
        )["predict_before_associate"]

        def _make_traj(track_id, category_num, unmatch_length, xy):
            bbox = types.SimpleNamespace(global_xyz=[xy[0], xy[1], 0.0], category="car")
            return types.SimpleNamespace(
                track_id=track_id,
                category_num=category_num,
                unmatch_length=unmatch_length,
                bboxes=[bbox],
                predict=lambda: None,
            )

        traj_1 = _make_traj(track_id=1, category_num=0, unmatch_length=0, xy=(3.0, 4.0))
        traj_2 = _make_traj(track_id=2, category_num=5, unmatch_length=2, xy=(0.0, 5.0))
        captured = {}

        def _fake_predict_with_mamba(*args, **kwargs):
            captured["state_buckets"] = kwargs["state_buckets"]
            raise RuntimeError("stop after capture")

        tracker = types.SimpleNamespace(
            device="cpu",
            filter_mode="mamba",
            _noise_audit_pending=None,
            all_trajs={1: traj_1, 2: traj_2},
            _extract_track_history=lambda trajs: (
                _PredictTorch.zeros((len(trajs), 3, 12), dtype=_PredictTorch.float32),
                _PredictTorch.ones(len(trajs) * 3, dtype=_PredictTorch.bool),
                _PredictTorch.ones(len(trajs) * 3, dtype=_PredictTorch.bool),
            ),
            _batch_kf_states=lambda track_ids: (
                _PredictTorch.zeros((len(track_ids), 6, 1), dtype=_PredictTorch.float32),
                _PredictTorch.zeros((len(track_ids), 6, 6), dtype=_PredictTorch.float32),
                _PredictTorch.zeros((len(track_ids), 3, 1), dtype=_PredictTorch.float32),
                _PredictTorch.zeros((len(track_ids), 3, 3), dtype=_PredictTorch.float32),
                _PredictTorch.zeros((len(track_ids), 2, 1), dtype=_PredictTorch.float32),
                _PredictTorch.zeros((len(track_ids), 2, 2), dtype=_PredictTorch.float32),
            ),
            _unbatch_kf_states=lambda *args, **kwargs: None,
            _write_predicted_state_to_bbox=lambda *args, **kwargs: None,
            _stage_noise_audit_samples=lambda *args, **kwargs: None,
            mamba_ekf=types.SimpleNamespace(
                kf=types.SimpleNamespace(
                    pos_filter=types.SimpleNamespace(B=0),
                    siz_filter=types.SimpleNamespace(B=0),
                    ori_filter=types.SimpleNamespace(B=0),
                    init_states=lambda *args, **kwargs: None,
                ),
                predict_with_mamba=_fake_predict_with_mamba,
            ),
        )
        tracker.mamba_ekf.kf.B = 0

        with self.assertRaisesRegex(RuntimeError, "stop after capture"):
            predict_before_associate(tracker, [tracker.all_trajs[1], tracker.all_trajs[2]], delta_t=0.5)

        self.assertEqual(captured["state_buckets"], ["matched", "unmatched"])

    def _load_predict_with_mamba(self, *, torch_namespace, apply_bounded_impl):
        return _load_class_methods(
            REPO_ROOT / "kalmanfilter" / "mamba_adaptive_kf.py",
            "MambaDecoupledEKF",
            ["predict_with_mamba"],
            extra_namespace={
                "Dict": Dict,
                "Optional": Optional,
                "Tuple": Tuple,
                "Tensor": object,
                "torch": torch_namespace,
                "_covariance_trace_batch": lambda cov: cov.trace_value if cov is not None else None,
                "apply_bounded_residuals": apply_bounded_impl,
            },
        )["predict_with_mamba"]

    def test_predict_with_mamba_mamba_mode_applies_closure_to_raw_outputs_and_preserves_audit_contract(self):
        captured = {}
        closure_cfg = {"ENABLED": True, "PROFILES": {"stable_large": {"matched": {"q_pos": [0.8, 1.8]}}}}
        class_ids = [0]
        state_buckets = ["matched"]

        def _fake_apply_bounded_residuals(**kwargs):
            captured["called"] = True
            captured["raw_q_pos_trace"] = kwargs["raw_tensors"]["Q_pos"].trace_value
            captured["prior_q_pos_trace"] = kwargs["prior_tensors"]["Q_pos"].trace_value
            captured["prior_r_pos_trace"] = kwargs["prior_tensors"]["R_pos"].trace_value
            captured["prior_r_siz_trace"] = kwargs["prior_tensors"]["R_siz"].trace_value
            captured["prior_r_ori_trace"] = kwargs["prior_tensors"]["R_ori"].trace_value
            captured["class_ids"] = kwargs["class_ids"]
            captured["state_buckets"] = kwargs["state_buckets"]
            captured["closure_cfg"] = kwargs["closure_cfg"]
            return {
                **kwargs["raw_tensors"],
                "Q_pos": _FakeCovariance(2.0, "bounded_q"),
                "R_pos": _FakeCovariance(3.0, "bounded_rp"),
                "R_siz": _FakeCovariance(4.0, "bounded_rs"),
                "R_ori": _FakeCovariance(5.0, "bounded_ro"),
            }

        predict_with_mamba = self._load_predict_with_mamba(
            torch_namespace=_PredictTorch,
            apply_bounded_impl=_fake_apply_bounded_residuals,
        )

        class _TrackHistory:
            dtype = "float32"

            def size(self, dim):
                self_dim = 1
                if dim != 0:
                    raise AssertionError(f"Unexpected dim: {dim}")
                return self_dim

        tracker = types.SimpleNamespace(
            mamba=lambda *args, **kwargs: {
                "Q_pos": _FakeCovariance(10.0, "raw_q"),
                "Q_siz": _FakeCovariance(20.0, "raw_qs"),
                "Q_ori": _FakeCovariance(30.0, "raw_qo"),
                "R_pos": _FakeCovariance(40.0, "raw_rp"),
                "R_siz": _FakeCovariance(50.0, "raw_rs"),
                "R_ori": _FakeCovariance(60.0, "raw_ro"),
                "embedding": "embed",
            },
            base_noise_cfg={"MAMBA_CLOSURE": closure_cfg},
            _get_base_noise=lambda bsize, dtype, class_ids: (
                _FakeCovariance(1.0, "prior_q"),
                _FakeCovariance(2.0, "prior_rp"),
                _FakeCovariance(3.0, "prior_qs"),
                _FakeCovariance(4.0, "prior_rs"),
                _FakeCovariance(5.0, "prior_qo"),
                _FakeCovariance(6.0, "prior_ro"),
            ),
            kf=types.SimpleNamespace(
                predict=lambda *args, **kwargs: ("px", "pP", "sx", "sP", "ox", "oP")
            ),
        )

        mamba_out, *_ = predict_with_mamba(
            tracker,
            _TrackHistory(),
            delta_t=0.5,
            class_ids=class_ids,
            mode="mamba",
            state_buckets=state_buckets,
        )

        self.assertTrue(captured["called"])
        self.assertEqual(captured["raw_q_pos_trace"], 10.0)
        self.assertEqual(captured["prior_q_pos_trace"], 1.0)
        self.assertEqual(captured["prior_r_pos_trace"], 2.0)
        self.assertEqual(captured["prior_r_siz_trace"], 4.0)
        self.assertEqual(captured["prior_r_ori_trace"], 6.0)
        self.assertIs(captured["class_ids"], class_ids)
        self.assertIs(captured["state_buckets"], state_buckets)
        self.assertIs(captured["closure_cfg"], closure_cfg)
        self.assertEqual(mamba_out["Q_pos"].trace_value, 2.0)
        self.assertEqual(mamba_out["R_pos"].trace_value, 3.0)
        self.assertEqual(mamba_out["R_siz"].trace_value, 4.0)
        self.assertEqual(mamba_out["R_ori"].trace_value, 5.0)
        self.assertEqual(mamba_out["noise_audit_values"]["q_pos"], 2.0)
        self.assertEqual(mamba_out["noise_audit_values"]["r_pos"], 3.0)
        self.assertEqual(mamba_out["noise_audit_values"]["r_siz"], 4.0)
        self.assertEqual(mamba_out["noise_audit_values"]["r_ori"], 5.0)
        self.assertEqual(mamba_out["noise_audit_priors"]["q_pos"], 1.0)
        self.assertEqual(mamba_out["noise_audit_priors"]["r_pos"], 2.0)
        self.assertEqual(mamba_out["noise_audit_priors"]["r_siz"], 4.0)
        self.assertEqual(mamba_out["noise_audit_priors"]["r_ori"], 6.0)
        self.assertNotIn("noise_audit_raw_values", mamba_out)

    def test_predict_with_mamba_pure_dekf_mode_skips_closure_and_keeps_prior_audit_semantics(self):
        def _forbidden_apply_bounded_residuals(**kwargs):
            raise AssertionError("closure must not run in pure_dekf mode")

        predict_with_mamba = self._load_predict_with_mamba(
            torch_namespace=_PredictTorch,
            apply_bounded_impl=_forbidden_apply_bounded_residuals,
        )

        class _TrackHistory:
            dtype = "float32"

            def size(self, dim):
                if dim != 0:
                    raise AssertionError(f"Unexpected dim: {dim}")
                return 1

        tracker = types.SimpleNamespace(
            mamba=lambda *args, **kwargs: {
                "Q_pos": _FakeCovariance(10.0, "raw_q"),
                "Q_siz": _FakeCovariance(20.0, "raw_qs"),
                "Q_ori": _FakeCovariance(30.0, "raw_qo"),
                "R_pos": _FakeCovariance(40.0, "raw_rp"),
                "R_siz": _FakeCovariance(50.0, "raw_rs"),
                "R_ori": _FakeCovariance(60.0, "raw_ro"),
                "embedding": "embed",
            },
            base_noise_cfg={"MAMBA_CLOSURE": {"ENABLED": True}},
            _get_base_noise=lambda bsize, dtype, class_ids: (
                _FakeCovariance(1.0, "prior_q"),
                _FakeCovariance(2.0, "prior_rp"),
                _FakeCovariance(3.0, "prior_qs"),
                _FakeCovariance(4.0, "prior_rs"),
                _FakeCovariance(5.0, "prior_qo"),
                _FakeCovariance(6.0, "prior_ro"),
            ),
            kf=types.SimpleNamespace(
                predict=lambda *args, **kwargs: ("px", "pP", "sx", "sP", "ox", "oP")
            ),
        )

        mamba_out, *_ = predict_with_mamba(
            tracker,
            _TrackHistory(),
            delta_t=0.5,
            class_ids=[0],
            mode="pure_dekf",
            state_buckets=["matched"],
        )

        self.assertEqual(mamba_out["Q_pos"].trace_value, 1.0)
        self.assertEqual(mamba_out["noise_audit_values"]["q_pos"], 1.0)
        self.assertEqual(mamba_out["noise_audit_priors"]["q_pos"], 1.0)

    def test_predict_with_mamba_fusion_mode_skips_closure(self):
        def _forbidden_apply_bounded_residuals(**kwargs):
            raise AssertionError("closure must not run in fusion mode")

        predict_with_mamba = self._load_predict_with_mamba(
            torch_namespace=_FusionTorch,
            apply_bounded_impl=_forbidden_apply_bounded_residuals,
        )

        class _TrackHistory:
            dtype = "float32"

            def size(self, dim):
                if dim != 0:
                    raise AssertionError(f"Unexpected dim: {dim}")
                return 1

        tracker = types.SimpleNamespace(
            mamba=lambda *args, **kwargs: {
                "Q_pos": _FakeCovariance(10.0, "raw_q"),
                "Q_siz": _FakeCovariance(20.0, "raw_qs"),
                "Q_ori": _FakeCovariance(30.0, "raw_qo"),
                "R_pos": _FakeCovariance(40.0, "raw_rp"),
                "R_siz": _FakeCovariance(50.0, "raw_rs"),
                "R_ori": _FakeCovariance(60.0, "raw_ro"),
                "embedding": "embed",
            },
            base_noise_cfg={
                "MAMBA_CLOSURE": {"ENABLED": True},
                "FUSION": {"STRICT_RATIO": {0: 2.0}},
            },
            _get_base_noise=lambda bsize, dtype, class_ids: (
                _FakeCovariance(1.0, "prior_q"),
                _FakeCovariance(2.0, "prior_rp"),
                _FakeCovariance(3.0, "prior_qs"),
                _FakeCovariance(4.0, "prior_rs"),
                _FakeCovariance(5.0, "prior_qo"),
                _FakeCovariance(6.0, "prior_ro"),
            ),
            kf=types.SimpleNamespace(
                predict=lambda *args, **kwargs: ("px", "pP", "sx", "sP", "ox", "oP")
            ),
        )

        mamba_out, *_ = predict_with_mamba(
            tracker,
            _TrackHistory(),
            delta_t=0.5,
            class_ids=[0],
            mode="fusion",
            state_buckets=["matched"],
        )

        self.assertAlmostEqual(mamba_out["noise_audit_values"]["q_pos"], 1.0)
        self.assertAlmostEqual(mamba_out["noise_audit_priors"]["q_pos"], 1.0)

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
