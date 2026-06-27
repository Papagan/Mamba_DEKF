import ast
import json
import math
import pathlib
import tempfile
import unittest

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from kalmanfilter.noise_audit import NoiseAuditAccumulator
from kalmanfilter.bounded_residual import get_family_ratio_bounds


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TRACKING_CAT_ID_MAP = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "motorcycle": 3,
    "bus": 4,
    "trailer": 5,
    "truck": 6,
}


class _TorchStub:
    Tensor = object


class _FunctionalStub:
    @staticmethod
    def relu(value):
        return _ArrayTensor(np.maximum(np.asarray(value), 0.0))


class _ArrayTensor:
    __array_priority__ = 1000

    def __init__(self, value):
        self._value = np.asarray(value, dtype=np.float64)

    def squeeze(self, axis=None):
        return _ArrayTensor(np.squeeze(self._value, axis=axis))

    def pow(self, exponent):
        return _ArrayTensor(np.power(self._value, exponent))

    def mean(self):
        return _ArrayTensor(np.asarray(self._value.mean(), dtype=np.float64))

    def sum(self):
        return _ArrayTensor(np.asarray(self._value.sum(), dtype=np.float64))

    def item(self):
        return float(np.asarray(self._value).reshape(-1)[0])

    def tolist(self):
        return self._value.tolist()

    def __float__(self):
        return self.item()

    def __array__(self, dtype=None):
        return np.asarray(self._value, dtype=dtype)

    def __len__(self):
        return len(self._value)

    @property
    def shape(self):
        return self._value.shape

    def __add__(self, other):
        return _ArrayTensor(self._value + np.asarray(other))

    def __radd__(self, other):
        return _ArrayTensor(np.asarray(other) + self._value)

    def __sub__(self, other):
        return _ArrayTensor(self._value - np.asarray(other))

    def __rsub__(self, other):
        return _ArrayTensor(np.asarray(other) - self._value)

    def __mul__(self, other):
        return _ArrayTensor(self._value * np.asarray(other))

    def __rmul__(self, other):
        return _ArrayTensor(np.asarray(other) * self._value)

    def __truediv__(self, other):
        return _ArrayTensor(self._value / np.asarray(other))

    def __rtruediv__(self, other):
        return _ArrayTensor(np.asarray(other) / self._value)

    def __neg__(self):
        return _ArrayTensor(-self._value)


class _NumpyTorchStub:
    Tensor = _ArrayTensor
    float32 = np.float32

    @staticmethod
    def tensor(value, dtype=None):
        return _ArrayTensor(np.asarray(value, dtype=np.float64 if dtype is None else dtype))

    @staticmethod
    def ones(*shape, dtype=None):
        return _ArrayTensor(np.ones(shape, dtype=np.float64 if dtype is None else dtype))

    @staticmethod
    def round(value):
        return _ArrayTensor(np.round(np.asarray(value)))

    @staticmethod
    def clamp(value, min=None, max=None):
        lo = -np.inf if min is None else min
        hi = np.inf if max is None else max
        return _ArrayTensor(np.clip(np.asarray(value), lo, hi))

    @staticmethod
    def log(value):
        return _ArrayTensor(np.log(np.asarray(value)))

    @staticmethod
    def abs(value):
        return _ArrayTensor(np.abs(np.asarray(value)))

    @staticmethod
    def cos(value):
        return _ArrayTensor(np.cos(np.asarray(value)))

    @staticmethod
    def isfinite(value):
        return bool(np.isfinite(np.asarray(value)).all())


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


class NoiseAuditTrainTest(unittest.TestCase):
    def _load_loss_helpers(self):
        torch_namespace = torch if torch is not None else _NumpyTorchStub()
        return _load_module_functions(
            REPO_ROOT / "training" / "losses.py",
            [
                "wrap_to_pi_torch",
                "wrapped_orientation_nll",
                "log_ratio_anchor_loss",
                "circular_orientation_state_loss",
                "orientation_saturation_penalty",
            ],
            extra_namespace={
                "math": math,
                "torch": torch_namespace,
                "F": torch.nn.functional if torch is not None else _FunctionalStub(),
            },
        )

    def _load_train_helpers(self):
        mamba_helpers = _load_module_functions(
            REPO_ROOT / "kalmanfilter" / "mamba_adaptive_kf.py",
            ["build_noise_audit_samples"],
        )
        noise_prior_helpers = _load_module_functions(
            REPO_ROOT / "kalmanfilter" / "noise_priors.py",
            ["category_to_tracking_name"],
            extra_namespace={"TRACKING_CAT_ID_MAP": TRACKING_CAT_ID_MAP},
        )
        return _load_module_functions(
            REPO_ROOT / "training" / "train.py",
            [
                "_build_noise_audit_cfg",
                "_new_train_noise_audit_accumulator",
                "_resolve_train_noise_audit_state",
                "_record_train_noise_audit_samples",
                "_dump_train_noise_audit_if_needed",
                "_trace_covariance_batch",
                "_compute_ratio_anchor_regularization",
                "_resolve_closure_ratio_bounds",
                "_compute_closure_ratio_regularization",
            ],
            extra_namespace={
                "build_noise_audit_samples": mamba_helpers["build_noise_audit_samples"],
                "category_to_tracking_name": noise_prior_helpers["category_to_tracking_name"],
                "NoiseAuditAccumulator": NoiseAuditAccumulator,
                "torch": torch or _TorchStub(),
                "F": torch.nn.functional if torch is not None else _FunctionalStub(),
                "get_family_ratio_bounds": get_family_ratio_bounds,
                "log_ratio_anchor_loss": self._load_loss_helpers()["log_ratio_anchor_loss"],
                "log_ratio_bound_loss": _load_module_functions(
                    REPO_ROOT / "training" / "losses.py",
                    ["log_ratio_bound_loss"],
                    extra_namespace={
                        "math": math,
                        "torch": torch if torch is not None else _NumpyTorchStub(),
                        "F": torch.nn.functional if torch is not None else _FunctionalStub(),
                    },
                )["log_ratio_bound_loss"],
            },
        )

    def test_disabled_training_audit_helpers_are_noop(self):
        helpers = self._load_train_helpers()
        build_cfg = helpers["_build_noise_audit_cfg"]
        new_accumulator = helpers["_new_train_noise_audit_accumulator"]
        dump_audit = helpers["_dump_train_noise_audit_if_needed"]

        self.assertEqual(build_cfg(None), {})
        self.assertEqual(build_cfg({"AUDIT": {"NOISE_AUDIT": {"ENABLED": False}}}), {"ENABLED": False})
        self.assertIsNone(new_accumulator({"ENABLED": False}))

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "train_noise_audit.json"
            dump_audit(None, {"TRAIN_OUTPUT_PATH": str(out)})
            self.assertFalse(out.exists())

    def test_training_summary_projects_detector_teacher_forcing_to_matched(self):
        helpers = self._load_train_helpers()
        record_samples = helpers["_record_train_noise_audit_samples"]
        dump_audit = helpers["_dump_train_noise_audit_if_needed"]

        acc = NoiseAuditAccumulator()
        record_samples(
            noise_audit=acc,
            filter_mode="mamba",
            audit_state="matched",
            categories=["vehicle.bus", "truck"],
            class_ids=[4, 6],
            history_mask=[
                [True, True, True, False],
                [True, True, False, False],
            ],
            q_pos=[2.0, 10.0],
            r_pos=[3.0, 30.0],
            r_siz=[4.0, 40.0],
            r_ori=[5.0, 50.0],
            prior_q_pos=[1.0, 5.0],
            prior_r_pos=[1.5, 15.0],
            prior_r_siz=[2.0, 20.0],
            prior_r_ori=[2.5, 25.0],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "train_noise_audit.json"
            dump_audit(acc, {"TRAIN_OUTPUT_PATH": str(out)})
            payload = json.loads(out.read_text(encoding="utf-8"))

        buckets = payload["buckets"]
        self.assertEqual(len(buckets), 2)

        bus_bucket = next(bucket for bucket in buckets if bucket["class_name"] == "bus")
        truck_bucket = next(bucket for bucket in buckets if bucket["class_name"] == "truck")

        self.assertEqual(bus_bucket["split"], "train")
        self.assertEqual(bus_bucket["mode"], "mamba")
        self.assertEqual(bus_bucket["state"], "matched")
        self.assertEqual(bus_bucket["history_len"], 3)
        self.assertAlmostEqual(bus_bucket["families"]["q_pos"]["median"], 2.0)
        self.assertAlmostEqual(bus_bucket["ratios"]["q_pos"]["median"], 2.0)

        self.assertEqual(truck_bucket["split"], "train")
        self.assertEqual(truck_bucket["state"], "matched")
        self.assertEqual(truck_bucket["history_len"], 2)
        self.assertAlmostEqual(truck_bucket["families"]["r_ori"]["median"], 50.0)
        self.assertAlmostEqual(truck_bucket["ratios"]["r_ori"]["median"], 2.0)

    def test_training_summary_projects_gt_noisy_teacher_forcing_to_unmatched(self):
        helpers = self._load_train_helpers()
        record_samples = helpers["_record_train_noise_audit_samples"]

        acc = NoiseAuditAccumulator()
        record_samples(
            noise_audit=acc,
            filter_mode="fusion",
            audit_state="unmatched",
            categories=["car"],
            class_ids=[0],
            history_mask=[[True, True, True, True]],
            q_pos=[8.0],
            r_pos=[6.0],
            r_siz=[4.0],
            r_ori=[2.0],
            prior_q_pos=[4.0],
            prior_r_pos=[3.0],
            prior_r_siz=[2.0],
            prior_r_ori=[1.0],
        )

        bucket = acc.to_summary()["buckets"][0]
        self.assertEqual(bucket["split"], "train")
        self.assertEqual(bucket["mode"], "fusion")
        self.assertEqual(bucket["state"], "unmatched")
        self.assertEqual(bucket["history_len"], 4)
        self.assertAlmostEqual(bucket["ratios"]["q_pos"]["median"], 2.0)

    def test_call_path_helpers_project_training_modes_to_shared_state_axis(self):
        helpers = self._load_train_helpers()
        resolve_state = helpers["_resolve_train_noise_audit_state"]

        self.assertEqual(
            resolve_state(True, object(), object()),
            "matched",
        )
        self.assertEqual(
            resolve_state(False, object(), object()),
            "unmatched",
        )
        self.assertEqual(
            resolve_state(True, None, object()),
            "unmatched",
        )
        self.assertEqual(
            resolve_state(True, object(), None),
            "unmatched",
        )

    def test_epoch_audit_accumulator_is_fresh_and_output_overwrites_prior_epoch(self):
        helpers = self._load_train_helpers()
        new_accumulator = helpers["_new_train_noise_audit_accumulator"]
        record_samples = helpers["_record_train_noise_audit_samples"]
        dump_audit = helpers["_dump_train_noise_audit_if_needed"]

        with tempfile.TemporaryDirectory() as tmpdir:
            out = pathlib.Path(tmpdir) / "train_noise_audit.json"
            cfg = {"ENABLED": True, "TRAIN_OUTPUT_PATH": str(out)}

            first_acc = new_accumulator(cfg)
            second_acc = new_accumulator(cfg)

            self.assertIsNotNone(first_acc)
            self.assertIsNotNone(second_acc)
            self.assertIsNot(first_acc, second_acc)

            record_samples(
                noise_audit=first_acc,
                filter_mode="mamba",
                audit_state="matched",
                categories=["bus"],
                class_ids=[4],
                history_mask=[[True, True, False]],
                q_pos=[2.0],
                r_pos=[3.0],
                r_siz=[4.0],
                r_ori=[5.0],
                prior_q_pos=[1.0],
                prior_r_pos=[1.5],
                prior_r_siz=[2.0],
                prior_r_ori=[2.5],
            )
            dump_audit(first_acc, cfg)

            record_samples(
                noise_audit=second_acc,
                filter_mode="mamba",
                audit_state="unmatched",
                categories=["truck"],
                class_ids=[6],
                history_mask=[[True, False, False]],
                q_pos=[10.0],
                r_pos=[20.0],
                r_siz=[30.0],
                r_ori=[40.0],
                prior_q_pos=[5.0],
                prior_r_pos=[10.0],
                prior_r_siz=[15.0],
                prior_r_ori=[20.0],
            )
            dump_audit(second_acc, cfg)

            payload = json.loads(out.read_text(encoding="utf-8"))

        self.assertEqual(len(payload["buckets"]), 1)
        bucket = payload["buckets"][0]
        self.assertEqual(bucket["class_name"], "truck")
        self.assertEqual(bucket["state"], "unmatched")

    def test_wrapped_gaussian_orientation_loss_handles_pi_boundary(self):
        helpers = self._load_loss_helpers()
        wrapped_orientation_nll = helpers["wrapped_orientation_nll"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        pred = tensor([[3.13]], dtype=torch.float32 if torch is not None else np.float32)
        gt = tensor([[-3.13]], dtype=torch.float32 if torch is not None else np.float32)
        var = tensor([[[0.1]]], dtype=torch.float32 if torch is not None else np.float32)
        loss = wrapped_orientation_nll(pred, gt, var)

        is_finite = bool(torch.isfinite(loss).all().item()) if torch is not None else _NumpyTorchStub.isfinite(loss)
        self.assertTrue(is_finite)
        self.assertLess(float(loss.item()), 0.01)

    def test_log_ratio_anchor_loss_is_zero_when_gamma_is_one(self):
        helpers = self._load_loss_helpers()
        log_ratio_anchor_loss = helpers["log_ratio_anchor_loss"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        gamma = tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float32 if torch is not None else np.float32)
        loss = log_ratio_anchor_loss(gamma)

        self.assertAlmostEqual(float(loss.item()), 0.0, places=6)

    def test_circular_orientation_state_loss_wraps_pi_boundary(self):
        helpers = self._load_loss_helpers()
        circular_orientation_state_loss = helpers["circular_orientation_state_loss"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        pred = tensor([[3.13]], dtype=torch.float32 if torch is not None else np.float32)
        gt = tensor([[-3.13]], dtype=torch.float32 if torch is not None else np.float32)
        loss = circular_orientation_state_loss(pred, gt)

        self.assertLess(float(loss.item()), 0.01)

    def test_orientation_saturation_penalty_is_zero_below_threshold(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        penalty = orientation_saturation_penalty(
            tensor([[3.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        self.assertAlmostEqual(float(penalty.item()), 0.0, places=6)

    def test_orientation_saturation_penalty_is_positive_above_threshold(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        penalty = orientation_saturation_penalty(
            tensor([[25.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        self.assertGreater(float(penalty.item()), 0.0)

    def test_orientation_saturation_penalty_grows_with_kappa_excess(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        small = orientation_saturation_penalty(
            tensor([[6.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        large = orientation_saturation_penalty(
            tensor([[12.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )

        self.assertGreater(float(large.item()), float(small.item()))

    def test_state_prediction_loss_exposes_orientation_tensor_components(self):
        if torch is None:
            source = (REPO_ROOT / "training" / "losses.py").read_text(encoding="utf-8")
            self.assertIn('"loss_ori_state_tensor"', source)
            self.assertIn('"loss_ori_wrapped_tensor"', source)
            self.assertIn('"loss_ori_tensor"', source)
            return

        from training.losses import StatePredictionLoss

        loss_fn = StatePredictionLoss(w_pos=0.0, w_siz=0.0, w_ori=1.0, w_vel=0.0, w_nis=0.0)
        batch = 2

        pos_x_pred = torch.zeros(batch, 6, 1, dtype=torch.float32)
        pos_P_pred = torch.eye(6, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
        siz_x_pred = torch.zeros(batch, 3, 1, dtype=torch.float32)
        siz_P_pred = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
        ori_x_pred = torch.zeros(batch, 2, 1, dtype=torch.float32)
        ori_P_pred = torch.eye(2, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
        gt_next_pos = torch.zeros(batch, 3, dtype=torch.float32)
        gt_next_siz = torch.zeros(batch, 3, dtype=torch.float32)
        gt_next_ori = torch.full((batch, 1), 0.1, dtype=torch.float32)
        r_pos = torch.eye(5, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
        r_siz = torch.eye(3, dtype=torch.float32).unsqueeze(0).repeat(batch, 1, 1)
        r_ori = torch.full((batch, 1, 1), 0.5, dtype=torch.float32)
        kappa_ori = torch.full((batch, 1), 1.5, dtype=torch.float32)

        _, detail = loss_fn(
            pos_x_pred,
            pos_P_pred,
            siz_x_pred,
            siz_P_pred,
            ori_x_pred,
            ori_P_pred,
            gt_next_pos,
            gt_next_siz,
            gt_next_ori,
            r_pos,
            r_siz,
            r_ori,
            kappa_ori=kappa_ori,
            use_wrapped_orientation_nll=True,
        )

        self.assertIn("loss_ori_state_tensor", detail)
        self.assertIn("loss_ori_wrapped_tensor", detail)
        self.assertTrue(torch.is_tensor(detail["loss_ori_state_tensor"]))
        self.assertTrue(torch.is_tensor(detail["loss_ori_wrapped_tensor"]))
        self.assertEqual(detail["loss_ori_state_tensor"].ndim, 0)
        self.assertEqual(detail["loss_ori_wrapped_tensor"].ndim, 0)

    def test_ratio_anchor_regularization_reports_family_losses(self):
        if torch is None:
            self.skipTest("torch unavailable in unit-test interpreter")
        helpers = self._load_train_helpers()
        trace_covariance_batch = helpers["_trace_covariance_batch"]
        ratio_regularization = helpers["_compute_ratio_anchor_regularization"]

        def scaled_eye(scale):
            return torch.eye(2, dtype=torch.float32).unsqueeze(0) * (scale / 2.0)

        closure_cfg = {
            "ENABLED": True,
            "RATIO_ANCHOR_WEIGHT": 1.0,
            "PROFILES": {
                "heavy_long": {
                    "unmatched": {
                        "q_pos": [0.8, 1.2],
                        "r_pos": [0.8, 1.2],
                        "r_siz": [0.8, 1.2],
                        "r_ori": [0.8, 1.2],
                    }
                }
            },
        }
        total, detail = ratio_regularization(
            raw_tensors={
                "Q_pos": scaled_eye(3.0),
                "R_pos": scaled_eye(4.0),
                "R_siz": scaled_eye(5.0),
                "R_ori": scaled_eye(6.0),
            },
            prior_tensors={
                "Q_pos_base": scaled_eye(1.0),
                "R_pos_base": scaled_eye(1.0),
                "R_siz_base": scaled_eye(1.0),
                "R_ori_base": scaled_eye(1.0),
            },
            class_ids=torch.tensor([5], dtype=torch.int64),
            state_buckets=["unmatched"],
            closure_cfg=closure_cfg,
        )

        self.assertAlmostEqual(trace_covariance_batch(scaled_eye(3.0)).item(), 3.0)
        self.assertGreater(total.item(), 0.0)
        self.assertEqual(
            set(detail.keys()),
            {"loss_ratio_q_pos", "loss_ratio_r_pos", "loss_ratio_r_siz", "loss_ratio_r_ori"},
        )
        for value in detail.values():
            self.assertGreaterEqual(value.item(), 0.0)
        self.assertGreater(detail["loss_ratio_r_ori"].item(), detail["loss_ratio_q_pos"].item())

    def test_closure_ratio_regularization_sums_all_seven_ratio_heads(self):
        if torch is None:
            self.skipTest("torch unavailable in unit-test interpreter")
        helpers = self._load_train_helpers()
        closure_regularization = helpers["_compute_closure_ratio_regularization"]

        ratio_value = 2.0
        base_term = abs(math.log(ratio_value))
        closure_cfg = {
            "ENABLED": True,
            "RATIO_ANCHOR_WEIGHT": 1.0,
            "RATIO_BOUND_WEIGHT": 0.0,
            "PROFILES": {
                "heavy_long": {
                    "matched": {
                        "q_pos": [0.5, 3.0],
                        "r_pos": [0.5, 3.0],
                        "r_siz": [0.5, 3.0],
                        "r_ori": [0.5, 3.0],
                    }
                }
            },
        }
        ratios = {
            "q_pos_xyz": torch.tensor([[ratio_value]], dtype=torch.float32),
            "q_pos_vxyz": torch.tensor([[ratio_value]], dtype=torch.float32),
            "r_pos_xyz": torch.tensor([[ratio_value]], dtype=torch.float32),
            "r_pos_vxy": torch.tensor([[ratio_value]], dtype=torch.float32),
            "r_siz_lw": torch.tensor([[ratio_value]], dtype=torch.float32),
            "r_siz_h": torch.tensor([[ratio_value]], dtype=torch.float32),
            "r_ori": torch.tensor([[ratio_value]], dtype=torch.float32),
        }

        total_anchor, total_bound, detail = closure_regularization(
            ratios=ratios,
            class_ids=torch.tensor([5], dtype=torch.int64),
            state_buckets=["matched"],
            closure_cfg=closure_cfg,
        )

        self.assertAlmostEqual(total_bound.item(), 0.0, places=6)
        self.assertAlmostEqual(total_anchor.item(), 7.0 * base_term, places=6)
        self.assertAlmostEqual(detail["loss_ratio_anchor_q_pos"].item(), 2.0 * base_term, places=6)
        self.assertAlmostEqual(detail["loss_ratio_anchor_r_pos"].item(), 2.0 * base_term, places=6)
        self.assertAlmostEqual(detail["loss_ratio_anchor_r_siz"].item(), 2.0 * base_term, places=6)
        self.assertAlmostEqual(detail["loss_ratio_anchor_r_ori"].item(), 1.0 * base_term, places=6)

    def test_matched_kf_band_overrides_broad_profile_bounds_during_closure_training(self):
        if torch is None:
            self.skipTest("torch unavailable in unit-test interpreter")
        helpers = self._load_train_helpers()
        resolve_bounds = helpers["_resolve_closure_ratio_bounds"]

        closure_cfg = {
            "ENABLED": True,
            "MATCHED_KF_BAND": {
                "ENABLED": True,
                "FAMILIES": {
                    "q_pos": [0.93, 1.07],
                    "r_pos": [0.94, 1.06],
                },
                "CLASS_OVERRIDES": {
                    5: {
                        "r_pos": [0.96, 1.04],
                    }
                },
            },
            "PROFILES": {
                "heavy_long": {
                    "matched": {
                        "q_pos": [0.5, 3.0],
                        "r_pos": [0.5, 3.0],
                    },
                    "unmatched": {
                        "r_pos": [0.8, 1.3],
                    },
                }
            },
        }

        self.assertEqual(
            resolve_bounds(5, "matched", "q_pos", closure_cfg),
            (0.93, 1.07),
        )
        self.assertEqual(
            resolve_bounds(5, "matched", "r_pos", closure_cfg),
            (0.96, 1.04),
        )
        self.assertEqual(
            resolve_bounds(5, "unmatched", "r_pos", closure_cfg),
            (0.8, 1.3),
        )


if __name__ == "__main__":
    unittest.main()
