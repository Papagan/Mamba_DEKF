import ast
import json
import pathlib
import tempfile
import unittest

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
        return value


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
            ],
            extra_namespace={
                "build_noise_audit_samples": mamba_helpers["build_noise_audit_samples"],
                "category_to_tracking_name": noise_prior_helpers["category_to_tracking_name"],
                "NoiseAuditAccumulator": NoiseAuditAccumulator,
                "torch": torch or _TorchStub(),
                "F": torch.nn.functional if torch is not None else _FunctionalStub(),
                "get_family_ratio_bounds": get_family_ratio_bounds,
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


if __name__ == "__main__":
    unittest.main()
