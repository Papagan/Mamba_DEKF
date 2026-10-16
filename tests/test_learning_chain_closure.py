"""Behavioral regressions for train/cache/inference state closure (CPU only)."""

import ast
import copy
from pathlib import Path
from types import MethodType, SimpleNamespace
from typing import Dict, List, Tuple
import unittest

import numpy as np
import yaml

try:
    import torch
except ImportError:
    torch = None

from tracker.compat_utils import (
    extract_bbox_history_fields, sync_bbox_fields_from_state, uses_filter_state_output,
    score_for_unmatched_fake_bbox, use_mctrack_exact_unmatch_update,
)
from tools.finalize_association_replay import build_replay_plan

ROOT = Path(__file__).resolve().parents[1]


def load_methods(relative_path, names, extra=None):
    namespace = dict(np=np, torch=torch, BBox=object, Trajectory=object,
                     Dict=Dict, List=List, Tuple=Tuple,
                     sync_bbox_fields_from_state=sync_bbox_fields_from_state,
                     extract_bbox_history_fields=extract_bbox_history_fields,
                     score_for_unmatched_fake_bbox=score_for_unmatched_fake_bbox,
                     use_mctrack_exact_unmatch_update=use_mctrack_exact_unmatch_update,
                     copy=copy)
    namespace.update(extra or {})
    for node in ast.parse((ROOT / relative_path).read_text()).body:
        if isinstance(node, ast.ClassDef):
            for method in node.body:
                if isinstance(method, ast.FunctionDef) and method.name in names:
                    exec(compile(ast.Module(body=[method], type_ignores=[]), relative_path, "exec"), namespace)
    return {name: namespace[name] for name in names}


class ClosurePolicyTest(unittest.TestCase):
    def test_compatibility_switch_never_selects_legacy_for_learned_filter(self):
        for compat in ("mctrack", "default"):
            self.assertTrue(uses_filter_state_output({
                "FILTER_MODE": "mamba_multihead_closure", "TRACKER_COMPAT_MODE": compat,
            }))
        self.assertFalse(uses_filter_state_output({"FILTER_MODE": "pure_dekf"}))
        with self.assertRaisesRegex(ValueError, "cannot overwrite"):
            uses_filter_state_output({"STATE_OUTPUT_MODE": "legacy_mctrack"})

    def test_finalization_writes_the_actual_production_head_path(self):
        bootstrap, commands = build_replay_plan(
            ROOT / "config/train_nuscenes.yaml",
            ROOT / "config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml",
            "/tmp/replay-plan-test", 2, "/tmp/bootstrap-test.yaml",
        )
        final_command = commands[1]
        self.assertEqual(final_command[final_command.index("--output") + 1],
                         bootstrap["MAMBA_ASSOCIATION_HEAD"]["CHECKPOINT_PATH"])
        self.assertNotIn("REQUIRE_TRAINING_SOURCE", bootstrap["MAMBA_ASSOCIATION_HEAD"])
        production = yaml.safe_load((ROOT / "config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml").read_text())
        self.assertEqual(production["MAMBA_ASSOCIATION_HEAD"]["REQUIRE_TRAINING_SOURCE"], "tracker_replay")

    def test_train_and_runtime_enable_same_noise_policy_and_history_windows(self):
        train = yaml.safe_load((ROOT / "config/train_nuscenes.yaml").read_text())
        runtime = yaml.safe_load((ROOT / "config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml").read_text())
        for key in ("FORCE_COAST_PRIOR_ONLY", "FORCE_PRIOR_STATES", "ACTIVE_CLASS_STATES"):
            self.assertEqual(train["BASE_NOISE"]["MAMBA_CLOSURE"][key], runtime["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"][key])
        self.assertFalse(runtime["ASSOCIATION_HEAD_AUDIT"]["SHADOW_MODE"])
        for category, window in train["DATA"]["CLASS_WINDOW"].items():
            for key in ("MIN_HISTORY_LEN", "MAX_HISTORY_LEN"):
                self.assertEqual(window[key], runtime["RESIDUAL_HISTORY"]["CLASS_WINDOW"][category][key])

    def test_filter_history_contract_allows_mc_lifecycle_ablation(self):
        from tracker.association_runtime_manifest import validate_association_runtime_contract
        contract = dict(state_output_mode="filter", tracker_compat_mode="mctrack",
                        filter_mode="mamba_multihead_closure", history_source="fusion",
                        expected_bev_cost_mode="geometric")
        runtime = dict(STATE_OUTPUT_MODE="filter", TRACKER_COMPAT_MODE="default",
                       FILTER_MODE="mamba_multihead_closure")
        self.assertEqual(validate_association_runtime_contract(contract, runtime)["validation_status"], "valid")


@unittest.skipIf(torch is None, "requires CPU torch")
class LearnedStateClosureTest(unittest.TestCase):
    def test_actual_multistep_training_has_finite_loss_and_motion_gradient(self):
        from training.train import training_step
        from training.losses import JointLoss
        from kalmanfilter.mamba_adaptive_kf import TemporalMamba
        torch.manual_seed(7)
        cfg = yaml.safe_load((ROOT / "config/train_nuscenes.yaml").read_text())
        model = TemporalMamba(d_model=16, n_mamba_layers=1, embed_dim=8, num_classes=7,
                              force_gru=True, base_noise_cfg=cfg["BASE_NOISE"])
        current = torch.tensor([[0., 0., 1., 1., 0., 0.], [0., 0., 1., 1., 0., 0.]])
        sizes = torch.tensor([[4., 2., 1.5], [4., 2., 1.5]])
        future = torch.tensor([[[.6, 0., 1.], [1.3, 0., 1.], [2.1, 0., 1.]]]).expand(2, -1, -1).clone()
        batch = dict(
            track_history=torch.randn(2, 8, 12) * .1,
            gt_current_state_pos=current, obs_current_state_pos=current,
            gt_current_state_siz=sizes, obs_current_state_siz=sizes,
            gt_current_state_ori=torch.zeros(2, 2), obs_current_state_ori=torch.zeros(2, 2),
            gt_future_pos=future, gt_future_siz=sizes[:, None].expand(-1, 3, -1),
            gt_future_ori=torch.zeros(2, 3, 1), delta_ts_future=torch.full((2, 3), .5),
            obs_future_pos=torch.cat([future, torch.ones(2, 3, 2) * .1], dim=2),
            obs_future_siz=sizes[:, None].expand(-1, 3, -1), obs_future_ori=torch.zeros(2, 3, 1),
            obs_future_score=torch.full((2, 3), .8), obs_future_match=torch.ones(2, 3, dtype=torch.bool),
            future_mask=torch.ones(2, 3, dtype=torch.bool),
            instance_token=["one", "two"], category=["car", "truck"],
            state_bucket=["matched", "unmatched"], is_detection_driven=torch.ones(2, dtype=torch.bool),
            history_mask=torch.tensor([[False] * 7 + [True], [False] * 6 + [True, True]]),
            history_match_mask=torch.tensor([[False] * 7 + [True], [False] * 6 + [True, False]]),
        )
        loss_fn = JointLoss(physics_scale=1., residual_supervision=cfg["LOSS"]["RESIDUAL_SUPERVISION"])
        for epoch in (0, 20):
            model.zero_grad()
            loss, _ = training_step(model, batch, loss_fn, torch.device("cpu"),
                                    noise_cfg=cfg.get("TRAIN_NOISE"), base_noise_cfg=cfg["BASE_NOISE"],
                                    epoch=epoch, warmup_epochs=1, transition_epochs=1,
                                    filter_mode=cfg["FILTER_MODE"])
            self.assertTrue(torch.isfinite(loss).item())
            loss.backward()
            gradient = model.head_delta_pos[-1].weight.grad
            self.assertTrue(torch.isfinite(gradient).all().item())
            self.assertGreater(gradient.norm().item(), 0.)

    def test_real_to_coast_omega_uses_real_detection_timestamps(self):
        from tests.test_nuscenes_audit_gt_enrichment import DatasetAuditSidecarTest, _converted_bbox
        loader = DatasetAuditSidecarTest()._load_dataset_module(lambda boxes, cfg: boxes)
        raw_timestamp = 1500000000000000
        scene = [dict(frame_id=0, timestamp=raw_timestamp, cur_sample_token="sample",
                      transform_matrix={}, bboxes=[_converted_bbox(global_yaw=0.)])]
        cfg = dict(CATEGORY_MAP_TO_NUMBER={"car": 0}, TRACKING_MODE="ONLINE",
                   THRESHOLD={"INPUT_SCORE": {"ONLINE": {0: .1}}}, DATASET="nuscenes")
        frame = loader.BaseVersionTrackingDataset("scene", scene, cfg)[0]
        real = frame.bboxes[0]
        self.assertEqual(real.timestamp, raw_timestamp)
        fake = copy.deepcopy(real)
        fake.timestamp += 500000
        fake.frame_id, fake.is_fake, fake.global_yaw_fusion = 1, True, .1
        fake.global_xyz_lwh_yaw_fusion = list(fake.global_xyz_lwh_yaw_fusion)
        fake.global_xyz_lwh_yaw_fusion[6] = .1
        tracker = SimpleNamespace(history_len=8, device="cpu", tracker_compat_mode="mctrack", frame_rate=2,
                                  default_window_cfg={"MIN_HISTORY_LEN": 4, "MAX_HISTORY_LEN": 8}, runtime_window_cfg={})
        for name, fn in load_methods("tracker/base_tracker.py", ["_extract_track_history", "_class_window_cfg", "_normalize_delta_t"]).items():
            setattr(tracker, name, MethodType(fn, tracker))
        history, _, _ = tracker._extract_track_history([SimpleNamespace(bboxes=[real, fake])])
        self.assertAlmostEqual(history[0, -1, 10].item(), .2)

    def make_bbox(self, index=0, category="car"):
        bbox = SimpleNamespace(category=category, global_velocity=[0.0, 0.0],
                               global_velocity_fusion=[0.0, 0.0], frame_id=index,
                               timestamp=index * .5, det_score=.8, is_fake=False)
        sync_bbox_fields_from_state(bbox, [float(index), 0., 1., 4., 2., 1.5, .1], update_predict=True)
        return bbox

    def test_prediction_update_and_coast_preserve_learned_values_in_both_modes(self):
        methods = load_methods("tracker/base_tracker.py", ["_write_predicted_state_to_bbox", "_write_updated_state_to_bbox"])
        coast = load_methods("tracker/trajectory.py", ["unmatch_update"])["unmatch_update"]
        pos = torch.tensor([1., 2., 3., 4., 5., 0.]).view(1, 6, 1)
        size = torch.tensor([4.5, 2.5, 1.8]).view(1, 3, 1)
        ori = torch.tensor([.7, .2]).view(1, 2, 1)
        expected = [1., 2., 3., 4.5, 2.5, 1.8, .7]
        for compat in ("mctrack", "default"):
            with self.subTest(compat=compat):
                tracker = SimpleNamespace(filter_state_output=True, tracker_compat_mode=compat)
                bbox = self.make_bbox()
                trajectory = SimpleNamespace(bboxes=[bbox], track_length=1)
                methods["_write_predicted_state_to_bbox"](tracker, trajectory, pos, size, ori)
                np.testing.assert_allclose(bbox.global_xyz_lwh_yaw_predict, expected)
                self.assertAlmostEqual(bbox.global_xyz_lwh_yaw_fusion[2], 1.)
                # Missing frame must consume the prediction without changing past posterior.
                trajectory.__dict__.update(
                    unmatch_length=0, _tracker_compat_mode=compat, cfg={}, category_num=0,
                    matched_scores=[], _cache_bbox_len=10, status_flag=1,
                    _confirmed_track_length=3, _max_unmatch_len=3, _max_predict_len=5,
                    record_coast_residual=lambda **kw: None,
                    cal_diff_velocity=lambda: [0., 0.], cal_curve_velocity=lambda: [0., 0.],
                )
                coast(trajectory, 1, .5)
                fake = trajectory.bboxes[-1]
                np.testing.assert_allclose(fake.global_xyz_lwh_yaw_fusion, expected)
                self.assertEqual(fake.global_velocity_fusion, [4., 5.])
                matched = self.make_bbox(2)
                methods["_write_updated_state_to_bbox"](tracker, matched, pos, size, ori)
                xyz, vel, lwh, yaw = extract_bbox_history_fields(matched, compat)
                np.testing.assert_allclose(xyz + lwh + [yaw], expected)
                self.assertEqual(vel, [4., 5.])

    def test_track_history_uses_class_window_and_not_mutated_prediction(self):
        methods = load_methods("tracker/base_tracker.py", ["_extract_track_history", "_class_window_cfg"])
        tracker = SimpleNamespace(history_len=8, device=torch.device("cpu"),
                                  tracker_compat_mode="default",
                                  default_window_cfg={"MIN_HISTORY_LEN": 4, "MAX_HISTORY_LEN": 8},
                                  runtime_window_cfg={"bicycle": {"MIN_HISTORY_LEN": 3, "MAX_HISTORY_LEN": 6}},
                                  _normalize_delta_t=lambda value: float(value), frame_rate=2)
        for name, method in methods.items():
            setattr(tracker, name, MethodType(method, tracker))
        boxes = [self.make_bbox(index, "bicycle") for index in range(10)]
        for box in boxes:
            box.filter_state_output = True
        boxes[-1].global_xyz = [999., 999., 999.] # prediction, not historical posterior
        history, mask, _ = tracker._extract_track_history([SimpleNamespace(bboxes=boxes)])
        self.assertEqual(int(mask.sum()), 6)
        self.assertEqual(history[0, -1, 0].item(), 0.)
        self.assertEqual(history[0, -1, 2].item(), 1.)

    def test_orientation_R_has_gradient_and_identical_train_runtime_semantics(self):
        from kalmanfilter.mamba_adaptive_kf import TemporalMamba
        cfg = yaml.safe_load((ROOT / "config/train_nuscenes.yaml").read_text())
        model = TemporalMamba(d_model=16, n_mamba_layers=1, embed_dim=8, num_classes=7,
                              force_gru=True, base_noise_cfg=cfg["BASE_NOISE"]).eval()
        history = torch.zeros(1, 8, 12)
        kwargs = dict(class_ids=torch.tensor([1]), state_buckets=["matched"], mode="mamba_multihead_closure")
        trained = model(history, apply_force_prior=False, **kwargs)
        runtime = model(history, apply_force_prior=True, **kwargs)
        self.assertTrue(torch.allclose(trained["R_ori"], runtime["R_ori"]))
        self.assertLess(trained["R_ori"].item(), .01)
        trained["R_ori"].sum().backward()
        gradient = model.head_bank.family_heads["r_ori"][1].proj.bias.grad
        self.assertGreater(float(gradient.abs().sum()), 0.)

    def test_bootstrap_fusion_cache_writes_the_same_filter_state_as_tracker(self):
        from tools.augment_tracklet_cache_with_fusion import (
            augment_tracklet, get_base_covariances, init_kf_from_detection,
        )
        base = yaml.safe_load((ROOT / "config/train_nuscenes.yaml").read_text())["BASE_NOISE"]
        base = dict(base, STATE_OUTPUT_MODE="filter")
        frames = []
        for index in range(3):
            matched = index < 2
            xyz, lwh, yaw = [float(index), 0., 1. + index], [4. + index, 2., 1.5], .1 + index * .4
            feature = xyz + [1., 0., 0.] + lwh + [yaw, 0., .8]
            frames.append(dict(frame_id=index, timestamp=index * .5, is_matched=matched,
                               det_global_xyz=xyz if matched else None, det_lwh=lwh if matched else None,
                               det_yaw=yaw if matched else None, det_velocity=[1., 0.],
                               det_score=.8 if matched else 0., obs_feature_12=feature))
        augmented, _ = augment_tracklet(dict(category="car", frames=frames), base, torch.device("cpu"))
        self.assertEqual(augmented["fusion_state_output_mode"], "filter")
        self.assertEqual(augmented["fusion_feature_schema_version"], 2)
        kf = init_kf_from_detection(frames[0], torch.device("cpu"))
        cov = get_base_covariances(base, "car", torch.device("cpu"))
        predict_args = {key: cov[key + "_base"] for key in ("Q_pos", "Q_siz", "Q_ori")}
        kf.predict(torch.tensor([.5]), **predict_args)
        updated = kf.update(
            torch.tensor(frames[1]["det_global_xyz"] + [1., 0.]).view(1, 5, 1),
            torch.tensor(frames[1]["det_lwh"]).view(1, 3, 1), torch.tensor([[[frames[1]["det_yaw"]]]]),
            **{key: cov[key + "_base"] for key in ("R_pos", "R_siz", "R_ori")},
        )
        writer = load_methods("tracker/base_tracker.py", ["_write_updated_state_to_bbox"])["_write_updated_state_to_bbox"]
        bbox = self.make_bbox(1)
        writer(SimpleNamespace(filter_state_output=True), bbox, updated[0], updated[2], updated[4])
        cached = augmented["frames"][1]
        np.testing.assert_allclose(cached["fusion_global_xyz"], bbox.global_xyz_lwh_yaw_fusion[:3])
        np.testing.assert_allclose(cached["fusion_lwh"], bbox.lwh_fusion)
        self.assertAlmostEqual(cached["fusion_yaw"], bbox.global_yaw_fusion)
        predicted = kf.predict(torch.tensor([.5]), **predict_args)
        cached_coast = augmented["frames"][2]
        np.testing.assert_allclose(cached_coast["fusion_global_xyz"], predicted[0][0, :3, 0])
        self.assertAlmostEqual(cached_coast["fusion_yaw"], predicted[4][0, 0, 0].item())

    def test_birth_filter_does_not_import_omega_from_before_lifecycle(self):
        from tools.augment_tracklet_cache_with_fusion import init_kf_from_detection
        frame = dict(det_global_xyz=[1., 2., 3.], det_velocity=[1., 0.],
                     det_lwh=[4., 2., 1.5], det_yaw=.3, obs_feature_12=[0.] * 10 + [9., .8])
        kf = init_kf_from_detection(frame, torch.device("cpu"))
        self.assertEqual(kf.ori_filter.x[0, 1, 0].item(), 0.)


if __name__ == "__main__":
    unittest.main()
