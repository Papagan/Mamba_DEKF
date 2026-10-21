"""Two real Gloo workers exercise conditional-head rollout gradients on CPU."""

import copy
from datetime import timedelta
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel
import yaml

from training.distributed import TrainingRollout, all_finite, reduce_epoch_metrics
from tracker.device_runtime import (
    resolve_inference_gpu_ids, initialize_inference_worker, result_run_directory,
)


def _worker(rank, rendezvous):
    from kalmanfilter.mamba_adaptive_kf import TemporalMamba
    from training.losses import JointLoss
    from training.train import training_step
    torch.set_num_threads(1)
    torch.manual_seed(10 + rank)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=timedelta(seconds=60))
    try:
        cfg = yaml.safe_load((Path(__file__).resolve().parents[1] / "config/train_nuscenes.yaml").read_text())
        model = TemporalMamba(d_model=16, n_mamba_layers=1, embed_dim=8, num_classes=7,
                              force_gru=True, base_noise_cfg=cfg["BASE_NOISE"])
        loss_fn = JointLoss(physics_scale=1., residual_supervision=cfg["LOSS"]["RESIDUAL_SUPERVISION"])
        rollout = TrainingRollout(model, loss_fn, training_step, torch.device("cpu"))
        parallel = DistributedDataParallel(rollout, find_unused_parameters=True, broadcast_buffers=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        current = torch.tensor([[0., 0., 1., 1., 0., 0.]]).expand(2, -1).clone()
        sizes = torch.tensor([[4., 2., 1.5]]).expand(2, -1).clone()
        future = torch.tensor([[[.6, 0., 1.], [1.3, 0., 1.]]]).expand(2, -1, -1).clone()
        batch = dict(
            track_history=torch.randn(2, 8, 12) * .1,
            gt_current_state_pos=current, obs_current_state_pos=current,
            gt_current_state_siz=sizes, obs_current_state_siz=sizes,
            gt_current_state_ori=torch.zeros(2, 2), obs_current_state_ori=torch.zeros(2, 2),
            gt_future_pos=future, gt_future_siz=sizes[:, None].expand(-1, 2, -1),
            gt_future_ori=torch.zeros(2, 2, 1), delta_ts_future=torch.full((2, 2), .5),
            obs_future_pos=torch.cat([future, torch.ones(2, 2, 2) * .1], dim=2),
            obs_future_siz=sizes[:, None].expand(-1, 2, -1), obs_future_ori=torch.zeros(2, 2, 1),
            obs_future_score=torch.full((2, 2), .8),
            obs_future_match=torch.ones(2, 2, dtype=torch.bool),
            future_mask=torch.ones(2, 2, dtype=torch.bool),
            instance_token=["one", "two"], category=["car", "car"] if rank == 0 else ["truck", "truck"],
            state_bucket=["matched", "unmatched"],
            is_detection_driven=torch.ones(2, dtype=torch.bool),
            history_mask=torch.tensor([[False] * 7 + [True], [False] * 6 + [True, True]]),
            history_match_mask=torch.tensor([[False] * 7 + [True], [False] * 6 + [True, False]]),
        )
        before = copy.deepcopy(model.state_dict())
        for epoch in (0, 20, 21):
            optimizer.zero_grad()
            loss = parallel(batch, base_noise_cfg=cfg["BASE_NOISE"],
                            noise_cfg=cfg["TRAINING"]["NOISE_MAP"], epoch=epoch,
                            warmup_epochs=1, transition_epochs=1, filter_mode=cfg["FILTER_MODE"])
            assert all_finite([loss], torch.device("cpu"), True)
            loss.backward()
            assert all_finite([p.grad for p in model.parameters()], torch.device("cpu"), True)
            optimizer.step()
        flattened = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(flattened) for _ in range(2)]
        dist.all_gather(gathered, flattened)
        assert torch.equal(gathered[0], gathered[1]), "Workers diverged"
        assert not torch.equal(before["head_delta_pos.2.weight"], model.state_dict()["head_delta_pos.2.weight"])
        metrics, batches, failures = reduce_epoch_metrics(
            {"loss_total": float(rank + 1)}, 2, rank, torch.device("cpu"))
        assert metrics == {"loss_total": 3.0} and batches == 4 and failures == 1
        # A bad loss on just one rank must be detected by both before backward.
        assert not all_finite([torch.tensor(float("nan") if rank else 1.)], torch.device("cpu"), True)
    finally:
        dist.destroy_process_group()


class DistributedTrainingTest(unittest.TestCase):
    def test_conditional_multistep_rollout_synchronizes_two_workers(self):
        with tempfile.TemporaryDirectory() as directory:
            mp.spawn(_worker, args=("file://" + directory + "/rendezvous",), nprocs=2, join=True)

    def test_inference_workers_use_visible_gpu_indices(self):
        with patch("tracker.device_runtime.torch.cuda.is_available", return_value=True), \
             patch("tracker.device_runtime.torch.cuda.device_count", return_value=2), \
             patch("tracker.device_runtime.torch.cuda.set_device") as set_device, \
             patch("tracker.device_runtime.torch.set_num_threads"), \
             patch("tracker.device_runtime.multiprocessing.current_process") as process:
            self.assertEqual(resolve_inference_gpu_ids("auto"), [0, 1])
            with self.assertRaises(ValueError):
                resolve_inference_gpu_ids([2])
            for identity in (1, 2, 3, 4):
                process.return_value._identity = (identity,)
                initialize_inference_worker([0, 1])
                set_device.assert_called_with((identity - 1) % 2)

    def test_results_directory_does_not_duplicate_dataset_or_drop_root(self):
        for value in ("/data/results", "/data/results/", "/data/results/nuscenes",
                      "/data/results/nuscenes/"):
            self.assertEqual(result_run_directory(value, "nuscenes", "run"),
                             "/data/results/nuscenes/run")

    def test_resume_restores_next_learning_rate(self):
        from training.train import LinearLR, CosineAnnealingLR, SequentialLR

        def build():
            optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=.01)
            scheduler = SequentialLR(
                optimizer,
                schedulers=[LinearLR(optimizer, start_factor=.01, total_iters=2),
                            CosineAnnealingLR(optimizer, T_max=8)],
                milestones=[2],
            )
            return optimizer, scheduler

        optimizer, scheduler = build()
        for _ in range(5):
            optimizer.step()
            scheduler.step()
        optimizer_state = copy.deepcopy(optimizer.state_dict())
        scheduler_state = copy.deepcopy(scheduler.state_dict())
        resumed_optimizer, resumed_scheduler = build()
        resumed_optimizer.load_state_dict(optimizer_state)
        resumed_scheduler.load_state_dict(scheduler_state)
        for _ in range(3):
            optimizer.step()
            scheduler.step()
            resumed_optimizer.step()
            resumed_scheduler.step()
            self.assertEqual(optimizer.param_groups[0]["lr"],
                             resumed_optimizer.param_groups[0]["lr"])
