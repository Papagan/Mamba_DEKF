"""Small torchrun adapter; the entire differentiable rollout is one DDP forward."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist
from torch import nn


class TrainingRollout(nn.Module):
    """Expose only the final loss to DDP's unused-parameter traversal."""

    def __init__(self, model, loss_fn, step_fn, device):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.step_fn = step_fn
        self.device = device
        self.detail = {}

    def forward(self, batch, **kwargs):
        loss, self.detail = self.step_fn(
            self.model, batch, self.loss_fn, self.device, **kwargs
        )
        return loss


def initialize_distributed(device_override=None):
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 1:
        if device_override and device_override != "cuda":
            raise ValueError("torchrun selects LOCAL_RANK; omit --device for multi-GPU training")
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise RuntimeError("Each torchrun worker needs a visible CUDA GPU")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        # Rank zero validates the full, unpadded validation set and saves weights.
        dist.init_process_group("nccl", timeout=timedelta(hours=2))
    else:
        device = torch.device(device_override or ("cuda" if torch.cuda.is_available() else "cpu"))
    return device, rank, world_size


def all_finite(tensors, device, distributed=False):
    checks = [torch.isfinite(value.detach()).all() for value in tensors if value is not None]
    valid = torch.stack(checks).all() if checks else torch.ones((), device=device, dtype=torch.bool)
    flag = valid.to(device=device, dtype=torch.int32)
    if distributed:
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


class NullWriter:
    def add_scalar(self, *args, **kwargs):
        pass

    def flush(self):
        pass

    def close(self):
        pass


def reduce_epoch_metrics(metrics, batches, failures, device):
    """Return sums across ranks so primary logs describe the whole training set."""
    keys = sorted(metrics)
    values = torch.tensor([metrics[key] for key in keys] + [batches, failures],
                          device=device, dtype=torch.float64)
    dist.all_reduce(values, op=dist.ReduceOp.SUM)
    values = values.cpu().tolist()
    return dict(zip(keys, values[:-2])), int(values[-2]), int(values[-1])
