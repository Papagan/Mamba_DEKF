"""Assign persistent inference workers to visible GPUs without CUDA-before-fork."""

import multiprocessing
from pathlib import Path

import torch


def result_run_directory(save_path, dataset, run_name):
    """Accept a results root or dataset results directory, with or without '/'."""
    base = Path(save_path).expanduser()
    if base.name != dataset:
        base = base / dataset
    return str(base / run_name)


def resolve_inference_gpu_ids(setting):
    if setting is None:
        return []
    count = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if setting == "auto":
        return list(range(count))
    if not isinstance(setting, (list, tuple)):
        raise ValueError("INFERENCE_GPU_IDS must be auto or a list of visible CUDA indices")
    ids = [int(value) for value in setting]
    if len(set(ids)) != len(ids) or any(value < 0 or value >= count for value in ids):
        raise ValueError(f"Invalid INFERENCE_GPU_IDS={ids} for {count} visible GPUs")
    return ids


def initialize_inference_worker(gpu_ids):
    if gpu_ids:
        identity = multiprocessing.current_process()._identity
        worker_index = identity[-1] - 1 if identity else 0
        torch.cuda.set_device(gpu_ids[worker_index % len(gpu_ids)])
        # Each scene is small; avoid CPU BLAS oversubscription across workers.
        torch.set_num_threads(1)
