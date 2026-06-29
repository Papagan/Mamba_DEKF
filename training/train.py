#!/usr/bin/env python3
# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — Training Entry Point
#
# Usage:
#   # Step 1: Extract GT tracklets (run once, cached to disk)
#   python training/train.py --config config/train_nuscenes.yaml --extract-only
#
#   # Step 2: Train
#   python training/train.py --config config/train_nuscenes.yaml
#
#   # Step 3: Resume from checkpoint
#   python training/train.py --config config/train_nuscenes.yaml \
#       --resume checkpoints/mamba_dekf/checkpoint_epoch10.pt
# ------------------------------------------------------------------------

import os
import sys
import time
import yaml
import logging
import argparse
import warnings

import torch

# SequentialLR internally passes epoch to sub-schedulers, triggering a
# deprecation warning in PyTorch 2.1+. Harmless, suppressed here.
warnings.filterwarnings("ignore", message=".*epoch parameter.*")
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR

# Project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from training.gt_tracklet_dataset import (
    extract_gt_tracklets_nuscenes,
    TrackletDataset,
    tracklet_collate_fn,
)
from training.det_tracklet_dataset import (
    DetectionTrackletDataset,
    detection_tracklet_collate_fn,
)
from training.losses import (
    JointLoss,
    log_ratio_anchor_loss,
    log_ratio_bound_loss,
    orientation_saturation_penalty,
    wrap_to_pi_torch,
)
from kalmanfilter.mamba_adaptive_kf import (
    TemporalMamba,
    DecoupledAdaptiveKF,
    build_noise_audit_samples,
)
from kalmanfilter.checkpoint_compat import (
    adapt_num_class_state_dict,
    filter_heads_only_state_dict,
)
from kalmanfilter.noise_audit import NoiseAuditAccumulator
from kalmanfilter.noise_priors import (
    build_base_covariances,
    category_to_tracking_name,
    categories_to_class_ids,
)
from kalmanfilter.bounded_residual import get_family_ratio_bounds
from training.class_state_metrics import (
    class_state_bucket_key,
    init_class_state_metric_accumulator,
    update_class_state_metric_accumulator,
    finalize_class_state_metric_accumulator,
)

logger = logging.getLogger("train")


# ======================================================================
# Training step
# ======================================================================


def _build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


def _new_train_noise_audit_accumulator(audit_cfg):
    if not bool((audit_cfg or {}).get("ENABLED", False)):
        return None
    return NoiseAuditAccumulator()


def _resolve_train_noise_audit_state(
    use_det_update,
    obs_future_pos,
    obs_future_match,
):
    """
    Project the training teacher-forcing regime onto the shared audit state axis.

    This is not a literal tracker lifecycle state. It maps detector/fusion
    teacher-forced observations to the shared "matched" bucket and the
    GT-noisy fallback path to the shared "unmatched" bucket so training and
    inference summaries stay schema-compatible.
    """
    if use_det_update and obs_future_pos is not None and obs_future_match is not None:
        return "matched"
    return "unmatched"


def _record_train_noise_audit_samples(
    noise_audit,
    *,
    filter_mode,
    audit_state,
    categories,
    class_ids,
    history_mask,
    q_pos,
    r_pos,
    r_siz,
    r_ori,
    prior_q_pos,
    prior_r_pos,
    prior_r_siz,
    prior_r_ori,
):
    if noise_audit is None:
        return

    # `audit_state` is already projected onto the shared matched/unmatched
    # axis so the JSON stays comparable with inference-side summaries.
    def _to_list(values):
        if values is None:
            return None
        if hasattr(values, "detach"):
            values = values.detach()
        if hasattr(values, "cpu"):
            values = values.cpu()
        if hasattr(values, "tolist"):
            return values.tolist()
        return list(values)

    history_rows = _to_list(history_mask) or []
    history_lens = [sum(1 for value in row if bool(value)) for row in history_rows]
    class_id_list = _to_list(class_ids) or []
    class_names = [category_to_tracking_name(category) for category in categories]

    samples = build_noise_audit_samples(
        mode=filter_mode,
        traj_labels=class_id_list,
        matched_mask=[True] * len(class_names),
        history_lens=history_lens,
        q_pos=_to_list(q_pos),
        r_pos=_to_list(r_pos),
        r_siz=_to_list(r_siz),
        r_ori=_to_list(r_ori),
        prior_q_pos=_to_list(prior_q_pos),
        prior_r_pos=_to_list(prior_r_pos),
        prior_r_siz=_to_list(prior_r_siz),
        prior_r_ori=_to_list(prior_r_ori),
    )

    for class_name, sample in zip(class_names, samples):
        noise_audit.add_sample(
            split="train",
            mode=sample["mode"],
            class_id=sample["class_id"],
            class_name=class_name,
            state=audit_state,
            history_len=sample["history_len"],
            families=sample["families"],
            prior_families=sample["prior_families"],
        )


def _dump_train_noise_audit_if_needed(noise_audit, audit_cfg, logger_obj=None):
    if noise_audit is None:
        return

    output_path = (audit_cfg or {}).get(
        "TRAIN_OUTPUT_PATH", "debug/train_noise_audit.json"
    )
    try:
        noise_audit.write_json(output_path)
    except Exception as exc:
        if (audit_cfg or {}).get("STRICT", False):
            raise
        message = f"Failed to write training noise audit to {output_path}: {exc}"
        if logger_obj is not None:
            logger_obj.warning(message)
        else:
            print(message)


def _trace_covariance_batch(cov: torch.Tensor) -> torch.Tensor:
    return cov.diagonal(dim1=-2, dim2=-1).sum(-1)


def _compute_ratio_anchor_regularization(
    *,
    raw_tensors: dict,
    prior_tensors: dict,
    class_ids: torch.Tensor,
    state_buckets: list,
    closure_cfg: dict,
) -> tuple:
    device = class_ids.device
    dtype = next(iter(raw_tensors.values())).dtype
    zero = torch.zeros((), device=device, dtype=dtype)
    detail = {
        "loss_ratio_q_pos": zero,
        "loss_ratio_r_pos": zero,
        "loss_ratio_r_siz": zero,
        "loss_ratio_r_ori": zero,
    }
    if not closure_cfg or not bool(closure_cfg.get("ENABLED", False)):
        return zero, detail

    weight = float(closure_cfg.get("RATIO_ANCHOR_WEIGHT", 0.0))
    if weight <= 0.0:
        return zero, detail

    class_id_list = [
        int(value.item()) if hasattr(value, "item") else int(value)
        for value in class_ids
    ]
    state_bucket_list = [str(bucket) for bucket in state_buckets]
    if len(class_id_list) != len(state_bucket_list):
        raise ValueError("class_ids and state_buckets must have the same batch length")

    family_specs = {
        "loss_ratio_q_pos": ("Q_pos", "Q_pos_base", "q_pos"),
        "loss_ratio_r_pos": ("R_pos", "R_pos_base", "r_pos"),
        "loss_ratio_r_siz": ("R_siz", "R_siz_base", "r_siz"),
        "loss_ratio_r_ori": ("R_ori", "R_ori_base", "r_ori"),
    }
    total = zero
    eps = torch.tensor(1e-8, device=device, dtype=dtype)

    for detail_key, (raw_key, prior_key, family_name) in family_specs.items():
        raw_cov = raw_tensors.get(raw_key)
        prior_cov = prior_tensors.get(prior_key)
        if raw_cov is None or prior_cov is None:
            continue

        min_ratios = []
        max_ratios = []
        mask = []
        has_bounded_sample = False
        for class_id, state_bucket in zip(class_id_list, state_bucket_list):
            bounds = get_family_ratio_bounds(class_id, state_bucket, family_name, closure_cfg)
            if bounds is None:
                min_ratios.append(0.0)
                max_ratios.append(float("inf"))
                mask.append(False)
                continue
            has_bounded_sample = True
            min_ratios.append(bounds[0])
            max_ratios.append(bounds[1])
            mask.append(True)

        if not has_bounded_sample:
            continue

        min_ratio_t = torch.tensor(min_ratios, device=device, dtype=dtype)
        max_ratio_t = torch.tensor(max_ratios, device=device, dtype=dtype)
        mask_t = torch.tensor(mask, device=device, dtype=torch.bool)

        raw_trace = _trace_covariance_batch(raw_cov)
        prior_trace = torch.clamp(_trace_covariance_batch(prior_cov), min=eps)
        raw_ratio = raw_trace / prior_trace
        valid_ratio = torch.isfinite(raw_ratio) & (raw_ratio > eps)
        safe_ratio = torch.where(valid_ratio, raw_ratio, max_ratio_t)

        log_ratio = torch.log(torch.clamp(safe_ratio, min=eps))
        log_min = torch.log(torch.clamp(min_ratio_t, min=eps))
        log_max = torch.log(torch.clamp(max_ratio_t, min=eps))
        lower_violation = F.relu(log_min - log_ratio)
        upper_violation = F.relu(log_ratio - log_max)
        penalty = (lower_violation.pow(2) + upper_violation.pow(2))[mask_t]
        if penalty.numel() == 0:
            continue
        family_loss = penalty.mean() * weight
        detail[detail_key] = family_loss
        total = total + family_loss

    return total, detail


def _resolve_closure_ratio_bounds(
    class_id: int,
    state_bucket: str,
    family_name: str,
    closure_cfg: dict,
):
    bounds = get_family_ratio_bounds(class_id, state_bucket, family_name, closure_cfg)
    matched_band = (closure_cfg or {}).get("MATCHED_KF_BAND", {})
    if str(state_bucket) != "matched" or not bool(matched_band.get("ENABLED", False)):
        return bounds

    families = matched_band.get("FAMILIES", {}) or {}
    band_bounds = families.get(family_name)
    class_overrides = matched_band.get("CLASS_OVERRIDES", {}) or {}
    class_cfg = class_overrides.get(int(class_id), class_overrides.get(str(int(class_id)), {})) or {}
    band_bounds = class_cfg.get(family_name, band_bounds)

    if not isinstance(band_bounds, (list, tuple)) or len(band_bounds) != 2:
        return bounds
    return float(band_bounds[0]), float(band_bounds[1])


def _compute_closure_ratio_regularization(
    *,
    ratios: dict,
    class_ids: torch.Tensor,
    state_buckets: list,
    closure_cfg: dict,
) -> tuple:
    device = class_ids.device
    dtype = next(iter(ratios.values())).dtype if ratios else torch.float32
    zero = torch.zeros((), device=device, dtype=dtype)
    detail = {
        "loss_ratio_anchor_q_pos": zero,
        "loss_ratio_anchor_r_pos": zero,
        "loss_ratio_anchor_r_siz": zero,
        "loss_ratio_anchor_r_ori": zero,
        "loss_ratio_bound_q_pos": zero,
        "loss_ratio_bound_r_pos": zero,
        "loss_ratio_bound_r_siz": zero,
        "loss_ratio_bound_r_ori": zero,
    }
    if not ratios or not closure_cfg or not bool(closure_cfg.get("ENABLED", False)):
        return zero, zero, detail

    anchor_weight = float(closure_cfg.get("RATIO_ANCHOR_WEIGHT", 0.0))
    bound_weight = float(closure_cfg.get("RATIO_BOUND_WEIGHT", 0.0))
    lower_weight = float(closure_cfg.get("RATIO_BOUND_LOWER_WEIGHT", 0.25))
    upper_weight = float(closure_cfg.get("RATIO_BOUND_UPPER_WEIGHT", 1.0))
    if anchor_weight <= 0.0 and bound_weight <= 0.0:
        return zero, zero, detail

    class_id_list = [
        int(value.item()) if hasattr(value, "item") else int(value)
        for value in class_ids
    ]
    state_bucket_list = [str(bucket) for bucket in state_buckets]
    if len(class_id_list) != len(state_bucket_list):
        raise ValueError("class_ids and state_buckets must have the same batch length")

    family_groups = {
        "q_pos": ("q_pos_xyz", "q_pos_vxyz"),
        "r_pos": ("r_pos_xyz", "r_pos_vxy"),
        "r_siz": ("r_siz_lw", "r_siz_h"),
        "r_ori": ("r_ori",),
    }

    total_anchor = zero
    total_bound = zero
    for family_name, ratio_keys in family_groups.items():
        min_ratios = []
        max_ratios = []
        mask = []
        has_bounded_sample = False
        for class_id, state_bucket in zip(class_id_list, state_bucket_list):
            bounds = _resolve_closure_ratio_bounds(class_id, state_bucket, family_name, closure_cfg)
            if bounds is None:
                min_ratios.append(1.0)
                max_ratios.append(1.0)
                mask.append(False)
                continue
            has_bounded_sample = True
            min_ratios.append(bounds[0])
            max_ratios.append(bounds[1])
            mask.append(True)

        if not has_bounded_sample:
            continue

        mask_t = torch.tensor(mask, device=device, dtype=torch.bool)
        min_ratio_t = torch.tensor(min_ratios, device=device, dtype=dtype)[mask_t]
        max_ratio_t = torch.tensor(max_ratios, device=device, dtype=dtype)[mask_t]

        anchor_terms = []
        bound_terms = []
        for ratio_key in ratio_keys:
            gamma = ratios.get(ratio_key)
            if gamma is None:
                continue
            gamma_masked = gamma[mask_t]
            if gamma_masked.numel() == 0:
                continue
            if anchor_weight > 0.0:
                anchor_terms.append(log_ratio_anchor_loss(gamma_masked))
            if bound_weight > 0.0:
                bound_terms.append(
                    log_ratio_bound_loss(
                        gamma_masked,
                        min_ratio_t,
                        max_ratio_t,
                        lower_weight=lower_weight,
                        upper_weight=upper_weight,
                    )
                )

        if anchor_terms:
            family_anchor = torch.stack(anchor_terms).sum() * anchor_weight
            detail[f"loss_ratio_anchor_{family_name}"] = family_anchor
            total_anchor = total_anchor + family_anchor
        if bound_terms:
            family_bound = torch.stack(bound_terms).sum() * bound_weight
            detail[f"loss_ratio_bound_{family_name}"] = family_bound
            total_bound = total_bound + family_bound

    return total_anchor, total_bound, detail


def resolve_runtime_contract_filter_mode(cfg, train_tracker_compat_mode):
    valid_modes = {"mamba", "pure_dekf", "fusion", "mamba_multihead_closure"}

    configured_mode = str((cfg or {}).get("FILTER_MODE", "")).strip().lower()
    if configured_mode in valid_modes:
        return configured_mode

    compat_fallback = str(train_tracker_compat_mode).strip().lower()
    if compat_fallback in valid_modes:
        return compat_fallback

    return "mamba"


def resolve_orientation_curriculum_weights(epoch: int, closure_cfg: dict) -> dict:
    cfg = closure_cfg or {}
    warmup_epochs = int(cfg.get("ORI_WARMUP_EPOCHS", 0))
    transition_epochs = int(cfg.get("ORI_TRANSITION_EPOCHS", 0))
    base_state = float(cfg.get("ORI_STATE_WEIGHT", 1.0))
    base_wrapped = float(cfg.get("ORI_WRAPPED_NLL_WEIGHT", 1.0))

    if epoch < warmup_epochs:
        alpha = 0.0
    elif transition_epochs <= 0:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, (epoch - warmup_epochs + 1) / float(transition_epochs)))

    return {
        "alpha": alpha,
        "state_weight": base_state * (1.0 - alpha),
        "wrapped_weight": base_wrapped * alpha,
    }

def training_step(
    mamba: TemporalMamba,
    batch: dict,
    loss_fn: JointLoss,
    device: torch.device,
    noise_cfg: dict = None,
    base_noise_cfg: dict = None,
    epoch: int = 0,
    warmup_epochs: int = 0,
    transition_epochs: int = 0,
    vel_warmup_epochs: int = 3,
    train_noise_audit: NoiseAuditAccumulator = None,
    filter_mode: str = "mamba",
    emit_sample_metrics: bool = False,
) -> tuple:
    """
    Multi-step KF rollout training with noisy teacher forcing.

    Mamba runs ONCE to predict Q/R/kappa from the history window.
    The KF is initialised from a NOISY GT state (simulating tracking
    uncertainty), then runs K predict-update steps. Noisy GT measurements
    are fed to KF.update (teacher forcing), but the NLL loss at each step
    compares predictions to CLEAN GT.

    Confidence warmup (epoch < warmup_epochs):
      Q/R/kappa are detached → constant noise → gradients only flow through
      state predictions (x_pred, P_pred), forcing Mamba backbone to learn
      accurate feature representations before the uncertainty heads are
      released. Orientation uses angle_loss (1-cos) instead of Von Mises NLL.

    Returns:
        loss   : scalar tensor
        detail : dict of sub-loss values
    """
    B = batch["track_history"].shape[0]
    K = batch["gt_future_pos"].shape[1]  # rollout steps from dataset

    history = batch["track_history"].to(device)            # [B, T, 12]
    gt_pos = batch["gt_current_state_pos"].to(device)      # [B, 6]
    gt_siz = batch["gt_current_state_siz"].to(device)      # [B, 3]
    gt_ori = batch["gt_current_state_ori"].to(device)      # [B, 2]
    obs_pos = batch.get("obs_current_state_pos", batch["gt_current_state_pos"]).to(device)
    obs_siz = batch.get("obs_current_state_siz", batch["gt_current_state_siz"]).to(device)
    obs_ori = batch.get("obs_current_state_ori", batch["gt_current_state_ori"]).to(device)
    gt_future_pos = batch["gt_future_pos"].to(device)      # [B, K, 3]
    gt_future_siz = batch["gt_future_siz"].to(device)      # [B, K, 3]
    gt_future_ori = batch["gt_future_ori"].to(device)      # [B, K, 1]
    obs_future_pos = batch.get("obs_future_pos", None)
    obs_future_siz = batch.get("obs_future_siz", None)
    obs_future_ori = batch.get("obs_future_ori", None)
    obs_future_match = batch.get("obs_future_match", None)
    future_mask = batch.get("future_mask", None)
    if obs_future_pos is not None:
        obs_future_pos = obs_future_pos.to(device)
    if obs_future_siz is not None:
        obs_future_siz = obs_future_siz.to(device)
    if obs_future_ori is not None:
        obs_future_ori = obs_future_ori.to(device)
    if obs_future_match is not None:
        obs_future_match = obs_future_match.to(device)
    if future_mask is not None:
        future_mask = future_mask.to(device)
    delta_ts_future = batch["delta_ts_future"].to(device)  # [B, K]
    instance_tokens = batch["instance_token"]              # list of B strings
    categories = batch["category"]                          # list of B strings
    current_range = batch.get("current_range", torch.zeros(B)).to(device)
    detection_driven_mask = batch.get("is_detection_driven", torch.zeros(B, dtype=torch.bool)).to(device)
    history_mask = batch.get("history_mask", torch.ones(history.shape[:2], dtype=torch.bool)).to(device)
    history_match_mask = batch.get("history_match_mask", history_mask).to(device)
    use_det_update = bool((base_noise_cfg or {}).get("DETECTION_UPDATE", {}).get("ENABLED", True))
    miss_r_scale = float((base_noise_cfg or {}).get("DETECTION_UPDATE", {}).get("MISS_R_SCALE", 1000.0))

    class_ids = categories_to_class_ids(categories, device=device)  # [B]

    # ---- Step 1: TemporalMamba forward (ONCE) → Q/R/embedding ----
    branch_name = str(filter_mode).strip().lower()
    use_closure_loss_path = branch_name == "mamba_multihead_closure"
    closure_cfg = (base_noise_cfg or {}).get("MAMBA_CLOSURE", {})
    audit_state = _resolve_train_noise_audit_state(
        use_det_update,
        obs_future_pos,
        obs_future_match,
    )
    state_buckets = [audit_state] * B

    mamba_out = mamba(
        history,
        class_ids=class_ids,
        current_range=current_range,
        detection_driven_mask=detection_driven_mask,
        history_mask=history_mask,
        history_match_mask=history_match_mask,
        state_buckets=state_buckets,
        apply_force_prior=not bool(closure_cfg.get("TRAIN_MATCHED_HEAD", {}).get("ENABLED", False)),
        mode=branch_name,
    )

    # ---- Step 2: Init KF from GT state at frame T (+ noise) ----
    if noise_cfg is None:
        noise_cfg = {
            "POS_STD_XY": [
                [0.5210, 0.5045],  # car
                [0.2745, 0.2731],  # pedestrian
                [0.5562, 0.5444],  # bicycle
                [0.8646, 0.8695],  # motorcycle
                [1.3037, 1.2327],  # bus
                [0.9590, 0.8735],  # trailer
                [0.6580, 0.6269],  # truck
            ],
            "SIZ_STD_LW": [
                [0.2417, 0.0799],  # car
                [0.0567, 0.0422],  # pedestrian
                [0.1118, 0.0661],  # bicycle
                [0.1866, 0.0860],  # motorcycle
                [1.2094, 0.1465],  # bus
                [1.9336, 0.1782],  # trailer
                [0.7685, 0.1569],  # truck
            ],
            "VEL_STD_XY": [
                [0.7551, 0.7017],  # car
                [0.2494, 0.2409],  # pedestrian
                [0.3933, 0.3942],  # bicycle
                [0.8261, 0.8865],  # motorcycle
                [1.2134, 1.1046],  # bus
                [0.4770, 0.3982],  # trailer
                [0.6409, 0.6169],  # truck
            ],
            "ORI_STD": [0.50, 0.91, 0.86, 0.96, 0.48, 0.53, 0.45],
            "MEAS_MULTIPLIER": 0.7
        }
    train_noise_cfg = {
        "Q": (base_noise_cfg or {}).get("Q", {"POS": [0.5, 0.5, 0.5, 1.5, 1.5, 1.5], "SIZ": [0.05, 0.05, 0.05], "ORI": [0.1]}),
        "R": noise_cfg,
        "CONDITIONAL_NOISE": (base_noise_cfg or {}).get("CONDITIONAL_NOISE", {}),
    }
    noise_bundle = build_base_covariances(
        base_noise_cfg=train_noise_cfg,
        class_ids=class_ids,
        dtype=history.dtype,
        device=device,
        track_history=history,
        current_range=current_range,
        detection_driven_mask=detection_driven_mask,
        history_mask=history_mask,
        history_match_mask=history_match_mask,
    )
    noise_scale_pos_xyz = noise_bundle["pos_std_xyz"]
    noise_scale_siz_lwh = noise_bundle["siz_std_lwh"]
    noise_scale_vel_xy = noise_bundle["vel_std_xy"]
    noise_scale_ori = noise_bundle["ori_std"].view(B, 1, 1)
    ori_curriculum = resolve_orientation_curriculum_weights(epoch=epoch, closure_cfg=closure_cfg)
    ori_state_weight = float(ori_curriculum["state_weight"])
    ori_wrapped_weight = float(ori_curriculum["wrapped_weight"])
    ori_curriculum_weight_sum = max(ori_state_weight + ori_wrapped_weight, 1e-8)

    runtime_prior_cov = mamba_out.get("prior_covariances") or {}
    audit_prior_q_pos = runtime_prior_cov.get("Q_pos", noise_bundle["Q_pos_base"])
    audit_prior_r_pos = runtime_prior_cov.get("R_pos", noise_bundle["R_pos_base"])
    audit_prior_r_siz = runtime_prior_cov.get("R_siz", noise_bundle["R_siz_base"])
    audit_prior_r_ori = runtime_prior_cov.get("R_ori", noise_bundle["R_ori_base"])

    if train_noise_audit is not None:
        # Training does not have literal tracker matched/unmatched lifecycle
        # states here. Project the teacher-forcing mode onto that shared axis
        # so the audit schema remains comparable with inference output.
        _record_train_noise_audit_samples(
            train_noise_audit,
            filter_mode=filter_mode,
            audit_state=audit_state,
            categories=categories,
            class_ids=class_ids,
            history_mask=history_mask,
            q_pos=mamba_out["Q_pos"].diagonal(dim1=-2, dim2=-1).sum(-1),
            r_pos=mamba_out["R_pos"].diagonal(dim1=-2, dim2=-1).sum(-1),
            r_siz=mamba_out["R_siz"].diagonal(dim1=-2, dim2=-1).sum(-1),
            r_ori=mamba_out["R_ori"].diagonal(dim1=-2, dim2=-1).sum(-1),
            prior_q_pos=audit_prior_q_pos.diagonal(dim1=-2, dim2=-1).sum(-1),
            prior_r_pos=audit_prior_r_pos.diagonal(dim1=-2, dim2=-1).sum(-1),
            prior_r_siz=audit_prior_r_siz.diagonal(dim1=-2, dim2=-1).sum(-1),
            prior_r_ori=audit_prior_r_ori.diagonal(dim1=-2, dim2=-1).sum(-1),
        )

    # delta_pos gives Mamba a direct gradient path through loss_pos,
    # not only through R_pos via the KF update.
    # Detached during warmup so Mamba backbone learns features first.
    delta_pos = mamba_out["delta_pos"].unsqueeze(-1)                  # [B, 6, 1]
    if epoch < warmup_epochs:
        delta_pos = delta_pos.detach()

    pos_x0 = obs_pos.unsqueeze(-1) + delta_pos                       # [B, 6, 1]
    siz_x0 = obs_siz.unsqueeze(-1)                                   # [B, 3, 1]
    ori_x0 = obs_ori.unsqueeze(-1)                                   # [B, 2, 1]

    # Category noise (added on top of delta, as extra perturbation)
    pos_noise_std_6 = torch.cat(
        [noise_scale_pos_xyz, torch.zeros(B, 3, device=device)], dim=1
    ).unsqueeze(-1)  # [B,6,1]
    pos_x0 = pos_x0 + torch.randn_like(pos_x0) * pos_noise_std_6
    siz_x0 = siz_x0 + torch.randn_like(siz_x0) * noise_scale_siz_lwh.unsqueeze(-1)
    ori_x0 = ori_x0 + torch.randn_like(ori_x0) * noise_scale_ori * torch.tensor(
        [1.0, 0.0], device=device).view(1, 2, 1)  # noise on theta, not omega

    pos_P0 = torch.eye(6, device=device).unsqueeze(0).expand(B, -1, -1).clone()
    pos_P0[:, 3, 3] = 10.0   # vx variance: high — K can infer speed from position
    pos_P0[:, 4, 4] = 10.0   # vy variance
    siz_P0 = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1).clone() * 0.1
    ori_P0 = torch.eye(2, device=device).unsqueeze(0).expand(B, -1, -1).clone() * 0.1

    kf = DecoupledAdaptiveKF(batch_size=B, device=device)
    kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)

    # Measurement noise for teacher forcing — scaled proportionally to std mapped from category
    meas_mult = noise_cfg.get("MEAS_MULTIPLIER", 0.7)
    meas_noise_pos = noise_scale_pos_xyz.unsqueeze(-1) * meas_mult  # [B, 3, 1]
    meas_noise_siz = noise_scale_siz_lwh.unsqueeze(-1) * meas_mult  # [B, 3, 1]
    meas_noise_ori = noise_scale_ori * meas_mult
    meas_noise_vel = noise_scale_vel_xy.unsqueeze(-1) * meas_mult    # [B, 2, 1]

    # ---- Confidence warmup + smooth unfreeze ----
    # Hard warmup: alpha=0 (fully detached uncertainty heads).
    # Transition: alpha ramps 0->1 across transition_epochs.
    # Post-transition: alpha=1 (fully trainable uncertainty heads).
    in_warmup = epoch < warmup_epochs
    if epoch < warmup_epochs:
        unfreeze_alpha = 0.0
    elif transition_epochs <= 0:
        unfreeze_alpha = 1.0
    else:
        prog = (epoch - warmup_epochs + 1) / float(transition_epochs)
        unfreeze_alpha = float(max(0.0, min(1.0, prog)))

    # Velocity NLL warmup: delay for N epochs after uncertainty-unseal to let
    # position+orientation NLL stabilise before adding velocity supervision.
    vel_active = epoch >= warmup_epochs + vel_warmup_epochs

    # ---- Step 3: K-step KF rollout with teacher forcing ----
    Q_pos = mamba_out["Q_pos"]
    Q_siz = mamba_out["Q_siz"]
    Q_ori = mamba_out["Q_ori"]
    R_pos = mamba_out["R_pos"]
    R_siz = mamba_out["R_siz"]
    R_ori = mamba_out["R_ori"]
    kappa_ori = mamba_out["kappa_ori"]
    if use_closure_loss_path:
        regularization_ratios = dict(mamba_out.get("ratios") or {})
        closure_prior_r_ori = runtime_prior_cov.get("R_ori", noise_bundle["R_ori_base"])
        prior_r_ori = torch.clamp(
            closure_prior_r_ori.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True),
            min=1e-8,
        )
        effective_r_ori = (
            mamba_out["R_ori"].diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True) / prior_r_ori
        )
        regularization_ratios["r_ori"] = effective_r_ori
        ratio_anchor_loss, ratio_bound_loss, ratio_detail = _compute_closure_ratio_regularization(
            ratios=regularization_ratios,
            class_ids=class_ids,
            state_buckets=state_buckets,
            closure_cfg=closure_cfg,
        )
        ratio_anchor_detail = {
            "loss_ratio_q_pos": ratio_detail["loss_ratio_anchor_q_pos"],
            "loss_ratio_r_pos": ratio_detail["loss_ratio_anchor_r_pos"],
            "loss_ratio_r_siz": ratio_detail["loss_ratio_anchor_r_siz"],
            "loss_ratio_r_ori": ratio_detail["loss_ratio_anchor_r_ori"],
        }
    else:
        ratio_anchor_loss, ratio_anchor_detail = _compute_ratio_anchor_regularization(
            raw_tensors={
                "Q_pos": Q_pos,
                "R_pos": R_pos,
                "R_siz": R_siz,
                "R_ori": R_ori,
            },
            prior_tensors={
                "Q_pos_base": noise_bundle["Q_pos_base"],
                "R_pos_base": noise_bundle["R_pos_base"],
                "R_siz_base": noise_bundle["R_siz_base"],
                "R_ori_base": noise_bundle["R_ori_base"],
            },
            class_ids=class_ids,
            state_buckets=state_buckets,
            closure_cfg=closure_cfg,
        )
        ratio_bound_loss = torch.zeros((), device=device, dtype=history.dtype)
        ratio_detail = {}

    # Smoothly release gradients after warmup (avoid abrupt loss cliff).
    def _blend_detach(x: torch.Tensor, alpha: float) -> torch.Tensor:
        xd = x.detach()
        return xd + (x - xd) * alpha

    Q_pos = _blend_detach(Q_pos, unfreeze_alpha)
    Q_siz = _blend_detach(Q_siz, unfreeze_alpha)
    Q_ori = _blend_detach(Q_ori, unfreeze_alpha)
    R_pos = _blend_detach(R_pos, unfreeze_alpha)
    R_siz = _blend_detach(R_siz, unfreeze_alpha)
    R_ori = _blend_detach(R_ori, unfreeze_alpha)
    kappa_ori = _blend_detach(kappa_ori, unfreeze_alpha)

    # GT velocity for all rollout steps (constant-velocity approximation over K≤3 frames)
    vel_gt = gt_pos[:, 3:5]  # [B, 2]

    total_state_loss = 0.0
    total_contrast_loss = 0.0
    detail_accum = {}
    detail_contrastive = {"loss_contrastive": 0.0, "n_valid_anchors": 0}
    total_weight = 0.0
    total_aux_loss_tensor = torch.zeros((), device=device, dtype=history.dtype)
    sample_loss_sum = torch.zeros(B, device=device, dtype=history.dtype)
    sample_loss_count = torch.zeros(B, device=device, dtype=history.dtype)

    # Accumulate loss tensors across rollout steps for proper gradient flow.
    for k in range(K):
        active_k = future_mask[:, k].bool() if future_mask is not None else torch.ones(B, dtype=torch.bool, device=device)
        active_count = int(active_k.sum().item())
        if active_count == 0:
            continue
        step_weight = active_count / float(max(B, 1))
        total_weight += step_weight

        # Per-sample delta_t preserves individual object dynamics
        dt_k = delta_ts_future[:, k].clone()                  # [B]
        dt_k[~active_k] = 1e-6

        # KF predict with same Q for all steps
        pos_x_pred, pos_P_pred, siz_x_pred, siz_P_pred, ori_x_pred, ori_P_pred = \
            kf.predict(dt_k, Q_pos=Q_pos, Q_siz=Q_siz, Q_ori=Q_ori)

        # NLL vs GT at this rollout step
        loss_k, detail_k = loss_fn(
            pos_x_pred[active_k], pos_P_pred[active_k],
            siz_x_pred[active_k], siz_P_pred[active_k],
            ori_x_pred[active_k], ori_P_pred[active_k],
            gt_future_pos[:, k, :][active_k], gt_future_siz[:, k, :][active_k], gt_future_ori[:, k, :][active_k],
            mamba_out["embedding"][active_k] if k == 0 else None,
            [tok for idx, tok in enumerate(instance_tokens) if bool(active_k[idx].item())] if k == 0 else None,
            R_pos=R_pos[active_k], R_siz=R_siz[active_k], R_ori=R_ori[active_k],
            kappa_ori=kappa_ori[active_k],
            gt_next_vel=vel_gt[active_k] if vel_active else None,
            in_warmup=in_warmup,
            ori_nll_alpha=unfreeze_alpha,
            class_ids=class_ids[active_k],
            use_wrapped_orientation_nll=use_closure_loss_path,
            return_per_sample=emit_sample_metrics,
        )

        if use_closure_loss_path:
            blended_ori_tensor = (
                ori_state_weight * detail_k["loss_ori_state_tensor"]
                + ori_wrapped_weight * detail_k["loss_ori_wrapped_tensor"]
            ) / ori_curriculum_weight_sum
            ori_delta_tensor = (
                blended_ori_tensor - detail_k["loss_ori_wrapped_tensor"]
            ) * float(loss_fn.state_loss.w_ori)
            loss_k = loss_k + loss_fn.physics_scale * ori_delta_tensor
            detail_k["loss_ori"] = blended_ori_tensor.item()
            detail_k["loss_state"] = detail_k["loss_state"] + ori_delta_tensor.item()
            detail_k["loss_total"] = loss_k.item()
            detail_k["loss_ori_curriculum"] = blended_ori_tensor.item()
            if emit_sample_metrics and "loss_total_per_sample" in detail_k:
                blended_ori_per_sample = (
                    ori_state_weight * detail_k["_loss_ori_state_per_sample"]
                    + ori_wrapped_weight * detail_k["_loss_ori_wrapped_per_sample"]
                ) / ori_curriculum_weight_sum
                ori_delta_per_sample = (
                    blended_ori_per_sample - detail_k["_loss_ori_wrapped_per_sample"]
                ) * float(loss_fn.state_loss.w_ori)
                detail_k["loss_state_per_sample"] = (
                    detail_k["loss_state_per_sample"] + ori_delta_per_sample
                )
                detail_k["loss_total_per_sample"] = (
                    detail_k["loss_total_per_sample"]
                    + loss_fn.physics_scale * ori_delta_per_sample
                )

            aux_pos = F.huber_loss(
                pos_x_pred[active_k, 0:3, 0],
                gt_future_pos[:, k, :][active_k],
                reduction="mean",
            )
            aux_siz = F.huber_loss(
                siz_x_pred[active_k, :, 0],
                gt_future_siz[:, k, :][active_k],
                reduction="mean",
            )
            aux_ori = (
                1.0
                - torch.cos(
                    wrap_to_pi_torch(
                        ori_x_pred[active_k, 0, 0] - gt_future_ori[:, k, 0][active_k]
                    )
                )
            ).mean()
            aux_loss_k = 0.05 * aux_pos + 0.02 * aux_siz + 0.02 * aux_ori
            total_aux_loss_tensor = total_aux_loss_tensor + aux_loss_k * step_weight
            detail_accum["loss_aux_pos"] = detail_accum.get("loss_aux_pos", 0.0) + aux_pos.item() * step_weight
            detail_accum["loss_aux_siz"] = detail_accum.get("loss_aux_siz", 0.0) + aux_siz.item() * step_weight
            detail_accum["loss_aux_ori"] = detail_accum.get("loss_aux_ori", 0.0) + aux_ori.item() * step_weight
            detail_accum["loss_aux_total"] = detail_accum.get("loss_aux_total", 0.0) + aux_loss_k.item() * step_weight

        if emit_sample_metrics:
            sample_loss_k = detail_k.get("loss_total_per_sample", detail_k.get("loss_state_per_sample"))
            if sample_loss_k is not None:
                # Class selection uses sample-local physics state loss only.
                # Contrastive and scalar/global regularizers are batch-dependent and excluded.
                sample_loss_sum[active_k] = sample_loss_sum[active_k] + sample_loss_k.detach()
                sample_loss_count[active_k] = sample_loss_count[active_k] + 1.0

        total_state_loss += detail_k["loss_state"] * step_weight   # for logging only
        if k == 0:
            total_loss_tensor = loss_k * step_weight
            total_contrast_loss = detail_k["loss_contrastive"]
        else:
            total_loss_tensor = total_loss_tensor + loss_k * step_weight

        for key, val in detail_k.items():
            if key == "loss_contrastive":
                detail_contrastive["loss_contrastive"] = max(
                    detail_contrastive["loss_contrastive"], val)
            elif key == "n_valid_anchors":
                detail_contrastive["n_valid_anchors"] = max(
                    detail_contrastive["n_valid_anchors"], int(val))
            elif torch.is_tensor(val):
                continue
            elif key == "loss_total":
                pass
            else:
                detail_accum[key] = detail_accum.get(key, 0.0) + val * step_weight

        # Monitor delta_pos norm (per-step, tracked inside rollout).
        detail_accum["delta_pos_norm"] = detail_accum.get("delta_pos_norm", 0.0) + \
            mamba_out["delta_pos"].norm(dim=-1).mean().item() * step_weight

        # Noisy teacher forcing with category-aware measurement noise
        if use_det_update and obs_future_pos is not None and obs_future_match is not None:
            match_k = obs_future_match[:, k].bool() & active_k  # [B]

            pred_obs_pos = torch.cat([
                pos_x_pred[:, 0:3, 0],
                pos_x_pred[:, 3:5, 0],
            ], dim=1)  # [B,5]
            pred_obs_siz = siz_x_pred[:, :, 0]       # [B,3]
            pred_obs_ori = ori_x_pred[:, 0:1, 0]     # [B,1]

            z_pos_vals = pred_obs_pos.clone()
            z_siz_vals = pred_obs_siz.clone()
            z_ori_vals = pred_obs_ori.clone()

            if match_k.any():
                z_pos_vals[match_k] = obs_future_pos[:, k, :][match_k]
                z_siz_vals[match_k] = obs_future_siz[:, k, :][match_k]
                z_ori_vals[match_k] = obs_future_ori[:, k, :][match_k]

            z_pos_k = z_pos_vals.unsqueeze(-1)
            z_siz_k = z_siz_vals.unsqueeze(-1)
            z_ori_k = z_ori_vals.unsqueeze(-1)

            R_pos_step = R_pos.clone()
            R_siz_step = R_siz.clone()
            R_ori_step = R_ori.clone()
            miss_k = ~match_k
            if miss_k.any():
                R_pos_step[miss_k] = R_pos_step[miss_k] * miss_r_scale
                R_siz_step[miss_k] = R_siz_step[miss_k] * miss_r_scale
                R_ori_step[miss_k] = R_ori_step[miss_k] * miss_r_scale

            kf.update(z_pos_k, z_siz_k, z_ori_k,
                      R_pos=R_pos_step, R_siz=R_siz_step, R_ori=R_ori_step)
        else:
            pred_obs_pos = torch.cat([
                pos_x_pred[:, 0:3, 0],
                pos_x_pred[:, 3:5, 0],
            ], dim=1)  # [B,5]
            pred_obs_siz = siz_x_pred[:, :, 0]
            pred_obs_ori = ori_x_pred[:, 0:1, 0]
            gt_vel_xy = gt_pos[:, 3:5]  # [B, 2] — GT velocity
            z_pos_vals = torch.cat([
                gt_future_pos[:, k, :] + torch.randn(B, 3, device=device) * meas_noise_pos.squeeze(-1),
                gt_vel_xy + torch.randn(B, 2, device=device) * meas_noise_vel.squeeze(-1),
            ], dim=1)
            z_siz_vals = gt_future_siz[:, k, :] + torch.randn(B, 3, device=device) * meas_noise_siz.squeeze(-1)
            z_ori_vals = gt_future_ori[:, k, :] + torch.randn(B, 1, device=device) * meas_noise_ori.squeeze(-1)
            if (~active_k).any():
                z_pos_vals[~active_k] = pred_obs_pos[~active_k]
                z_siz_vals[~active_k] = pred_obs_siz[~active_k]
                z_ori_vals[~active_k] = pred_obs_ori[~active_k]
            z_pos_k = z_pos_vals.unsqueeze(-1)
            z_siz_k = z_siz_vals.unsqueeze(-1)
            z_ori_k = z_ori_vals.unsqueeze(-1)
            R_pos_step = R_pos.clone()
            R_siz_step = R_siz.clone()
            R_ori_step = R_ori.clone()
            if (~active_k).any():
                R_pos_step[~active_k] = R_pos_step[~active_k] * miss_r_scale
                R_siz_step[~active_k] = R_siz_step[~active_k] * miss_r_scale
                R_ori_step[~active_k] = R_ori_step[~active_k] * miss_r_scale
            kf.update(z_pos_k, z_siz_k, z_ori_k,
                      R_pos=R_pos_step, R_siz=R_siz_step, R_ori=R_ori_step)

    # ---- Logging detail (uses .item() floats, no gradient impact) ----
    norm = max(total_weight, 1e-6)
    detail = {k: v / norm for k, v in detail_accum.items()}
    detail["loss_contrastive"] = detail_contrastive["loss_contrastive"]
    detail["n_valid_anchors"] = detail_contrastive["n_valid_anchors"]
    detail["loss_total"] = (
        loss_fn.physics_scale * detail["loss_state"] +
        loss_fn.lambda_contrast * detail["loss_contrastive"]
    )

    if use_closure_loss_path:
        kappa_reg = torch.zeros((), device=device, dtype=history.dtype)
        ori_saturation_reg_weight = float(closure_cfg.get("ORI_SATURATION_REG_WEIGHT", 0.0))
        ori_max_effective_kappa = float(closure_cfg.get("ORI_MAX_EFFECTIVE_KAPPA", 5.0))
        if ori_saturation_reg_weight > 0.0:
            ori_saturation_reg = ori_saturation_reg_weight * orientation_saturation_penalty(
                mamba_out["kappa_ori_unc"],
                max_effective_kappa=ori_max_effective_kappa,
            )
        else:
            ori_saturation_reg = torch.zeros((), device=device, dtype=history.dtype)
    else:
        # κ overconfidence penalty: fires when pre-clamp κ > 5.0 (R_ori < 0.2).
        # Must use kappa_ori_unc (pre-clamp) — kappa_ori is already clamped to ≤ 5.0
        # so ReLU(kappa_ori - 5.0) would always be zero.
        kappa_reg = 5e-3 * F.relu(mamba_out["kappa_ori_unc"] - 5.0).mean()
        ori_saturation_reg = torch.zeros((), device=device, dtype=history.dtype)
    detail["loss_kappa_reg"] = kappa_reg.item()
    detail["loss_ori_saturation_reg"] = ori_saturation_reg.item()

    # Δpos L2 penalty: mild regularisation to keep delta_pos bounded.
    delta_pos_reg = 0.01 * mamba_out["delta_pos"].pow(2).mean()
    detail["loss_delta_pos_reg"] = delta_pos_reg.item()
    for key, value in ratio_anchor_detail.items():
        detail[key] = value.item()
    for key, value in ratio_detail.items():
        detail[key] = value.item()
    detail["loss_ratio_anchor"] = ratio_anchor_loss.item()
    detail["loss_ratio_bound"] = ratio_bound_loss.item()
    detail["ori_curriculum_alpha"] = ori_curriculum["alpha"] if use_closure_loss_path else 0.0
    detail["ori_state_weight"] = ori_curriculum["state_weight"] if use_closure_loss_path else 0.0
    detail["ori_wrapped_weight"] = ori_curriculum["wrapped_weight"] if use_closure_loss_path else 0.0

    # ---- Step 4: Q/R/kappa variance monitor ----
    with torch.no_grad():
        for name, mat in [
            ("Q_pos", mamba_out["Q_pos"]), ("Q_siz", mamba_out["Q_siz"]),
            ("R_pos", mamba_out["R_pos"]), ("R_siz", mamba_out["R_siz"]),
        ]:
            diag = mat.diagonal(dim1=-2, dim2=-1)
            detail[f"std_{name}"] = diag.std(dim=0).mean().item()
        detail["std_kappa"] = mamba_out["kappa_ori_unc"].std(dim=0).mean().item()
        detail["mean_kappa"] = mamba_out["kappa_ori_unc"].mean().item()
        detail["cond_pos_scale"] = noise_bundle["scales"]["pos"].mean().item()
        detail["cond_vel_scale"] = noise_bundle["scales"]["vel"].mean().item()
        detail["cond_ori_scale"] = noise_bundle["scales"]["ori"].mean().item()
        detail["cond_siz_scale"] = noise_bundle["scales"]["siz"].mean().item()
        detail["matched_ratio"] = noise_bundle["scales"]["matched_ratio"].mean().item()
        detail["effective_r_ori_mean"] = mamba_out["R_ori"].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
        detail["effective_kappa_mean"] = kappa_ori.mean().item()
        detail["effective_kappa_std"] = kappa_ori.std(dim=0).mean().item()

    # ---- Backward loss: base rollout objective plus branch-local regularisers.
    real_loss = (
        total_loss_tensor / norm
        + total_aux_loss_tensor / norm
        + kappa_reg
        + ori_saturation_reg
        + delta_pos_reg
        + ratio_anchor_loss
        + ratio_bound_loss
    )
    detail["loss_real"] = real_loss.item()
    detail["unfreeze_alpha"] = unfreeze_alpha
    detail["effective_rollout_weight"] = float(total_weight)
    detail["_class_ids"] = [int(v.item()) for v in class_ids.detach().cpu()]
    detail["_state_buckets"] = list(state_buckets)
    if emit_sample_metrics:
        sample_loss_avg = torch.where(
            sample_loss_count > 0,
            sample_loss_sum / torch.clamp(sample_loss_count, min=1.0),
            torch.zeros_like(sample_loss_sum),
        )
        detail["_sample_loss_real"] = sample_loss_avg.detach().cpu().tolist()
        detail["_sample_loss_real_valid"] = (sample_loss_count > 0).detach().cpu().tolist()
    if use_closure_loss_path and "ratios" in mamba_out:
        ratios = mamba_out.get("ratios") or {}
        for ratio_name in ["q_pos_xyz", "q_pos_vxyz", "r_pos_xyz", "r_pos_vxy", "r_siz_lw", "r_siz_h", "r_ori"]:
            if ratio_name in ratios:
                detail[f"_sample_{ratio_name}"] = ratios[ratio_name].detach().view(-1).cpu().tolist()
    return real_loss, detail


# ======================================================================
# Validation
# ======================================================================

@torch.no_grad()
def validate(
    mamba: TemporalMamba,
    val_loader: DataLoader,
    loss_fn: JointLoss,
    device: torch.device,
    noise_cfg: dict = None,
    base_noise_cfg: dict = None,
    filter_mode: str = "mamba",
) -> dict:
    """Run validation and return average losses."""
    mamba.eval()
    totals = {}
    class_state_acc = init_class_state_metric_accumulator()
    n_batches = 0

    for batch in val_loader:
        _, detail = training_step(mamba, batch, loss_fn, device, noise_cfg, base_noise_cfg,
                                  epoch=999, warmup_epochs=0, transition_epochs=0,
                                  filter_mode=filter_mode,
                                  emit_sample_metrics=True)  # fully unfrozen
        class_ids_detail = detail.get("_class_ids")
        state_buckets_detail = detail.get("_state_buckets")
        if class_ids_detail is not None and state_buckets_detail is not None:
            per_sample_metrics = {}
            if "_sample_loss_real" in detail:
                loss_values = detail["_sample_loss_real"]
                loss_valid = detail.get("_sample_loss_real_valid", [True] * len(loss_values))
                per_sample_metrics["loss_real"] = [
                    value if valid else None
                    for value, valid in zip(loss_values, loss_valid)
                ]
            for name in ["q_pos_xyz", "q_pos_vxyz", "r_pos_xyz", "r_pos_vxy", "r_siz_lw", "r_siz_h", "r_ori"]:
                sample_key = f"_sample_{name}"
                if sample_key in detail:
                    per_sample_metrics[f"{name}_mean"] = detail[sample_key]
            if per_sample_metrics:
                update_class_state_metric_accumulator(
                    class_state_acc,
                    class_ids=class_ids_detail,
                    state_buckets=state_buckets_detail,
                    metrics=per_sample_metrics,
                )
        for k, v in detail.items():
            if str(k).startswith("_"):
                continue
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    mamba.train()
    if n_batches == 0:
        return {}
    averaged = {k: v / n_batches for k, v in totals.items() if not str(k).startswith("_")}
    averaged.update({
        f"class_state/{k}": v
        for k, v in finalize_class_state_metric_accumulator(class_state_acc).items()
    })
    return averaged


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TemporalMamba for Mamba-Decoupled-EKF Track")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--extract-only", action="store_true", help="Only extract GT tracklets, then exit")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument(
        "--resume-heads-only",
        type=str,
        default=None,
        help="Load compatible non-backbone weights from a checkpoint and train from epoch 0",
    )
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()
    if args.resume and args.resume_heads_only:
        parser.error("--resume and --resume-heads-only are mutually exclusive")

    # ---- Load config ----
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # ---- Device ----
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Logging: console + file ----
    log_dir = cfg.get("TRAINING", {}).get("SAVE_DIR", "checkpoints/mamba_dekf/")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "train.log")
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(ch)
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(fh)
    logger.info(f"Logging to {log_path}")
    logger.info(f"Device: {device}")

    # ---- Extract GT tracklets ----
    data_cfg = cfg["DATA"]
    train_source = data_cfg.get("TRAIN_SOURCE", "gt")
    val_source = data_cfg.get("VAL_SOURCE", train_source)
    train_pkl = data_cfg.get("TRAIN_TRACKLET_PKL", None)
    val_pkl = data_cfg.get("VAL_TRACKLET_PKL", None)

    if train_source == "gt":
        train_pkl = extract_gt_tracklets_nuscenes(
            nusc_version=data_cfg["NUSC_VERSION"],
            nusc_dataroot=data_cfg["NUSC_DATAROOT"],
            split=data_cfg["TRAIN_SPLIT"],
            output_dir=data_cfg["TRACKLET_CACHE_DIR"],
            category_filter=data_cfg.get("CATEGORY_FILTER", None),
        )
    elif not train_pkl:
        raise ValueError("DATA.TRAIN_TRACKLET_PKL is required when TRAIN_SOURCE=det")

    if val_source == "gt":
        val_pkl = extract_gt_tracklets_nuscenes(
            nusc_version=data_cfg["NUSC_VERSION"],
            nusc_dataroot=data_cfg["NUSC_DATAROOT"],
            split=data_cfg["VAL_SPLIT"],
            output_dir=data_cfg["TRACKLET_CACHE_DIR"],
            category_filter=data_cfg.get("CATEGORY_FILTER", None),
        )
    elif not val_pkl:
        raise ValueError("DATA.VAL_TRACKLET_PKL is required when VAL_SOURCE=det")

    if args.extract_only:
        logger.info("Extraction complete. Exiting.")
        return

    # ---- Datasets ----
    model_cfg = cfg["MODEL"]
    history_len = model_cfg.get("HISTORY_LEN", 10)
    train_cfg = cfg["TRAINING"]
    rollout_steps = train_cfg.get("ROLLOUT_STEPS", 1)
    train_adaptive_windows = bool(data_cfg.get("TRAIN_ADAPTIVE_WINDOWS", False))
    val_adaptive_windows = bool(data_cfg.get("VAL_ADAPTIVE_WINDOWS", False))
    min_history_len = int(data_cfg.get("MIN_HISTORY_LEN", history_len))
    min_rollout_steps = int(data_cfg.get("MIN_ROLLOUT_STEPS", rollout_steps))
    class_window_cfg = data_cfg.get("CLASS_WINDOW", {})
    history_source = str(data_cfg.get("HISTORY_SOURCE", "det")).strip().lower()
    init_state_source = str(data_cfg.get("INIT_STATE_SOURCE", "det")).strip().lower()
    train_tracker_compat_mode = str(data_cfg.get("TRAIN_TRACKER_COMPAT_MODE", "default")).strip().lower()

    if train_source == "det":
        train_dataset = DetectionTrackletDataset(
            train_pkl,
            history_len=history_len,
            rollout_steps=rollout_steps,
            require_current_match=bool(data_cfg.get("REQUIRE_CURRENT_MATCH", True)),
            min_history_match_ratio=float(data_cfg.get("MIN_HISTORY_MATCH_RATIO", 0.25)),
            adaptive_windows=train_adaptive_windows,
            min_history_len=min_history_len,
            min_rollout_steps=min_rollout_steps,
            class_window_cfg=class_window_cfg if train_adaptive_windows else {},
            history_source=history_source,
            init_state_source=init_state_source,
        )
        train_collate_fn = detection_tracklet_collate_fn
    else:
        train_dataset = TrackletDataset(train_pkl, history_len=history_len,
                                        rollout_steps=rollout_steps,
                                        adaptive_windows=train_adaptive_windows,
                                        min_history_len=min_history_len,
                                        min_rollout_steps=min_rollout_steps,
                                        class_window_cfg=class_window_cfg if train_adaptive_windows else {})
        train_collate_fn = tracklet_collate_fn

    if val_source == "det":
        val_dataset = DetectionTrackletDataset(
            val_pkl,
            history_len=history_len,
            rollout_steps=rollout_steps,
            require_current_match=bool(data_cfg.get("REQUIRE_CURRENT_MATCH", True)),
            min_history_match_ratio=float(data_cfg.get("MIN_HISTORY_MATCH_RATIO", 0.25)),
            adaptive_windows=val_adaptive_windows,
            min_history_len=min_history_len,
            min_rollout_steps=min_rollout_steps,
            class_window_cfg=class_window_cfg if val_adaptive_windows else {},
            history_source=history_source,
            init_state_source=init_state_source,
        )
        val_collate_fn = detection_tracklet_collate_fn
    else:
        val_dataset = TrackletDataset(val_pkl, history_len=history_len,
                                      rollout_steps=rollout_steps,
                                      adaptive_windows=val_adaptive_windows,
                                      min_history_len=min_history_len,
                                      min_rollout_steps=min_rollout_steps,
                                      class_window_cfg=class_window_cfg if val_adaptive_windows else {})
        val_collate_fn = tracklet_collate_fn

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=train_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=val_collate_fn,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_dataset)} samples, {len(train_loader)} batches "
        f"(Tmax={history_len}, Kmax={rollout_steps}, adaptive={train_adaptive_windows}, "
        f"Tmin={min_history_len}, Kmin={min_rollout_steps}, source={train_source}, "
        f"history_source={history_source}, init_state_source={init_state_source}, "
        f"tracker_compat={train_tracker_compat_mode})"
    )
    logger.info(
        f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches "
        f"(Tmax={history_len}, Kmax={rollout_steps}, adaptive={val_adaptive_windows}, "
        f"Tmin={min_history_len}, Kmin={min_rollout_steps}, source={val_source}, "
        f"history_source={history_source}, init_state_source={init_state_source}, "
        f"tracker_compat={train_tracker_compat_mode})"
    )

    # ---- Model: only TemporalMamba is trained ----
    mamba = TemporalMamba(
        d_model=model_cfg.get("D_MODEL", 64),
        d_state=model_cfg.get("D_STATE", 16),
        d_conv=model_cfg.get("D_CONV", 4),
        expand=model_cfg.get("EXPAND", 2),
        n_mamba_layers=model_cfg.get("N_MAMBA_LAYERS", 3),
        embed_dim=model_cfg.get("EMBED_DIM", 32),
        min_diag_q=model_cfg.get("MIN_DIAG_Q", 0.1),
        min_diag_r=model_cfg.get("MIN_DIAG_R", 0.1),
        num_classes=model_cfg.get("NUM_CLASSES", 10),
        min_diag_siz=model_cfg.get("MIN_DIAG_SIZ", 0.05),
        min_kappa=model_cfg.get("MIN_KAPPA", 0.1),
        base_noise_cfg=cfg.get("BASE_NOISE", None),
    ).to(device)
    backbone_type = "mamba_ssm" if getattr(mamba, "mamba_layers", None) is not None else "fallback_gru"
    require_mamba_ssm = bool(train_cfg.get("REQUIRE_MAMBA_SSM", False))
    if require_mamba_ssm and backbone_type != "mamba_ssm":
        raise RuntimeError(
            "TRAINING.REQUIRE_MAMBA_SSM=true but TemporalMamba was built with "
            "fallback_gru. Verify `from mamba_ssm.modules.mamba_simple import Mamba` "
            "works in the same Python environment used for training."
        )

    n_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    logger.info(f"TemporalMamba: {n_params:,} trainable parameters (backbone={backbone_type})")

    # ---- Loss ----
    loss_cfg = cfg.get("LOSS", {})
    loss_fn = JointLoss(
        w_pos=loss_cfg.get("W_POS", 1.0),
        w_siz=loss_cfg.get("W_SIZ", 0.5),
        w_ori=loss_cfg.get("W_ORI", 1.0),
        w_vel=loss_cfg.get("W_VEL", 0.3),
        w_nis=loss_cfg.get("W_NIS", 0.0),
        lambda_contrast=loss_cfg.get("LAMBDA_CONTRAST", 0.1),
        temperature=loss_cfg.get("INFONCE_TEMPERATURE", 0.07),
        physics_scale=loss_cfg.get("PHYSICS_SCALE", 5.0),
        hard_negative_topk=loss_cfg.get("HARD_NEGATIVE_TOPK", 0),
        class_weights=loss_cfg.get("CLASS_WEIGHTS", None),
    ).to(device)
    vel_warmup_epochs = loss_cfg.get("VEL_WARMUP_EPOCHS", 3)

    # ---- Optimizer (separated LR groups) ----
    # Mamba SSM backbone: low LR (5e-5) to prevent early covariance divergence.
    # All other params (projections, norms, heads): base LR (1e-3).
    base_lr = train_cfg.get("LR", 1e-3)
    mamba_lr = train_cfg.get("MAMBA_LR", 5e-5)
    weight_decay = train_cfg.get("WEIGHT_DECAY", 1e-4)

    mamba_backbone_params = []
    other_params = []
    for name, p in mamba.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("mamba_layers") or name.startswith("fallback_gru"):
            mamba_backbone_params.append(p)
        else:
            other_params.append(p)

    optimizer = torch.optim.AdamW([
        {"params": other_params, "lr": base_lr},
        {"params": mamba_backbone_params, "lr": mamba_lr},
    ], weight_decay=weight_decay)

    logger.info(
        f"Optimizer: base_lr={base_lr}, mamba_lr={mamba_lr}, "
        f"mamba_params={sum(p.numel() for p in mamba_backbone_params):,}, "
        f"other_params={sum(p.numel() for p in other_params):,}"
    )

    # ---- Scheduler: linear warmup → cosine annealing ----
    epochs = train_cfg.get("EPOCHS", 50)
    warmup_epochs = train_cfg.get("WARMUP_EPOCHS", 3)
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.01, total_iters=warmup_epochs,
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=max(epochs - warmup_epochs, 1),
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    # ---- TensorBoard ----
    tb_dir = os.path.join(log_dir, "tensorboard")
    writer = SummaryWriter(tb_dir)
    logger.info(f"TensorBoard logs → {tb_dir}  (view: tensorboard --logdir {tb_dir})")

    grad_clip = train_cfg.get("GRAD_CLIP_NORM", 1.0)
    log_every = train_cfg.get("LOG_EVERY", 50)
    save_every = train_cfg.get("SAVE_EVERY", 5)
    save_dir = train_cfg.get("SAVE_DIR", "checkpoints/mamba_dekf/")
    os.makedirs(save_dir, exist_ok=True)

    # ---- Resume ----
    start_epoch = 0
    if args.resume_heads_only:
        ckpt = torch.load(args.resume_heads_only, map_location=device)
        resume_state = ckpt.get("model_state_dict", ckpt)
        resume_state, adapted_keys = adapt_num_class_state_dict(
            resume_state,
            mamba.state_dict(),
        )
        filtered_state, skipped = filter_heads_only_state_dict(
            resume_state,
            mamba.state_dict(),
        )
        missing, unexpected = mamba.load_state_dict(filtered_state, strict=False)
        logger.info(
            "Loaded heads-only checkpoint from %s: loaded=%d, "
            "skipped_backbone=%d, skipped_missing=%d, skipped_shape=%d",
            args.resume_heads_only,
            len(filtered_state),
            len(skipped["backbone"]),
            len(skipped["missing"]),
            len(skipped["shape_mismatch"]),
        )
        if adapted_keys:
            logger.info(
                "Adapted class-count-dependent checkpoint tensors on heads-only load: %s",
                ", ".join(adapted_keys),
            )
        if missing:
            logger.info("Heads-only load leaves %d model keys initialized from scratch", len(missing))
        if unexpected:
            logger.warning("Unexpected keys during heads-only load: %s", unexpected)
        runtime_contract = ckpt.get("runtime_contract", None)
        if runtime_contract:
            logger.info(f"Heads-only checkpoint runtime_contract: {runtime_contract}")
        logger.info("Heads-only load starts training from epoch 0 with a fresh optimizer")
    elif args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        resume_state = ckpt["model_state_dict"]
        resume_state, adapted_keys = adapt_num_class_state_dict(
            resume_state,
            mamba.state_dict(),
        )
        mamba.load_state_dict(resume_state)
        if adapted_keys:
            logger.info(
                "Adapted class-count-dependent checkpoint tensors on resume: %s",
                ", ".join(adapted_keys),
            )
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Could not load optimizer state (likely old checkpoint with "
                f"different param groups): {e}. Optimizer will start fresh."
            )
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")
        runtime_contract = ckpt.get("runtime_contract", None)
        if runtime_contract:
            logger.info(f"Checkpoint runtime_contract: {runtime_contract}")

    # ---- Training loop ----
    best_val_loss = float("inf")
    runtime_contract = {
        "tracker_compat_mode": train_tracker_compat_mode,
        "history_source": history_source,
        "init_state_source": init_state_source,
        "filter_mode": resolve_runtime_contract_filter_mode(cfg, train_tracker_compat_mode),
        "backbone_type": backbone_type,
        "future_update_source": "det" if bool((cfg.get("BASE_NOISE", {}) or {}).get("DETECTION_UPDATE", {}).get("ENABLED", True)) else "gt_noisy",
        "train_source": train_source,
        "val_source": val_source,
        "expected_bev_cost_mode": str(data_cfg.get("EXPECTED_BEV_COST_MODE", "geometric")).strip().lower(),
    }
    train_filter_mode = runtime_contract["filter_mode"]
    train_noise_audit_cfg = _build_noise_audit_cfg(cfg)

    for epoch in range(start_epoch, epochs):
        mamba.train()
        train_noise_audit = _new_train_noise_audit_accumulator(train_noise_audit_cfg)
        epoch_loss = 0.0
        epoch_detail = {}
        n_batches = 0
        nan_count = 0
        t0 = time.time()
        noise_cfg = train_cfg.get("NOISE_MAP", None)
        base_noise_cfg = cfg.get("BASE_NOISE", None)
        warmup_epochs = train_cfg.get("WARMUP_UNCERTAINTY_EPOCHS", 0)
        transition_epochs = train_cfg.get("WARMUP_TRANSITION_EPOCHS", 0)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, detail = training_step(
                mamba, batch, loss_fn, device, noise_cfg, base_noise_cfg,
                epoch=epoch, warmup_epochs=warmup_epochs,
                transition_epochs=transition_epochs,
                vel_warmup_epochs=vel_warmup_epochs,
                train_noise_audit=train_noise_audit,
                filter_mode=train_filter_mode)

            # Check for NaN / Inf loss before backward
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                logger.warning(
                    f"NaN/Inf loss at epoch {epoch}, batch {batch_idx}. Skipping."
                )
                optimizer.zero_grad()
                continue

            loss.backward()

            # NaN / Inf guard on gradients — skip batch if any param has bad grad
            grads_ok = True
            for p in mamba.parameters():
                if p.grad is not None:
                    if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                        grads_ok = False
                        break
            if not grads_ok:
                nan_count += 1
                logger.warning(
                    f"NaN/Inf gradient at epoch {epoch}, batch {batch_idx}. Skipping."
                )
                optimizer.zero_grad()
                continue

            # Gradient clipping (safety lock: prevents covariance divergence)
            nn.utils.clip_grad_norm_(mamba.parameters(), max_norm=grad_clip)

            optimizer.step()

            epoch_loss += detail["loss_total"]
            for k, v in detail.items():
                if str(k).startswith("_"):
                    continue
                epoch_detail[k] = epoch_detail.get(k, 0.0) + v
            n_batches += 1

            if (batch_idx + 1) % log_every == 0:
                avg = epoch_loss / n_batches
                lr = optimizer.param_groups[0]["lr"]
                writer.add_scalar("train/loss_batch", avg, epoch * len(train_loader) + batch_idx)
                writer.add_scalar("train/lr", lr, epoch * len(train_loader) + batch_idx)
                logger.info(
                    f"  [{epoch+1}/{epochs}] batch {batch_idx+1}/{len(train_loader)} "
                    f"loss={avg:.4f} lr={lr:.2e}"
                )

        scheduler.step()

        # ---- Epoch summary ----
        dt_epoch = time.time() - t0
        avg_train = {k: v / max(n_batches, 1) for k, v in epoch_detail.items()}

        # ---- Validation ----
        avg_val = validate(
            mamba,
            val_loader,
            loss_fn,
            device,
            noise_cfg,
            base_noise_cfg,
            filter_mode=train_filter_mode,
        )

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({dt_epoch:.1f}s) | "
            f"loss={avg_train.get('loss_real', 0):.4f} "
            f"(raw={avg_train.get('loss_total', 0):.4f}) "
            f"pos={avg_train.get('loss_pos', 0):.4f} "
            f"siz={avg_train.get('loss_siz', 0):.4f} "
            f"ori={avg_train.get('loss_ori', 0):.4f} "
            f"vel={avg_train.get('loss_vel', 0):.4f} "
            f"nis={avg_train.get('loss_nis', 0):.4f} "
            f"ua={avg_train.get('unfreeze_alpha', 1.0):.2f} "
            f"contrast={avg_train.get('loss_contrastive', 0):.4f} | "
            f"val={avg_val.get('loss_real', avg_val.get('loss_total', 0)):.4f} "
            f"val_pos={avg_val.get('loss_pos', 0):.4f} | "
            f"k_m={avg_train.get('mean_kappa', 0):.2f} "
            f"k_s={avg_train.get('std_kappa', 0):.2f} "
            f"k_reg={avg_train.get('loss_kappa_reg', 0):.4f} "
            f"dp_reg={avg_train.get('loss_delta_pos_reg', 0):.4f} | "
            f"NaN={nan_count}"
        )

        # ---- TensorBoard per-epoch scalars ----
        step = epoch
        writer.add_scalar("train/loss_real", avg_train.get("loss_real", 0), step)
        writer.add_scalar("train/loss_raw", avg_train.get("loss_total", 0), step)
        writer.add_scalar("train/loss_pos", avg_train.get("loss_pos", 0), step)
        writer.add_scalar("train/loss_siz", avg_train.get("loss_siz", 0), step)
        writer.add_scalar("train/loss_ori", avg_train.get("loss_ori", 0), step)
        writer.add_scalar("train/loss_vel", avg_train.get("loss_vel", 0), step)
        writer.add_scalar("train/loss_nis", avg_train.get("loss_nis", 0), step)
        writer.add_scalar("train/loss_contrastive", avg_train.get("loss_contrastive", 0), step)
        writer.add_scalar("train/loss_kappa_reg", avg_train.get("loss_kappa_reg", 0), step)
        writer.add_scalar("train/loss_delta_pos_reg", avg_train.get("loss_delta_pos_reg", 0), step)
        writer.add_scalar("train/unfreeze_alpha", avg_train.get("unfreeze_alpha", 1.0), step)
        writer.add_scalar("val/loss_real", avg_val.get("loss_real", avg_val.get("loss_total", 0)), step)
        writer.add_scalar("val/loss_raw", avg_val.get("loss_total", 0), step)
        writer.add_scalar("val/loss_pos", avg_val.get("loss_pos", 0), step)
        writer.add_scalar("val/loss_siz", avg_val.get("loss_siz", 0), step)
        writer.add_scalar("val/loss_ori", avg_val.get("loss_ori", 0), step)
        writer.add_scalar("val/loss_vel", avg_val.get("loss_vel", 0), step)
        writer.add_scalar("val/loss_nis", avg_val.get("loss_nis", 0), step)
        writer.add_scalar("val/loss_contrastive", avg_val.get("loss_contrastive", 0), step)
        writer.add_scalar("val/loss_kappa_reg", avg_val.get("loss_kappa_reg", 0), step)
        writer.add_scalar("val/loss_delta_pos_reg", avg_val.get("loss_delta_pos_reg", 0), step)
        writer.add_scalar("train/epoch_time", dt_epoch, step)
        writer.add_scalar("train/nan_count", nan_count, step)
        # Q/R diagonal std — should stay > 0; zero = constant output (degenerate)
        for key in ["std_Q_pos", "std_Q_siz", "std_R_pos", "std_R_siz", "std_kappa", "mean_kappa"]:
            writer.add_scalar(f"train/{key}", avg_train.get(key, 0), step)

        _dump_train_noise_audit_if_needed(
            train_noise_audit,
            train_noise_audit_cfg,
            logger_obj=logger,
        )

        # ---- Checkpoint ----
        val_total = avg_val.get("loss_real", avg_val.get("loss_total", float("inf")))

        if (epoch + 1) % save_every == 0 or val_total < best_val_loss:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": mamba.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
                "runtime_contract": runtime_contract,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint → {ckpt_path}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = os.path.join(save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": mamba.state_dict(),
                "val_loss": avg_val,
                "runtime_contract": runtime_contract,
            }, best_path)
            logger.info(f"  New best model → {best_path} (val_loss={val_total:.4f})")

        # ---- Auto-unseal: toggle inference cost from "geometric" → "full" ----
        # Default behavior is dry-run to avoid mutating source config files.
        # Set TRAINING.AUTO_UNSEAL_WRITE_BACK=true to restore legacy write-back.
        auto_unseal_epoch = train_cfg.get("AUTO_UNSEAL_EPOCH", 0)
        if auto_unseal_epoch > 0 and (epoch + 1) == auto_unseal_epoch:
            auto_unseal_write_back = bool(train_cfg.get("AUTO_UNSEAL_WRITE_BACK", False))
            inf_cfg_path = train_cfg.get(
                "INFERENCE_CONFIG",
                "config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml",
            )
            if os.path.exists(inf_cfg_path):
                with open(inf_cfg_path, "r") as f:
                    inf_cfg = yaml.safe_load(f)
                old_mode = inf_cfg.get("THRESHOLD", {}).get("BEV", {}).get("COST_MODE", "unknown")
                inf_cfg.setdefault("THRESHOLD", {}).setdefault("BEV", {})["COST_MODE"] = "full"
                if auto_unseal_write_back:
                    with open(inf_cfg_path, "w") as f:
                        yaml.safe_dump(inf_cfg, f, default_flow_style=False, sort_keys=False)
                    logger.info(
                        f"  *** AUTO-UNSEAL: {inf_cfg_path} COST_MODE: {old_mode} → full ***"
                    )
                else:
                    preview_path = os.path.join(save_dir, "auto_unseal_preview.yaml")
                    with open(preview_path, "w") as f:
                        yaml.safe_dump(inf_cfg, f, default_flow_style=False, sort_keys=False)
                    logger.info(
                        "  *** AUTO-UNSEAL (dry-run): source config unchanged; "
                        f"preview saved to {preview_path} (COST_MODE: {old_mode} → full) ***"
                    )
            else:
                logger.warning(
                    f"  AUTO-UNSEAL skipped: {inf_cfg_path} not found"
                )

    writer.close()
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
