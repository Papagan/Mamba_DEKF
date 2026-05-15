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
from training.losses import JointLoss
from kalmanfilter.mamba_adaptive_kf import (
    TemporalMamba,
    DecoupledAdaptiveKF,
)

logger = logging.getLogger("train")


# ======================================================================
# Training step
# ======================================================================

def training_step(
    mamba: TemporalMamba,
    batch: dict,
    loss_fn: JointLoss,
    device: torch.device,
    noise_cfg: dict = None,
) -> tuple:
    """
    Multi-step KF rollout training with noisy teacher forcing.

    Mamba runs ONCE to predict Q/R/kappa from the history window.
    The KF is initialised from a NOISY GT state (simulating tracking
    uncertainty), then runs K predict-update steps. Noisy GT measurements
    are fed to KF.update (teacher forcing), but the NLL loss at each step
    compares predictions to CLEAN GT. The noise prevents collapse to
    trivial solutions (min-diag Q, κ→∞).

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
    gt_future_pos = batch["gt_future_pos"].to(device)      # [B, K, 3]
    gt_future_siz = batch["gt_future_siz"].to(device)      # [B, K, 3]
    gt_future_ori = batch["gt_future_ori"].to(device)      # [B, K, 1]
    delta_ts_future = batch["delta_ts_future"].to(device)  # [B, K]
    instance_tokens = batch["instance_token"]              # list of B strings
    categories = batch["category"]                          # list of B strings

    # Map nuScenes category names to integer class IDs for size embeddings
    _CAT_ID_MAP = {"car": 0, "pedestrian": 1, "bicycle": 2, "motorcycle": 3,
                   "bus": 4, "trailer": 5, "truck": 6}
    class_ids = torch.tensor(
        [_CAT_ID_MAP.get(c.split(".")[-1], 0) for c in categories],
        dtype=torch.long, device=device)                   # [B]

    # ---- Step 1: TemporalMamba forward (ONCE) → Q/R/embedding ----
    mamba_out = mamba(history, class_ids=class_ids)

    # ---- Step 2: Init KF from GT state at frame T (+ noise) ----
    if noise_cfg is None:
        noise_cfg = {
            "POS_STD": [0.51, 0.27, 0.55, 0.86, 1.26, 0.91, 0.64],
            "SIZ_STD": [0.16, 0.05, 0.09, 0.13, 0.67, 1.05, 0.46],
            "ORI_STD": [0.50, 0.91, 0.86, 0.96, 0.48, 0.53, 0.45],
            "MEAS_MULTIPLIER": 0.7
        }

    # Category-specific noise mapped from yaml config
    # [Car, Pedestrian, Bicycle, Motorcycle, Bus, Trailer, Truck]
    pos_std_map = torch.tensor(noise_cfg["POS_STD"], device=device)
    siz_std_map = torch.tensor(noise_cfg["SIZ_STD"], device=device)
    ori_std_map = torch.tensor(noise_cfg["ORI_STD"], device=device)

    # Broadcast category-specific noise to batch
    safe_class_ids = torch.clamp(class_ids, min=0, max=6)
    noise_scale_pos = pos_std_map[safe_class_ids].view(B, 1, 1)  # [B, 1, 1]
    noise_scale_siz = siz_std_map[safe_class_ids].view(B, 1, 1)
    noise_scale_ori = ori_std_map[safe_class_ids].view(B, 1, 1)

    pos_x0 = gt_pos.unsqueeze(-1)                                    # [B, 6, 1]
    siz_x0 = gt_siz.unsqueeze(-1)                                    # [B, 3, 1]
    ori_x0 = gt_ori.unsqueeze(-1)                                    # [B, 2, 1]

    # Add dynamically mapped category noise to position state ([x,y,z] only)
    pos_x0 = pos_x0 + torch.randn_like(pos_x0) * noise_scale_pos * torch.tensor(
        [1.0, 1.0, 1.0, 0.0, 0.0, 0.0], device=device).view(1, 6, 1)
    siz_x0 = siz_x0 + torch.randn_like(siz_x0) * noise_scale_siz
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
    meas_noise_pos = noise_scale_pos * meas_mult       # [B, 1, 1] — position noise
    meas_noise_siz = noise_scale_siz * meas_mult
    meas_noise_ori = noise_scale_ori * meas_mult
    # Velocity observation noise: 2× position std (4× variance) so KF
    # trusts velocity less than position (CenterPoint velocity is often zero).
    meas_noise_vel = noise_scale_pos * meas_mult * 2.0  # [B, 1, 1]

    # ---- Step 3: K-step KF rollout with teacher forcing ----
    Q_pos = mamba_out["Q_pos"]
    Q_siz = mamba_out["Q_siz"]
    Q_ori = mamba_out["Q_ori"]
    R_pos = mamba_out["R_pos"]
    R_siz = mamba_out["R_siz"]
    R_ori = mamba_out["R_ori"]

    total_state_loss = 0.0
    total_contrast_loss = 0.0
    detail_accum = {}
    detail_contrastive = {"loss_contrastive": 0.0, "n_valid_anchors": 0}

    for k in range(K):
        # Per-sample delta_t preserves individual object dynamics
        dt_k = delta_ts_future[:, k]                          # [B]

        # KF predict with same Q for all steps
        pos_x_pred, pos_P_pred, siz_x_pred, siz_P_pred, ori_x_pred, ori_P_pred = \
            kf.predict(dt_k, Q_pos=Q_pos, Q_siz=Q_siz, Q_ori=Q_ori)

        # NLL vs GT at this rollout step
        loss_k, detail_k = loss_fn(
            pos_x_pred, pos_P_pred, siz_x_pred, siz_P_pred, ori_x_pred, ori_P_pred,
            gt_future_pos[:, k, :], gt_future_siz[:, k, :], gt_future_ori[:, k, :],
            mamba_out["embedding"] if k == 0 else None,
            instance_tokens if k == 0 else None,
            R_pos=R_pos, R_siz=R_siz, R_ori=R_ori,
            kappa_ori=mamba_out["kappa_ori"],
        )

        total_state_loss += detail_k["loss_state"]
        if k == 0:
            total_contrast_loss = detail_k["loss_contrastive"]
        for key, val in detail_k.items():
            if key == "loss_contrastive":
                detail_contrastive["loss_contrastive"] = max(
                    detail_contrastive["loss_contrastive"], val)
            elif key == "n_valid_anchors":
                detail_contrastive["n_valid_anchors"] = max(
                    detail_contrastive["n_valid_anchors"], int(val))
            elif key == "loss_total":
                pass
            else:
                detail_accum[key] = detail_accum.get(key, 0.0) + val

        # Noisy teacher forcing with category-aware measurement noise
        # Position observation now 5D: [x, y, z, vx, vy]. Velocity is observed
        # directly to eliminate the KF velocity-inference lag.
        gt_vel_xy = gt_pos[:, 3:5]  # [B, 2] — GT velocity
        z_pos_k = torch.cat([
            gt_future_pos[:, k, :]                      # [B, 3]
            + torch.randn(B, 3, device=device) * meas_noise_pos.squeeze(-1),
            gt_vel_xy                                    # [B, 2]
            + torch.randn(B, 2, device=device) * meas_noise_vel.squeeze(-1),
        ], dim=1).unsqueeze(-1)  # [B, 5, 1]
        z_siz_k = gt_future_siz[:, k, :].unsqueeze(-1) \
            + torch.randn(B, 3, 1, device=device) * meas_noise_siz
        z_ori_k = gt_future_ori[:, k, :].unsqueeze(-1) \
            + torch.randn(B, 1, 1, device=device) * meas_noise_ori
        kf.update(z_pos_k, z_siz_k, z_ori_k,
                  R_pos=R_pos, R_siz=R_siz, R_ori=R_ori)

    # State losses averaged across rollout; contrastive kept raw
    detail = {k: v / K for k, v in detail_accum.items()}
    detail["loss_contrastive"] = detail_contrastive["loss_contrastive"]
    detail["n_valid_anchors"] = detail_contrastive["n_valid_anchors"]
    avg_state_loss = detail["loss_state"]
    detail["loss_total"] = (
        loss_fn.physics_scale * avg_state_loss +
        loss_fn.lambda_contrast * detail["loss_contrastive"]
    )

    # Weak κ² penalty prevents unbounded concentration growth.
    # 1e-5 is small enough to not affect the primary learning signal
    # but large enough to stop κ from reaching 2000+.
    kappa_reg = 1e-5 * (mamba_out["kappa_ori"] ** 2).mean()
    detail["loss_kappa_reg"] = kappa_reg.item()

    # ---- Step 4: Q/R/kappa variance monitor ----
    with torch.no_grad():
        for name, mat in [
            ("Q_pos", mamba_out["Q_pos"]), ("Q_siz", mamba_out["Q_siz"]),
            ("R_pos", mamba_out["R_pos"]), ("R_siz", mamba_out["R_siz"]),
        ]:
            diag = mat.diagonal(dim1=-2, dim2=-1)
            detail[f"std_{name}"] = diag.std(dim=0).mean().item()
        # Orientation: kappa std (R_ori = 1/kappa, Q_ori is static)
        detail["std_kappa"] = mamba_out["kappa_ori"].std(dim=0).mean().item()
        detail["mean_kappa"] = mamba_out["kappa_ori"].mean().item()

    real_loss = (loss_fn.physics_scale * avg_state_loss
                 + loss_fn.lambda_contrast * total_contrast_loss
                 + kappa_reg)
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
) -> dict:
    """Run validation and return average losses."""
    mamba.eval()
    totals = {}
    n_batches = 0

    for batch in val_loader:
        _, detail = training_step(mamba, batch, loss_fn, device, noise_cfg)
        for k, v in detail.items():
            totals[k] = totals.get(k, 0.0) + v
        n_batches += 1

    mamba.train()
    if n_batches == 0:
        return {}
    return {k: v / n_batches for k, v in totals.items()}


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Train TemporalMamba for Mamba-Decoupled-EKF Track")
    parser.add_argument("--config", type=str, required=True, help="Path to training config YAML")
    parser.add_argument("--extract-only", action="store_true", help="Only extract GT tracklets, then exit")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--device", type=str, default=None, help="Override device (cuda/cpu)")
    args = parser.parse_args()

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
    train_pkl = extract_gt_tracklets_nuscenes(
        nusc_version=data_cfg["NUSC_VERSION"],
        nusc_dataroot=data_cfg["NUSC_DATAROOT"],
        split=data_cfg["TRAIN_SPLIT"],
        output_dir=data_cfg["TRACKLET_CACHE_DIR"],
        category_filter=data_cfg.get("CATEGORY_FILTER", None),
    )
    val_pkl = extract_gt_tracklets_nuscenes(
        nusc_version=data_cfg["NUSC_VERSION"],
        nusc_dataroot=data_cfg["NUSC_DATAROOT"],
        split=data_cfg["VAL_SPLIT"],
        output_dir=data_cfg["TRACKLET_CACHE_DIR"],
        category_filter=data_cfg.get("CATEGORY_FILTER", None),
    )

    if args.extract_only:
        logger.info("Extraction complete. Exiting.")
        return

    # ---- Datasets ----
    model_cfg = cfg["MODEL"]
    history_len = model_cfg.get("HISTORY_LEN", 10)
    train_cfg = cfg["TRAINING"]
    rollout_steps = train_cfg.get("ROLLOUT_STEPS", 1)

    train_dataset = TrackletDataset(train_pkl, history_len=history_len,
                                    rollout_steps=rollout_steps)
    val_dataset = TrackletDataset(val_pkl, history_len=history_len,
                                  rollout_steps=rollout_steps)

    train_loader = DataLoader(
        train_dataset,
        batch_size=train_cfg["BATCH_SIZE"],
        shuffle=True,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=tracklet_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_cfg["BATCH_SIZE"],
        shuffle=False,
        num_workers=train_cfg.get("NUM_WORKERS", 4),
        collate_fn=tracklet_collate_fn,
        pin_memory=True,
    )

    logger.info(f"Train: {len(train_dataset)} samples, {len(train_loader)} batches (K={rollout_steps})")
    logger.info(f"Val:   {len(val_dataset)} samples, {len(val_loader)} batches (K={rollout_steps})")

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
    ).to(device)

    n_params = sum(p.numel() for p in mamba.parameters() if p.requires_grad)
    logger.info(f"TemporalMamba: {n_params:,} trainable parameters")

    # ---- Loss ----
    loss_cfg = cfg.get("LOSS", {})
    loss_fn = JointLoss(
        w_pos=loss_cfg.get("W_POS", 1.0),
        w_siz=loss_cfg.get("W_SIZ", 0.5),
        w_ori=loss_cfg.get("W_ORI", 50.0),
        lambda_contrast=loss_cfg.get("LAMBDA_CONTRAST", 0.1),
        temperature=loss_cfg.get("INFONCE_TEMPERATURE", 0.07),
        physics_scale=loss_cfg.get("PHYSICS_SCALE", 50.0),
    ).to(device)

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
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        mamba.load_state_dict(ckpt["model_state_dict"])
        try:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Could not load optimizer state (likely old checkpoint with "
                f"different param groups): {e}. Optimizer will start fresh."
            )
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info(f"Resumed from {args.resume}, epoch {start_epoch}")

    # ---- Training loop ----
    best_val_loss = float("inf")

    for epoch in range(start_epoch, epochs):
        mamba.train()
        epoch_loss = 0.0
        epoch_detail = {}
        n_batches = 0
        nan_count = 0
        t0 = time.time()
        noise_cfg = train_cfg.get("NOISE_MAP", None)

        for batch_idx, batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss, detail = training_step(mamba, batch, loss_fn, device, noise_cfg)

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
        avg_val = validate(mamba, val_loader, loss_fn, device, noise_cfg)

        logger.info(
            f"Epoch {epoch+1}/{epochs} ({dt_epoch:.1f}s) | "
            f"train_loss={avg_train.get('loss_total', 0):.4f} "
            f"pos={avg_train.get('loss_pos', 0):.4f} "
            f"siz={avg_train.get('loss_siz', 0):.4f} "
            f"ori={avg_train.get('loss_ori', 0):.4f} "
            f"contrast={avg_train.get('loss_contrastive', 0):.4f} | "
            f"val_loss={avg_val.get('loss_total', 0):.4f} "
            f"val_pos={avg_val.get('loss_pos', 0):.4f} | "
            f"kappa_m={avg_train.get('mean_kappa', 0):.3f} "
            f"kappa_s={avg_train.get('std_kappa', 0):.3f} | "
            f"NaN={nan_count}"
        )

        # ---- TensorBoard per-epoch scalars ----
        step = epoch
        writer.add_scalar("train/loss_total", avg_train.get("loss_total", 0), step)
        writer.add_scalar("train/loss_pos", avg_train.get("loss_pos", 0), step)
        writer.add_scalar("train/loss_siz", avg_train.get("loss_siz", 0), step)
        writer.add_scalar("train/loss_ori", avg_train.get("loss_ori", 0), step)
        writer.add_scalar("train/loss_contrastive", avg_train.get("loss_contrastive", 0), step)
        writer.add_scalar("val/loss_total", avg_val.get("loss_total", 0), step)
        writer.add_scalar("val/loss_pos", avg_val.get("loss_pos", 0), step)
        writer.add_scalar("train/epoch_time", dt_epoch, step)
        writer.add_scalar("train/nan_count", nan_count, step)
        # Q/R diagonal std — should stay > 0; zero = constant output (degenerate)
        for key in ["std_Q_pos", "std_Q_siz", "std_R_pos", "std_R_siz", "std_kappa", "mean_kappa"]:
            writer.add_scalar(f"train/{key}", avg_train.get(key, 0), step)

        # ---- Checkpoint ----
        val_total = avg_val.get("loss_total", float("inf"))

        if (epoch + 1) % save_every == 0 or val_total < best_val_loss:
            ckpt_path = os.path.join(save_dir, f"checkpoint_epoch{epoch+1}.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": mamba.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": avg_train,
                "val_loss": avg_val,
            }, ckpt_path)
            logger.info(f"  Saved checkpoint → {ckpt_path}")

        if val_total < best_val_loss:
            best_val_loss = val_total
            best_path = os.path.join(save_dir, "best.pt")
            torch.save({
                "epoch": epoch,
                "model_state_dict": mamba.state_dict(),
                "val_loss": avg_val,
            }, best_path)
            logger.info(f"  New best model → {best_path} (val_loss={val_total:.4f})")

        # ---- Auto-unseal: toggle inference cost from "geometric" → "full" ----
        auto_unseal_epoch = train_cfg.get("AUTO_UNSEAL_EPOCH", 0)
        if auto_unseal_epoch > 0 and (epoch + 1) == auto_unseal_epoch:
            inf_cfg_path = train_cfg.get("INFERENCE_CONFIG", "config/nuscenes.yaml")
            if os.path.exists(inf_cfg_path):
                with open(inf_cfg_path, "r") as f:
                    inf_cfg = yaml.safe_load(f)
                old_mode = inf_cfg.get("THRESHOLD", {}).get("BEV", {}).get("COST_MODE", "unknown")
                inf_cfg.setdefault("THRESHOLD", {}).setdefault("BEV", {})["COST_MODE"] = "full"
                with open(inf_cfg_path, "w") as f:
                    yaml.safe_dump(inf_cfg, f, default_flow_style=False, sort_keys=False)
                logger.info(
                    f"  *** AUTO-UNSEAL: {inf_cfg_path} COST_MODE: {old_mode} → full ***"
                )
            else:
                logger.warning(
                    f"  AUTO-UNSEAL skipped: {inf_cfg_path} not found"
                )

    writer.close()
    logger.info(f"Training complete. Best val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    main()
