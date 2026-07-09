#!/usr/bin/env python3
"""Train/evaluate the class-conditioned pairwise association head offline."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.association_dataset import (  # noqa: E402
    PairwiseAssociationDataset,
    pairwise_association_collate_fn,
)
from training.association_metrics import compute_association_metrics  # noqa: E402
from training.association_model import (  # noqa: E402
    ClassConditionedAssociationHeadBank,
    PairwiseAssociationModel,
)


LOGGER = logging.getLogger("train.association")


class MeanHistoryEncoder(nn.Module):
    """Small deterministic encoder for smoke tests and CPU dry-runs."""

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(12, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.SiLU(),
        )

    def forward(
        self,
        history,
        class_ids=None,
        current_range=None,
        detection_driven_mask=None,
        history_mask=None,
        history_match_mask=None,
        state_buckets=None,
        mode="mamba_multihead_closure",
        **kwargs,
    ):
        if history_mask is None:
            history_mask = history.abs().sum(dim=-1) > 0
        weights = history_mask.to(dtype=history.dtype).unsqueeze(-1)
        denom = torch.clamp(weights.sum(dim=1), min=1.0)
        pooled = (history * weights).sum(dim=1) / denom
        return {"embedding": self.proj(pooled)}


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _build_encoder(args, cfg: Dict[str, Any]) -> nn.Module:
    if bool(args.dry_run):
        return MeanHistoryEncoder(embed_dim=int(args.embed_dim))

    from kalmanfilter.mamba_adaptive_kf import Mamba, TemporalMamba

    if Mamba is None:
        raise RuntimeError(
            "mamba_ssm is not available. Association training must use the real "
            "Mamba backbone; use --dry-run only for smoke tests."
        )

    model_cfg = cfg.get("MODEL", {}) or {}
    return TemporalMamba(
        d_model=int(model_cfg.get("D_MODEL", 96)),
        d_state=int(model_cfg.get("D_STATE", 24)),
        d_conv=int(model_cfg.get("D_CONV", 4)),
        expand=int(model_cfg.get("EXPAND", 2)),
        n_mamba_layers=int(model_cfg.get("N_MAMBA_LAYERS", 6)),
        embed_dim=int(model_cfg.get("EMBED_DIM", args.embed_dim)),
        min_diag_q=float(model_cfg.get("MIN_DIAG_Q", 0.1)),
        min_diag_r=float(model_cfg.get("MIN_DIAG_R", 0.1)),
        num_classes=int(model_cfg.get("NUM_CLASSES", 7)),
        min_diag_siz=float(model_cfg.get("MIN_DIAG_SIZ", 0.15)),
        min_kappa=float(model_cfg.get("MIN_KAPPA", 0.5)),
        base_noise_cfg=cfg.get("BASE_NOISE", None),
        force_gru=False,
    )


def _load_backbone_checkpoint(model: PairwiseAssociationModel, checkpoint_path: str | None) -> Dict[str, Any]:
    if not checkpoint_path:
        return {"path": None, "loaded_keys": 0, "missing_keys": [], "unexpected_keys": []}
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Backbone checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(path, map_location="cpu")
    state = ckpt.get("model_state_dict", ckpt)
    encoder_state = model.temporal_encoder.state_dict()
    loadable = {}
    for key, value in state.items():
        stripped = key[len("temporal_encoder."):] if key.startswith("temporal_encoder.") else key
        if stripped in encoder_state and tuple(encoder_state[stripped].shape) == tuple(value.shape):
            loadable[stripped] = value
    result = model.temporal_encoder.load_state_dict(loadable, strict=False)
    return {
        "path": str(path),
        "loaded_keys": len(loadable),
        "missing_keys": list(result.missing_keys),
        "unexpected_keys": list(result.unexpected_keys),
    }


def _set_backbone_trainable(model: PairwiseAssociationModel, trainable: bool) -> None:
    for param in model.temporal_encoder.parameters():
        param.requires_grad = bool(trainable)


def _association_ranking_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    anchor_keys: List[str],
    *,
    margin: float,
) -> torch.Tensor:
    losses = []
    device = logits.device
    for anchor_key in sorted(set(anchor_keys)):
        indices = [idx for idx, key in enumerate(anchor_keys) if key == anchor_key]
        if len(indices) < 2:
            continue
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_logits = logits[idx_tensor]
        group_labels = labels[idx_tensor] > 0.5
        if not bool(group_labels.any().item()) or not bool((~group_labels).any().item()):
            continue
        pos_logits = group_logits[group_labels]
        neg_logits = group_logits[~group_labels]
        hardest_pos = pos_logits.min()
        hard_negatives = neg_logits
        losses.append(F.relu(float(margin) - hardest_pos + hard_negatives).mean())
    if not losses:
        return torch.zeros((), device=device, dtype=logits.dtype)
    return torch.stack(losses).mean()


def _records_from_logits(batch: Dict[str, Any], logits: torch.Tensor) -> List[Dict[str, Any]]:
    scores = torch.sigmoid(logits).detach().cpu().tolist()
    labels = batch["label"].detach().cpu().tolist()
    class_ids = batch["class_id"].detach().cpu().tolist()
    records = []
    for idx, score in enumerate(scores):
        records.append({
            "class_id": int(class_ids[idx]),
            "anchor_key": batch["anchor_keys"][idx],
            "label": int(labels[idx] > 0.5),
            "score": float(score),
            "negative_type": batch["negative_types"][idx],
        })
    return records


def _run_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    hard_negative_weight: float,
    ranking_margin: float,
    ranking_weight: float,
    epoch: int = 0,
    epochs: int = 1,
    log_every: int = 0,
    logger_obj=None,
) -> Dict[str, Any]:
    model.train(train)
    records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    phase = "train" if train else "val"
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        anchor_history = batch["anchor_history"].to(device)
        candidate_history = batch["candidate_history"].to(device)
        pair_features = batch["pair_features"].to(device)
        class_ids = batch["class_id"].to(device)
        labels = batch["label"].to(device)
        logits = model(
            anchor_history,
            candidate_history,
            pair_features,
            class_ids,
            state_buckets=batch["state_buckets"],
        )
        weights = torch.ones_like(labels)
        if hard_negative_weight != 1.0:
            hard_mask = torch.tensor(
                [item == "hard" for item in batch["negative_types"]],
                device=device,
                dtype=torch.bool,
            )
            weights[hard_mask] = float(hard_negative_weight)
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        ranking_loss = _association_ranking_loss_from_logits(
            logits,
            labels,
            batch["anchor_keys"],
            margin=float(ranking_margin),
        )
        loss = bce_loss + float(ranking_weight) * ranking_loss
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_count += count
        records.extend(_records_from_logits(batch, logits))
        if logger_obj is not None and train and log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
            logger_obj.info(
                "  [assoc %d/%d] batch %d/%d loss=%.4f bce=%.4f rank=%.4f lr=%.2e",
                epoch + 1,
                epochs,
                batch_idx + 1,
                len(loader),
                float(total_loss / max(total_count, 1)),
                float(bce_loss.detach().item()),
                float(ranking_loss.detach().item()),
                lr,
            )
    metrics = compute_association_metrics(records)
    metrics["loss"] = float(total_loss / max(total_count, 1))
    metrics["epoch_time"] = float(time.time() - t0)
    metrics["ranking"] = {
        "margin": float(ranking_margin),
        "weight": float(ranking_weight),
    }
    if logger_obj is not None and not train:
        overall = metrics.get("overall", {}) or {}
        logger_obj.info(
            "  [assoc %s] samples=%d loss=%.4f auc=%.4f top1=%.4f top3=%.4f hard_acc=%.4f time=%.1fs",
            phase,
            int(total_count),
            float(metrics.get("loss", 0.0)),
            float(overall.get("auc", 0.0) or 0.0),
            float(overall.get("top1", 0.0) or 0.0),
            float(overall.get("top3", 0.0) or 0.0),
            float(overall.get("hard_negative_accuracy", 0.0) or 0.0),
            float(metrics.get("epoch_time", 0.0)),
        )
    return metrics


def _get_logger(logger_obj=None):
    if logger_obj is not None:
        return logger_obj
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    return LOGGER


def train_pairwise_association(args, cfg: Dict[str, Any] | None = None, logger_obj=None) -> Dict[str, Any]:
    log = _get_logger(logger_obj)
    cfg = _load_yaml(args.train_config) if cfg is None else cfg
    assoc_cfg = cfg.get("ASSOCIATION_HEAD_TRAINING", {}) or {}
    args.train_pkl = args.train_pkl or assoc_cfg.get("TRAIN_PAIRWISE_PKL")
    args.val_pkl = args.val_pkl or assoc_cfg.get("VAL_PAIRWISE_PKL")
    args.output = args.output or assoc_cfg.get("OUTPUT")
    args.metrics_output = args.metrics_output or assoc_cfg.get("METRICS_OUTPUT")
    args.epochs = int(args.epochs if args.epochs is not None else assoc_cfg.get("EPOCHS", 10))
    args.batch_size = int(args.batch_size if args.batch_size is not None else assoc_cfg.get("BATCH_SIZE", 256))
    args.lr = float(args.lr if args.lr is not None else assoc_cfg.get("LR", 1.0e-4))
    args.weight_decay = float(args.weight_decay if args.weight_decay is not None else assoc_cfg.get("WEIGHT_DECAY", 1.0e-4))
    args.dropout = float(args.dropout if args.dropout is not None else assoc_cfg.get("DROPOUT", 0.1))
    args.hard_negative_weight = float(
        args.hard_negative_weight
        if args.hard_negative_weight is not None
        else assoc_cfg.get("HARD_NEGATIVE_WEIGHT", 2.0)
    )
    explicit_backbone_checkpoint = args.backbone_checkpoint
    args.backbone_checkpoint = args.backbone_checkpoint or assoc_cfg.get("BACKBONE_CHECKPOINT")
    if bool(args.dry_run) and explicit_backbone_checkpoint is None:
        args.backbone_checkpoint = None
    args.freeze_backbone = bool(args.freeze_backbone or assoc_cfg.get("FREEZE_BACKBONE", False))
    explicit_tensorboard_dir = args.tensorboard_dir
    args.tensorboard_dir = args.tensorboard_dir or assoc_cfg.get("TENSORBOARD_DIR")
    if bool(args.dry_run) and explicit_tensorboard_dir is None:
        args.tensorboard_dir = None
    args.ranking_margin = float(
        args.ranking_margin if args.ranking_margin is not None else assoc_cfg.get("RANKING_MARGIN", 0.2)
    )
    args.ranking_weight = float(
        args.ranking_weight if args.ranking_weight is not None else assoc_cfg.get("RANKING_WEIGHT", 0.0)
    )
    args.log_every = int(getattr(args, "log_every", None) or assoc_cfg.get("LOG_EVERY", 50))
    hidden_cfg = args.hidden_dims if args.hidden_dims is not None else assoc_cfg.get("HIDDEN_DIMS", [256, 128])
    if isinstance(hidden_cfg, (list, tuple)):
        hidden_dims = tuple(int(v) for v in hidden_cfg)
    else:
        hidden_dims = tuple(int(v) for v in str(hidden_cfg).split(",") if str(v).strip())
    missing = [
        name for name in ["train_pkl", "val_pkl", "output", "metrics_output"]
        if not getattr(args, name)
    ]
    if missing:
        raise ValueError(
            "Missing required association training settings: "
            + ", ".join(missing)
            + ". Provide CLI args or ASSOCIATION_HEAD_TRAINING in train config."
        )
    device = torch.device(args.device)

    train_ds = PairwiseAssociationDataset(args.train_pkl, history_len=int(args.history_len))
    val_ds = PairwiseAssociationDataset(args.val_pkl, history_len=int(args.history_len))
    log.info(
        "Association Stage B data: train=%d val=%d batch_size=%d num_workers=%d history_len=%d",
        len(train_ds),
        len(val_ds),
        int(args.batch_size),
        int(args.num_workers),
        int(args.history_len),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=pairwise_association_collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=pairwise_association_collate_fn,
    )

    encoder = _build_encoder(args, cfg)
    model = PairwiseAssociationModel(
        encoder,
        embed_dim=int(args.embed_dim if args.dry_run else (cfg.get("MODEL", {}) or {}).get("EMBED_DIM", args.embed_dim)),
        pair_feature_dim=7,
        num_classes=int((cfg.get("MODEL", {}) or {}).get("NUM_CLASSES", 7)),
        hidden_dims=hidden_dims,
        dropout=float(args.dropout),
        encoder_mode=str(cfg.get("FILTER_MODE", "mamba_multihead_closure")),
    ).to(device)
    backbone_load = _load_backbone_checkpoint(model, args.backbone_checkpoint)
    if args.freeze_backbone:
        _set_backbone_trainable(model, trainable=False)
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total_params = sum(param.numel() for param in model.parameters())
    log.info(
        "Association Stage B model: trainable=%d total=%d freeze_backbone=%s backbone_loaded_keys=%d",
        trainable_params,
        total_params,
        bool(args.freeze_backbone),
        int(backbone_load.get("loaded_keys", 0)),
    )
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir else None

    best_val = float("inf")
    best_payload = None
    for epoch in range(int(args.epochs)):
        t_epoch = time.time()
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            hard_negative_weight=float(args.hard_negative_weight),
            ranking_margin=float(args.ranking_margin),
            ranking_weight=float(args.ranking_weight),
            epoch=epoch,
            epochs=int(args.epochs),
            log_every=int(args.log_every),
            logger_obj=log,
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                hard_negative_weight=float(args.hard_negative_weight),
                ranking_margin=float(args.ranking_margin),
                ranking_weight=float(args.ranking_weight),
                epoch=epoch,
                epochs=int(args.epochs),
                log_every=0,
                logger_obj=log,
            )
        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
            "backbone_load": backbone_load,
        }
        overall_train = train_metrics.get("overall", {}) or {}
        overall_val = val_metrics.get("overall", {}) or {}
        log.info(
            "Assoc Epoch %d/%d (%.1fs) | loss=%.4f auc=%.4f top1=%.4f hard_acc=%.4f | "
            "val=%.4f val_auc=%.4f val_top1=%.4f val_hard_acc=%.4f",
            epoch + 1,
            int(args.epochs),
            float(time.time() - t_epoch),
            float(train_metrics.get("loss", 0.0)),
            float(overall_train.get("auc", 0.0) or 0.0),
            float(overall_train.get("top1", 0.0) or 0.0),
            float(overall_train.get("hard_negative_accuracy", 0.0) or 0.0),
            float(val_metrics.get("loss", 0.0)),
            float(overall_val.get("auc", 0.0) or 0.0),
            float(overall_val.get("top1", 0.0) or 0.0),
            float(overall_val.get("hard_negative_accuracy", 0.0) or 0.0),
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_metrics.get("loss", 0.0), epoch)
            writer.add_scalar("val/loss", val_metrics.get("loss", 0.0), epoch)
            writer.add_scalar("train/top1", train_metrics.get("overall", {}).get("top1", 0.0), epoch)
            writer.add_scalar("val/top1", val_metrics.get("overall", {}).get("top1", 0.0), epoch)
            writer.add_scalar("train/auc", train_metrics.get("overall", {}).get("auc", 0.0), epoch)
            writer.add_scalar("val/auc", val_metrics.get("overall", {}).get("auc", 0.0), epoch)
            writer.flush()
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_payload = payload
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "settings": vars(args),
                "metrics": payload,
                "runtime_contract": {
                    "filter_mode": str(cfg.get("FILTER_MODE", "mamba_multihead_closure")),
                    "history_source": str((cfg.get("DATA", {}) or {}).get("HISTORY_SOURCE", "det")).strip().lower(),
                    "init_state_source": str((cfg.get("DATA", {}) or {}).get("INIT_STATE_SOURCE", "det")).strip().lower(),
                    "backbone_checkpoint": args.backbone_checkpoint,
                    "freeze_backbone": bool(args.freeze_backbone),
                    "association_head": "class_conditioned_pairwise",
                },
                "backbone_load": backbone_load,
            }, output_path)
            log.info("  New best association head -> %s (val_loss=%.4f)", output_path, best_val)

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(best_payload or {}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if writer is not None:
        writer.close()
    log.info("Association Stage B complete. Best val_loss=%.4f output=%s", best_val, args.output)
    return {
        "best_val": best_val,
        "best_payload": best_payload,
        "output": args.output,
        "metrics_output": args.metrics_output,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Train class-conditioned pairwise association heads.")
    parser.add_argument("--train-pkl", default=None)
    parser.add_argument("--val-pkl", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics-output", default=None)
    parser.add_argument("--train-config", default="config/train_nuscenes.yaml")
    parser.add_argument("--history-len", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--backbone-checkpoint", default=None)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--ranking-margin", type=float, default=None)
    parser.add_argument("--ranking-weight", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_pairwise_association(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
