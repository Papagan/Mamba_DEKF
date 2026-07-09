#!/usr/bin/env python3
"""Train/evaluate the class-conditioned pairwise association head offline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

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


def _run_epoch(model, loader, optimizer, device, *, train: bool, hard_negative_weight: float) -> Dict[str, Any]:
    model.train(train)
    records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    for batch in loader:
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
        loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_count += count
        records.extend(_records_from_logits(batch, logits))
    metrics = compute_association_metrics(records)
    metrics["loss"] = float(total_loss / max(total_count, 1))
    return metrics


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
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--embed-dim", type=int, default=64)
    parser.add_argument("--hidden-dims", default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--hard-negative-weight", type=float, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    cfg = _load_yaml(args.train_config)
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
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )

    best_val = float("inf")
    best_payload = None
    for epoch in range(int(args.epochs)):
        train_metrics = _run_epoch(
            model,
            train_loader,
            optimizer,
            device,
            train=True,
            hard_negative_weight=float(args.hard_negative_weight),
        )
        with torch.no_grad():
            val_metrics = _run_epoch(
                model,
                val_loader,
                optimizer,
                device,
                train=False,
                hard_negative_weight=float(args.hard_negative_weight),
            )
        payload = {
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        }
        print(json.dumps(payload, ensure_ascii=False))
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
            }, output_path)

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(best_payload or {}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
