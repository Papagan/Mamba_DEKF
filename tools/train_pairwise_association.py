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
    AnchorGroupBatchSampler,
    PairwiseAssociationDataset,
    PrecomputedPairwiseAssociationDataset,
    pairwise_association_collate_fn,
    precomputed_pairwise_association_collate_fn,
)
from training.association_features import compose_pair_vectors  # noqa: E402
from kalmanfilter.checkpoint_compat import checkpoint_sha256  # noqa: E402
from training.association_metrics import compute_association_metrics  # noqa: E402
from training.association_model import (  # noqa: E402
    ClassConditionedAssociationHeadBank,
    PairwiseAssociationModel,
)


LOGGER = logging.getLogger("train.association")
CLASS_NAMES = ("car", "pedestrian", "bicycle", "motorcycle", "bus", "trailer", "truck")


def _pair_encoding_kwargs(batch, device):
    fields = (
        "anchor_current_range", "candidate_current_range",
        "anchor_history_mask", "anchor_history_match_mask",
        "candidate_history_mask", "anchor_detection_driven",
    )
    return {key: batch[key].to(device) for key in fields if key in batch}


def _is_hard_negative_type(value: Any) -> bool:
    return str(value).strip().lower() in {"hard", "inference_margin", "inference_topk"}


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


def _association_listwise_loss_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    anchor_keys: List[str],
) -> torch.Tensor:
    """Negative log probability assigned to the GT candidates of each anchor."""
    losses = []
    device = logits.device
    for anchor_key in sorted(set(anchor_keys)):
        indices = [idx for idx, key in enumerate(anchor_keys) if key == anchor_key]
        if not indices:
            continue
        idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
        group_logits = logits[idx_tensor]
        positive_mask = labels[idx_tensor] > 0.5
        if not bool(positive_mask.any().item()):
            continue
        log_normalizer = torch.logsumexp(group_logits, dim=0)
        positive_log_mass = torch.logsumexp(group_logits[positive_mask], dim=0)
        losses.append(log_normalizer - positive_log_mass)
    if not losses:
        return torch.zeros((), device=device, dtype=logits.dtype)
    return torch.stack(losses).mean()


def _association_bce_weights(
    labels: torch.Tensor,
    negative_types: List[str],
    *,
    hard_negative_weight: float,
    balance_positives: bool,
) -> torch.Tensor:
    """Build mean-one BCE weights with equal positive/negative total mass."""
    weights = torch.ones_like(labels)
    if hard_negative_weight != 1.0:
        hard_mask = torch.tensor(
            [_is_hard_negative_type(item) for item in negative_types],
            device=labels.device,
            dtype=torch.bool,
        )
        weights[hard_mask & (labels <= 0.5)] = float(hard_negative_weight)
    if balance_positives:
        positive_mask = labels > 0.5
        negative_mask = ~positive_mask
        if bool(positive_mask.any().item()) and bool(negative_mask.any().item()):
            positive_mass = torch.clamp(weights[positive_mask].sum(), min=1.0e-6)
            negative_mass = weights[negative_mask].sum()
            weights[positive_mask] *= negative_mass / positive_mass
    return weights / torch.clamp(weights.mean(), min=1.0e-6)


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
    listwise_weight: float = 0.0,
    balance_positives: bool = False,
    epoch: int = 0,
    epochs: int = 1,
    log_every: int = 0,
    logger_obj=None,
) -> Dict[str, Any]:
    model.train(train)
    records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    grad_norm_total = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
    logit_min_seen = float("inf")
    logit_max_seen = float("-inf")
    pos_score_sum = 0.0
    pos_score_count = 0
    neg_score_sum = 0.0
    neg_score_count = 0
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
            **_pair_encoding_kwargs(batch, device),
        )
        weights = _association_bce_weights(
            labels,
            batch["negative_types"],
            hard_negative_weight=float(hard_negative_weight),
            balance_positives=bool(balance_positives),
        )
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        ranking_loss = _association_ranking_loss_from_logits(
            logits,
            labels,
            batch["anchor_keys"],
            margin=float(ranking_margin),
        )
        listwise_loss = _association_listwise_loss_from_logits(
            logits, labels, batch["anchor_keys"]
        )
        loss = (
            bce_loss
            + float(ranking_weight) * ranking_loss
            + float(listwise_weight) * listwise_loss
        )
        scores_detached = torch.sigmoid(logits.detach())
        labels_detached = labels.detach() > 0.5
        if logits.numel() > 0:
            logit_min_seen = min(logit_min_seen, float(logits.detach().min().item()))
            logit_max_seen = max(logit_max_seen, float(logits.detach().max().item()))
        if bool(labels_detached.any().item()):
            pos_scores = scores_detached[labels_detached]
            pos_score_sum += float(pos_scores.sum().item())
            pos_score_count += int(pos_scores.numel())
        neg_mask = ~labels_detached
        if bool(neg_mask.any().item()):
            neg_scores = scores_detached[neg_mask]
            neg_score_sum += float(neg_scores.sum().item())
            neg_score_count += int(neg_scores.numel())
        grad_norm_value = 0.0
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            grad_norm_total += grad_norm_value
            grad_norm_max = max(grad_norm_max, grad_norm_value)
            grad_norm_count += 1
            optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_count += count
        records.extend(_records_from_logits(batch, logits))
        if logger_obj is not None and train and log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
            logger_obj.info(
                "  [assoc %d/%d] batch %d/%d loss=%.6g bce=%.6g rank=%.6g list=%.6g "
                "grad=%.3e logit=[%.2f,%.2f] score_pos=%.4f score_neg=%.4f lr=%.2e",
                epoch + 1,
                epochs,
                batch_idx + 1,
                len(loader),
                float(total_loss / max(total_count, 1)),
                float(bce_loss.detach().item()),
                float(ranking_loss.detach().item()),
                float(listwise_loss.detach().item()),
                float(grad_norm_value),
                float(logits.detach().min().item()) if logits.numel() > 0 else 0.0,
                float(logits.detach().max().item()) if logits.numel() > 0 else 0.0,
                float(scores_detached[labels_detached].mean().item()) if bool(labels_detached.any().item()) else 0.0,
                float(scores_detached[neg_mask].mean().item()) if bool(neg_mask.any().item()) else 0.0,
                lr,
            )
    metrics_t0 = time.time()
    metrics = compute_association_metrics(records)
    metrics_time = time.time() - metrics_t0
    metrics["loss"] = float(total_loss / max(total_count, 1))
    metrics["epoch_time"] = float(time.time() - t0)
    metrics["metrics_time"] = float(metrics_time)
    metrics["grad_norm_mean"] = float(grad_norm_total / max(grad_norm_count, 1))
    metrics["grad_norm_max"] = float(grad_norm_max)
    metrics["logit_min"] = float(logit_min_seen if logit_min_seen != float("inf") else 0.0)
    metrics["logit_max"] = float(logit_max_seen if logit_max_seen != float("-inf") else 0.0)
    metrics["score_pos_mean"] = float(pos_score_sum / max(pos_score_count, 1))
    metrics["score_neg_mean"] = float(neg_score_sum / max(neg_score_count, 1))
    metrics["ranking"] = {
        "margin": float(ranking_margin),
        "weight": float(ranking_weight),
    }
    metrics["listwise"] = {"weight": float(listwise_weight)}
    metrics["bce_balance_positives"] = bool(balance_positives)
    if logger_obj is not None and not train:
        overall = metrics.get("overall", {}) or {}
        logger_obj.info(
            "  [assoc %s] samples=%d loss=%.4f auc=%.4f top1=%.4f top3=%.4f hard_acc=%.4f time=%.1fs metrics=%.1fs",
            phase,
            int(total_count),
            float(metrics.get("loss", 0.0)),
            float(overall.get("auc", 0.0) or 0.0),
            float(overall.get("top1", 0.0) or 0.0),
            float(overall.get("top3", 0.0) or 0.0),
            float(overall.get("hard_negative_accuracy", 0.0) or 0.0),
            float(metrics.get("epoch_time", 0.0)),
            float(metrics.get("metrics_time", 0.0)),
        )
    return metrics


def _pair_vectors_from_batch(model, batch, device) -> torch.Tensor:
    anchor_history = batch["anchor_history"].to(device)
    candidate_history = batch["candidate_history"].to(device)
    pair_features = batch["pair_features"].to(device)
    class_ids = batch["class_id"].to(device)
    encoding_kwargs = _pair_encoding_kwargs(batch, device)
    track_kwargs = {}
    for batch_key, model_key in (
        ("anchor_current_range", "current_range"),
        ("anchor_history_mask", "history_mask"),
        ("anchor_history_match_mask", "history_match_mask"),
        ("anchor_detection_driven", "detection_driven_mask"),
    ):
        if batch_key in encoding_kwargs:
            track_kwargs[model_key] = encoding_kwargs[batch_key]
    detection_kwargs = {}
    if "candidate_current_range" in encoding_kwargs:
        detection_kwargs = {
            "current_range": encoding_kwargs["candidate_current_range"],
            "history_mask": encoding_kwargs["candidate_history_mask"],
            "history_match_mask": encoding_kwargs["candidate_history_mask"],
        }
    track_embedding = model.encode(
        anchor_history, class_ids, state_buckets=batch["state_buckets"],
        **track_kwargs,
    )
    det_embedding = model.encode(
        candidate_history,
        class_ids,
        state_buckets=["matched"] * int(class_ids.shape[0]),
        **detection_kwargs,
    )
    return compose_pair_vectors(track_embedding, det_embedding, pair_features)


def _precompute_pair_vectors(model, loader, device, *, logger_obj=None, label: str = "train") -> Dict[str, Any]:
    model.eval()
    pair_chunks: List[torch.Tensor] = []
    class_chunks: List[torch.Tensor] = []
    label_chunks: List[torch.Tensor] = []
    anchor_keys: List[str] = []
    negative_types: List[str] = []
    categories: List[str] = []
    sample_count = 0
    t0 = time.time()
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            pair_vectors = _pair_vectors_from_batch(model, batch, device).detach().cpu()
            pair_chunks.append(pair_vectors)
            class_chunks.append(batch["class_id"].detach().cpu())
            label_chunks.append(batch["label"].detach().cpu())
            anchor_keys.extend(batch["anchor_keys"])
            negative_types.extend(batch["negative_types"])
            categories.extend(batch["categories"])
            sample_count += int(pair_vectors.shape[0])
            if logger_obj is not None and (batch_idx + 1) % 100 == 0:
                logger_obj.info(
                    "  [assoc precompute %s] batch %d/%d samples=%d",
                    label,
                    batch_idx + 1,
                    len(loader),
                    sample_count,
                )
    out = {
        "pair_vector": torch.cat(pair_chunks, dim=0) if pair_chunks else torch.empty(0, 0),
        "class_id": torch.cat(class_chunks, dim=0) if class_chunks else torch.empty(0, dtype=torch.long),
        "label": torch.cat(label_chunks, dim=0) if label_chunks else torch.empty(0),
        "anchor_keys": anchor_keys,
        "negative_types": negative_types,
        "categories": categories,
    }
    if logger_obj is not None:
        logger_obj.info(
            "Association precompute %s complete: samples=%d pair_dim=%d time=%.1fs",
            label,
            sample_count,
            int(out["pair_vector"].shape[1]) if out["pair_vector"].ndim == 2 else 0,
            time.time() - t0,
        )
    return out


def _select_precomputed_class(samples: Dict[str, Any], class_id: int) -> Dict[str, Any]:
    """Select one class without recomputing the shared-backbone embeddings."""
    class_tensor = torch.as_tensor(samples["class_id"], dtype=torch.long)
    indices = torch.nonzero(class_tensor == int(class_id), as_tuple=False).flatten()
    index_list = indices.tolist()
    return {
        "pair_vector": torch.as_tensor(samples["pair_vector"])[indices],
        "class_id": class_tensor[indices],
        "label": torch.as_tensor(samples["label"], dtype=torch.float32)[indices],
        "anchor_keys": [samples["anchor_keys"][idx] for idx in index_list],
        "negative_types": [samples["negative_types"][idx] for idx in index_list],
        "categories": [samples["categories"][idx] for idx in index_list],
    }


def _make_precomputed_loader(
    samples: Dict[str, Any],
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    dataset = PrecomputedPairwiseAssociationDataset(samples)
    batch_sampler = AnchorGroupBatchSampler(
        dataset.anchor_keys,
        int(batch_size),
        shuffle=bool(shuffle),
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        collate_fn=precomputed_pairwise_association_collate_fn,
    )


def _make_pairwise_loader(
    dataset: PairwiseAssociationDataset,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    batch_sampler = AnchorGroupBatchSampler(
        dataset.anchor_keys,
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
    )
    return DataLoader(
        dataset,
        batch_sampler=batch_sampler,
        num_workers=int(num_workers),
        collate_fn=pairwise_association_collate_fn,
    )


def _parse_class_ids(value: Any, num_classes: int) -> List[int]:
    if value is None:
        class_ids = list(range(int(num_classes)))
    elif isinstance(value, str):
        class_ids = [int(item.strip()) for item in value.split(",") if item.strip()]
    else:
        class_ids = [int(item) for item in value]
    class_ids = list(dict.fromkeys(class_ids))
    invalid = [class_id for class_id in class_ids if class_id < 0 or class_id >= int(num_classes)]
    if invalid:
        raise ValueError(f"Association CLASS_IDS out of range [0, {num_classes - 1}]: {invalid}")
    if not class_ids:
        raise ValueError("Association CLASS_IDS must contain at least one class")
    return class_ids


def _cpu_state_dict(module: nn.Module) -> Dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def _save_association_checkpoint(
    model: PairwiseAssociationModel,
    args,
    cfg: Dict[str, Any],
    *,
    epoch: int,
    metrics: Dict[str, Any],
    backbone_load: Dict[str, Any],
    use_precomputed: bool,
) -> None:
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "settings": vars(args),
        "metrics": metrics,
        "runtime_contract": {
            "filter_mode": str(cfg.get("FILTER_MODE", "mamba_multihead_closure")),
            "history_source": str((cfg.get("DATA", {}) or {}).get("HISTORY_SOURCE", "det")).strip().lower(),
            "init_state_source": str((cfg.get("DATA", {}) or {}).get("INIT_STATE_SOURCE", "det")).strip().lower(),
            "tracker_compat_mode": str(
                (cfg.get("DATA", {}) or {}).get("TRAIN_TRACKER_COMPAT_MODE", "default")
            ).strip().lower(),
            "expected_bev_cost_mode": str(
                (cfg.get("DATA", {}) or {}).get("EXPECTED_BEV_COST_MODE", "geometric")
            ).strip().lower(),
            "backbone_checkpoint": args.backbone_checkpoint,
            "backbone_sha256": checkpoint_sha256(args.backbone_checkpoint),
            "freeze_backbone": bool(args.freeze_backbone),
            "precompute_embeddings": bool(use_precomputed),
            "association_head": "class_conditioned_pairwise",
            "association_training_mode": str(args.train_mode),
            "independent_class_selection": bool(args.train_mode == "per_class"),
        },
        "backbone_load": backbone_load,
    }, output_path)


def _train_heads_per_class(
    model: PairwiseAssociationModel,
    train_samples: Dict[str, Any],
    val_samples: Dict[str, Any],
    args,
    cfg: Dict[str, Any],
    device: torch.device,
    backbone_load: Dict[str, Any],
    writer,
    log,
) -> Dict[str, Any]:
    """Optimize and select every category head independently on frozen embeddings."""
    num_classes = int(model.association_heads.num_classes)
    class_ids = _parse_class_ids(args.class_ids, num_classes)
    class_results: Dict[str, Any] = {}

    for class_id in class_ids:
        train_class = _select_precomputed_class(train_samples, class_id)
        val_class = _select_precomputed_class(val_samples, class_id)
        train_count = int(train_class["class_id"].numel())
        val_count = int(val_class["class_id"].numel())
        class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
        if train_count == 0 or val_count == 0:
            message = (
                f"Association class {class_id} ({class_name}) has insufficient data: "
                f"train={train_count}, val={val_count}"
            )
            if args.require_all_classes:
                raise ValueError(message)
            log.warning("%s; leaving this head at initialization.", message)
            class_results[str(class_id)] = {
                "class_name": class_name,
                "status": "skipped_missing_data",
                "train_pairs": train_count,
                "val_pairs": val_count,
            }
            continue

        train_loader = _make_precomputed_loader(
            train_class,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
        )
        val_loader = _make_precomputed_loader(
            val_class,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        target_head = model.association_heads.heads[class_id]
        optimizer = torch.optim.AdamW(
            target_head.parameters(),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
        )
        best_val = float("inf")
        best_epoch = -1
        best_head_state = None
        best_train_metrics = None
        best_val_metrics = None
        log.info(
            "Association class %d/%d (%s): train=%d val=%d independent_epochs=%d",
            class_id,
            num_classes - 1,
            class_name,
            train_count,
            val_count,
            int(args.epochs),
        )

        for epoch in range(int(args.epochs)):
            train_metrics = _run_precomputed_epoch(
                model.association_heads,
                train_loader,
                optimizer,
                device,
                train=True,
                hard_negative_weight=float(args.hard_negative_weight),
                ranking_margin=float(args.ranking_margin),
                ranking_weight=float(args.ranking_weight),
                listwise_weight=float(args.listwise_weight),
                balance_positives=bool(args.balance_positives),
                epoch=epoch,
                epochs=int(args.epochs),
                log_every=int(args.log_every),
                logger_obj=log,
            )
            with torch.no_grad():
                val_metrics = _run_precomputed_epoch(
                    model.association_heads,
                    val_loader,
                    optimizer,
                    device,
                    train=False,
                    hard_negative_weight=float(args.hard_negative_weight),
                    ranking_margin=float(args.ranking_margin),
                    ranking_weight=float(args.ranking_weight),
                    listwise_weight=float(args.listwise_weight),
                    balance_positives=bool(args.balance_positives),
                    epoch=epoch,
                    epochs=int(args.epochs),
                    log_every=0,
                    logger_obj=log,
                )
            if writer is not None:
                prefix = f"class_{class_id}_{class_name}"
                writer.add_scalar(f"{prefix}/train_loss", train_metrics["loss"], epoch)
                writer.add_scalar(f"{prefix}/val_loss", val_metrics["loss"], epoch)
                writer.add_scalar(
                    f"{prefix}/val_top1", val_metrics.get("overall", {}).get("top1", 0.0), epoch
                )
                writer.add_scalar(
                    f"{prefix}/val_auc", val_metrics.get("overall", {}).get("auc", 0.0), epoch
                )
            if float(val_metrics["loss"]) < best_val:
                best_val = float(val_metrics["loss"])
                best_epoch = epoch
                best_head_state = _cpu_state_dict(target_head)
                best_train_metrics = train_metrics
                best_val_metrics = val_metrics

        if best_head_state is None:
            raise RuntimeError(f"Association class {class_id} did not produce a valid checkpoint")
        target_head.load_state_dict(best_head_state)
        class_results[str(class_id)] = {
            "class_name": class_name,
            "status": "trained",
            "train_pairs": train_count,
            "val_pairs": val_count,
            "best_epoch": int(best_epoch),
            "best_val_loss": float(best_val),
            "train": best_train_metrics,
            "val": best_val_metrics,
        }
        log.info(
            "  Selected %s head epoch %d/%d (val_loss=%.4f, val_top1=%.4f)",
            class_name,
            best_epoch + 1,
            int(args.epochs),
            best_val,
            float((best_val_metrics or {}).get("overall", {}).get("top1", 0.0) or 0.0),
        )

    combined_train_loader = _make_precomputed_loader(
        train_samples,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    combined_val_loader = _make_precomputed_loader(
        val_samples,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
    )
    # Evaluation does not use the optimizer; reuse the epoch runner to keep
    # joint and per-class metric semantics identical.
    eval_optimizer = torch.optim.AdamW(model.association_heads.parameters(), lr=float(args.lr))
    with torch.no_grad():
        combined_train_metrics = _run_precomputed_epoch(
            model.association_heads,
            combined_train_loader,
            eval_optimizer,
            device,
            train=False,
            hard_negative_weight=float(args.hard_negative_weight),
            ranking_margin=float(args.ranking_margin),
            ranking_weight=float(args.ranking_weight),
            listwise_weight=float(args.listwise_weight),
            balance_positives=bool(args.balance_positives),
            logger_obj=None,
        )
        combined_val_metrics = _run_precomputed_epoch(
            model.association_heads,
            combined_val_loader,
            eval_optimizer,
            device,
            train=False,
            hard_negative_weight=float(args.hard_negative_weight),
            ranking_margin=float(args.ranking_margin),
            ranking_weight=float(args.ranking_weight),
            listwise_weight=float(args.listwise_weight),
            balance_positives=bool(args.balance_positives),
            logger_obj=log,
        )
    payload = {
        "training_mode": "per_class",
        "class_ids": class_ids,
        "classes": class_results,
        "train": combined_train_metrics,
        "val": combined_val_metrics,
        "backbone_load": backbone_load,
    }
    _save_association_checkpoint(
        model,
        args,
        cfg,
        epoch=int(args.epochs) - 1,
        metrics=payload,
        backbone_load=backbone_load,
        use_precomputed=True,
    )
    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if writer is not None:
        writer.flush()
        writer.close()
    best_val = float(combined_val_metrics["loss"])
    log.info(
        "Association Stage B per-class training complete. Combined val_loss=%.4f output=%s",
        best_val,
        args.output,
    )
    return {
        "best_val": best_val,
        "best_payload": payload,
        "output": args.output,
        "metrics_output": args.metrics_output,
    }


def _run_precomputed_epoch(
    head,
    loader,
    optimizer,
    device,
    *,
    train: bool,
    hard_negative_weight: float,
    ranking_margin: float,
    ranking_weight: float,
    listwise_weight: float = 0.0,
    balance_positives: bool = False,
    epoch: int = 0,
    epochs: int = 1,
    log_every: int = 0,
    logger_obj=None,
) -> Dict[str, Any]:
    head.train(train)
    records: List[Dict[str, Any]] = []
    total_loss = 0.0
    total_count = 0
    grad_norm_total = 0.0
    grad_norm_max = 0.0
    grad_norm_count = 0
    logit_min_seen = float("inf")
    logit_max_seen = float("-inf")
    pos_score_sum = 0.0
    pos_score_count = 0
    neg_score_sum = 0.0
    neg_score_count = 0
    t0 = time.time()
    for batch_idx, batch in enumerate(loader):
        pair_vector = batch["pair_vector"].to(device)
        class_ids = batch["class_id"].to(device)
        labels = batch["label"].to(device)
        logits = head(pair_vector, class_ids)
        weights = _association_bce_weights(
            labels,
            batch["negative_types"],
            hard_negative_weight=float(hard_negative_weight),
            balance_positives=bool(balance_positives),
        )
        bce_loss = F.binary_cross_entropy_with_logits(logits, labels, weight=weights)
        ranking_loss = _association_ranking_loss_from_logits(
            logits,
            labels,
            batch["anchor_keys"],
            margin=float(ranking_margin),
        )
        listwise_loss = _association_listwise_loss_from_logits(
            logits, labels, batch["anchor_keys"]
        )
        loss = (
            bce_loss
            + float(ranking_weight) * ranking_loss
            + float(listwise_weight) * listwise_loss
        )
        scores_detached = torch.sigmoid(logits.detach())
        labels_detached = labels.detach() > 0.5
        if logits.numel() > 0:
            logit_min_seen = min(logit_min_seen, float(logits.detach().min().item()))
            logit_max_seen = max(logit_max_seen, float(logits.detach().max().item()))
        if bool(labels_detached.any().item()):
            pos_scores = scores_detached[labels_detached]
            pos_score_sum += float(pos_scores.sum().item())
            pos_score_count += int(pos_scores.numel())
        neg_mask = ~labels_detached
        if bool(neg_mask.any().item()):
            neg_scores = scores_detached[neg_mask]
            neg_score_sum += float(neg_scores.sum().item())
            neg_score_count += int(neg_scores.numel())
        grad_norm_value = 0.0
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            grad_norm_value = float(grad_norm.detach().item() if torch.is_tensor(grad_norm) else grad_norm)
            grad_norm_total += grad_norm_value
            grad_norm_max = max(grad_norm_max, grad_norm_value)
            grad_norm_count += 1
            optimizer.step()
        count = int(labels.shape[0])
        total_loss += float(loss.detach().item()) * count
        total_count += count
        scores = scores_detached.cpu().tolist()
        label_values = labels.detach().cpu().tolist()
        class_values = class_ids.detach().cpu().tolist()
        for idx, score in enumerate(scores):
            records.append({
                "class_id": int(class_values[idx]),
                "anchor_key": batch["anchor_keys"][idx],
                "label": int(label_values[idx] > 0.5),
                "score": float(score),
                "negative_type": batch["negative_types"][idx],
            })
        if logger_obj is not None and train and log_every > 0 and (batch_idx + 1) % int(log_every) == 0:
            lr = float(optimizer.param_groups[0]["lr"]) if optimizer.param_groups else 0.0
            logger_obj.info(
                "  [assoc-fast %d/%d] batch %d/%d loss=%.6g bce=%.6g rank=%.6g list=%.6g "
                "grad=%.3e logit=[%.2f,%.2f] score_pos=%.4f score_neg=%.4f lr=%.2e",
                epoch + 1,
                epochs,
                batch_idx + 1,
                len(loader),
                float(total_loss / max(total_count, 1)),
                float(bce_loss.detach().item()),
                float(ranking_loss.detach().item()),
                float(listwise_loss.detach().item()),
                float(grad_norm_value),
                float(logits.detach().min().item()) if logits.numel() > 0 else 0.0,
                float(logits.detach().max().item()) if logits.numel() > 0 else 0.0,
                float(scores_detached[labels_detached].mean().item()) if bool(labels_detached.any().item()) else 0.0,
                float(scores_detached[neg_mask].mean().item()) if bool(neg_mask.any().item()) else 0.0,
                lr,
            )
    metrics_t0 = time.time()
    metrics = compute_association_metrics(records)
    metrics_time = time.time() - metrics_t0
    metrics["loss"] = float(total_loss / max(total_count, 1))
    metrics["epoch_time"] = float(time.time() - t0)
    metrics["metrics_time"] = float(metrics_time)
    metrics["grad_norm_mean"] = float(grad_norm_total / max(grad_norm_count, 1))
    metrics["grad_norm_max"] = float(grad_norm_max)
    metrics["logit_min"] = float(logit_min_seen if logit_min_seen != float("inf") else 0.0)
    metrics["logit_max"] = float(logit_max_seen if logit_max_seen != float("-inf") else 0.0)
    metrics["score_pos_mean"] = float(pos_score_sum / max(pos_score_count, 1))
    metrics["score_neg_mean"] = float(neg_score_sum / max(neg_score_count, 1))
    metrics["ranking"] = {
        "margin": float(ranking_margin),
        "weight": float(ranking_weight),
    }
    metrics["listwise"] = {"weight": float(listwise_weight)}
    metrics["bce_balance_positives"] = bool(balance_positives)
    if logger_obj is not None and not train:
        overall = metrics.get("overall", {}) or {}
        logger_obj.info(
            "  [assoc-fast val] samples=%d loss=%.4f auc=%.4f top1=%.4f top3=%.4f hard_acc=%.4f time=%.1fs metrics=%.1fs",
            int(total_count),
            float(metrics.get("loss", 0.0)),
            float(overall.get("auc", 0.0) or 0.0),
            float(overall.get("top1", 0.0) or 0.0),
            float(overall.get("top3", 0.0) or 0.0),
            float(overall.get("hard_negative_accuracy", 0.0) or 0.0),
            float(metrics.get("epoch_time", 0.0)),
            float(metrics.get("metrics_time", 0.0)),
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
    args.listwise_weight = float(
        getattr(args, "listwise_weight", None)
        if getattr(args, "listwise_weight", None) is not None
        else assoc_cfg.get("LISTWISE_WEIGHT", 0.0)
    )
    args.balance_positives = bool(
        getattr(args, "balance_positives", False)
        or assoc_cfg.get("BALANCE_POSITIVES", False)
    )
    args.log_every = int(getattr(args, "log_every", None) or assoc_cfg.get("LOG_EVERY", 50))
    args.precompute_embeddings = bool(
        getattr(args, "precompute_embeddings", False)
        or assoc_cfg.get("PRECOMPUTE_EMBEDDINGS", False)
    )
    args.train_mode = str(
        getattr(args, "train_mode", None) or assoc_cfg.get("TRAIN_MODE", "joint")
    ).strip().lower()
    if args.train_mode not in {"joint", "per_class"}:
        raise ValueError("Association TRAIN_MODE must be 'joint' or 'per_class'")
    args.class_ids = (
        getattr(args, "class_ids", None)
        if getattr(args, "class_ids", None) is not None
        else assoc_cfg.get("CLASS_IDS")
    )
    args.require_all_classes = bool(
        getattr(args, "require_all_classes", False)
        or assoc_cfg.get("REQUIRE_ALL_CLASSES", False)
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

    configured_schema = assoc_cfg.get("REQUIRE_FEATURE_SCHEMA_VERSION")
    required_schema = (
        None if bool(args.dry_run) or configured_schema is None else int(configured_schema)
    )
    train_ds = PairwiseAssociationDataset(
        args.train_pkl, history_len=int(args.history_len),
        required_feature_schema_version=required_schema,
    )
    val_ds = PairwiseAssociationDataset(
        args.val_pkl, history_len=int(args.history_len),
        required_feature_schema_version=required_schema,
    )
    log.info(
        "Association Stage B data: train=%d val=%d epochs=%d batch_size=%d num_workers=%d history_len=%d",
        len(train_ds),
        len(val_ds),
        int(args.epochs),
        int(args.batch_size),
        int(args.num_workers),
        int(args.history_len),
    )
    log.info(
        "Association Stage B optim: lr=%.2e weight_decay=%.2e hard_negative_weight=%.2f "
        "ranking_margin=%.2f ranking_weight=%.2f listwise_weight=%.2f "
        "balance_positives=%s log_every=%d",
        float(args.lr),
        float(args.weight_decay),
        float(args.hard_negative_weight),
        float(args.ranking_margin),
        float(args.ranking_weight),
        float(args.listwise_weight),
        bool(args.balance_positives),
        int(args.log_every),
    )
    train_loader = _make_pairwise_loader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
    )
    val_loader = _make_pairwise_loader(
        val_ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
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
    use_precomputed = bool(args.freeze_backbone and args.precompute_embeddings)
    if args.train_mode == "per_class" and not use_precomputed:
        raise ValueError(
            "Association TRAIN_MODE=per_class requires FREEZE_BACKBONE=true and "
            "PRECOMPUTE_EMBEDDINGS=true so all category heads share an immutable backbone."
        )
    if args.train_mode == "per_class" and args.backbone_checkpoint and (
        int(backbone_load.get("loaded_keys", 0)) == 0
        or bool(backbone_load.get("missing_keys"))
    ):
        raise RuntimeError(
            "Association per-class training requires a complete shared-backbone load; "
            f"loaded_keys={backbone_load.get('loaded_keys', 0)}, "
            f"missing_keys={backbone_load.get('missing_keys', [])}"
        )
    if use_precomputed:
        train_samples = _precompute_pair_vectors(model, train_loader, device, logger_obj=log, label="train")
        val_samples = _precompute_pair_vectors(model, val_loader, device, logger_obj=log, label="val")
        train_loader = _make_precomputed_loader(
            train_samples,
            batch_size=int(args.batch_size),
            shuffle=True,
            num_workers=int(args.num_workers),
        )
        val_loader = _make_precomputed_loader(
            val_samples,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=int(args.num_workers),
        )
        trainable_module = model.association_heads
        log.info("Association Stage B fast path enabled: precomputed frozen-backbone pair vectors.")
    else:
        trainable_module = model

    writer = SummaryWriter(log_dir=args.tensorboard_dir) if args.tensorboard_dir else None
    if args.train_mode == "per_class":
        return _train_heads_per_class(
            model,
            train_samples,
            val_samples,
            args,
            cfg,
            device,
            backbone_load,
            writer,
            log,
        )

    optimizer = torch.optim.AdamW(
        [param for param in trainable_module.parameters() if param.requires_grad],
        lr=float(args.lr),
        weight_decay=float(args.weight_decay),
    )
    best_val = float("inf")
    best_payload = None
    for epoch in range(int(args.epochs)):
        t_epoch = time.time()
        if use_precomputed:
            train_metrics = _run_precomputed_epoch(
                model.association_heads,
                train_loader,
                optimizer,
                device,
                train=True,
                hard_negative_weight=float(args.hard_negative_weight),
                ranking_margin=float(args.ranking_margin),
                ranking_weight=float(args.ranking_weight),
                listwise_weight=float(args.listwise_weight),
                balance_positives=bool(args.balance_positives),
                epoch=epoch,
                epochs=int(args.epochs),
                log_every=int(args.log_every),
                logger_obj=log,
            )
        else:
            train_metrics = _run_epoch(
                model,
                train_loader,
                optimizer,
                device,
                train=True,
                hard_negative_weight=float(args.hard_negative_weight),
                ranking_margin=float(args.ranking_margin),
                ranking_weight=float(args.ranking_weight),
                listwise_weight=float(args.listwise_weight),
                balance_positives=bool(args.balance_positives),
                epoch=epoch,
                epochs=int(args.epochs),
                log_every=int(args.log_every),
                logger_obj=log,
            )
        with torch.no_grad():
            if use_precomputed:
                val_metrics = _run_precomputed_epoch(
                    model.association_heads,
                    val_loader,
                    optimizer,
                    device,
                    train=False,
                    hard_negative_weight=float(args.hard_negative_weight),
                    ranking_margin=float(args.ranking_margin),
                    ranking_weight=float(args.ranking_weight),
                    listwise_weight=float(args.listwise_weight),
                    balance_positives=bool(args.balance_positives),
                    epoch=epoch,
                    epochs=int(args.epochs),
                    log_every=0,
                    logger_obj=log,
                )
            else:
                val_metrics = _run_epoch(
                    model,
                    val_loader,
                    optimizer,
                    device,
                    train=False,
                    hard_negative_weight=float(args.hard_negative_weight),
                    ranking_margin=float(args.ranking_margin),
                    ranking_weight=float(args.ranking_weight),
                    listwise_weight=float(args.listwise_weight),
                    balance_positives=bool(args.balance_positives),
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
            "val=%.4f val_auc=%.4f val_top1=%.4f val_hard_acc=%.4f | "
            "grad=%.3e/%.3e score=%.4f/%.4f logits=[%.2f,%.2f]",
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
            float(train_metrics.get("grad_norm_mean", 0.0)),
            float(train_metrics.get("grad_norm_max", 0.0)),
            float(train_metrics.get("score_pos_mean", 0.0)),
            float(train_metrics.get("score_neg_mean", 0.0)),
            float(train_metrics.get("logit_min", 0.0)),
            float(train_metrics.get("logit_max", 0.0)),
        )
        if writer is not None:
            writer.add_scalar("train/loss", train_metrics.get("loss", 0.0), epoch)
            writer.add_scalar("val/loss", val_metrics.get("loss", 0.0), epoch)
            writer.add_scalar("train/top1", train_metrics.get("overall", {}).get("top1", 0.0), epoch)
            writer.add_scalar("val/top1", val_metrics.get("overall", {}).get("top1", 0.0), epoch)
            writer.add_scalar("train/auc", train_metrics.get("overall", {}).get("auc", 0.0), epoch)
            writer.add_scalar("val/auc", val_metrics.get("overall", {}).get("auc", 0.0), epoch)
            writer.add_scalar("train/grad_norm_mean", train_metrics.get("grad_norm_mean", 0.0), epoch)
            writer.add_scalar("train/grad_norm_max", train_metrics.get("grad_norm_max", 0.0), epoch)
            writer.add_scalar("train/score_pos_mean", train_metrics.get("score_pos_mean", 0.0), epoch)
            writer.add_scalar("train/score_neg_mean", train_metrics.get("score_neg_mean", 0.0), epoch)
            writer.add_scalar("train/logit_min", train_metrics.get("logit_min", 0.0), epoch)
            writer.add_scalar("train/logit_max", train_metrics.get("logit_max", 0.0), epoch)
            writer.add_scalar("val/score_pos_mean", val_metrics.get("score_pos_mean", 0.0), epoch)
            writer.add_scalar("val/score_neg_mean", val_metrics.get("score_neg_mean", 0.0), epoch)
            writer.add_scalar("val/logit_min", val_metrics.get("logit_min", 0.0), epoch)
            writer.add_scalar("val/logit_max", val_metrics.get("logit_max", 0.0), epoch)
            writer.flush()
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_payload = payload
            _save_association_checkpoint(
                model,
                args,
                cfg,
                epoch=epoch,
                metrics=payload,
                backbone_load=backbone_load,
                use_precomputed=use_precomputed,
            )
            log.info("  New best association head -> %s (val_loss=%.4f)", args.output, best_val)

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
    parser.add_argument("--precompute-embeddings", action="store_true")
    parser.add_argument("--train-mode", choices=("joint", "per_class"), default=None)
    parser.add_argument("--class-ids", default=None, help="Comma-separated class ids for per-class training.")
    parser.add_argument("--require-all-classes", action="store_true")
    parser.add_argument("--tensorboard-dir", default=None)
    parser.add_argument("--ranking-margin", type=float, default=None)
    parser.add_argument("--ranking-weight", type=float, default=None)
    parser.add_argument("--listwise-weight", type=float, default=None)
    parser.add_argument("--balance-positives", action="store_true")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    train_pairwise_association(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
