from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict


def _bool_cfg(value: Any, default: bool = False) -> bool:
    if value is None:
        return bool(default)
    return bool(value)


def _require_file(path: str | None, label: str) -> str:
    if not path:
        raise ValueError(f"ASSOCIATION_HEAD_TRAINING.{label} is required when RUN_AFTER_MAIN=true")
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Pairwise association cache not found: {resolved}. "
            "Generate it with tools/build_pairwise_association_cache.py before running Stage B."
        )
    return str(resolved)


def build_association_training_args(
    cfg: Dict[str, Any],
    *,
    config_path: str,
    device: str,
) -> SimpleNamespace:
    assoc_cfg = cfg.get("ASSOCIATION_HEAD_TRAINING", {}) or {}
    model_cfg = cfg.get("MODEL", {}) or {}
    train_pkl = _require_file(assoc_cfg.get("TRAIN_PAIRWISE_PKL"), "TRAIN_PAIRWISE_PKL")
    val_pkl = _require_file(assoc_cfg.get("VAL_PAIRWISE_PKL"), "VAL_PAIRWISE_PKL")
    return SimpleNamespace(
        train_pkl=train_pkl,
        val_pkl=val_pkl,
        output=assoc_cfg.get("OUTPUT"),
        metrics_output=assoc_cfg.get("METRICS_OUTPUT"),
        train_config=config_path,
        history_len=int(model_cfg.get("HISTORY_LEN", 8)),
        epochs=assoc_cfg.get("EPOCHS"),
        batch_size=assoc_cfg.get("BATCH_SIZE"),
        num_workers=int(assoc_cfg.get("NUM_WORKERS", 0)),
        lr=assoc_cfg.get("LR"),
        weight_decay=assoc_cfg.get("WEIGHT_DECAY"),
        embed_dim=int(model_cfg.get("EMBED_DIM", 64)),
        hidden_dims=assoc_cfg.get("HIDDEN_DIMS"),
        dropout=assoc_cfg.get("DROPOUT"),
        hard_negative_weight=assoc_cfg.get("HARD_NEGATIVE_WEIGHT"),
        backbone_checkpoint=assoc_cfg.get("BACKBONE_CHECKPOINT"),
        freeze_backbone=_bool_cfg(assoc_cfg.get("FREEZE_BACKBONE"), True),
        tensorboard_dir=assoc_cfg.get("TENSORBOARD_DIR"),
        ranking_margin=assoc_cfg.get("RANKING_MARGIN"),
        ranking_weight=assoc_cfg.get("RANKING_WEIGHT"),
        device=str(device),
        dry_run=False,
    )


def run_association_head_stage_if_requested(
    cfg: Dict[str, Any],
    *,
    config_path: str,
    device: str,
    runner: Callable[[SimpleNamespace, Dict[str, Any]], Any] | None = None,
    logger_obj=None,
) -> str:
    assoc_cfg = cfg.get("ASSOCIATION_HEAD_TRAINING", {}) or {}
    enabled = _bool_cfg(assoc_cfg.get("ENABLED"), False)
    run_after_main = _bool_cfg(assoc_cfg.get("RUN_AFTER_MAIN"), False)
    if not enabled or not run_after_main:
        if logger_obj is not None:
            logger_obj.info("Association head Stage B disabled.")
        return "disabled"

    args = build_association_training_args(cfg, config_path=config_path, device=device)
    if runner is None:
        from tools.train_pairwise_association import train_pairwise_association

        runner = train_pairwise_association

    if logger_obj is not None:
        logger_obj.info(
            "Starting association head Stage B: train_pkl=%s val_pkl=%s output=%s freeze_backbone=%s",
            args.train_pkl,
            args.val_pkl,
            args.output,
            args.freeze_backbone,
        )
    runner(args, cfg)
    if logger_obj is not None:
        logger_obj.info("Association head Stage B complete.")
    return "ran"
