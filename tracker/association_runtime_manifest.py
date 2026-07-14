from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping


AUDIT_SCHEMA_VERSION = 2


def sha256_file(path: str | Path) -> str | None:
    """Return a file's SHA-256 digest without propagating read failures."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as file_obj:
            for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except (OSError, TypeError, ValueError):
        return None


def _absolute_path(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _git_output(git_root: Path, *args: str) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(git_root),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def _normalize_string_mapping(values: Any) -> dict:
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    }


def _normalize_active_class_states(values: Any) -> dict:
    if not isinstance(values, Mapping):
        return {}

    normalized = {}
    for class_id, states in sorted(values.items(), key=lambda item: str(item[0])):
        if states is None:
            state_values = []
        elif isinstance(states, str):
            state_values = [states]
        else:
            state_values = states
        normalized[str(class_id)] = sorted(str(state) for state in state_values)
    return normalized


def _parse_hidden_dims(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None:
        return [256, 128]
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def build_association_runtime_manifest(
    *, cfg: dict, config_path: str, process_count: int, git_root: str | Path
) -> dict:
    head_cfg = cfg.get("MAMBA_ASSOCIATION_HEAD", {}) or {}
    mamba_cfg = cfg.get("MAMBA", {}) or {}
    bev_cfg = ((cfg.get("MATCHING", {}) or {}).get("BEV", {}) or {})

    normalized_config_path = _absolute_path(config_path)
    normalized_git_root = Path(git_root).expanduser().resolve()
    checkpoint_value = head_cfg.get("CHECKPOINT_PATH")
    checkpoint_path = None
    if checkpoint_value:
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = normalized_git_root / checkpoint
        checkpoint_path = str(checkpoint.resolve())

    git_commit = _git_output(normalized_git_root, "rev-parse", "HEAD")
    git_status = _git_output(
        normalized_git_root, "status", "--porcelain", "--untracked-files=normal"
    )

    embed_dim = int(mamba_cfg.get("EMBED_DIM", 32))
    num_classes = int(head_cfg.get("NUM_CLASSES", mamba_cfg.get("NUM_CLASSES", 7)))

    return {
        "audit_schema_version": AUDIT_SCHEMA_VERSION,
        "config_path": normalized_config_path,
        "config_sha256": sha256_file(normalized_config_path),
        "git_commit": git_commit,
        "git_dirty": None if git_status is None else bool(git_status),
        "association_checkpoint_path": checkpoint_path,
        "association_checkpoint_sha256": (
            sha256_file(checkpoint_path) if checkpoint_path is not None else None
        ),
        "association_head_architecture": {
            "input_dim": embed_dim * 4 + 7,
            "hidden_dims": _parse_hidden_dims(head_cfg.get("HIDDEN_DIMS")),
            "dropout": float(head_cfg.get("DROPOUT", 0.1)),
            "num_classes": num_classes,
        },
        "active_class_states": _normalize_active_class_states(
            head_cfg.get("ACTIVE_CLASS_STATES", {})
        ),
        "apply_mode": str(head_cfg.get("APPLY_MODE", "penalty")).strip().lower(),
        "cost_margin_eps": float(head_cfg.get("COST_MARGIN_EPS", 0.05)),
        "min_score": float(head_cfg.get("MIN_SCORE", 0.5)),
        "alpha": float(head_cfg.get("ALPHA", 0.05)),
        "max_delta": float(head_cfg.get("MAX_DELTA", 0.03)),
        "filter_mode": str(cfg.get("FILTER_MODE", "mamba")).strip().lower(),
        "tracker_compat_mode": str(
            cfg.get("TRACKER_COMPAT_MODE", "default")
        ).strip().lower(),
        "bev_matching_mode": bev_cfg.get("MATCHING_MODE"),
        "bev_cost_modes": _normalize_string_mapping(bev_cfg.get("COST_MODE", {})),
        "process_count": int(process_count),
    }


def _canonical_manifest(manifest: dict | None) -> str:
    return json.dumps(
        manifest,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def runtime_manifests_compatible(left: dict | None, right: dict | None) -> bool:
    try:
        return _canonical_manifest(left) == _canonical_manifest(right)
    except (TypeError, ValueError):
        return False
