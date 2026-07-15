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
    except (OSError, RuntimeError, TypeError, ValueError):
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


def _resolve_checkpoint_path(value: Any, git_root: Path) -> str | None:
    if not value:
        return None
    try:
        checkpoint = Path(value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = git_root / checkpoint
        return str(checkpoint.resolve())
    except (OSError, RuntimeError, TypeError, ValueError):
        return None


def _preflight_association_checkpoint(
    checkpoint_path: str | None,
) -> tuple[dict, dict]:
    result = {
        "loadable": False,
        "head_key_count": 0,
        "validation_error": None,
    }
    if checkpoint_path is None:
        result["validation_error"] = "association checkpoint path is empty"
        return result, {}

    path = Path(checkpoint_path)
    if not path.exists():
        result["validation_error"] = "association checkpoint does not exist"
        return result, {}
    if not path.is_file():
        result["validation_error"] = "association checkpoint path is not a file"
        return result, {}

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result["loadable"] = True
    except Exception as exc:
        result["validation_error"] = (
            "failed to load association checkpoint: "
            f"{type(exc).__name__}: {exc}"
        )
        return result, {}

    if not isinstance(checkpoint, Mapping):
        result["validation_error"] = "association checkpoint payload is not a mapping"
        return result, {}

    state = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state, Mapping):
        result["validation_error"] = "association checkpoint state is not a mapping"
    else:
        result["head_key_count"] = sum(
            1
            for key in state
            if str(key).startswith("association_heads.")
        )
        if result["head_key_count"] == 0:
            result["validation_error"] = (
                "association checkpoint has no association_heads.* keys"
            )

    try:
        settings = checkpoint.get("settings", {}) or {}
        checkpoint_settings = (
            dict(settings) if isinstance(settings, Mapping) else {}
        )
    except Exception:
        checkpoint_settings = {}
    return result, checkpoint_settings


def _normalize_string_mapping(values: Any) -> dict:
    if not isinstance(values, Mapping):
        return {}
    return {
        str(key): str(value)
        for key, value in sorted(values.items(), key=lambda item: str(item[0]))
    }


def _lookup_mapping_value(values: Mapping, key: Any, default: Any) -> Any:
    if key in values:
        return values[key]
    text_key = str(key)
    if text_key in values:
        return values[text_key]
    return default


def _normalize_active_class_states(values: Any) -> dict:
    if not isinstance(values, Mapping):
        return {}

    missing = object()
    effective_class_ids = set()
    for key in values:
        try:
            effective_class_ids.add(int(key))
        except (TypeError, ValueError):
            effective_class_ids.add(str(key))

    normalized = {}
    for class_id in sorted(effective_class_ids, key=str):
        states = _lookup_mapping_value(values, class_id, missing)
        if states is missing:
            continue
        if states is None:
            state_values = []
        elif isinstance(states, str):
            state_values = [states]
        else:
            state_values = states
        normalized[str(class_id)] = sorted(
            {str(state).strip().lower() for state in state_values}
        )
    return normalized


def _parse_hidden_dims(value: Any) -> list:
    if isinstance(value, (list, tuple)):
        return [int(item) for item in value]
    if value is None:
        return [256, 128]
    return [int(item) for item in str(value).split(",") if str(item).strip()]


def _effective_hidden_dims(head_cfg: Mapping, checkpoint_settings: Mapping) -> tuple:
    if "HIDDEN_DIMS" in head_cfg:
        value = head_cfg["HIDDEN_DIMS"]
        source = "config"
    elif "hidden_dims" in checkpoint_settings:
        value = checkpoint_settings["hidden_dims"]
        source = "checkpoint"
    else:
        value = [256, 128]
        source = "default"
    try:
        return _parse_hidden_dims(value), source
    except Exception:
        return [256, 128], "default"


def _effective_dropout(head_cfg: Mapping, checkpoint_settings: Mapping) -> tuple:
    if "DROPOUT" in head_cfg:
        value = head_cfg["DROPOUT"]
        source = "config"
    elif "dropout" in checkpoint_settings:
        value = checkpoint_settings["dropout"]
        source = "checkpoint"
    else:
        value = 0.1
        source = "default"
    try:
        return float(value), source
    except Exception:
        return 0.1, "default"


def build_association_runtime_manifest(
    *, cfg: dict, config_path: str, process_count: int, git_root: str | Path
) -> dict:
    head_cfg = cfg.get("MAMBA_ASSOCIATION_HEAD", {}) or {}
    mamba_cfg = cfg.get("MAMBA", {}) or {}
    bev_cfg = ((cfg.get("MATCHING", {}) or {}).get("BEV", {}) or {})

    normalized_config_path = _absolute_path(config_path)
    normalized_git_root = Path(git_root).expanduser().resolve()
    checkpoint_path = _resolve_checkpoint_path(
        head_cfg.get("CHECKPOINT_PATH"), normalized_git_root
    )

    git_commit = _git_output(normalized_git_root, "rev-parse", "HEAD")
    git_status = _git_output(
        normalized_git_root, "status", "--porcelain", "--untracked-files=normal"
    )

    embed_dim = int(mamba_cfg.get("EMBED_DIM", 32))
    num_classes = int(head_cfg.get("NUM_CLASSES", mamba_cfg.get("NUM_CLASSES", 7)))
    checkpoint_preflight, checkpoint_settings = (
        _preflight_association_checkpoint(checkpoint_path)
    )
    hidden_dims, hidden_dims_source = _effective_hidden_dims(
        head_cfg, checkpoint_settings
    )
    dropout, dropout_source = _effective_dropout(head_cfg, checkpoint_settings)

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
        "association_checkpoint_loadable": checkpoint_preflight["loadable"],
        "association_checkpoint_head_key_count": checkpoint_preflight[
            "head_key_count"
        ],
        "association_checkpoint_validation_error": checkpoint_preflight[
            "validation_error"
        ],
        "association_head_architecture": {
            "input_dim": embed_dim * 4 + 7,
            "hidden_dims": hidden_dims,
            "hidden_dims_source": hidden_dims_source,
            "dropout": dropout,
            "dropout_source": dropout_source,
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
