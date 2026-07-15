from __future__ import annotations

import hashlib
import json
import math
import subprocess
from numbers import Integral
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


def _merge_validation_errors(*errors: str | None) -> str | None:
    messages = [str(error) for error in errors if error]
    return "; ".join(messages) if messages else None


def _parse_hidden_dims(value: Any) -> tuple[int, ...]:
    if value is None:
        raw_dims = [256, 128]
    elif isinstance(value, (list, tuple)):
        raw_dims = value
    elif isinstance(value, Integral) and not isinstance(value, bool):
        raw_dims = [value]
    elif isinstance(value, str):
        raw_dims = [item.strip() for item in value.split(",") if item.strip()]
    else:
        raise ValueError("hidden_dims must be a sequence or comma-separated string")

    if not raw_dims:
        raise ValueError("hidden_dims must contain at least one dimension")
    try:
        hidden_dims = tuple(int(item) for item in raw_dims)
    except (TypeError, ValueError) as exc:
        raise ValueError("hidden_dims must contain only integers") from exc
    if any(isinstance(item, bool) for item in raw_dims) or any(
        dimension <= 0 for dimension in hidden_dims
    ):
        raise ValueError("hidden_dims must contain only positive integers")
    return hidden_dims


def resolve_association_head_parameters(
    head_cfg: Mapping, checkpoint_settings: Mapping
) -> tuple[dict, str | None]:
    """Resolve loader architecture settings without silently accepting bad values."""
    if "HIDDEN_DIMS" in head_cfg:
        hidden_value = head_cfg["HIDDEN_DIMS"]
        hidden_source = "config"
    elif "hidden_dims" in checkpoint_settings:
        hidden_value = checkpoint_settings["hidden_dims"]
        hidden_source = "checkpoint"
    else:
        hidden_value = [256, 128]
        hidden_source = "default"

    if "DROPOUT" in head_cfg:
        dropout_value = head_cfg["DROPOUT"]
        dropout_source = "config"
    elif "dropout" in checkpoint_settings:
        dropout_value = checkpoint_settings["dropout"]
        dropout_source = "checkpoint"
    else:
        dropout_value = 0.1
        dropout_source = "default"

    errors = []
    try:
        hidden_dims = _parse_hidden_dims(hidden_value)
    except (TypeError, ValueError) as exc:
        hidden_dims = None
        errors.append(f"invalid {hidden_source} hidden_dims: {exc}")

    try:
        dropout = float(dropout_value)
        if not math.isfinite(dropout) or not 0.0 <= dropout <= 1.0:
            raise ValueError("dropout must be finite and between 0 and 1")
    except (TypeError, ValueError) as exc:
        dropout = None
        errors.append(f"invalid {dropout_source} dropout: {exc}")

    return {
        "hidden_dims": hidden_dims,
        "hidden_dims_source": hidden_source,
        "dropout": dropout,
        "dropout_source": dropout_source,
    }, _merge_validation_errors(*errors)


def preflight_association_checkpoint(
    checkpoint_path: str | None,
    head_cfg: Mapping | None = None,
    category_map: Any = None,
) -> tuple[dict, dict]:
    result = {
        "loadable": False,
        "head_key_count": 0,
        "validation_error": None,
    }

    def finalize(
        checkpoint_settings: Mapping | None = None,
        settings_error: str | None = None,
    ) -> tuple[dict, dict]:
        normalized_settings = dict(checkpoint_settings or {})
        _, architecture_error = resolve_association_head_parameters(
            head_cfg or {}, normalized_settings
        )
        _, category_map_error = _canonical_class_id_to_name(category_map)
        result["validation_error"] = _merge_validation_errors(
            result["validation_error"],
            settings_error,
            architecture_error,
            category_map_error,
        )
        return result, normalized_settings

    if checkpoint_path is None:
        result["validation_error"] = "association checkpoint path is empty"
        return finalize()

    path = Path(checkpoint_path)
    if not path.exists():
        result["validation_error"] = "association checkpoint does not exist"
        return finalize()
    if not path.is_file():
        result["validation_error"] = "association checkpoint path is not a file"
        return finalize()

    try:
        import torch

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        result["loadable"] = True
    except Exception as exc:
        result["validation_error"] = (
            "failed to load association checkpoint: "
            f"{type(exc).__name__}: {exc}"
        )
        return finalize()

    if not isinstance(checkpoint, Mapping):
        result["validation_error"] = "association checkpoint payload is not a mapping"
        return finalize()

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

    settings = checkpoint.get("settings", {})
    settings_error = None
    if settings is None:
        checkpoint_settings = {}
    elif isinstance(settings, Mapping):
        try:
            checkpoint_settings = dict(settings)
        except Exception as exc:
            checkpoint_settings = {}
            settings_error = (
                "association checkpoint settings could not be copied: "
                f"{type(exc).__name__}: {exc}"
            )
    else:
        checkpoint_settings = {}
        settings_error = "association checkpoint settings must be a mapping"

    return finalize(checkpoint_settings, settings_error)


def _canonical_class_id_to_name(values: Any) -> tuple[dict, str | None]:
    if values is None:
        return {}, None
    if not isinstance(values, Mapping):
        return {}, "CATEGORY_MAP_TO_NUMBER must be a mapping"

    reversed_mapping = {}
    normalized_names = set()
    for raw_name, raw_class_id in values.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            return {}, "CATEGORY_MAP_TO_NUMBER requires a non-empty class name"
        class_name = raw_name.strip().lower()
        if class_name in normalized_names:
            return {}, f"CATEGORY_MAP_TO_NUMBER has duplicate class name {class_name!r}"
        if not isinstance(raw_class_id, Integral) or isinstance(raw_class_id, bool):
            return {}, "CATEGORY_MAP_TO_NUMBER requires an integer class id"
        class_id = int(raw_class_id)
        class_id_key = str(class_id)
        if class_id_key in reversed_mapping:
            return {}, f"CATEGORY_MAP_TO_NUMBER has duplicate class id {class_id}"
        normalized_names.add(class_name)
        reversed_mapping[class_id_key] = class_name
    return dict(sorted(reversed_mapping.items(), key=lambda item: int(item[0]))), None


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
        preflight_association_checkpoint(
            checkpoint_path,
            head_cfg,
            cfg.get("CATEGORY_MAP_TO_NUMBER", {}),
        )
    )
    architecture_parameters, _ = resolve_association_head_parameters(
        head_cfg, checkpoint_settings
    )
    class_id_to_name, _ = _canonical_class_id_to_name(
        cfg.get("CATEGORY_MAP_TO_NUMBER", {})
    )

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
            "hidden_dims": (
                list(architecture_parameters["hidden_dims"])
                if architecture_parameters["hidden_dims"] is not None
                else None
            ),
            "hidden_dims_source": architecture_parameters["hidden_dims_source"],
            "dropout": architecture_parameters["dropout"],
            "dropout_source": architecture_parameters["dropout_source"],
            "num_classes": num_classes,
        },
        "class_id_to_name": class_id_to_name,
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
