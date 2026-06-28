from __future__ import annotations

from typing import Iterable, Mapping, MutableMapping

_BACKBONE_PREFIXES = ("fallback_gru.", "mamba_layers.")


def _clone_like(value):
    if hasattr(value, "clone"):
        return value.clone()
    if hasattr(value, "copy"):
        return value.copy()
    raise TypeError("checkpoint tensor must support clone() or copy()")


def adapt_num_class_state_dict(
    state_dict: MutableMapping,
    model_state_dict: Mapping,
    *,
    param_names: Iterable[str] = ("raw_q_siz.weight", "raw_r_siz.weight"),
):
    """
    Reconcile class-count-dependent tensors between an older checkpoint and the
    current model definition.

    Typical case in this repo:
      - old checkpoint trained with NUM_CLASSES=10
      - current nuScenes runtime uses NUM_CLASSES=7

    Strategy:
      - keep the current model tensor as the target shape
      - copy the overlapping rows from the checkpoint
      - preserve current-model initialization for any extra rows
    """
    adapted = dict(state_dict)
    adapted_keys = []

    for name in param_names:
        if name not in adapted or name not in model_state_dict:
            continue

        ckpt_value = adapted[name]
        model_value = model_state_dict[name]
        ckpt_shape = tuple(getattr(ckpt_value, "shape", ()))
        model_shape = tuple(getattr(model_value, "shape", ()))

        if ckpt_shape == model_shape or len(ckpt_shape) == 0 or len(model_shape) == 0:
            continue
        if len(ckpt_shape) != len(model_shape):
            continue
        if ckpt_shape[1:] != model_shape[1:]:
            continue

        merged = _clone_like(model_value)
        rows = min(int(ckpt_shape[0]), int(model_shape[0]))
        merged[:rows] = ckpt_value[:rows]
        adapted[name] = merged
        adapted_keys.append(name)

    return adapted, adapted_keys


def filter_heads_only_state_dict(
    state_dict: Mapping,
    model_state_dict: Mapping,
    *,
    backbone_prefixes: Iterable[str] = _BACKBONE_PREFIXES,
):
    """
    Select checkpoint tensors that are safe to migrate across backbone changes.

    This is intended for GRU-fallback -> real Mamba training transitions:
    recurrent backbone tensors are skipped, while compatible projection,
    normalization, size/orientation, embedding, and closure-head tensors are
    reused when their shapes exactly match the current model.
    """
    filtered = {}
    skipped = {
        "backbone": [],
        "missing": [],
        "shape_mismatch": [],
    }

    for name, value in state_dict.items():
        if any(str(name).startswith(prefix) for prefix in backbone_prefixes):
            skipped["backbone"].append(name)
            continue
        if name not in model_state_dict:
            skipped["missing"].append(name)
            continue
        if tuple(getattr(value, "shape", ())) != tuple(getattr(model_state_dict[name], "shape", ())):
            skipped["shape_mismatch"].append(name)
            continue
        filtered[name] = value

    return filtered, skipped
