#!/usr/bin/env python3
"""Merge per-class closure head weights into a base Mamba checkpoint."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Mapping


def _clone_value(value):
    if hasattr(value, "clone"):
        return value.clone()
    return copy.deepcopy(value)


def is_class_head_key_for_class(key: str, class_id: int) -> bool:
    parts = str(key).split(".")
    if len(parts) < 5:
        return False
    if parts[0] != "head_bank" or parts[1] != "family_heads":
        return False
    return parts[3] == str(int(class_id))


def merge_class_head_state_dicts(
    base_state: Mapping,
    class_states: Mapping[int, Mapping],
) -> tuple[dict, dict]:
    merged = {key: _clone_value(value) for key, value in base_state.items()}
    copied = {}

    for class_id, state in sorted(class_states.items()):
        copied_keys = []
        for key in sorted(state):
            if not is_class_head_key_for_class(key, int(class_id)):
                continue
            if key not in merged:
                continue
            base_shape = tuple(getattr(merged[key], "shape", ()))
            source_shape = tuple(getattr(state[key], "shape", ()))
            if base_shape != source_shape:
                continue
            merged[key] = _clone_value(state[key])
            copied_keys.append(key)
        copied[int(class_id)] = copied_keys

    return merged, copied


def _parse_class_head(value: str) -> tuple[int, str]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--class-head must use class_id=checkpoint.pt")
    class_id_raw, path = value.split("=", 1)
    try:
        class_id = int(class_id_raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid class id: {class_id_raw}") from exc
    if class_id < 0:
        raise argparse.ArgumentTypeError("class id must be non-negative")
    return class_id, path


def _load_checkpoint(path: str, device: str):
    import torch

    return torch.load(path, map_location=device)


def _checkpoint_state_dict(ckpt):
    return ckpt.get("model_state_dict", ckpt)


def merge_checkpoints(base_path: str, class_head_specs: list[tuple[int, str]], output_path: str, device: str = "cpu"):
    import torch

    base_ckpt = _load_checkpoint(base_path, device)
    base_state = _checkpoint_state_dict(base_ckpt)
    class_states = {}
    sources = {}
    for class_id, ckpt_path in class_head_specs:
        class_ckpt = _load_checkpoint(ckpt_path, device)
        class_states[class_id] = _checkpoint_state_dict(class_ckpt)
        sources[class_id] = str(Path(ckpt_path))

    merged_state, copied = merge_class_head_state_dicts(base_state, class_states)
    if not any(copied.values()):
        raise RuntimeError("no class head keys were copied; check class ids and checkpoint compatibility")

    output_ckpt = (
        dict(base_ckpt)
        if isinstance(base_ckpt, dict) and "model_state_dict" in base_ckpt
        else {"model_state_dict": base_ckpt}
    )
    output_ckpt["model_state_dict"] = merged_state
    runtime_contract = dict(output_ckpt.get("runtime_contract", {}) or {})
    runtime_contract["merged_class_heads"] = {
        str(class_id): {
            "source": sources[class_id],
            "copied_keys": copied.get(class_id, []),
        }
        for class_id in sorted(sources)
    }
    output_ckpt["runtime_contract"] = runtime_contract

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output_ckpt, output)
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge per-class closure heads into a base checkpoint.")
    parser.add_argument("--base", required=True, help="Base checkpoint path, usually checkpoints/mamba_dekf/best.pt")
    parser.add_argument(
        "--class-head",
        action="append",
        type=_parse_class_head,
        required=True,
        help="Class-specific checkpoint as class_id=path. Can be passed multiple times.",
    )
    parser.add_argument("--output", required=True, help="Output merged checkpoint path")
    parser.add_argument("--device", default="cpu", help="torch map_location, default: cpu")
    args = parser.parse_args()

    copied = merge_checkpoints(args.base, args.class_head, args.output, device=args.device)
    total = sum(len(keys) for keys in copied.values())
    print(f"Merged {total} class-head tensors -> {args.output}")
    for class_id, keys in sorted(copied.items()):
        print(f"  class {class_id}: {len(keys)} tensors")


if __name__ == "__main__":
    main()
