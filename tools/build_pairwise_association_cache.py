#!/usr/bin/env python3
"""Build offline pairwise association samples from detection tracklet cache.

This is stage-1 infrastructure for class-conditioned association heads. It
does not affect training or inference unless its output is explicitly used.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.pairwise_association_cache import build_pairwise_association_samples


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_default_history_len(train_cfg: Dict[str, Any], fallback: int) -> int:
    return int((train_cfg.get("MODEL", {}) or {}).get("HISTORY_LEN", fallback))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pairwise association cache.")
    parser.add_argument("--input", required=True, help="Detection/fusion tracklet pkl")
    parser.add_argument("--output", required=True, help="Output pairwise samples pkl")
    parser.add_argument("--summary-output", default=None, help="Optional summary json")
    parser.add_argument("--train-config", default="config/train_nuscenes.yaml")
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument("--future-step", type=int, default=1)
    parser.add_argument("--hard-negative-distance", type=float, default=4.0)
    parser.add_argument("--max-hard-negatives", type=int, default=4)
    parser.add_argument("--max-easy-negatives", type=int, default=2)
    parser.add_argument("--allow-current-miss", action="store_true")
    parser.add_argument("--allow-future-miss", action="store_true")
    args = parser.parse_args()

    train_cfg = _load_yaml(args.train_config)
    history_len = int(args.history_len or _resolve_default_history_len(train_cfg, 8))

    with open(args.input, "rb") as f:
        tracklets = pickle.load(f)
    if not isinstance(tracklets, list):
        raise TypeError(f"Expected list tracklets in {args.input}, got {type(tracklets).__name__}")

    samples, summary = build_pairwise_association_samples(
        tracklets,
        history_len=history_len,
        future_step=int(args.future_step),
        hard_negative_distance=float(args.hard_negative_distance),
        max_hard_negatives=int(args.max_hard_negatives),
        max_easy_negatives=int(args.max_easy_negatives),
        require_current_match=not bool(args.allow_current_miss),
        require_future_match=not bool(args.allow_future_miss),
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)

    summary_payload = {
        "input": args.input,
        "output": str(output_path),
        "settings": {
            "history_len": history_len,
            "future_step": int(args.future_step),
            "hard_negative_distance": float(args.hard_negative_distance),
            "max_hard_negatives": int(args.max_hard_negatives),
            "max_easy_negatives": int(args.max_easy_negatives),
            "require_current_match": not bool(args.allow_current_miss),
            "require_future_match": not bool(args.allow_future_miss),
        },
        **summary,
    }

    summary_text = json.dumps(summary_payload, indent=2, ensure_ascii=False)
    if args.summary_output:
        summary_path = Path(args.summary_output)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(summary_text + "\n", encoding="utf-8")
    print(summary_text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
