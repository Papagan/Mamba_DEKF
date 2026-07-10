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


DEFAULT_HARD_NEGATIVE_DISTANCE_BY_CLASS = {
    "car": 6.0,
    "pedestrian": 4.0,
    "bicycle": 5.0,
    "motorcycle": 5.0,
    "bus": 12.0,
    "trailer": 10.0,
    "truck": 12.0,
}


def _load_yaml(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_default_history_len(train_cfg: Dict[str, Any], fallback: int) -> int:
    return int((train_cfg.get("MODEL", {}) or {}).get("HISTORY_LEN", fallback))


def _resolve_default_history_source(train_cfg: Dict[str, Any], fallback: str) -> str:
    return str((train_cfg.get("DATA", {}) or {}).get("HISTORY_SOURCE", fallback)).strip().lower()


def _parse_class_float_map(raw: str | None) -> Dict[str, float]:
    if raw is None or str(raw).strip() == "":
        return {}
    text = str(raw).strip()
    if text.startswith("{"):
        data = json.loads(text)
        return {str(key): float(value) for key, value in data.items()}
    out: Dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        key, value = item.split("=", 1)
        out[str(key).strip()] = float(value)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Build pairwise association cache.")
    parser.add_argument("--input", required=True, help="Detection/fusion tracklet pkl")
    parser.add_argument("--output", required=True, help="Output pairwise samples pkl")
    parser.add_argument("--summary-output", default=None, help="Optional summary json")
    parser.add_argument("--train-config", default="config/train_nuscenes.yaml")
    parser.add_argument("--history-len", type=int, default=0)
    parser.add_argument(
        "--history-source",
        default=None,
        choices=["det", "fusion"],
        help="Anchor history source. Defaults to DATA.HISTORY_SOURCE from --train-config.",
    )
    parser.add_argument(
        "--pair-geometry-source",
        default="predicted_track_candidate",
        choices=["predicted_track_candidate", "track_candidate", "future_candidate"],
        help=(
            "Geometry used for pair features. 'predicted_track_candidate' mirrors "
            "the pre-association predicted track state; 'future_candidate' keeps "
            "the old detection-detection behavior."
        ),
    )
    parser.add_argument("--future-step", type=int, default=1)
    parser.add_argument("--hard-negative-distance", type=float, default=4.0)
    parser.add_argument(
        "--hard-negative-distance-by-class",
        default=None,
        help=(
            "Per-class hard-negative radius, e.g. "
            "'car=6,pedestrian=4,bicycle=5,motorcycle=5,bus=12,trailer=10,truck=12'. "
            "Defaults to class-aware radii tuned from pairwise audit."
        ),
    )
    parser.add_argument("--max-hard-negatives", type=int, default=4)
    parser.add_argument("--max-easy-negatives", type=int, default=2)
    parser.add_argument(
        "--max-pairs-per-class",
        default=None,
        help="Optional class caps, e.g. 'car=60000,pedestrian=50000'. Defaults to no cap.",
    )
    parser.add_argument("--allow-current-miss", action="store_true")
    parser.add_argument("--allow-future-miss", action="store_true")
    args = parser.parse_args()

    train_cfg = _load_yaml(args.train_config)
    history_len = int(args.history_len or _resolve_default_history_len(train_cfg, 8))
    history_source = args.history_source or _resolve_default_history_source(train_cfg, "det")

    with open(args.input, "rb") as f:
        tracklets = pickle.load(f)
    if not isinstance(tracklets, list):
        raise TypeError(f"Expected list tracklets in {args.input}, got {type(tracklets).__name__}")

    class_hard_dist = dict(DEFAULT_HARD_NEGATIVE_DISTANCE_BY_CLASS)
    class_hard_dist.update(_parse_class_float_map(args.hard_negative_distance_by_class))
    max_pairs_per_class = {
        key: int(value)
        for key, value in _parse_class_float_map(args.max_pairs_per_class).items()
    }

    samples, summary = build_pairwise_association_samples(
        tracklets,
        history_len=history_len,
        history_source=history_source,
        pair_geometry_source=str(args.pair_geometry_source),
        future_step=int(args.future_step),
        hard_negative_distance=float(args.hard_negative_distance),
        hard_negative_distance_by_class=class_hard_dist,
        max_hard_negatives=int(args.max_hard_negatives),
        max_easy_negatives=int(args.max_easy_negatives),
        max_pairs_per_class=max_pairs_per_class,
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
            "history_source": history_source,
            "pair_geometry_source": str(args.pair_geometry_source),
            "future_step": int(args.future_step),
            "hard_negative_distance": float(args.hard_negative_distance),
            "hard_negative_distance_by_class": class_hard_dist,
            "max_hard_negatives": int(args.max_hard_negatives),
            "max_easy_negatives": int(args.max_easy_negatives),
            "max_pairs_per_class": max_pairs_per_class,
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
