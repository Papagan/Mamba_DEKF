#!/usr/bin/env python3
"""Audit pairwise association cache volume and class balance."""

from __future__ import annotations

import argparse
import json
import pickle
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from training.pairwise_association_cache import summarize_pairwise_association_samples


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit pairwise association cache.")
    parser.add_argument("--input", required=True, help="Pairwise association .pkl file")
    parser.add_argument("--output", default=None, help="Optional summary .json path")
    parser.add_argument("--train-config", help="Also fail on stale schema or missing required classes")
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list samples in {args.input}, got {type(samples).__name__}")
    if args.train_config:
        with open(args.train_config, encoding="utf-8") as stream:
            cfg = yaml.safe_load(stream)
        validate_training_samples(samples, cfg)

    summary = {
        "input": args.input,
        **summarize_pairwise_association_samples(samples),
    }
    text = json.dumps(summary, indent=2, ensure_ascii=False)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


def validate_training_samples(samples, cfg):
    """Fail before motion training if its subsequent Stage B cannot use the cache."""
    policy = cfg.get("ASSOCIATION_HEAD_TRAINING", {})
    if not samples:
        raise ValueError("Empty association cache")
    schema = policy.get("REQUIRE_FEATURE_SCHEMA_VERSION")
    if schema is not None and any(s.get("feature_schema_version") != schema for s in samples):
        raise ValueError(f"Association feature schema must be {schema}; rebuild this cache")
    if policy.get("REQUIRE_ALL_CLASSES", False):
        required = set(policy.get("CLASS_IDS", range(cfg.get("MODEL", {}).get("NUM_CLASSES", 7))))
        present = {int(s.get("class_id", -1)) for s in samples}
        missing = sorted(required - present)
        if missing:
            raise ValueError(f"Association cache lacks required classes: {missing}")


if __name__ == "__main__":
    raise SystemExit(main())
