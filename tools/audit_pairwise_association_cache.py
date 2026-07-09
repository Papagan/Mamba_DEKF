#!/usr/bin/env python3
"""Audit pairwise association cache volume and class balance."""

from __future__ import annotations

import argparse
import json
import pickle
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
    args = parser.parse_args()

    with open(args.input, "rb") as f:
        samples = pickle.load(f)
    if not isinstance(samples, list):
        raise TypeError(f"Expected list samples in {args.input}, got {type(samples).__name__}")

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


if __name__ == "__main__":
    raise SystemExit(main())
