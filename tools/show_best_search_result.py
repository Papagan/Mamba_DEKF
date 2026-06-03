#!/usr/bin/env python3
"""
Show the best available ByteTrack search result from a run directory or JSON file.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, List, Tuple


def is_finite_number(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def resolve_input(path: str) -> Tuple[str, str]:
    abspath = os.path.abspath(path)
    if os.path.isdir(abspath):
        ranked = os.path.join(abspath, "results_ranked.json")
        partial = os.path.join(abspath, "results_partial.json")
        if os.path.exists(ranked):
            return ranked, "ranked"
        if os.path.exists(partial):
            return partial, "partial"
        raise FileNotFoundError(
            f"No results_ranked.json or results_partial.json found under: {abspath}"
        )

    if not os.path.exists(abspath):
        raise FileNotFoundError(abspath)

    name = os.path.basename(abspath)
    if name == "results_ranked.json":
        return abspath, "ranked"
    if name == "results_partial.json":
        return abspath, "partial"
    raise ValueError(
        "Input must be a search run directory, results_ranked.json, or results_partial.json"
    )


def load_best(path: str, mode: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if mode == "ranked":
        best = data.get("best", None)
        if not isinstance(best, dict):
            raise ValueError(f"`best` not found or invalid in: {path}")
        return best

    if not isinstance(data, list) or not data:
        raise ValueError(f"results_partial.json is empty or invalid: {path}")

    candidates: List[Dict[str, Any]] = [x for x in data if isinstance(x, dict)]
    if not candidates:
        raise ValueError(f"No valid trial entries in: {path}")

    return max(
        candidates,
        key=lambda x: x.get("objective", float("-inf"))
        if is_finite_number(x.get("objective", None))
        else float("-inf"),
    )


def print_summary(best: Dict[str, Any], source_path: str, source_mode: str) -> None:
    print(f"source={source_path}")
    print(f"source_mode={source_mode}")
    print(f"trial={best.get('trial')}")
    print(f"stage={best.get('stage')}")
    print(f"status={best.get('status')}")
    print(f"objective={best.get('objective')}")
    print(f"objective_raw_mean={best.get('objective_raw_mean')}")
    print(f"objective_penalized_mean={best.get('objective_penalized_mean')}")
    print(f"objective_penalized_std={best.get('objective_penalized_std')}")
    print(f"hard_ok={best.get('hard_ok')}")

    overall = best.get("overall", {})
    if isinstance(overall, dict) and overall:
        print(f"overall.idf1={overall.get('idf1')}")
        print(f"overall.mota={overall.get('mota')}")
        print(f"overall.recall={overall.get('recall')}")
        print(f"overall.precision={overall.get('precision')}")

    print(f"result_json={best.get('result_json')}")
    print("params=" + json.dumps(best.get("params", {}), ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Show best available ByteTrack search result from a run directory or result JSON."
    )
    parser.add_argument(
        "input",
        help="Search run directory, results_ranked.json, or results_partial.json",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print full best trial as JSON.",
    )
    args = parser.parse_args()

    source_path, source_mode = resolve_input(args.input)
    best = load_best(source_path, source_mode)

    if args.json:
        print(json.dumps(best, indent=2, ensure_ascii=False))
        return

    print_summary(best, source_path, source_mode)


if __name__ == "__main__":
    main()
