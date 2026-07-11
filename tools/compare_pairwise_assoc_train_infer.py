#!/usr/bin/env python3
"""Compare pairwise association training cache with inference audit output."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    values = list(values)
    if not values:
        return {"min": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(arr)),
        "p10": float(np.quantile(arr, 0.10)),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "max": float(np.max(arr)),
    }


def _load_train_cache(path: Path) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        data = pickle.load(f)
    if not isinstance(data, list):
        raise TypeError(f"Expected list samples in {path}, got {type(data).__name__}")
    return data


def _class_key(class_id: int, class_name: str) -> str:
    return f"{int(class_id)}:{class_name}"


def _summarize_train(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    grouped = defaultdict(list)
    for sample in samples:
        class_id = int(sample.get("class_id", -1))
        class_name = str(sample.get("category", class_id))
        grouped[_class_key(class_id, class_name)].append(sample)

    out = {}
    for key, items in sorted(grouped.items()):
        labels = [int(item.get("label", 0)) for item in items]
        negatives = [item for item in items if int(item.get("label", 0)) == 0]
        hard_negatives = [item for item in negatives if str(item.get("negative_type", "")).lower() == "hard"]
        center_distances = [float(item.get("center_distance", 0.0)) for item in items]
        pos_center_distances = [
            float(item.get("center_distance", 0.0))
            for item in items
            if int(item.get("label", 0)) == 1
        ]
        hard_center_distances = [
            float(item.get("center_distance", 0.0))
            for item in hard_negatives
        ]
        count = max(len(items), 1)
        out[key] = {
            "sample_count": int(len(items)),
            "positive_count": int(sum(labels)),
            "negative_count": int(len(negatives)),
            "hard_negative_count": int(len(hard_negatives)),
            "positive_ratio": float(sum(labels) / count),
            "hard_negative_ratio": float(len(hard_negatives) / count),
            "center_distance_quantiles": _quantiles(center_distances),
            "positive_center_distance_quantiles": _quantiles(pos_center_distances),
            "hard_center_distance_quantiles": _quantiles(hard_center_distances),
        }
    return out


def _summarize_infer(audit: Dict[str, Any]) -> Dict[str, Any]:
    out = defaultdict(dict)
    for bucket in audit.get("buckets", []):
        class_id = int(bucket.get("class_id", -1))
        class_name = str(bucket.get("class_name", class_id))
        state_bucket = str(bucket.get("state_bucket", "unknown"))
        key = _class_key(class_id, class_name)
        out[key][state_bucket] = {
            "pair_count": int(bucket.get("pair_count", 0)),
            "active_pair_count": int(bucket.get("active_pair_count", 0)),
            "penalized_pair_count": int(bucket.get("penalized_pair_count", 0)),
            "active_ratio": float(bucket.get("active_ratio", 0.0)),
            "penalized_ratio": float(bucket.get("penalized_ratio", 0.0)),
            "avg_score": float(bucket.get("avg_score", 0.0)),
            "avg_delta": float(bucket.get("avg_delta", 0.0)),
            "score_quantiles": bucket.get("score_quantiles", {}),
            "delta_quantiles": bucket.get("delta_quantiles", {}),
            "sample_count": int(bucket.get("sample_count", 0)),
            "samples": bucket.get("samples", []),
        }
    return dict(out)


def _diagnose(train_summary: Dict[str, Any], infer_summary: Dict[str, Any]) -> List[str]:
    diagnoses = []
    matched = infer_summary.get("matched", {})
    unmatched = infer_summary.get("unmatched", {})
    pos_ratio = float(train_summary.get("positive_ratio", 0.0))
    hard_ratio = float(train_summary.get("hard_negative_ratio", 0.0))
    matched_score = float(matched.get("avg_score", 0.0))
    matched_penalty = float(matched.get("penalized_ratio", 0.0))
    unmatched_score = float(unmatched.get("avg_score", 0.0))

    if pos_ratio < 0.10:
        diagnoses.append("train_positive_ratio_too_low")
    if hard_ratio < 0.10:
        diagnoses.append("hard_negative_ratio_too_low")
    if matched and matched_score < 0.20 and matched_penalty > 0.60:
        diagnoses.append("likely_distribution_mismatch")
    if matched and unmatched and matched_score <= unmatched_score * 1.5:
        diagnoses.append("weak_matched_unmatched_separation")
    if matched and matched_score >= 0.50 and matched_penalty < 0.30:
        diagnoses.append("candidate_for_safe_activation")
    if not diagnoses:
        diagnoses.append("needs_manual_review")
    return diagnoses


def build_summary(train_cache: Path, infer_audit: Path) -> Dict[str, Any]:
    train = _summarize_train(_load_train_cache(train_cache))
    audit = json.loads(infer_audit.read_text(encoding="utf-8"))
    infer = _summarize_infer(audit)
    keys = sorted(set(train.keys()) | set(infer.keys()))
    classes = {}
    for key in keys:
        train_summary = train.get(key, {})
        infer_summary = infer.get(key, {})
        classes[key] = {
            "train": train_summary,
            "infer": infer_summary,
            "diagnosis": _diagnose(train_summary, infer_summary),
        }
    return {
        "schema_version": 1,
        "train_cache": str(train_cache),
        "infer_audit": str(infer_audit),
        "classes": classes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Pairwise association train cache pkl.")
    parser.add_argument("--infer-audit", required=True, help="Association head inference audit JSON.")
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    args = parser.parse_args()

    summary = build_summary(Path(args.train_cache), Path(args.infer_audit))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote pairwise association train/infer comparison to {output}")


if __name__ == "__main__":
    main()
