#!/usr/bin/env python3
"""Compare pairwise association training cache with inference audit output."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np


DEFAULT_MIN_KNOWN_CHANGED = 100
MAX_HARM_PRECISION = 0.20


def _quantiles(values: Iterable[float]) -> Dict[str, float]:
    values = [float(value) for value in values if math.isfinite(float(value))]
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


def _is_hard_negative_type(value: Any) -> bool:
    return str(value).strip().lower() in {"hard", "inference_margin", "inference_topk"}


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
        hard_negatives = [item for item in negatives if _is_hard_negative_type(item.get("negative_type", ""))]
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


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _summarize_track_state_scores(audit: Dict[str, Any]) -> Dict[str, Any]:
    out = defaultdict(dict)
    pair_buckets = audit.get("pair_buckets", audit.get("buckets", []))
    for bucket in pair_buckets or []:
        class_id = int(bucket.get("class_id", -1))
        class_name = str(bucket.get("class_name", class_id))
        state_bucket = str(bucket.get("state_bucket", "unknown"))
        key = _class_key(class_id, class_name)
        out[key][state_bucket] = {
            "pair_count": int(bucket.get("pair_count", 0)),
            "active_pair_count": int(bucket.get("active_pair_count", 0)),
            "penalized_pair_count": int(bucket.get("penalized_pair_count", 0)),
            "active_ratio": _finite_float(bucket.get("active_ratio", 0.0)),
            "penalized_ratio": _finite_float(bucket.get("penalized_ratio", 0.0)),
            "avg_score": _finite_float(bucket.get("avg_score", 0.0)),
            "avg_delta": _finite_float(bucket.get("avg_delta", 0.0)),
            "avg_cost_before": _finite_float(
                bucket.get("avg_cost_before", 0.0)
            ),
            "avg_cost_after": _finite_float(bucket.get("avg_cost_after", 0.0)),
            "score_quantiles": bucket.get("score_quantiles", {}),
            "delta_quantiles": bucket.get("delta_quantiles", {}),
            "cost_before_quantiles": bucket.get("cost_before_quantiles", {}),
            "cost_after_quantiles": bucket.get("cost_after_quantiles", {}),
            "sample_count": int(bucket.get("sample_count", 0)),
            "samples": bucket.get("samples", []),
        }
    return dict(out)


def _summarize_decisions(audit: Dict[str, Any]) -> Dict[str, Any]:
    counters = defaultdict(
        lambda: {
            "decision_bucket_count": 0,
            "total_decisions": 0,
            "known_changed_count": 0,
            "improved_count": 0,
            "damaged_count": 0,
        }
    )
    for bucket in audit.get("decision_buckets", []) or []:
        class_id = int(bucket.get("class_id", -1))
        class_name = str(bucket.get("class_name", class_id))
        summary = counters[_class_key(class_id, class_name)]
        summary["decision_bucket_count"] += 1
        for field in (
            "total_decisions",
            "known_changed_count",
            "improved_count",
            "damaged_count",
        ):
            summary[field] += max(int(bucket.get(field, 0)), 0)

    out = {}
    for key, summary in counters.items():
        known_changed = summary["known_changed_count"]
        summary["net_improvement"] = (
            summary["improved_count"] - summary["damaged_count"]
        )
        summary["harm_precision"] = float(
            summary["damaged_count"] / max(known_changed, 1)
        )
        out[key] = summary
    return out


def _audit_errors_status(audit: Dict[str, Any]) -> Dict[str, Any]:
    errors = audit.get("audit_errors")
    if not isinstance(errors, dict):
        return {"valid": False, "count": None}

    count = errors.get("count")
    if type(count) is not int or count < 0:
        return {"valid": False, "count": None}

    if "audit_error_count" in audit:
        top_level_count = audit["audit_error_count"]
        if (
            type(top_level_count) is not int
            or top_level_count < 0
            or top_level_count != count
        ):
            return {"valid": False, "count": None}
    return {"valid": True, "count": count}


def _decision_evidence_status(audit: Dict[str, Any]) -> Dict[str, Any]:
    schema_version = int(audit.get("schema_version", 1))
    decision_buckets = audit.get("decision_buckets", [])
    has_decision_buckets = isinstance(decision_buckets, list) and bool(
        decision_buckets
    )
    available = bool(audit.get("decision_evidence_available", False))
    complete = bool(audit.get("decision_evidence_complete", False))
    manifest_compatible = audit.get("manifest_compatible") is True
    audit_errors = _audit_errors_status(audit)
    incomplete_reasons = list(
        audit.get("decision_evidence_incomplete_reasons", []) or []
    )
    if not audit_errors["valid"]:
        if "invalid_audit_errors" not in incomplete_reasons:
            incomplete_reasons.append("invalid_audit_errors")
    elif audit_errors["count"] > 0 and "audit_errors" not in incomplete_reasons:
        incomplete_reasons.append("audit_errors")
    usable = bool(
        schema_version == 2
        and available
        and complete
        and manifest_compatible
        and audit_errors["valid"]
        and audit_errors["count"] == 0
        and has_decision_buckets
    )
    return {
        "available": available,
        "complete": complete,
        "usable": usable,
        "manifest_compatible": manifest_compatible,
        "audit_errors_valid": audit_errors["valid"],
        "audit_error_count": audit_errors["count"],
        "decision_bucket_count": len(decision_buckets)
        if isinstance(decision_buckets, list)
        else 0,
        "incomplete_reasons": incomplete_reasons,
    }


def _diagnose(
    train_summary: Dict[str, Any],
    decision_summary: Dict[str, Any],
    evidence_usable: bool,
    min_known_changed: int,
) -> List[str]:
    diagnoses = []
    pos_ratio = float(train_summary.get("positive_ratio", 0.0))
    hard_ratio = float(train_summary.get("hard_negative_ratio", 0.0))

    if pos_ratio < 0.10:
        diagnoses.append("train_positive_ratio_too_low")
    if hard_ratio < 0.10:
        diagnoses.append("hard_negative_ratio_too_low")

    known_changed = int(decision_summary.get("known_changed_count", 0))
    improved = int(decision_summary.get("improved_count", 0))
    damaged = int(decision_summary.get("damaged_count", 0))
    harm_precision = _finite_float(
        decision_summary.get("harm_precision", 0.0)
    )
    if not evidence_usable or known_changed == 0:
        diagnoses.append("insufficient_decision_evidence")
    elif (
        known_changed >= min_known_changed
        and improved > damaged
        and harm_precision <= MAX_HARM_PRECISION
    ):
        diagnoses.append("candidate_for_shadow_ab_activation")
    else:
        diagnoses.append("shadow_decisions_not_safe")
    return diagnoses


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def build_summary(
    train_cache: Path,
    infer_audit: Path,
    min_known_changed: int = DEFAULT_MIN_KNOWN_CHANGED,
) -> Dict[str, Any]:
    if min_known_changed < DEFAULT_MIN_KNOWN_CHANGED:
        raise ValueError(
            f"min_known_changed must be at least {DEFAULT_MIN_KNOWN_CHANGED}"
        )
    train = _summarize_train(_load_train_cache(train_cache))
    audit = json.loads(infer_audit.read_text(encoding="utf-8"))
    track_state_scores = _summarize_track_state_scores(audit)
    decisions = _summarize_decisions(audit)
    evidence = _decision_evidence_status(audit)
    keys = sorted(
        set(train.keys()) | set(track_state_scores.keys()) | set(decisions.keys())
    )
    classes = {}
    for key in keys:
        train_summary = train.get(key, {})
        decision_summary = decisions.get(key, {})
        classes[key] = {
            "train": train_summary,
            "track_state_scores": track_state_scores.get(key, {}),
            "decision_evidence": decision_summary,
            "diagnosis": _diagnose(
                train_summary,
                decision_summary,
                evidence_usable=evidence["usable"],
                min_known_changed=min_known_changed,
            ),
        }
    return _json_safe({
        "schema_version": 2,
        "infer_audit_schema_version": int(audit.get("schema_version", 1)),
        "train_cache": str(train_cache),
        "infer_audit": str(infer_audit),
        "decision_evidence": evidence,
        "decision_gates": {
            "min_known_changed": int(min_known_changed),
            "max_harm_precision": MAX_HARM_PRECISION,
            "require_improved_greater_than_damaged": True,
        },
        "classes": classes,
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-cache", required=True, help="Pairwise association train cache pkl.")
    parser.add_argument("--infer-audit", required=True, help="Association head inference audit JSON.")
    parser.add_argument("--output", required=True, help="Output JSON summary path.")
    parser.add_argument(
        "--min-known-changed",
        type=int,
        default=DEFAULT_MIN_KNOWN_CHANGED,
        help="Minimum known changed decisions required for a shadow A/B candidate.",
    )
    args = parser.parse_args()
    if args.min_known_changed < DEFAULT_MIN_KNOWN_CHANGED:
        parser.error(
            f"--min-known-changed must be at least {DEFAULT_MIN_KNOWN_CHANGED}"
        )

    summary = build_summary(
        Path(args.train_cache),
        Path(args.infer_audit),
        min_known_changed=args.min_known_changed,
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    print(f"Wrote pairwise association train/infer comparison to {output}")


if __name__ == "__main__":
    main()
