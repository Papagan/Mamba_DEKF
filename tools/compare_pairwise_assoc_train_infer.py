#!/usr/bin/env python3
"""Compare pairwise association training cache with inference audit output."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


DEFAULT_MIN_KNOWN_CHANGED = 100
MAX_HARM_PRECISION = 0.20
TRACK_STATE_BUCKETS = {"matched", "unmatched"}
DECISION_COUNTER_FIELDS = (
    "total_decisions",
    "changed_decisions",
    "changed_detection_count",
    "changed_to_match_count",
    "changed_to_unmatched_count",
    "threshold_crossing_count",
    "known_gt_decisions",
    "improved_count",
    "damaged_count",
    "neutral_count",
    "unknown_count",
    "known_changed_count",
)


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


def _empty_decision_summary() -> Dict[str, int]:
    return {
        "decision_bucket_count": 0,
        **{field: 0 for field in DECISION_COUNTER_FIELDS},
    }


def _finish_decision_summary(summary: Dict[str, int]) -> Dict[str, Any]:
    result = dict(summary)
    known_changed = result["known_changed_count"]
    result["net_improvement"] = (
        result["improved_count"] - result["damaged_count"]
    )
    result["harm_precision"] = float(
        result["damaged_count"] / max(known_changed, 1)
    )
    return result


def _summarize_decisions(
    decision_buckets: List[Dict[str, Any]],
) -> Dict[str, Any]:
    by_class_state = defaultdict(lambda: defaultdict(_empty_decision_summary))
    for bucket in decision_buckets:
        key = _class_key(bucket["class_id"], bucket["class_name"])
        state = bucket["track_state_bucket"]
        summary = by_class_state[key][state]
        summary["decision_bucket_count"] += 1
        for field in DECISION_COUNTER_FIELDS:
            summary[field] += bucket[field]

    out = {}
    for key, state_counters in by_class_state.items():
        class_summary = _empty_decision_summary()
        by_state = {}
        for state, summary in sorted(state_counters.items()):
            by_state[state] = _finish_decision_summary(summary)
            class_summary["decision_bucket_count"] += summary[
                "decision_bucket_count"
            ]
            for field in DECISION_COUNTER_FIELDS:
                class_summary[field] += summary[field]
        out[key] = {
            "decision_evidence": _finish_decision_summary(class_summary),
            "decision_evidence_by_state": by_state,
        }
    return out


def _strict_nonnegative_int(value: Any, field: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{field} must be a native integer")
    if value < 0:
        raise ValueError(f"{field} must be non-negative")
    return value


def _strict_finite_number(value: Any, field: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(value):
        raise TypeError(f"{field} must be a finite native number")
    return float(value)


def _validate_decision_bucket(bucket: Any) -> Dict[str, Any]:
    if not isinstance(bucket, Mapping):
        raise TypeError("decision bucket must be a mapping")
    for field in ("class_id", "class_name", "track_state_bucket"):
        if field not in bucket:
            raise ValueError(f"missing required field {field}")

    class_id = _strict_nonnegative_int(bucket["class_id"], "class_id")
    class_name = bucket["class_name"]
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")
    state = bucket["track_state_bucket"]
    if state not in TRACK_STATE_BUCKETS:
        raise ValueError("track_state_bucket must be matched or unmatched")

    normalized = {
        "class_id": class_id,
        "class_name": class_name,
        "track_state_bucket": state,
    }
    for field in DECISION_COUNTER_FIELDS:
        if field not in bucket:
            raise ValueError(f"missing required counter {field}")
        normalized[field] = _strict_nonnegative_int(bucket[field], field)

    total = normalized["total_decisions"]
    outcome_total = sum(
        normalized[field]
        for field in (
            "improved_count",
            "damaged_count",
            "neutral_count",
            "unknown_count",
        )
    )
    if outcome_total != total:
        raise ValueError("outcome counts must sum to total_decisions")
    known_gt = (
        normalized["improved_count"]
        + normalized["damaged_count"]
        + normalized["neutral_count"]
    )
    if normalized["known_gt_decisions"] != known_gt:
        raise ValueError("known_gt_decisions does not match known outcomes")

    changed = normalized["changed_decisions"]
    if changed > total:
        raise ValueError("changed_decisions cannot exceed total_decisions")
    change_category_total = sum(
        normalized[field]
        for field in (
            "changed_detection_count",
            "changed_to_match_count",
            "changed_to_unmatched_count",
        )
    )
    if change_category_total != changed:
        raise ValueError("change category counts must sum to changed_decisions")
    if normalized["threshold_crossing_count"] > changed:
        raise ValueError("threshold_crossing_count cannot exceed changed_decisions")

    improved_or_damaged = (
        normalized["improved_count"] + normalized["damaged_count"]
    )
    known_changed = normalized["known_changed_count"]
    if not (
        improved_or_damaged
        <= known_changed
        <= improved_or_damaged + normalized["neutral_count"]
        and known_changed <= changed
    ):
        raise ValueError("known_changed_count is inconsistent with outcomes")

    if "net_improvement" in bucket:
        net = _strict_finite_number(bucket["net_improvement"], "net_improvement")
        if net != normalized["improved_count"] - normalized["damaged_count"]:
            raise ValueError("net_improvement is inconsistent with outcomes")
    if "harm_precision" in bucket:
        harm = _strict_finite_number(bucket["harm_precision"], "harm_precision")
        expected_harm = normalized["damaged_count"] / max(known_changed, 1)
        if not math.isclose(harm, expected_harm, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("harm_precision is inconsistent with outcomes")
    return normalized


def _audit_errors_status(audit: Mapping[str, Any]) -> Dict[str, Any]:
    errors = audit.get("audit_errors")
    if not isinstance(errors, Mapping):
        return {"valid": False, "count": None, "reason": "not_mapping"}

    count = errors.get("count")
    if type(count) is not int or count < 0:
        return {"valid": False, "count": None, "reason": "invalid_count"}
    sample_count = errors.get("sample_count")
    if type(sample_count) is not int or sample_count < 0:
        return {
            "valid": False,
            "count": None,
            "reason": "invalid_sample_count",
        }
    samples = errors.get("samples")
    if not isinstance(samples, list):
        return {"valid": False, "count": None, "reason": "invalid_samples"}
    if sample_count != len(samples) or sample_count > count:
        return {
            "valid": False,
            "count": None,
            "reason": "inconsistent_sample_count",
        }
    if count == 0 and samples:
        return {
            "valid": False,
            "count": None,
            "reason": "samples_with_zero_count",
        }

    if "audit_error_count" in audit:
        top_level_count = audit["audit_error_count"]
        if (
            type(top_level_count) is not int
            or top_level_count < 0
            or top_level_count != count
        ):
            return {
                "valid": False,
                "count": None,
                "reason": "top_level_count_mismatch",
            }
    return {"valid": True, "count": count, "reason": None}


def _decision_evidence_status(
    audit: Any,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    if not isinstance(audit, Mapping):
        return (
            {
                "available": False,
                "complete": False,
                "usable": False,
                "manifest_compatible": False,
                "audit_errors_valid": False,
                "audit_error_count": None,
                "decision_bucket_count": 0,
                "incomplete_reasons": ["invalid_audit_root"],
            },
            [],
        )

    incomplete_reasons = []

    def reject(reason: str) -> None:
        if reason not in incomplete_reasons:
            incomplete_reasons.append(reason)

    schema_version = audit.get("schema_version")
    if type(schema_version) is not int or schema_version != 2:
        reject("invalid_schema_version")

    available_value = audit.get("decision_evidence_available")
    available = type(available_value) is bool and available_value is True
    if not available:
        reject("invalid_decision_evidence_available")
    complete_value = audit.get("decision_evidence_complete")
    complete = type(complete_value) is bool and complete_value is True
    if not complete:
        reject("invalid_decision_evidence_complete")
    manifest_value = audit.get("manifest_compatible")
    manifest_compatible = type(manifest_value) is bool and manifest_value is True
    if not manifest_compatible:
        reject("invalid_manifest_compatible")

    if "runtime_manifest" not in audit:
        reject("missing_runtime_manifest")
    elif not isinstance(audit["runtime_manifest"], Mapping):
        reject("invalid_runtime_manifest")
    if "manifest_initialized" in audit:
        initialized = audit["manifest_initialized"]
        if type(initialized) is not bool or initialized is not True:
            reject("invalid_manifest_initialized")

    source_reasons = audit.get("decision_evidence_incomplete_reasons")
    if not isinstance(source_reasons, list):
        reject("invalid_decision_evidence_incomplete_reasons")
    elif source_reasons:
        reject("decision_evidence_incomplete_reasons_not_empty")

    decision_buckets = audit.get("decision_buckets", [])
    validated_buckets = []
    if not isinstance(decision_buckets, list) or not decision_buckets:
        reject("invalid_decision_buckets")
    else:
        bucket_errors = []
        for index, bucket in enumerate(decision_buckets):
            try:
                validated_buckets.append(_validate_decision_bucket(bucket))
            except (TypeError, ValueError) as exc:
                bucket_errors.append(f"decision_bucket[{index}]: {exc}")
        if bucket_errors:
            reject("invalid_decision_buckets")
            incomplete_reasons.extend(bucket_errors)

    audit_errors = _audit_errors_status(audit)
    if not audit_errors["valid"]:
        reject("invalid_audit_errors")
        reject(f"invalid_audit_errors:{audit_errors['reason']}")
    elif audit_errors["count"] > 0:
        reject("audit_errors")

    status = {
        "available": available,
        "complete": complete,
        "usable": not incomplete_reasons,
        "manifest_compatible": manifest_compatible,
        "audit_errors_valid": audit_errors["valid"],
        "audit_error_count": audit_errors["count"],
        "decision_bucket_count": len(decision_buckets)
        if isinstance(decision_buckets, list)
        else 0,
        "incomplete_reasons": incomplete_reasons,
    }
    return status, validated_buckets


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


def _class_diagnosis(
    train_summary: Dict[str, Any],
    diagnosis_by_state: Dict[str, List[str]],
) -> List[str]:
    diagnoses = []
    pos_ratio = float(train_summary.get("positive_ratio", 0.0))
    hard_ratio = float(train_summary.get("hard_negative_ratio", 0.0))
    if pos_ratio < 0.10:
        diagnoses.append("train_positive_ratio_too_low")
    if hard_ratio < 0.10:
        diagnoses.append("hard_negative_ratio_too_low")

    state_gates = [
        next(
            diagnosis
            for diagnosis in state_diagnoses
            if diagnosis
            in {
                "insufficient_decision_evidence",
                "candidate_for_shadow_ab_activation",
                "shadow_decisions_not_safe",
            }
        )
        for state_diagnoses in diagnosis_by_state.values()
    ]
    if not state_gates or "insufficient_decision_evidence" in state_gates:
        diagnoses.append("insufficient_decision_evidence")
    elif all(gate == "candidate_for_shadow_ab_activation" for gate in state_gates):
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
    audit_mapping = audit if isinstance(audit, Mapping) else {}
    track_state_scores = _summarize_track_state_scores(audit_mapping)
    evidence, validated_buckets = _decision_evidence_status(audit)
    decisions = _summarize_decisions(validated_buckets)
    keys = sorted(
        set(train.keys()) | set(track_state_scores.keys()) | set(decisions.keys())
    )
    classes = {}
    for key in keys:
        train_summary = train.get(key, {})
        class_decisions = decisions.get(key, {})
        decision_summary = class_decisions.get("decision_evidence", {})
        decision_by_state = class_decisions.get(
            "decision_evidence_by_state", {}
        )
        diagnosis_by_state = {
            state: _diagnose(
                train_summary,
                state_summary,
                evidence_usable=evidence["usable"],
                min_known_changed=min_known_changed,
            )
            for state, state_summary in decision_by_state.items()
        }
        classes[key] = {
            "train": train_summary,
            "track_state_scores": track_state_scores.get(key, {}),
            "decision_evidence": decision_summary,
            "decision_evidence_by_state": decision_by_state,
            "diagnosis_by_state": diagnosis_by_state,
            "diagnosis": _class_diagnosis(
                train_summary, diagnosis_by_state
            ),
        }
    return _json_safe({
        "schema_version": 2,
        "infer_audit_schema_version": audit_mapping.get("schema_version"),
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
