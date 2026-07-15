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
MAX_CLASS_ID = 2**31 - 1
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


def _load_audit_json(path: Path) -> Tuple[Any, List[str]]:
    duplicate_keys = []

    def strict_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                duplicate_keys.append(key)
            else:
                result[key] = value
        return result

    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            parse_int=_bounded_json_int,
            object_pairs_hook=strict_object,
        )
    except (OSError, UnicodeError, json.JSONDecodeError):
        return {}, ["invalid_audit_json"]
    return data, ["duplicate_json_key"] if duplicate_keys else []


def _class_key(class_id: int, class_name: str) -> str:
    return f"{class_id}:{class_name}"


class _ClassIdentityError(ValueError):
    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


def _bounded_class_id(
    value: Any, field: str, *, allow_canonical_string: bool = False
) -> int:
    if type(value) is int:
        class_id = value
    elif allow_canonical_string and isinstance(value, str):
        if (
            not value
            or len(value) > len(str(MAX_CLASS_ID))
            or not all("0" <= char <= "9" for char in value)
            or (len(value) > 1 and value.startswith("0"))
        ):
            raise ValueError(f"{field} must be a canonical decimal class id")
        class_id = int(value)
    else:
        expected = (
            "a native integer or canonical decimal string"
            if allow_canonical_string
            else "a native integer"
        )
        raise TypeError(f"{field} must be {expected}")
    if not 0 <= class_id <= MAX_CLASS_ID:
        raise ValueError(f"{field} must be between 0 and {MAX_CLASS_ID}")
    return class_id


def _canonical_class_identity_status(
    audit: Mapping[str, Any],
) -> Tuple[Dict[str, Any], Dict[int, str]]:
    reasons = []
    normalized = {}
    runtime_manifest = audit.get("runtime_manifest")
    if not isinstance(runtime_manifest, Mapping):
        reasons.append("invalid_runtime_manifest")
        raw_mapping = None
    elif "class_id_to_name" not in runtime_manifest:
        reasons.append("missing_class_id_to_name")
        raw_mapping = None
    else:
        raw_mapping = runtime_manifest["class_id_to_name"]
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            reasons.append("invalid_class_id_to_name")

    seen_names = set()
    if isinstance(raw_mapping, Mapping):
        for raw_class_id, raw_class_name in raw_mapping.items():
            try:
                class_id = _bounded_class_id(
                    raw_class_id,
                    "class_id_to_name key",
                    allow_canonical_string=True,
                )
            except (TypeError, ValueError):
                reasons.append("invalid_class_id_to_name_class_id")
                continue
            if (
                not isinstance(raw_class_name, str)
                or not raw_class_name
                or raw_class_name != raw_class_name.strip()
            ):
                reasons.append("invalid_class_id_to_name_class_name")
                continue
            if class_id in normalized:
                reasons.append("duplicate_class_id_to_name_class_id")
                continue
            if raw_class_name in seen_names:
                reasons.append("duplicate_class_id_to_name_class_name")
                continue
            normalized[class_id] = raw_class_name
            seen_names.add(raw_class_name)

    reasons = list(dict.fromkeys(reasons))
    valid = not reasons
    status = {
        "valid": valid,
        "incomplete_reasons": reasons,
        "canonical_class_id_to_name": {
            str(class_id): class_name
            for class_id, class_name in sorted(normalized.items())
        },
    }
    return status, normalized


def _bucket_class_identity(
    bucket: Mapping[str, Any],
    *,
    id_field: str,
    name_field: str,
    source: str,
    canonical_classes: Mapping[int, str],
) -> Tuple[int, str]:
    try:
        class_id = _bounded_class_id(bucket.get(id_field), id_field)
    except (TypeError, ValueError) as exc:
        raise _ClassIdentityError(
            f"{source}_bucket_invalid_class_identity", str(exc)
        ) from exc
    class_name = bucket.get(name_field)
    if not isinstance(class_name, str) or not class_name:
        raise _ClassIdentityError(
            f"{source}_bucket_invalid_class_identity",
            f"{name_field} must be a non-empty string",
        )
    if canonical_classes and canonical_classes.get(class_id) != class_name:
        raise _ClassIdentityError(
            f"{source}_bucket_class_identity_mismatch",
            f"{source} bucket class id/name does not match runtime manifest",
        )
    return class_id, class_name


def _is_hard_negative_type(value: Any) -> bool:
    return str(value).strip().lower() in {"hard", "inference_margin", "inference_topk"}


def _summarize_train(
    samples: List[Dict[str, Any]], canonical_classes: Mapping[int, str]
) -> Tuple[Dict[str, Any], List[str]]:
    grouped = defaultdict(list)
    reasons = []
    for sample in samples:
        if not isinstance(sample, Mapping):
            reasons.append("train_bucket_invalid_class_identity")
            continue
        try:
            class_id, class_name = _bucket_class_identity(
                sample,
                id_field="class_id",
                name_field="category",
                source="train",
                canonical_classes=canonical_classes,
            )
        except _ClassIdentityError as exc:
            reasons.append(exc.reason)
            continue
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
    return out, list(dict.fromkeys(reasons))


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _summarize_track_state_scores(
    audit: Dict[str, Any], canonical_classes: Mapping[int, str]
) -> Tuple[Dict[str, Any], List[str]]:
    out = defaultdict(dict)
    reasons = []
    pair_buckets = audit.get("pair_buckets", audit.get("buckets", []))
    if not isinstance(pair_buckets, list):
        return {}, ["invalid_pair_buckets"]
    for bucket in pair_buckets:
        if not isinstance(bucket, Mapping):
            reasons.append("pair_bucket_invalid_class_identity")
            continue
        try:
            class_id, class_name = _bucket_class_identity(
                bucket,
                id_field="class_id",
                name_field="class_name",
                source="pair",
                canonical_classes=canonical_classes,
            )
        except _ClassIdentityError as exc:
            reasons.append(exc.reason)
            continue
        state_bucket = bucket.get("state_bucket")
        if (
            not isinstance(state_bucket, str)
            or state_bucket not in TRACK_STATE_BUCKETS
        ):
            reasons.append("invalid_pair_bucket_state")
            continue
        key = _class_key(class_id, class_name)
        try:
            out[key][state_bucket] = {
                "pair_count": int(bucket.get("pair_count", 0)),
                "active_pair_count": int(bucket.get("active_pair_count", 0)),
                "penalized_pair_count": int(bucket.get("penalized_pair_count", 0)),
                "active_ratio": _finite_float(bucket.get("active_ratio", 0.0)),
                "penalized_ratio": _finite_float(
                    bucket.get("penalized_ratio", 0.0)
                ),
                "avg_score": _finite_float(bucket.get("avg_score", 0.0)),
                "avg_delta": _finite_float(bucket.get("avg_delta", 0.0)),
                "avg_cost_before": _finite_float(
                    bucket.get("avg_cost_before", 0.0)
                ),
                "avg_cost_after": _finite_float(
                    bucket.get("avg_cost_after", 0.0)
                ),
                "score_quantiles": bucket.get("score_quantiles", {}),
                "delta_quantiles": bucket.get("delta_quantiles", {}),
                "cost_before_quantiles": bucket.get(
                    "cost_before_quantiles", {}
                ),
                "cost_after_quantiles": bucket.get("cost_after_quantiles", {}),
                "sample_count": int(bucket.get("sample_count", 0)),
                "samples": bucket.get("samples", []),
            }
        except (TypeError, ValueError, OverflowError):
            reasons.append("invalid_pair_bucket")
            out[key].pop(state_bucket, None)
    return dict(out), list(dict.fromkeys(reasons))


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


def _validate_decision_bucket(
    bucket: Any, canonical_classes: Mapping[int, str]
) -> Dict[str, Any]:
    if not isinstance(bucket, Mapping):
        raise TypeError("decision bucket must be a mapping")
    for field in ("class_id", "class_name", "track_state_bucket"):
        if field not in bucket:
            raise ValueError(f"missing required field {field}")

    class_id, class_name = _bucket_class_identity(
        bucket,
        id_field="class_id",
        name_field="class_name",
        source="decision",
        canonical_classes=canonical_classes,
    )
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


def _active_class_states_status(
    audit: Mapping[str, Any], canonical_classes: Mapping[int, str]
) -> Tuple[Dict[str, Any], Dict[int, set]]:
    reasons = []
    runtime_manifest = audit.get("runtime_manifest")
    if not isinstance(runtime_manifest, Mapping):
        reasons.append("invalid_runtime_manifest")
        active_class_states = None
    elif "active_class_states" not in runtime_manifest:
        reasons.append("missing_active_class_states")
        active_class_states = None
    else:
        active_class_states = runtime_manifest["active_class_states"]
        if not isinstance(active_class_states, Mapping):
            reasons.append("invalid_active_class_states")

    normalized = {}
    if isinstance(active_class_states, Mapping):
        for raw_class_id, raw_states in active_class_states.items():
            try:
                class_id = _bounded_class_id(
                    raw_class_id,
                    "active_class_states key",
                    allow_canonical_string=True,
                )
            except (TypeError, ValueError):
                reasons.append("invalid_active_class_states_class_id")
                continue
            if canonical_classes and class_id not in canonical_classes:
                reasons.append("active_class_not_in_class_id_to_name")
                continue
            if class_id in normalized:
                reasons.append("duplicate_active_class_states_class_id")
                continue
            if not isinstance(raw_states, list):
                reasons.append("invalid_active_class_states_states")
                continue
            if any(
                not isinstance(state, str) or state not in TRACK_STATE_BUCKETS
                for state in raw_states
            ):
                reasons.append("invalid_active_class_states_state")
                continue
            normalized[class_id] = set(raw_states)

    reasons = list(dict.fromkeys(reasons))
    valid = not reasons
    status = {
        "valid": valid,
        "class_level_usable": valid,
        "incomplete_reasons": reasons,
        "active_class_states": {
            str(class_id): sorted(states)
            for class_id, states in sorted(normalized.items())
        },
    }
    return status, normalized


def _decision_evidence_status(
    audit: Any, canonical_classes: Mapping[int, str]
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[str]]:
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
    if "manifest_initialized" not in audit:
        reject("missing_manifest_initialized")
    else:
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
    class_identity_reasons = []
    if not isinstance(decision_buckets, list) or not decision_buckets:
        reject("invalid_decision_buckets")
    else:
        bucket_errors = []
        for index, bucket in enumerate(decision_buckets):
            try:
                validated_buckets.append(
                    _validate_decision_bucket(bucket, canonical_classes)
                )
            except _ClassIdentityError as exc:
                class_identity_reasons.append(exc.reason)
                bucket_errors.append(f"decision_bucket[{index}]: {exc}")
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
    return (
        status,
        validated_buckets,
        list(dict.fromkeys(class_identity_reasons)),
    )


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
    active_states: Any,
    class_level_usable: bool,
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
        for state, state_diagnoses in diagnosis_by_state.items()
        if active_states is not None and state in active_states
    ]
    if (
        not class_level_usable
        or active_states is None
        or not active_states
        or len(state_gates) != len(active_states)
        or "insufficient_decision_evidence" in state_gates
    ):
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


def _bounded_json_int(value: str) -> Any:
    digits = value[1:] if value.startswith("-") else value
    if len(digits) > len(str(MAX_CLASS_ID)):
        return value
    try:
        return int(value)
    except ValueError:
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
    train_samples = _load_train_cache(train_cache)
    audit, json_reasons = _load_audit_json(infer_audit)
    audit_mapping = audit if isinstance(audit, Mapping) else {}
    identity_status, canonical_classes = _canonical_class_identity_status(
        audit_mapping
    )
    train, train_identity_reasons = _summarize_train(
        train_samples, canonical_classes
    )
    track_state_scores, pair_identity_reasons = _summarize_track_state_scores(
        audit_mapping, canonical_classes
    )
    evidence, validated_buckets, decision_identity_reasons = (
        _decision_evidence_status(audit, canonical_classes)
    )
    activation_status, active_class_states = _active_class_states_status(
        audit_mapping, canonical_classes
    )
    identity_reasons = list(
        dict.fromkeys(
            json_reasons
            + identity_status["incomplete_reasons"]
            + train_identity_reasons
            + pair_identity_reasons
            + decision_identity_reasons
        )
    )
    identity_status["valid"] = not identity_reasons
    identity_status["incomplete_reasons"] = identity_reasons
    if identity_reasons:
        evidence["usable"] = False
        if "invalid_class_identity" not in evidence["incomplete_reasons"]:
            evidence["incomplete_reasons"].append("invalid_class_identity")
        activation_status["class_level_usable"] = False
        if (
            "class_identity_validation_failed"
            not in activation_status["incomplete_reasons"]
        ):
            activation_status["incomplete_reasons"].append(
                "class_identity_validation_failed"
            )
    decisions = _summarize_decisions(validated_buckets)
    manifest_active_keys = {
        _class_key(class_id, canonical_classes[class_id])
        for class_id in active_class_states
        if class_id in canonical_classes
    }
    keys = sorted(
        set(train.keys())
        | set(track_state_scores.keys())
        | set(decisions.keys())
        | manifest_active_keys
    )
    classes = {}
    for key in keys:
        train_summary = train.get(key, {})
        class_decisions = decisions.get(key, {})
        decision_summary = class_decisions.get(
            "decision_evidence", _finish_decision_summary(_empty_decision_summary())
        )
        decision_by_state = dict(
            class_decisions.get("decision_evidence_by_state", {})
        )
        class_id = _bounded_class_id(
            key.split(":", 1)[0], "report class id", allow_canonical_string=True
        )
        active_states = active_class_states.get(class_id)
        if active_states is not None:
            for state in active_states:
                decision_by_state.setdefault(
                    state, _finish_decision_summary(_empty_decision_summary())
                )
        reported_states = set(decision_by_state)
        if active_states is not None:
            reported_states.update(active_states)
        diagnosis_by_state = {
            state: _diagnose(
                train_summary,
                decision_by_state.get(state, {}),
                evidence_usable=evidence["usable"],
                min_known_changed=min_known_changed,
            )
            for state in sorted(reported_states)
        }
        classes[key] = {
            "train": train_summary,
            "track_state_scores": track_state_scores.get(key, {}),
            "decision_evidence": decision_summary,
            "decision_evidence_by_state": decision_by_state,
            "diagnosis_by_state": diagnosis_by_state,
            "diagnosis": _class_diagnosis(
                train_summary,
                diagnosis_by_state,
                active_states,
                activation_status["class_level_usable"],
            ),
        }
    return _json_safe(
        {
            "schema_version": 2,
            "infer_audit_schema_version": audit_mapping.get("schema_version"),
            "train_cache": str(train_cache),
            "infer_audit": str(infer_audit),
            "decision_evidence": evidence,
            "class_identity_validation": identity_status,
            "class_state_activation": activation_status,
            "decision_gates": {
                "min_known_changed": int(min_known_changed),
                "max_harm_precision": MAX_HARM_PRECISION,
                "require_improved_greater_than_damaged": True,
                "class_candidate_requires_all_manifest_active_states": True,
            },
            "classes": classes,
        }
    )


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
