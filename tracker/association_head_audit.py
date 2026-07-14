import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tracker.association_runtime_manifest import runtime_manifests_compatible


_DECISION_OUTCOMES = {"improved", "damaged", "neutral", "unknown"}
_TRACK_STATE_BUCKETS = {"matched", "unmatched"}
_DECISION_COUNTER_FIELDS = (
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


def _json_safe(value, *, path="value"):
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{path} must be finite")
        return value
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{path} keys must be strings")
            result[key] = _json_safe(item, path=f"{path}.{key}")
        return result
    if isinstance(value, (list, tuple)):
        return [
            _json_safe(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(f"{path} is not JSON serializable: {type(value).__name__}")


def _required(record, field_name):
    if field_name not in record:
        raise ValueError(f"decision is missing required field {field_name!r}")
    return record[field_name]


def _strict_bool(value, field_name, *, allow_none=False):
    if allow_none and value is None:
        return None
    if not isinstance(value, (bool, np.bool_)):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{field_name} must be a boolean{suffix}")
    return bool(value)


def _nonnegative_int(value, field_name, *, allow_none=False):
    if allow_none and value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        suffix = " or None" if allow_none else ""
        raise TypeError(f"{field_name} must be a non-negative integer{suffix}")
    result = int(value)
    if result < 0:
        raise ValueError(f"{field_name} must be non-negative")
    return result


def _optional_finite_float(value, field_name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(f"{field_name} must be a finite number or None")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{field_name} must be finite")
    return result


def _finite_float(value, field_name):
    result = _optional_finite_float(value, field_name)
    if result is None:
        raise TypeError(f"{field_name} must be a finite number")
    return result


def _optional_rank(value, field_name):
    result = _nonnegative_int(value, field_name, allow_none=True)
    if result == 0:
        raise ValueError(f"{field_name} must be positive or None")
    return result


def _optional_pair_label(value, field_name):
    if value is None:
        return None
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{field_name} must be 0, 1, or None")
    result = int(value)
    if result not in {0, 1}:
        raise ValueError(f"{field_name} must be 0, 1, or None")
    return result


def _normalize_decision(record):
    if not isinstance(record, Mapping):
        raise TypeError("decision must be a mapping")

    class_id = _nonnegative_int(_required(record, "class_id"), "class_id")
    class_name = _required(record, "class_name")
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("class_name must be a non-empty string")
    class_name = class_name.strip()

    state_bucket = _required(record, "track_state_bucket")
    if not isinstance(state_bucket, str) or state_bucket not in _TRACK_STATE_BUCKETS:
        raise ValueError("track_state_bucket must be matched or unmatched")

    assignment_changed = _strict_bool(
        _required(record, "assignment_changed"), "assignment_changed"
    )
    production_index = _nonnegative_int(
        _required(record, "production_det_index"),
        "production_det_index",
        allow_none=True,
    )
    shadow_index = _nonnegative_int(
        _required(record, "shadow_det_index"),
        "shadow_det_index",
        allow_none=True,
    )
    if assignment_changed != (production_index != shadow_index):
        raise ValueError(
            "assignment_changed must equal whether production_det_index and "
            "shadow_det_index differ"
        )

    threshold_crossing = _strict_bool(
        _required(record, "threshold_crossing"),
        "threshold_crossing",
        allow_none=True,
    )
    if threshold_crossing and not assignment_changed:
        raise ValueError("threshold_crossing cannot be true for an unchanged decision")

    outcome = _required(record, "outcome")
    if not isinstance(outcome, str) or outcome not in _DECISION_OUTCOMES:
        raise ValueError(
            "outcome must be one of improved, damaged, neutral, or unknown"
        )
    if outcome in {"improved", "damaged"} and not assignment_changed:
        raise ValueError(f"{outcome} outcome requires an assignment change")

    normalized = _json_safe(record, path="decision")
    normalized.update(
        {
            "class_id": class_id,
            "class_name": class_name,
            "track_state_bucket": state_bucket,
            "assignment_changed": assignment_changed,
            "production_det_index": production_index,
            "shadow_det_index": shadow_index,
            "threshold_crossing": threshold_crossing,
            "outcome": outcome,
        }
    )
    for field_name in (
        "production_selected_cost",
        "shadow_selected_cost",
        "production_selected_association_score",
        "shadow_selected_association_score",
        "original_cost_gap",
    ):
        if field_name in record:
            normalized[field_name] = _optional_finite_float(
                record[field_name], field_name
            )
    for field_name in ("geometric_candidate_rank", "shadow_candidate_rank"):
        if field_name in record:
            normalized[field_name] = _optional_rank(record[field_name], field_name)
    for field_name in ("production_pair_label", "shadow_pair_label"):
        if field_name in record:
            normalized[field_name] = _optional_pair_label(
                record[field_name], field_name
            )
    return normalized


def _manifest_differences(left, right, path=""):
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        differences = []
        for key in sorted(set(left) | set(right)):
            key_path = f"{path}.{key}" if path else str(key)
            if key not in left:
                differences.append(f"{key_path}: target=<missing>, incoming={right[key]!r}")
            elif key not in right:
                differences.append(f"{key_path}: target={left[key]!r}, incoming=<missing>")
            else:
                differences.extend(
                    _manifest_differences(left[key], right[key], key_path)
                )
        return differences
    if left != right:
        label = path or "runtime_manifest"
        return [f"{label}: target={left!r}, incoming={right!r}"]
    return []


def _extend_bounded(target, values, limit):
    if limit <= 0:
        return
    remaining = limit - len(target)
    if remaining > 0:
        target.extend(values[:remaining])


def _list_field(mapping, field_name, *, required=False):
    if required and field_name not in mapping:
        raise ValueError(f"audit state is missing required field {field_name!r}")
    value = mapping.get(field_name, [])
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list")
    return value


def _validate_pair_bucket_state(item):
    if not isinstance(item, Mapping):
        raise TypeError("pair bucket state must be a mapping")
    for field_name in (
        "class_id",
        "class_name",
        "state_bucket",
        "pair_count",
        "finite_pair_count",
        "active_pair_count",
        "penalized_pair_count",
        "avg_score",
        "avg_delta",
        "avg_cost_before",
        "avg_cost_after",
    ):
        if field_name not in item:
            raise ValueError(f"pair bucket is missing required field {field_name!r}")

    normalized = dict(item)
    normalized["class_id"] = _nonnegative_int(item["class_id"], "class_id")
    for field_name in ("class_name", "state_bucket"):
        value = item[field_name]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{field_name} must be a non-empty string")
        normalized[field_name] = value.strip()

    for field_name in (
        "pair_count",
        "finite_pair_count",
        "active_pair_count",
        "penalized_pair_count",
    ):
        normalized[field_name] = _nonnegative_int(item[field_name], field_name)
    pair_count = normalized["pair_count"]
    for field_name in (
        "finite_pair_count",
        "active_pair_count",
        "penalized_pair_count",
    ):
        if normalized[field_name] > pair_count:
            raise ValueError(f"{field_name} cannot exceed pair_count")

    for field_name in (
        "avg_score",
        "avg_delta",
        "avg_cost_before",
        "avg_cost_after",
    ):
        normalized[field_name] = _finite_float(item[field_name], field_name)

    for field_name in (
        "score_quantiles",
        "delta_quantiles",
        "cost_before_quantiles",
        "cost_after_quantiles",
    ):
        quantiles = item.get(field_name, {}) or {}
        if not isinstance(quantiles, Mapping):
            raise TypeError(f"{field_name} must be a mapping")
        quantiles = _json_safe(quantiles, path=field_name)
        normalized[field_name] = {
            key: _finite_float(value, f"{field_name}.{key}")
            for key, value in quantiles.items()
        }

    samples = _list_field(item, "samples")
    normalized["samples"] = []
    for index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise TypeError(f"samples[{index}] must be a mapping")
        normalized["samples"].append(
            _json_safe(sample, path=f"samples[{index}]")
        )
    return normalized


def _validate_decision_bucket_state(item):
    if not isinstance(item, Mapping):
        raise TypeError("decision bucket state must be a mapping")
    for field_name in ("class_id", "class_name", "track_state_bucket"):
        if field_name not in item:
            raise ValueError(
                f"decision bucket is missing required field {field_name!r}"
            )
    class_id = _nonnegative_int(item["class_id"], "class_id")
    class_name = item["class_name"]
    state_bucket = item["track_state_bucket"]
    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError("decision bucket class_name must be non-empty")
    if state_bucket not in _TRACK_STATE_BUCKETS:
        raise ValueError("decision bucket track_state_bucket is invalid")

    counters = {}
    for field_name in _DECISION_COUNTER_FIELDS:
        if field_name not in item:
            raise ValueError(
                f"decision bucket is missing required counter {field_name!r}"
            )
        counters[field_name] = _nonnegative_int(item[field_name], field_name)

    total = counters["total_decisions"]
    outcome_total = sum(
        counters[field_name]
        for field_name in (
            "improved_count",
            "damaged_count",
            "neutral_count",
            "unknown_count",
        )
    )
    if outcome_total != total:
        raise ValueError("outcome counts must sum to total_decisions")
    expected_known = (
        counters["improved_count"]
        + counters["damaged_count"]
        + counters["neutral_count"]
    )
    if counters["known_gt_decisions"] != expected_known:
        raise ValueError(
            "known_gt_decisions must equal improved_count + damaged_count + "
            "neutral_count"
        )
    changed = counters["changed_decisions"]
    if changed > total:
        raise ValueError("changed_decisions cannot exceed total_decisions")
    change_categories = (
        "changed_detection_count",
        "changed_to_match_count",
        "changed_to_unmatched_count",
    )
    for field_name in change_categories:
        if counters[field_name] > changed:
            raise ValueError(f"{field_name} cannot exceed changed_decisions")
    if sum(counters[field_name] for field_name in change_categories) > changed:
        raise ValueError(
            "changed assignment categories cannot exceed changed_decisions"
        )
    known_changed = counters["known_changed_count"]
    if known_changed > changed:
        raise ValueError("known_changed_count cannot exceed changed_decisions")
    minimum_known_changed = counters["improved_count"] + counters["damaged_count"]
    maximum_known_changed = minimum_known_changed + counters["neutral_count"]
    if not minimum_known_changed <= known_changed <= maximum_known_changed:
        raise ValueError(
            "known_changed_count must include all improved/damaged changes and "
            "at most changed neutral decisions"
        )
    if counters["threshold_crossing_count"] > total:
        raise ValueError("threshold_crossing_count cannot exceed total_decisions")

    raw_decisions = item.get("raw_decisions", item.get("samples", []))
    if not isinstance(raw_decisions, list):
        raise TypeError("raw_decisions must be a list")
    normalized_decisions = []
    bucket_key = (class_id, class_name.strip(), state_bucket)
    for index, record in enumerate(raw_decisions):
        decision = _normalize_decision(record)
        decision_key = (
            decision["class_id"],
            decision["class_name"],
            decision["track_state_bucket"],
        )
        if decision_key != bucket_key:
            raise ValueError(
                f"raw_decisions[{index}] does not match decision bucket metadata"
            )
        normalized_decisions.append(decision)

    association_scores = _list_field(item, "association_scores")
    normalized_scores = [
        _json_safe(value, path=f"association_scores[{index}]")
        for index, value in enumerate(association_scores)
    ]
    cost_gaps = _list_field(item, "cost_gaps")
    normalized_cost_gaps = [
        _finite_float(value, f"cost_gaps[{index}]")
        for index, value in enumerate(cost_gaps)
    ]
    return {
        "class_id": class_id,
        "class_name": class_name.strip(),
        "track_state_bucket": state_bucket,
        **counters,
        "raw_decisions": normalized_decisions,
        "association_scores": normalized_scores,
        "cost_gaps": normalized_cost_gaps,
    }


@dataclass
class _Bucket:
    class_id: int
    class_name: str
    state_bucket: str
    pair_count: int = 0
    finite_pair_count: int = 0
    active_pair_count: int = 0
    penalized_pair_count: int = 0
    score_sum: float = 0.0
    delta_sum: float = 0.0
    cost_before_sum: float = 0.0
    cost_after_sum: float = 0.0
    scores: list = field(default_factory=list)
    deltas: list = field(default_factory=list)
    cost_befores: list = field(default_factory=list)
    cost_afters: list = field(default_factory=list)
    samples: list = field(default_factory=list)

    def add(
        self,
        *,
        score: float,
        delta: float,
        cost_before: float,
        cost_after: float,
        active: bool,
        finite: bool,
        sample=None,
        max_samples: int = 0,
    ) -> None:
        self.pair_count += 1
        if finite:
            self.finite_pair_count += 1
        if active:
            self.active_pair_count += 1
        if delta > 0.0:
            self.penalized_pair_count += 1
        self.score_sum += float(score)
        self.delta_sum += float(delta)
        self.cost_before_sum += float(cost_before)
        self.cost_after_sum += float(cost_after)
        self.scores.append(float(score))
        self.deltas.append(float(delta))
        self.cost_befores.append(float(cost_before))
        self.cost_afters.append(float(cost_after))
        if sample is not None and len(self.samples) < int(max_samples):
            self.samples.append(dict(sample))

    @staticmethod
    def _quantiles(values):
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

    def to_summary(self):
        pair_count = max(int(self.pair_count), 1)
        active_count = max(int(self.active_pair_count), 1)
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "state_bucket": str(self.state_bucket),
            "pair_count": int(self.pair_count),
            "finite_pair_count": int(self.finite_pair_count),
            "active_pair_count": int(self.active_pair_count),
            "penalized_pair_count": int(self.penalized_pair_count),
            "active_ratio": float(self.active_pair_count / pair_count),
            "penalized_ratio": float(self.penalized_pair_count / active_count),
            "avg_score": float(self.score_sum / pair_count),
            "avg_delta": float(self.delta_sum / pair_count),
            "avg_cost_before": float(self.cost_before_sum / pair_count),
            "avg_cost_after": float(self.cost_after_sum / pair_count),
            "score_quantiles": self._quantiles(self.scores),
            "delta_quantiles": self._quantiles(self.deltas),
            "cost_before_quantiles": self._quantiles(self.cost_befores),
            "cost_after_quantiles": self._quantiles(self.cost_afters),
            "sample_count": int(len(self.samples)),
            "samples": list(self.samples),
        }


@dataclass
class _DecisionBucket:
    class_id: int
    class_name: str
    track_state_bucket: str
    total_decisions: int = 0
    changed_decisions: int = 0
    changed_detection_count: int = 0
    changed_to_match_count: int = 0
    changed_to_unmatched_count: int = 0
    threshold_crossing_count: int = 0
    known_gt_decisions: int = 0
    improved_count: int = 0
    damaged_count: int = 0
    neutral_count: int = 0
    unknown_count: int = 0
    known_changed_count: int = 0
    raw_decisions: list = field(default_factory=list)
    association_scores: list = field(default_factory=list)
    cost_gaps: list = field(default_factory=list)

    def add(self, record, *, max_samples):
        changed = record["assignment_changed"]
        production_index = record["production_det_index"]
        shadow_index = record["shadow_det_index"]
        outcome = record["outcome"]

        self.total_decisions += 1
        self.changed_decisions += int(changed)
        self.changed_detection_count += int(
            changed and production_index is not None and shadow_index is not None
        )
        self.changed_to_match_count += int(
            changed and production_index is None and shadow_index is not None
        )
        self.changed_to_unmatched_count += int(
            changed and production_index is not None and shadow_index is None
        )
        self.threshold_crossing_count += int(record["threshold_crossing"] is True)
        self.known_gt_decisions += int(outcome != "unknown")
        setattr(self, f"{outcome}_count", getattr(self, f"{outcome}_count") + 1)
        self.known_changed_count += int(changed and outcome != "unknown")

        if max_samples <= 0:
            return
        if len(self.raw_decisions) < max_samples:
            self.raw_decisions.append(dict(record))
        production_score = record.get("production_selected_association_score")
        shadow_score = record.get("shadow_selected_association_score")
        if (
            len(self.association_scores) < max_samples
            and (production_score is not None or shadow_score is not None)
        ):
            self.association_scores.append(
                {
                    "production_selected_association_score": production_score,
                    "shadow_selected_association_score": shadow_score,
                }
            )
        cost_gap = record.get("original_cost_gap")
        if cost_gap is not None and len(self.cost_gaps) < max_samples:
            self.cost_gaps.append(float(cost_gap))

    def to_summary(self):
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "track_state_bucket": str(self.track_state_bucket),
            "total_decisions": int(self.total_decisions),
            "changed_decisions": int(self.changed_decisions),
            "changed_detection_count": int(self.changed_detection_count),
            "changed_to_match_count": int(self.changed_to_match_count),
            "changed_to_unmatched_count": int(self.changed_to_unmatched_count),
            "threshold_crossing_count": int(self.threshold_crossing_count),
            "known_gt_decisions": int(self.known_gt_decisions),
            "improved_count": int(self.improved_count),
            "damaged_count": int(self.damaged_count),
            "neutral_count": int(self.neutral_count),
            "unknown_count": int(self.unknown_count),
            "known_changed_count": int(self.known_changed_count),
            "net_improvement": int(self.improved_count - self.damaged_count),
            "harm_precision": float(
                self.damaged_count / max(self.known_changed_count, 1)
            ),
            "sample_count": int(len(self.raw_decisions)),
            "samples": list(self.raw_decisions),
            "raw_decisions": list(self.raw_decisions),
            "association_scores": list(self.association_scores),
            "cost_gaps": list(self.cost_gaps),
        }


class AssociationHeadAuditAccumulator:
    def __init__(
        self,
        *,
        max_samples_per_bucket: int = 0,
        runtime_manifest=None,
        strict_manifest: bool = False,
    ):
        self._buckets = {}
        self._decision_buckets = {}
        self.max_samples_per_bucket = int(max_samples_per_bucket)
        self.runtime_manifest = (
            None
            if runtime_manifest is None
            else _json_safe(runtime_manifest, path="runtime_manifest")
        )
        if self.runtime_manifest is not None and not isinstance(
            self.runtime_manifest, dict
        ):
            raise TypeError("runtime_manifest must be a mapping or None")
        self.strict_manifest = bool(strict_manifest)
        self._manifest_compatible = True
        self._manifest_mismatches = []

    def add_pair(
        self,
        *,
        class_id: int,
        class_name: str,
        state_bucket: str,
        score: float,
        delta: float,
        cost_before: float,
        cost_after: float,
        active: bool,
        finite: bool,
        sample=None,
    ) -> None:
        normalized_class_id = _nonnegative_int(class_id, "class_id")
        if not isinstance(class_name, str) or not class_name.strip():
            raise ValueError("class_name must be a non-empty string")
        if not isinstance(state_bucket, str) or not state_bucket.strip():
            raise ValueError("state_bucket must be a non-empty string")
        normalized_score = _finite_float(score, "score")
        normalized_delta = _finite_float(delta, "delta (penalty)")
        normalized_cost_before = _finite_float(
            cost_before, "cost_before (base_cost)"
        )
        normalized_cost_after = _finite_float(
            cost_after, "cost_after (corrected_cost)"
        )
        normalized_active = _strict_bool(active, "active")
        normalized_finite = _strict_bool(finite, "finite")
        normalized_sample = None
        if sample is not None:
            if not isinstance(sample, Mapping):
                raise TypeError("sample must be a JSON-serializable mapping or None")
            normalized_sample = _json_safe(sample, path="sample")

        key = (normalized_class_id, class_name.strip(), state_bucket.strip())
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                class_id=normalized_class_id,
                class_name=class_name.strip(),
                state_bucket=state_bucket.strip(),
            )
            self._buckets[key] = bucket
        bucket.add(
            score=normalized_score,
            delta=normalized_delta,
            cost_before=normalized_cost_before,
            cost_after=normalized_cost_after,
            active=normalized_active,
            finite=normalized_finite,
            sample=normalized_sample,
            max_samples=self.max_samples_per_bucket,
        )

    def add_decision(self, record) -> None:
        decision = _normalize_decision(record)
        key = (
            decision["class_id"],
            decision["class_name"],
            decision["track_state_bucket"],
        )
        bucket = self._decision_buckets.get(key)
        if bucket is None:
            bucket = _DecisionBucket(
                class_id=decision["class_id"],
                class_name=decision["class_name"],
                track_state_bucket=decision["track_state_bucket"],
            )
            self._decision_buckets[key] = bucket
        bucket.add(decision, max_samples=self.max_samples_per_bucket)

    def to_summary(self):
        pair_buckets = [bucket.to_summary() for bucket in self._buckets.values()]
        pair_buckets.sort(
            key=lambda item: (
                item["class_id"],
                item["state_bucket"],
                item["class_name"],
            )
        )
        decision_buckets = [
            bucket.to_summary() for bucket in self._decision_buckets.values()
        ]
        decision_buckets.sort(
            key=lambda item: (
                item["class_id"],
                item["track_state_bucket"],
                item["class_name"],
            )
        )
        decision_evidence = any(
            item["total_decisions"] > 0 for item in decision_buckets
        )
        return {
            "schema_version": 2,
            "runtime_manifest": self.runtime_manifest,
            "pair_buckets": pair_buckets,
            # Preserve the schema-v1 key for existing pair-diagnostic readers.
            "buckets": pair_buckets,
            "decision_buckets": decision_buckets,
            "decision_evidence_available": bool(
                decision_evidence and self._manifest_compatible
            ),
            "manifest_compatible": bool(self._manifest_compatible),
            "manifest_mismatches": list(self._manifest_mismatches),
        }

    def export_state(self):
        return self.to_summary()

    def merge_state(self, state):
        if state is None:
            return
        validated = self._validate_merge_state(state)
        manifest_result = self._manifest_merge_result(validated)

        for item in validated["pair_buckets"]:
            key = (
                item["class_id"],
                item["class_name"],
                item["state_bucket"],
            )
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    class_id=item["class_id"],
                    class_name=item["class_name"],
                    state_bucket=item["state_bucket"],
                )
                self._buckets[key] = bucket

            pair_count = item["pair_count"]
            bucket.pair_count += pair_count
            bucket.finite_pair_count += item["finite_pair_count"]
            bucket.active_pair_count += item["active_pair_count"]
            bucket.penalized_pair_count += item["penalized_pair_count"]
            bucket.score_sum += item["avg_score"] * pair_count
            bucket.delta_sum += item["avg_delta"] * pair_count
            bucket.cost_before_sum += item["avg_cost_before"] * pair_count
            bucket.cost_after_sum += item["avg_cost_after"] * pair_count
            # Scene-level states are already summarized; keep bounded proxy
            # points for merged quantiles instead of expanding millions of
            # pairs back into memory.
            for attr, field_name, avg_name in (
                ("scores", "score_quantiles", "avg_score"),
                ("deltas", "delta_quantiles", "avg_delta"),
                ("cost_befores", "cost_before_quantiles", "avg_cost_before"),
                ("cost_afters", "cost_after_quantiles", "avg_cost_after"),
            ):
                quantiles = item.get(field_name, {}) or {}
                values = [
                    quantiles.get("min", item.get(avg_name, 0.0)),
                    quantiles.get("p10", item.get(avg_name, 0.0)),
                    quantiles.get("p50", item.get(avg_name, 0.0)),
                    quantiles.get("p90", item.get(avg_name, 0.0)),
                    quantiles.get("max", item.get(avg_name, 0.0)),
                ]
                getattr(bucket, attr).extend(values)
            _extend_bounded(
                bucket.samples,
                item["samples"],
                self.max_samples_per_bucket,
            )

        for item in validated["decision_buckets"]:
            self._apply_decision_bucket(item)

        (
            self.runtime_manifest,
            self._manifest_compatible,
            self._manifest_mismatches,
        ) = manifest_result

    def _validate_merge_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("audit state must be a mapping")
        if "schema_version" not in state:
            raise ValueError("audit state is missing required schema_version")
        schema_version = _nonnegative_int(
            state["schema_version"], "schema_version"
        )
        if schema_version not in {1, 2}:
            raise ValueError(
                f"schema_version must be 1 or 2, got {schema_version}"
            )

        manifest_present = "runtime_manifest" in state
        if schema_version == 2 and not manifest_present:
            raise ValueError("schema v2 state is missing runtime_manifest")
        runtime_manifest = state.get("runtime_manifest")
        if runtime_manifest is not None:
            runtime_manifest = _json_safe(
                runtime_manifest, path="runtime_manifest"
            )
            if not isinstance(runtime_manifest, dict):
                raise TypeError("runtime_manifest must be a mapping or None")

        if schema_version == 1:
            if state.get("decision_buckets"):
                raise ValueError("schema v1 state cannot contain decision_buckets")
            pair_items = _list_field(state, "buckets", required=True)
            decision_items = []
        else:
            pair_items = _list_field(state, "pair_buckets", required=True)
            decision_items = _list_field(
                state, "decision_buckets", required=True
            )

        return {
            "schema_version": schema_version,
            "manifest_present": manifest_present,
            "runtime_manifest": runtime_manifest,
            "pair_buckets": [
                _validate_pair_bucket_state(item) for item in pair_items
            ],
            "decision_buckets": [
                _validate_decision_bucket_state(item) for item in decision_items
            ],
        }

    def _manifest_merge_result(self, validated):
        if validated["schema_version"] == 1 and not validated["manifest_present"]:
            return (
                self.runtime_manifest,
                self._manifest_compatible,
                list(self._manifest_mismatches),
            )
        incoming = validated["runtime_manifest"]
        if self.runtime_manifest is None and incoming is not None:
            return (
                incoming,
                self._manifest_compatible,
                list(self._manifest_mismatches),
            )
        if runtime_manifests_compatible(self.runtime_manifest, incoming):
            return (
                self.runtime_manifest,
                self._manifest_compatible,
                list(self._manifest_mismatches),
            )

        differences = _manifest_differences(self.runtime_manifest, incoming)
        message = "runtime manifest mismatch: " + "; ".join(differences)
        if self.strict_manifest:
            raise ValueError(message)
        mismatches = list(self._manifest_mismatches)
        for difference in differences:
            if difference not in mismatches:
                mismatches.append(difference)
        return self.runtime_manifest, False, mismatches

    def _apply_decision_bucket(self, item):
        class_id = item["class_id"]
        class_name = item["class_name"]
        state_bucket = item["track_state_bucket"]
        key = (class_id, class_name, state_bucket)
        bucket = self._decision_buckets.get(key)
        if bucket is None:
            bucket = _DecisionBucket(
                class_id=class_id,
                class_name=class_name,
                track_state_bucket=state_bucket,
            )
            self._decision_buckets[key] = bucket

        for field_name in _DECISION_COUNTER_FIELDS:
            setattr(
                bucket,
                field_name,
                getattr(bucket, field_name) + item[field_name],
            )
        _extend_bounded(
            bucket.raw_decisions,
            item["raw_decisions"],
            self.max_samples_per_bucket,
        )
        _extend_bounded(
            bucket.association_scores,
            item["association_scores"],
            self.max_samples_per_bucket,
        )
        _extend_bounded(
            bucket.cost_gaps,
            item["cost_gaps"],
            self.max_samples_per_bucket,
        )

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
