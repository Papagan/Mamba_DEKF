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
        key = (int(class_id), str(class_name), str(state_bucket))
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                class_id=int(class_id),
                class_name=str(class_name),
                state_bucket=str(state_bucket),
            )
            self._buckets[key] = bucket
        bucket.add(
            score=float(score),
            delta=float(delta),
            cost_before=float(cost_before),
            cost_after=float(cost_after),
            active=bool(active),
            finite=bool(finite),
            sample=sample,
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
        if not state:
            return
        if not isinstance(state, Mapping):
            raise TypeError("audit state must be a mapping")
        self._merge_manifest(state)
        pair_items = state.get("pair_buckets")
        if pair_items is None:
            pair_items = state.get("buckets", [])
        for item in pair_items:
            key = (
                int(item["class_id"]),
                str(item["class_name"]),
                str(item["state_bucket"]),
            )
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    class_id=int(item["class_id"]),
                    class_name=str(item["class_name"]),
                    state_bucket=str(item["state_bucket"]),
                )
                self._buckets[key] = bucket

            pair_count = int(item.get("pair_count", 0))
            bucket.pair_count += pair_count
            bucket.finite_pair_count += int(item.get("finite_pair_count", 0))
            bucket.active_pair_count += int(item.get("active_pair_count", 0))
            bucket.penalized_pair_count += int(item.get("penalized_pair_count", 0))
            bucket.score_sum += float(item.get("avg_score", 0.0)) * pair_count
            bucket.delta_sum += float(item.get("avg_delta", 0.0)) * pair_count
            bucket.cost_before_sum += float(item.get("avg_cost_before", 0.0)) * pair_count
            bucket.cost_after_sum += float(item.get("avg_cost_after", 0.0)) * pair_count
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
                getattr(bucket, attr).extend(float(value) for value in values)
            samples = [
                _json_safe(sample, path="pair_bucket.samples")
                for sample in item.get("samples", [])
            ]
            _extend_bounded(bucket.samples, samples, self.max_samples_per_bucket)

        for item in state.get("decision_buckets", []):
            self._merge_decision_bucket(item)

    def _merge_manifest(self, state):
        is_legacy = int(state.get("schema_version", 1)) == 1
        if is_legacy and "runtime_manifest" not in state:
            return

        incoming = state.get("runtime_manifest")
        if incoming is not None:
            incoming = _json_safe(incoming, path="runtime_manifest")
            if not isinstance(incoming, dict):
                raise TypeError("runtime_manifest must be a mapping or None")

        if self.runtime_manifest is None and incoming is not None:
            self.runtime_manifest = incoming
            return
        if runtime_manifests_compatible(self.runtime_manifest, incoming):
            return

        differences = _manifest_differences(self.runtime_manifest, incoming)
        message = "runtime manifest mismatch: " + "; ".join(differences)
        if self.strict_manifest:
            raise ValueError(message)
        self._manifest_compatible = False
        for difference in differences:
            if difference not in self._manifest_mismatches:
                self._manifest_mismatches.append(difference)

    def _merge_decision_bucket(self, item):
        if not isinstance(item, Mapping):
            raise TypeError("decision bucket state must be a mapping")
        class_id = _nonnegative_int(item.get("class_id"), "class_id")
        class_name = item.get("class_name")
        state_bucket = item.get("track_state_bucket")
        if not isinstance(class_name, str) or not class_name:
            raise ValueError("decision bucket class_name must be non-empty")
        if state_bucket not in _TRACK_STATE_BUCKETS:
            raise ValueError("decision bucket track_state_bucket is invalid")
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
            value = _nonnegative_int(item.get(field_name, 0), field_name)
            setattr(bucket, field_name, getattr(bucket, field_name) + value)

        raw_decisions = item.get("raw_decisions", item.get("samples", []))
        normalized_decisions = [
            _normalize_decision(record) for record in raw_decisions
        ]
        score_samples = [
            _json_safe(value, path="decision_bucket.association_scores")
            for value in item.get("association_scores", [])
        ]
        cost_gaps = [
            float(_json_safe(value, path="decision_bucket.cost_gaps"))
            for value in item.get("cost_gaps", [])
        ]
        _extend_bounded(
            bucket.raw_decisions,
            normalized_decisions,
            self.max_samples_per_bucket,
        )
        _extend_bounded(
            bucket.association_scores,
            score_samples,
            self.max_samples_per_bucket,
        )
        _extend_bounded(bucket.cost_gaps, cost_gaps, self.max_samples_per_bucket)

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )
