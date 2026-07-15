import json
import math
import os
import shutil
import stat
import struct
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path

import numpy as np

from tracker.association_runtime_manifest import runtime_manifests_compatible


_DECISION_OUTCOMES = {"improved", "damaged", "neutral", "unknown"}
_TRACK_STATE_BUCKETS = {"matched", "unmatched"}
_MANIFEST_UNSET = object()
_HISTOGRAM_TYPE = "fixed_float_histogram_v1"
_HISTOGRAM_BIN_COUNT = 4096
_HISTOGRAM_SHIFT = 52
_UINT64_MASK = (1 << 64) - 1
_FLOAT_SIGN_MASK = 1 << 63
_QUANTILE_KEYS = ("min", "p10", "p50", "p90", "max")
_PAIR_METRICS = ("score", "delta", "cost_before", "cost_after")
_EXACT_SUM_UNIT = "2^-1074"
_EXACT_SUM_DENOMINATOR = 1 << 1074
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
    return 0.0 if result == 0.0 else result


def _finite_float(value, field_name):
    result = _optional_finite_float(value, field_name)
    if result is None:
        raise TypeError(f"{field_name} must be a finite number")
    return result


def _strict_int(value, field_name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(
        value, (int, np.integer)
    ):
        raise TypeError(f"{field_name} must be an integer")
    return int(value)


def _float_to_exact_units(value):
    numerator, denominator = float(value).as_integer_ratio()
    return numerator * (_EXACT_SUM_DENOMINATOR // denominator)


def _exact_average(units, count, field_name):
    if count == 0:
        return 0.0
    try:
        result = float(Fraction(int(units), int(count) * _EXACT_SUM_DENOMINATOR))
    except OverflowError as error:
        raise ValueError(f"{field_name} average must remain finite") from error
    if not np.isfinite(result):
        raise ValueError(f"{field_name} average must remain finite")
    return 0.0 if result == 0.0 else result


def _float_close(left, right):
    return math.isclose(
        float(left),
        float(right),
        rel_tol=1e-15,
        abs_tol=float.fromhex("0x0.0000000000001p-1022"),
    )


def _float_bin(value):
    bits = struct.unpack(">Q", struct.pack(">d", float(value)))[0]
    if bits & _FLOAT_SIGN_MASK:
        ordered = (~bits) & _UINT64_MASK
    else:
        ordered = bits ^ _FLOAT_SIGN_MASK
    return ordered >> _HISTOGRAM_SHIFT


def _bound_matches_histogram_bin(value, index):
    if value == 0.0:
        return index in {_float_bin(-0.0), _float_bin(0.0)}
    return _float_bin(value) == index


def _bin_midpoint(index):
    ordered = (int(index) << _HISTOGRAM_SHIFT) + (1 << (_HISTOGRAM_SHIFT - 1))
    if ordered & _FLOAT_SIGN_MASK:
        bits = ordered ^ _FLOAT_SIGN_MASK
    else:
        bits = (~ordered) & _UINT64_MASK
    return struct.unpack(">d", struct.pack(">Q", bits))[0]


def _finite_lerp(left, right, fraction):
    scale = max(abs(float(left)), abs(float(right)))
    if scale == 0.0:
        return 0.0
    normalized = (
        (float(left) / scale) * (1.0 - float(fraction))
        + (float(right) / scale) * float(fraction)
    )
    return _finite_float(normalized * scale, "quantile interpolation")


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
                differences.append(
                    f"{key_path}: target=<missing>, "
                    f"incoming={right[key]!r} ({type(right[key]).__name__})"
                )
            elif key not in right:
                differences.append(
                    f"{key_path}: target={left[key]!r} "
                    f"({type(left[key]).__name__}), incoming=<missing>"
                )
            else:
                differences.extend(
                    _manifest_differences(left[key], right[key], key_path)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = []
        for index in range(max(len(left), len(right))):
            item_path = f"{path}[{index}]"
            if index >= len(left):
                differences.append(
                    f"{item_path}: target=<missing>, incoming={right[index]!r} "
                    f"({type(right[index]).__name__})"
                )
            elif index >= len(right):
                differences.append(
                    f"{item_path}: target={left[index]!r} "
                    f"({type(left[index]).__name__}), incoming=<missing>"
                )
            else:
                differences.extend(
                    _manifest_differences(left[index], right[index], item_path)
                )
        return differences
    if type(left) is not type(right) or left != right:
        label = path or "runtime_manifest"
        return [
            f"{label}: target={left!r} ({type(left).__name__}), "
            f"incoming={right!r} ({type(right).__name__})"
        ]
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


def _normalize_audit_error(record):
    if not isinstance(record, Mapping):
        raise TypeError("audit error must be a mapping")
    stage = record.get("stage")
    error_type = record.get("type")
    message = record.get("message")
    context = record.get("context", {})
    for field_name, value in (
        ("stage", stage),
        ("type", error_type),
        ("message", message),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"audit error {field_name} must be a non-empty string")
    if not isinstance(context, Mapping):
        raise TypeError("audit error context must be a mapping")
    return {
        "stage": stage.strip(),
        "type": error_type.strip(),
        "message": message.strip(),
        "context": _json_safe(context, path="audit_error.context"),
    }


def _validate_audit_errors(state):
    value = state.get("audit_errors", {"count": state.get("audit_error_count", 0), "samples": []})
    if not isinstance(value, Mapping):
        raise TypeError("audit_errors must be a mapping")
    count = _nonnegative_int(value.get("count", 0), "audit_errors.count")
    samples = _list_field(value, "samples")
    normalized_samples = [_normalize_audit_error(sample) for sample in samples]
    if len(normalized_samples) > count:
        raise ValueError("audit_errors samples cannot exceed count")
    legacy_count = state.get("audit_error_count", count)
    legacy_count = _nonnegative_int(legacy_count, "audit_error_count")
    if legacy_count != count:
        raise ValueError("audit_error_count must equal audit_errors.count")
    return {"count": count, "samples": normalized_samples}


def _copy_xattrs_best_effort(source, destination):
    if not all(
        hasattr(os, name) for name in ("listxattr", "getxattr", "setxattr")
    ):
        return
    try:
        names = os.listxattr(source)
    except OSError:
        return
    for name in names:
        try:
            os.setxattr(destination, name, os.getxattr(source, name))
        except OSError:
            continue


@dataclass
class _FixedFloatHistogram:
    count: int = 0
    minimum: float = None
    maximum: float = None
    counts: list = field(
        default_factory=lambda: [0] * _HISTOGRAM_BIN_COUNT
    )

    def add(self, value):
        value = float(value)
        if value == 0.0:
            value = 0.0
        index = _float_bin(value)
        self.count += 1
        self.counts[index] += 1
        self.minimum = value if self.minimum is None else min(self.minimum, value)
        self.maximum = value if self.maximum is None else max(self.maximum, value)

    def merge(self, other):
        if other.count == 0:
            return
        self.count += other.count
        self.minimum = (
            other.minimum
            if self.minimum is None
            else min(self.minimum, other.minimum)
        )
        self.maximum = (
            other.maximum
            if self.maximum is None
            else max(self.maximum, other.maximum)
        )
        if self.minimum == 0.0:
            self.minimum = 0.0
        if self.maximum == 0.0:
            self.maximum = 0.0
        for index, count in enumerate(other.counts):
            if count:
                self.counts[index] += count

    def _value_at_rank(self, rank):
        if rank <= 0:
            return float(self.minimum)
        if rank >= self.count - 1:
            return float(self.maximum)
        remaining = int(rank)
        for index, count in enumerate(self.counts):
            if remaining < count:
                value = _bin_midpoint(index)
                if not np.isfinite(value):
                    return _finite_lerp(
                        self.minimum,
                        self.maximum,
                        rank / max(self.count - 1, 1),
                    )
                return float(np.clip(value, self.minimum, self.maximum))
            remaining -= count
        raise ValueError("histogram rank exceeds count")

    def quantiles(self):
        if self.count == 0:
            return {}
        result = {"min": float(self.minimum), "max": float(self.maximum)}
        for name, quantile in (("p10", 0.10), ("p50", 0.50), ("p90", 0.90)):
            rank = quantile * (self.count - 1)
            lower_rank = int(np.floor(rank))
            upper_rank = int(np.ceil(rank))
            lower = self._value_at_rank(lower_rank)
            upper = self._value_at_rank(upper_rank)
            result[name] = _finite_lerp(lower, upper, rank - lower_rank)
        return {key: result[key] for key in _QUANTILE_KEYS}

    def export_state(self):
        return {
            "type": _HISTOGRAM_TYPE,
            "bin_count": _HISTOGRAM_BIN_COUNT,
            "count": int(self.count),
            "min": None if self.minimum is None else float(self.minimum),
            "max": None if self.maximum is None else float(self.maximum),
            "bins": [
                [index, int(count)]
                for index, count in enumerate(self.counts)
                if count
            ],
        }


def _validate_histogram_state(value, field_name):
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    required = {"type", "bin_count", "count", "min", "max", "bins"}
    if set(value) != required:
        raise ValueError(
            f"{field_name} fields must be exactly {sorted(required)}"
        )
    if value["type"] != _HISTOGRAM_TYPE:
        raise ValueError(f"{field_name}.type must be {_HISTOGRAM_TYPE}")
    bin_count = _nonnegative_int(value["bin_count"], f"{field_name}.bin_count")
    if bin_count != _HISTOGRAM_BIN_COUNT:
        raise ValueError(
            f"{field_name}.bin_count must be {_HISTOGRAM_BIN_COUNT}"
        )
    count = _nonnegative_int(value["count"], f"{field_name}.count")
    bins = _list_field(value, "bins", required=True)
    histogram = _FixedFloatHistogram(count=count)
    previous_index = -1
    bin_total = 0
    for position, entry in enumerate(bins):
        entry_name = f"{field_name}.bins[{position}]"
        if not isinstance(entry, list) or len(entry) != 2:
            raise TypeError(f"{entry_name} must be a two-item list")
        index = _nonnegative_int(entry[0], f"{entry_name}[0]")
        entry_count = _nonnegative_int(entry[1], f"{entry_name}[1]")
        if index >= _HISTOGRAM_BIN_COUNT:
            raise ValueError(f"{entry_name}[0] exceeds histogram capacity")
        if index <= previous_index:
            raise ValueError(f"{field_name}.bins must be strictly ordered")
        if entry_count == 0:
            raise ValueError(f"{entry_name}[1] must be positive")
        histogram.counts[index] = entry_count
        bin_total += entry_count
        previous_index = index
    if bin_total != count:
        raise ValueError(f"{field_name}.bins counts must sum to count")

    if count == 0:
        if value["min"] is not None or value["max"] is not None or bins:
            raise ValueError(f"{field_name} empty histogram must have null bounds")
        return histogram

    histogram.minimum = _finite_float(value["min"], f"{field_name}.min")
    histogram.maximum = _finite_float(value["max"], f"{field_name}.max")
    if histogram.minimum > histogram.maximum:
        raise ValueError(f"{field_name}.min cannot exceed max")
    if not _bound_matches_histogram_bin(histogram.minimum, bins[0][0]):
        raise ValueError(f"{field_name}.min does not match first occupied bin")
    if not _bound_matches_histogram_bin(histogram.maximum, bins[-1][0]):
        raise ValueError(f"{field_name}.max does not match last occupied bin")
    return histogram


def _validate_pair_bucket_state(item, *, schema_version):
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

    if schema_version == 2:
        if item.get("aggregate_sum_unit") != _EXACT_SUM_UNIT:
            raise ValueError(f"aggregate_sum_unit must be {_EXACT_SUM_UNIT}")
        aggregate_sums = item.get("aggregate_sums")
        if not isinstance(aggregate_sums, Mapping):
            raise TypeError("aggregate_sums must be a mapping")
        if set(aggregate_sums) != set(_PAIR_METRICS):
            raise ValueError("aggregate_sums must contain all pair metrics")
        normalized["aggregate_sums"] = {
            metric: _strict_int(
                aggregate_sums[metric], f"aggregate_sums.{metric}"
            )
            for metric in _PAIR_METRICS
        }
        for metric in _PAIR_METRICS:
            if pair_count == 0 and normalized["aggregate_sums"][metric] != 0:
                raise ValueError(
                    f"aggregate_sums.{metric} must be zero when pair_count is zero"
                )
            expected_average = _exact_average(
                normalized["aggregate_sums"][metric], pair_count, metric
            )
            if not _float_close(normalized[f"avg_{metric}"], expected_average):
                raise ValueError(f"avg_{metric} must match aggregate_sums.{metric}")
    else:
        normalized["aggregate_sums"] = {
            metric: _float_to_exact_units(normalized[f"avg_{metric}"])
            * pair_count
            for metric in _PAIR_METRICS
        }

    if schema_version == 2:
        if item.get("quantile_method") != _HISTOGRAM_TYPE:
            raise ValueError(f"quantile_method must be {_HISTOGRAM_TYPE}")
        complete = _strict_bool(
            item.get("quantile_sketch_complete"), "quantile_sketch_complete"
        )
        sketches = item.get("quantile_sketches")
        if not isinstance(sketches, Mapping):
            raise TypeError("quantile_sketches must be a mapping")
        if set(sketches) != set(_PAIR_METRICS):
            raise ValueError("quantile_sketches must contain all pair metrics")
        normalized_sketches = {
            metric: _validate_histogram_state(
                sketches[metric], f"quantile_sketches.{metric}"
            )
            for metric in _PAIR_METRICS
        }
        for metric, sketch in normalized_sketches.items():
            if sketch.count != pair_count:
                raise ValueError(
                    f"quantile_sketches.{metric}.count must equal pair_count"
                )
            if pair_count:
                average = normalized[f"avg_{metric}"]
                if average < sketch.minimum and not _float_close(
                    average, sketch.minimum
                ):
                    raise ValueError(f"avg_{metric} cannot be below histogram min")
                if average > sketch.maximum and not _float_close(
                    average, sketch.maximum
                ):
                    raise ValueError(f"avg_{metric} cannot exceed histogram max")
        if not complete:
            raise ValueError(
                "quantile_sketch_complete must be true when histogram counts "
                "equal pair_count"
            )
        expected_semantics = (
            "approximate_fixed_histogram_with_exact_min_max"
            if complete
            else "unavailable_incomplete_sketch"
        )
        if item.get("quantile_semantics") != expected_semantics:
            raise ValueError("quantile_semantics does not match sketch completeness")
        normalized["quantile_sketch_complete"] = complete
        normalized["quantile_sketches"] = normalized_sketches
        for metric in _PAIR_METRICS:
            field_name = f"{metric}_quantiles"
            quantiles = item.get(field_name)
            if not isinstance(quantiles, Mapping):
                raise TypeError(f"{field_name} must be a mapping")
            quantiles = _json_safe(quantiles, path=field_name)
            expected = normalized_sketches[metric].quantiles() if complete else {}
            if quantiles != expected:
                raise ValueError(f"{field_name} must match its quantile sketch")
            normalized[field_name] = dict(expected)
    else:
        normalized["quantile_sketch_complete"] = False
        normalized["quantile_sketches"] = {
            metric: _FixedFloatHistogram() for metric in _PAIR_METRICS
        }
        for metric in _PAIR_METRICS:
            normalized[f"{metric}_quantiles"] = {}

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
    if sum(counters[field_name] for field_name in change_categories) != changed:
        raise ValueError(
            "changed assignment categories must equal changed_decisions"
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
    if counters["threshold_crossing_count"] > changed:
        raise ValueError("threshold_crossing_count cannot exceed changed_decisions")

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
    score_sum: int = 0
    delta_sum: int = 0
    cost_before_sum: int = 0
    cost_after_sum: int = 0
    histograms: dict = field(
        default_factory=lambda: {
            metric: _FixedFloatHistogram() for metric in _PAIR_METRICS
        }
    )
    quantile_sketch_complete: bool = True
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
        self.score_sum += _float_to_exact_units(score)
        self.delta_sum += _float_to_exact_units(delta)
        self.cost_before_sum += _float_to_exact_units(cost_before)
        self.cost_after_sum += _float_to_exact_units(cost_after)
        self.histograms["score"].add(score)
        self.histograms["delta"].add(delta)
        self.histograms["cost_before"].add(cost_before)
        self.histograms["cost_after"].add(cost_after)
        if sample is not None and len(self.samples) < int(max_samples):
            self.samples.append(dict(sample))

    def to_summary(self):
        pair_count = int(self.pair_count)
        ratio_count = max(pair_count, 1)
        active_count = max(int(self.active_pair_count), 1)
        complete = bool(
            self.quantile_sketch_complete
            and all(
                histogram.count == self.pair_count
                for histogram in self.histograms.values()
            )
        )
        return {
            "class_id": int(self.class_id),
            "class_name": str(self.class_name),
            "state_bucket": str(self.state_bucket),
            "pair_count": int(self.pair_count),
            "finite_pair_count": int(self.finite_pair_count),
            "active_pair_count": int(self.active_pair_count),
            "penalized_pair_count": int(self.penalized_pair_count),
            "active_ratio": float(self.active_pair_count / ratio_count),
            "penalized_ratio": float(self.penalized_pair_count / active_count),
            "avg_score": _exact_average(self.score_sum, pair_count, "score"),
            "avg_delta": _exact_average(self.delta_sum, pair_count, "delta"),
            "avg_cost_before": _exact_average(
                self.cost_before_sum, pair_count, "cost_before"
            ),
            "avg_cost_after": _exact_average(
                self.cost_after_sum, pair_count, "cost_after"
            ),
            "aggregate_sum_unit": _EXACT_SUM_UNIT,
            "aggregate_sums": {
                "score": int(self.score_sum),
                "delta": int(self.delta_sum),
                "cost_before": int(self.cost_before_sum),
                "cost_after": int(self.cost_after_sum),
            },
            "quantile_method": _HISTOGRAM_TYPE,
            "quantile_semantics": (
                "approximate_fixed_histogram_with_exact_min_max"
                if complete
                else "unavailable_incomplete_sketch"
            ),
            "quantile_sketch_complete": complete,
            "quantile_sketches": {
                metric: histogram.export_state()
                for metric, histogram in self.histograms.items()
            },
            "score_quantiles": (
                self.histograms["score"].quantiles() if complete else {}
            ),
            "delta_quantiles": (
                self.histograms["delta"].quantiles() if complete else {}
            ),
            "cost_before_quantiles": (
                self.histograms["cost_before"].quantiles() if complete else {}
            ),
            "cost_after_quantiles": (
                self.histograms["cost_after"].quantiles() if complete else {}
            ),
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
        runtime_manifest=_MANIFEST_UNSET,
        strict_manifest: bool = False,
    ):
        self._buckets = {}
        self._decision_buckets = {}
        self.max_samples_per_bucket = int(max_samples_per_bucket)
        self._manifest_initialized = runtime_manifest is not _MANIFEST_UNSET
        manifest_value = (
            None if runtime_manifest is _MANIFEST_UNSET else runtime_manifest
        )
        self.runtime_manifest = (
            None
            if manifest_value is None
            else _json_safe(manifest_value, path="runtime_manifest")
        )
        if self.runtime_manifest is not None and not isinstance(
            self.runtime_manifest, dict
        ):
            raise TypeError("runtime_manifest must be a mapping or None")
        self.strict_manifest = bool(strict_manifest)
        self._manifest_compatible = True
        self._manifest_mismatches = []
        self._decision_evidence_complete = True
        self._decision_evidence_incomplete_reasons = []
        self._audit_error_count = 0
        self._audit_error_samples = []

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
        self._manifest_initialized = True

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
        self._manifest_initialized = True

    def add_error(
        self,
        error=None,
        *,
        stage=None,
        error_type=None,
        message=None,
        context=None,
    ) -> None:
        if isinstance(error, Mapping):
            if any(value is not None for value in (stage, error_type, message, context)):
                raise TypeError("mapping audit errors cannot include keyword fields")
            record = error
        else:
            if isinstance(error, BaseException):
                derived_type = type(error).__name__
                derived_message = str(error) or derived_type
            elif error is None:
                derived_type = error_type
                derived_message = message
            else:
                derived_type = type(error).__name__
                derived_message = str(error) or derived_type
            record = {
                "stage": stage,
                "type": error_type or derived_type,
                "message": message or derived_message,
                "context": {} if context is None else context,
            }
        normalized = _normalize_audit_error(record)
        self._audit_error_count += 1
        _extend_bounded(
            self._audit_error_samples,
            [normalized],
            self.max_samples_per_bucket,
        )
        self._decision_evidence_complete = False
        if "audit_errors" not in self._decision_evidence_incomplete_reasons:
            self._decision_evidence_incomplete_reasons.append("audit_errors")
        self._manifest_initialized = True

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
        summary = {
            "schema_version": 2,
            "runtime_manifest": self.runtime_manifest,
            "manifest_initialized": bool(self._manifest_initialized),
            "pair_buckets": pair_buckets,
            # Preserve the schema-v1 key for existing pair-diagnostic readers.
            "buckets": pair_buckets,
            "decision_buckets": decision_buckets,
            "decision_evidence_available": bool(
                decision_evidence
                and self._manifest_compatible
                and self._decision_evidence_complete
            ),
            "decision_evidence_complete": bool(
                self._decision_evidence_complete
            ),
            "decision_evidence_incomplete_reasons": list(
                self._decision_evidence_incomplete_reasons
            ),
            "manifest_compatible": bool(self._manifest_compatible),
            "manifest_mismatches": list(self._manifest_mismatches),
            "audit_error_count": int(self._audit_error_count),
            "audit_errors": {
                "count": int(self._audit_error_count),
                "sample_count": int(len(self._audit_error_samples)),
                "samples": list(self._audit_error_samples),
            },
        }
        return _json_safe(summary, path="summary")

    def export_state(self):
        return self.to_summary()

    def merge_state(self, state):
        if state is None:
            return
        validated = self._validate_merge_state(state)
        if validated["no_op"]:
            return
        pair_merge_plan = self._prepare_pair_merge(
            validated["pair_buckets"]
        )
        manifest_result = self._manifest_merge_result(validated)
        evidence_result = self._decision_evidence_merge_result(
            validated, manifest_result
        )
        next_audit_error_count = (
            self._audit_error_count + validated["audit_errors"]["count"]
        )

        for item, aggregate_sums in pair_merge_plan:
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
            bucket.score_sum = aggregate_sums["score_sum"]
            bucket.delta_sum = aggregate_sums["delta_sum"]
            bucket.cost_before_sum = aggregate_sums["cost_before_sum"]
            bucket.cost_after_sum = aggregate_sums["cost_after_sum"]
            for metric in _PAIR_METRICS:
                bucket.histograms[metric].merge(
                    item["quantile_sketches"][metric]
                )
            bucket.quantile_sketch_complete = bool(
                bucket.quantile_sketch_complete
                and item["quantile_sketch_complete"]
            )
            _extend_bounded(
                bucket.samples,
                item["samples"],
                self.max_samples_per_bucket,
            )

        if manifest_result["discard_decisions"]:
            self._decision_buckets = {}
        elif manifest_result["accept_decisions"]:
            for item in validated["decision_buckets"]:
                self._apply_decision_bucket(item)

        self.runtime_manifest = manifest_result["runtime_manifest"]
        self._manifest_initialized = manifest_result["manifest_initialized"]
        self._manifest_compatible = manifest_result["manifest_compatible"]
        self._manifest_mismatches = manifest_result["manifest_mismatches"]
        self._decision_evidence_complete = evidence_result["complete"]
        self._decision_evidence_incomplete_reasons = evidence_result["reasons"]
        self._audit_error_count = next_audit_error_count
        _extend_bounded(
            self._audit_error_samples,
            validated["audit_errors"]["samples"],
            self.max_samples_per_bucket,
        )

    def _validate_merge_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("audit state must be a mapping")
        state = _json_safe(state, path="audit_state")
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
        audit_errors = _validate_audit_errors(state)

        manifest_initialized_present = "manifest_initialized" in state
        if manifest_initialized_present:
            manifest_initialized = _strict_bool(
                state["manifest_initialized"], "manifest_initialized"
            )
        else:
            # Schema v1 predates this field; schema v2 states without it used
            # runtime_manifest as an implicit initialized baseline.
            manifest_initialized = schema_version == 2
        has_bucket_data = bool(
            pair_items or decision_items or audit_errors["count"]
        )
        if (
            manifest_initialized_present
            and not manifest_initialized
            and has_bucket_data
        ):
            raise ValueError(
                "manifest_initialized=false state cannot contain pair or "
                "decision data"
            )
        no_op = bool(
            manifest_initialized_present
            and not manifest_initialized
            and not has_bucket_data
        )

        pair_buckets = [
            _validate_pair_bucket_state(item, schema_version=schema_version)
            for item in pair_items
        ]
        decision_buckets = [
            _validate_decision_bucket_state(item) for item in decision_items
        ]
        source_manifest_compatible = state.get("manifest_compatible", True)
        source_manifest_compatible = _strict_bool(
            source_manifest_compatible, "manifest_compatible"
        )
        source_manifest_mismatches = _list_field(
            state, "manifest_mismatches"
        )
        if not all(isinstance(item, str) for item in source_manifest_mismatches):
            raise TypeError("manifest_mismatches must contain strings")
        decision_total = sum(
            item["total_decisions"] for item in decision_buckets
        )
        source_evidence_complete = state.get(
            "decision_evidence_complete",
            state.get("decision_evidence_available", decision_total > 0),
        )
        source_evidence_complete = _strict_bool(
            source_evidence_complete, "decision_evidence_complete"
        )
        source_evidence_reasons = _list_field(
            state, "decision_evidence_incomplete_reasons"
        )
        if not all(isinstance(item, str) for item in source_evidence_reasons):
            raise TypeError(
                "decision_evidence_incomplete_reasons must contain strings"
            )

        return {
            "schema_version": schema_version,
            "manifest_present": manifest_present,
            "manifest_initialized": manifest_initialized,
            "no_op": no_op,
            "runtime_manifest": runtime_manifest,
            "manifest_compatible": source_manifest_compatible,
            "manifest_mismatches": list(source_manifest_mismatches),
            "decision_evidence_complete": source_evidence_complete,
            "decision_evidence_incomplete_reasons": list(
                source_evidence_reasons
            ),
            "decision_total": decision_total,
            "pair_buckets": pair_buckets,
            "decision_buckets": decision_buckets,
            "audit_errors": audit_errors,
        }

    def _prepare_pair_merge(self, pair_items):
        staged_sums = {}
        plan = []
        aggregate_fields = tuple(
            (metric, f"{metric}_sum") for metric in _PAIR_METRICS
        )
        for item in pair_items:
            key = (
                item["class_id"],
                item["class_name"],
                item["state_bucket"],
            )
            if key not in staged_sums:
                bucket = self._buckets.get(key)
                staged_sums[key] = {
                    sum_name: 0 if bucket is None else getattr(bucket, sum_name)
                    for _, sum_name in aggregate_fields
                }

            next_sums = dict(staged_sums[key])
            for label, sum_name in aggregate_fields:
                increment = item["aggregate_sums"][label]
                next_sums[sum_name] += increment
            staged_sums[key] = next_sums
            plan.append((item, next_sums))
        return plan

    def _manifest_merge_result(self, validated):
        if not validated["manifest_initialized"]:
            return {
                "runtime_manifest": self.runtime_manifest,
                "manifest_initialized": self._manifest_initialized,
                "manifest_compatible": self._manifest_compatible,
                "manifest_mismatches": list(self._manifest_mismatches),
                "accept_decisions": False,
                "discard_decisions": not self._manifest_compatible,
            }

        incoming = validated["runtime_manifest"]
        if self._manifest_initialized:
            differences = _manifest_differences(
                self.runtime_manifest, incoming
            )
            direct_compatible = bool(
                runtime_manifests_compatible(self.runtime_manifest, incoming)
                and not differences
            )
            baseline = self.runtime_manifest
        else:
            differences = []
            direct_compatible = True
            baseline = incoming

        source_compatible = validated["manifest_compatible"]
        if self.strict_manifest and (not direct_compatible or not source_compatible):
            details = list(differences) + list(validated["manifest_mismatches"])
            if not details:
                details = ["incoming state reports manifest_compatible=False"]
            raise ValueError("runtime manifest mismatch: " + "; ".join(details))

        mismatches = list(self._manifest_mismatches)
        for difference in differences + validated["manifest_mismatches"]:
            if difference not in mismatches:
                mismatches.append(difference)
        incoming_compatible = bool(direct_compatible and source_compatible)
        globally_compatible = bool(
            self._manifest_compatible and incoming_compatible
        )
        return {
            "runtime_manifest": baseline,
            "manifest_initialized": True,
            "manifest_compatible": globally_compatible,
            "manifest_mismatches": mismatches,
            "accept_decisions": globally_compatible,
            "discard_decisions": not globally_compatible,
        }

    def _decision_evidence_merge_result(self, validated, manifest_result):
        complete = self._decision_evidence_complete
        reasons = list(self._decision_evidence_incomplete_reasons)

        def mark_incomplete(reason):
            nonlocal complete
            complete = False
            if reason not in reasons:
                reasons.append(reason)

        if validated["schema_version"] == 1:
            mark_incomplete("schema_v1_state")
        else:
            if validated["audit_errors"]["count"] > 0:
                mark_incomplete("audit_errors")
            if validated["decision_total"] == 0:
                mark_incomplete("pair_only_state")
            if not validated["decision_evidence_complete"]:
                incoming_reasons = validated[
                    "decision_evidence_incomplete_reasons"
                ]
                if incoming_reasons:
                    for reason in incoming_reasons:
                        mark_incomplete(reason)
                else:
                    mark_incomplete("upstream_incomplete_evidence")
            if not manifest_result["accept_decisions"]:
                mark_incomplete("manifest_mismatch")

        return {"complete": complete, "reasons": reasons}

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
        try:
            target_stat = path.stat()
        except FileNotFoundError:
            target_stat = None
        payload = json.dumps(
            self.to_summary(), indent=2, sort_keys=True, allow_nan=False
        )
        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            if target_stat is not None:
                try:
                    shutil.copystat(path, temporary_path)
                except OSError:
                    pass
                _copy_xattrs_best_effort(path, temporary_path)
                os.chmod(temporary_path, stat.S_IMODE(target_stat.st_mode))
                with temporary_path.open("rb") as handle:
                    os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
