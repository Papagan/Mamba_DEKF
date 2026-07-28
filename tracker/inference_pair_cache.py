"""Pure construction of inference-time association candidate groups."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from collections.abc import Mapping
from numbers import Integral, Real
from pathlib import Path

import numpy as np


INFERENCE_PAIR_CACHE_SCHEMA_VERSION = 1
TRACK_STATES = {"matched", "unmatched"}

_REQUIRED_PAYLOAD_FIELDS = (
    "scene_id",
    "sample_token",
    "frame_id",
    "track_ids",
    "class_ids",
    "class_names",
    "state_buckets",
    "track_gt_tokens",
    "det_gt_tokens",
    "pair_indices",
    "pair_vectors",
    "track_embeddings",
    "det_embeddings",
)


def _native_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _nonempty_string(name, value):
    value = _native_scalar(value)
    if type(value) is not str or not value.strip() or value.strip() != value:
        raise ValueError(f"{name} must be a nonblank trimmed built-in string")
    return value


def _frame_id(value):
    value = _native_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError("frame_id must be a non-negative integer")
    value = int(value)
    if value < 0:
        raise ValueError("frame_id must be a non-negative integer")
    return value


def _sequence(name, values, expected_count=None):
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional sequence")
    if expected_count is not None and array.shape[0] != expected_count:
        raise ValueError(f"{name} must contain {expected_count} values")
    return [_native_scalar(value) for value in array]


def _nonnegative_integers(name, values, expected_count=None, *, unique=False):
    normalized = []
    for value in _sequence(name, values, expected_count):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
            raise ValueError(f"{name} must contain non-negative integers")
        value = int(value)
        if value < 0:
            raise ValueError(f"{name} must contain non-negative integers")
        normalized.append(value)
    if unique and len(set(normalized)) != len(normalized):
        raise ValueError(f"{name} must contain unique values")
    return normalized


def _class_names(values, expected_count):
    normalized = _sequence("class_names", values, expected_count)
    return [_nonempty_string("class_names", value) for value in normalized]


def _states(values, expected_count):
    normalized = _sequence("state_buckets", values, expected_count)
    if any(type(value) is not str or value not in TRACK_STATES for value in normalized):
        raise ValueError("state_buckets values must be matched or unmatched")
    return normalized


def _tokens(name, values, expected_count):
    normalized = _sequence(name, values, expected_count)
    for value in normalized:
        if value is not None and (
            type(value) is not str or not value or value.strip() != value
        ):
            raise ValueError(f"{name} must contain canonical string tokens or None")
    return normalized


def _matrix(name, value, *, shape=None, finite=False, keep_original=False):
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError(f"{name} must contain real numeric values")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} must have shape {shape}, got {array.shape}")
    normalized = _normalized_float64_array(name, array, finite=finite)
    if keep_original:
        return normalized, np.array(array, copy=True)
    return normalized


def _finite_python_float(name, value):
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} cannot convert to Python float") from exc
    if not np.isfinite(converted):
        raise ValueError(f"{name} must convert to a finite Python float")
    if converted == 0.0 and value != 0:
        raise ValueError(f"{name} underflows Python float")
    return 0.0 if converted == 0.0 else converted


def _normalized_float64_array(name, array, *, finite):
    original_finite = np.isfinite(array)
    if finite and not np.all(original_finite):
        raise ValueError(f"{name} must contain only finite values")
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        normalized = array.astype(np.float64, copy=True)
    if np.any(original_finite & ~np.isfinite(normalized)):
        raise ValueError(f"{name} must convert to finite Python float values")
    if np.any(original_finite & (array != 0) & (normalized == 0.0)):
        raise ValueError(f"{name} underflows Python float")
    normalized[normalized == 0.0] = 0.0
    return normalized


def _ranked_finite_entries(base_costs, original_base_costs, track_index, pair_entries):
    entries = []
    originals_by_float = {}
    for det_index, _ in pair_entries:
        converted = base_costs[track_index, det_index]
        if not np.isfinite(converted):
            continue
        converted = converted.item()
        original = original_base_costs[track_index, det_index]
        if (
            converted in originals_by_float
            and original != originals_by_float[converted]
        ):
            raise ValueError(
                "base_cost_matrix has distinct candidate costs that convert to "
                "the same Python float"
            )
        originals_by_float[converted] = original
        entries.append((converted, det_index))
    return sorted(entries, key=lambda item: (item[0], item[1]))


def _contains_raw_boolean(value):
    if isinstance(value, (bool, np.bool_)):
        return True
    if isinstance(value, np.ndarray):
        if np.issubdtype(value.dtype, np.bool_):
            return bool(value.size)
        if value.dtype == object:
            return any(_contains_raw_boolean(item) for item in value.flat)
        return False
    if isinstance(value, (list, tuple)):
        return any(_contains_raw_boolean(item) for item in value)
    return False


def _index_pairs(name, value):
    if _contains_raw_boolean(value):
        raise TypeError(f"{name} must not contain boolean indices")
    array = np.asarray(value)
    if array.ndim == 1 and array.size == 0:
        return np.empty((0, 2), dtype=np.int64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must contain integer indices")
    return np.array(array, dtype=np.int64, copy=True)


def _validate_pair_indices(pair_indices, track_count, det_count):
    seen = set()
    for track_index, det_index in pair_indices:
        track_index = int(track_index)
        det_index = int(det_index)
        pair = (track_index, det_index)
        if pair in seen:
            raise ValueError(f"pair_indices contains duplicate pair {pair}")
        if track_index < 0 or track_index >= track_count:
            raise ValueError(
                f"pair_indices track index {track_index} is outside [0, {track_count})"
            )
        if det_index < 0 or det_index >= det_count:
            raise ValueError(
                f"pair_indices detection index {det_index} is outside [0, {det_count})"
            )
        seen.add(pair)
    return seen


def _production_cost_array(value, match_count):
    array = np.asarray(value)
    if array.ndim != 1 or array.shape[0] != match_count:
        raise ValueError(f"production_costs must have shape ({match_count},)")
    if not np.issubdtype(array.dtype, np.number) or np.issubdtype(
        array.dtype, np.complexfloating
    ):
        raise TypeError("production_costs must contain real numeric values")
    return _normalized_float64_array("production_costs", array, finite=True)


def _assignment_map(matches, costs, track_count, det_count, candidate_pairs):
    result = {}
    seen_detections = set()
    for (track_index, det_index), cost in zip(matches, costs):
        track_index = int(track_index)
        det_index = int(det_index)
        if track_index < 0 or track_index >= track_count:
            raise ValueError(
                "production_matches track index "
                f"{track_index} is outside [0, {track_count})"
            )
        if det_index < 0 or det_index >= det_count:
            raise ValueError(
                "production_matches detection index "
                f"{det_index} is outside [0, {det_count})"
            )
        if track_index in result:
            raise ValueError(
                f"duplicate production assignment for track index {track_index}"
            )
        if det_index in seen_detections:
            raise ValueError(
                f"duplicate production assignment for detection index {det_index}"
            )
        if (track_index, det_index) not in candidate_pairs:
            raise ValueError(
                "production match "
                f"({track_index}, {det_index}) is not represented by pair_indices"
            )
        result[track_index] = (det_index, cost.item())
        seen_detections.add(det_index)
    return result


def _thresholds_by_class(thresholds, class_ids):
    if not isinstance(thresholds, Mapping):
        raise TypeError("thresholds must be a mapping")

    normalized = {}
    for raw_class_id, raw_threshold in thresholds.items():
        class_id = _native_scalar(raw_class_id)
        if isinstance(class_id, (bool, np.bool_)) or not isinstance(class_id, Integral):
            raise ValueError("threshold keys must be non-negative integer class ids")
        class_id = int(class_id)
        if class_id < 0:
            raise ValueError("threshold keys must be non-negative integer class ids")
        raw_threshold = _native_scalar(raw_threshold)
        if isinstance(raw_threshold, (bool, np.bool_)) or not isinstance(
            raw_threshold, Real
        ):
            raise TypeError(
                f"threshold for class id {class_id} must be a real numeric scalar"
            )
        threshold = _finite_python_float(
            f"threshold for class id {class_id}", raw_threshold
        )
        normalized[class_id] = threshold

    for class_id in set(class_ids):
        if class_id not in normalized:
            raise ValueError(f"thresholds is missing class id {class_id}")
    return normalized


def _candidate_target(track_gt_token, pair_entries, det_gt_tokens):
    if track_gt_token is None:
        return "unknown", None
    if any(det_gt_tokens[det_index] is None for det_index, _ in pair_entries):
        return "unknown", None
    matching_det_indices = [
        det_index
        for det_index, _ in pair_entries
        if det_gt_tokens[det_index] == track_gt_token
    ]
    if not matching_det_indices:
        return "no_match", None
    if len(matching_det_indices) > 1:
        return "unknown", None
    return "matched", matching_det_indices[0]


def build_candidate_groups(
    *,
    payload,
    base_cost_matrix,
    production_matches,
    production_costs,
    thresholds,
    include_embeddings=True,
    include_pair_vectors=True,
):
    """Build JSON/pickle-safe candidate groups without mutating inputs."""
    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    missing_fields = [name for name in _REQUIRED_PAYLOAD_FIELDS if name not in payload]
    if missing_fields:
        raise ValueError(f"payload is missing required field {missing_fields[0]}")
    for name, value in (
        ("include_embeddings", include_embeddings),
        ("include_pair_vectors", include_pair_vectors),
    ):
        if not isinstance(value, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean")

    track_ids = _nonnegative_integers("track_ids", payload["track_ids"], unique=True)
    track_count = len(track_ids)
    class_ids = _nonnegative_integers("class_ids", payload["class_ids"], track_count)
    class_names = _class_names(payload["class_names"], track_count)
    states = _states(payload["state_buckets"], track_count)
    track_gt_tokens = _tokens(
        "track_gt_tokens", payload["track_gt_tokens"], track_count
    )

    track_embeddings = _matrix(
        "track_embeddings", payload["track_embeddings"], finite=True
    )
    if track_embeddings.shape[0] != track_count:
        raise ValueError(
            f"track_embeddings must have {track_count} rows, "
            f"got {track_embeddings.shape[0]}"
        )
    if track_embeddings.shape[1] == 0:
        raise ValueError("embedding width must be positive")
    det_embeddings = _matrix("det_embeddings", payload["det_embeddings"], finite=True)
    det_count = det_embeddings.shape[0]
    if det_embeddings.shape[1] != track_embeddings.shape[1]:
        raise ValueError(
            "track_embeddings and det_embeddings must have the same embedding width"
        )
    det_gt_tokens = _tokens("det_gt_tokens", payload["det_gt_tokens"], det_count)

    base_costs, original_base_costs = _matrix(
        "base_cost_matrix",
        base_cost_matrix,
        shape=(track_count, det_count),
        keep_original=True,
    )
    pair_indices = _index_pairs("pair_indices", payload["pair_indices"])
    candidate_pairs = _validate_pair_indices(pair_indices, track_count, det_count)
    pair_vectors = _matrix("pair_vectors", payload["pair_vectors"], finite=True)
    if pair_vectors.shape[0] != pair_indices.shape[0]:
        raise ValueError(
            "pair_vectors must contain one row per pair_indices entry "
            f"({pair_indices.shape[0]})"
        )
    if pair_vectors.shape[1] == 0:
        raise ValueError("pair_vectors width must be positive")

    scores_value = payload.get("association_scores")
    scores = (
        None
        if scores_value is None
        else _matrix("association_scores", scores_value, shape=(track_count, det_count))
    )
    threshold_by_class = _thresholds_by_class(thresholds, class_ids)

    matches = _index_pairs("production_matches", production_matches)
    costs = _production_cost_array(production_costs, matches.shape[0])
    assignments = _assignment_map(
        matches, costs, track_count, det_count, candidate_pairs
    )
    for track_index, (det_index, _) in assignments.items():
        if not np.isfinite(base_costs[track_index, det_index]):
            raise ValueError(
                "production assignment base_cost_matrix value must be finite"
            )

    pairs_by_track = {track_index: [] for track_index in range(track_count)}
    for pair_position, (track_index, det_index) in enumerate(pair_indices):
        pairs_by_track[int(track_index)].append((int(det_index), pair_position))

    scene_id = _nonempty_string("scene_id", payload["scene_id"])
    sample_token = _nonempty_string("sample_token", payload["sample_token"])
    frame_id = _frame_id(payload["frame_id"])

    groups = []
    for track_index in range(track_count):
        pair_entries = sorted(pairs_by_track[track_index])
        finite_entries = _ranked_finite_entries(
            base_costs, original_base_costs, track_index, pair_entries
        )
        ranks = {
            det_index: rank
            for rank, (_, det_index) in enumerate(finite_entries, start=1)
        }
        best_cost = finite_entries[0][0] if finite_entries else None
        target_status, target_det_index = _candidate_target(
            track_gt_tokens[track_index], pair_entries, det_gt_tokens
        )

        assignment = assignments.get(track_index)
        selected_det_index = None if assignment is None else assignment[0]
        threshold = threshold_by_class[class_ids[track_index]]
        candidates = []
        for det_index, pair_position in pair_entries:
            base_cost = base_costs[track_index, det_index]
            finite = bool(np.isfinite(base_cost))
            raw_cost = base_cost.item() if finite else None
            cost_gap = None
            if finite and best_cost is not None:
                cost_gap = _finite_python_float("cost_gap", raw_cost - best_cost)
            association_score = None
            if scores is not None and np.isfinite(scores[track_index, det_index]):
                association_score = scores[track_index, det_index].item()
            candidate = {
                "det_index": det_index,
                "geometric_cost": raw_cost if finite else None,
                "geometric_threshold": threshold,
                "finite": finite,
                "inside_geometric_gate": bool(finite and raw_cost <= threshold),
                "candidate_rank": ranks.get(det_index),
                "best_cost": best_cost,
                "cost_gap": cost_gap,
                "association_score": association_score,
                "production_selected": det_index == selected_det_index,
                "det_gt_token": det_gt_tokens[det_index],
                "gt_label": (
                    1
                    if target_status == "matched" and det_index == target_det_index
                    else 0
                    if target_status in {"matched", "no_match"}
                    else None
                ),
            }
            if include_embeddings:
                candidate["det_embedding"] = det_embeddings[det_index].tolist()
            if include_pair_vectors:
                candidate["pair_vector"] = pair_vectors[pair_position].tolist()
            candidates.append(candidate)

        group = {
            "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "scene_id": scene_id,
            "sample_token": sample_token,
            "frame_id": frame_id,
            "track_id": track_ids[track_index],
            "class_id": class_ids[track_index],
            "class_name": class_names[track_index],
            "state_bucket": states[track_index],
            "track_gt_token": track_gt_tokens[track_index],
            "production_det_index": selected_det_index,
            "production_match_cost": None if assignment is None else assignment[1],
            "candidate_count": len(candidates),
            "finite_candidate_count": len(finite_entries),
            "gt_target_det_index": target_det_index,
            "gt_target_status": target_status,
            "candidates": candidates,
        }
        if include_embeddings:
            group["track_embedding"] = track_embeddings[track_index].tolist()
        groups.append(group)

    return groups


_GROUP_FIELDS = {
    "schema_version",
    "scene_id",
    "sample_token",
    "frame_id",
    "track_id",
    "class_id",
    "class_name",
    "state_bucket",
    "track_gt_token",
    "production_det_index",
    "production_match_cost",
    "candidate_count",
    "finite_candidate_count",
    "gt_target_det_index",
    "gt_target_status",
    "candidates",
}
_CANDIDATE_FIELDS = {
    "det_index",
    "geometric_cost",
    "geometric_threshold",
    "finite",
    "inside_geometric_gate",
    "candidate_rank",
    "best_cost",
    "cost_gap",
    "association_score",
    "production_selected",
    "det_gt_token",
    "gt_label",
}
_ERROR_FIELDS = {
    "scene_id",
    "sample_token",
    "frame_id",
    "stage",
    "error_type",
    "message",
}
_STATE_FIELDS = {
    "schema_version",
    "runtime_manifest",
    "observed_metadata",
    "groups",
    "error_count",
    "error_samples",
}
_OBSERVED_METADATA_FIELDS = {
    "scene_id",
    "sample_token",
    "frame_id",
    "track_id",
    "fingerprint",
    "class_id",
    "class_name",
    "state_bucket",
    "gt_target_status",
}
_GROUP_STATUSES = {"matched", "no_match", "unknown"}
_PICKLE_PROTOCOL = 4


def _require_exact_fields(name, value, required, optional=()):
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    keys = set(value.keys())
    allowed = set(required) | set(optional)
    missing = set(required) - keys
    extra = keys - allowed
    if missing:
        raise ValueError(f"{name} is missing required field {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"{name} has unexpected field {sorted(extra, key=str)[0]}")


def _strict_nonnegative_integer(name, value):
    value = _native_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise TypeError(f"{name} must be a non-negative integer")
    normalized = int(value)
    if normalized < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return normalized


def _strict_positive_integer(name, value):
    normalized = _strict_nonnegative_integer(name, value)
    if normalized == 0:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _strict_bool(name, value):
    if type(value) is not bool:
        raise TypeError(f"{name} must be a built-in bool")
    return value


def _canonical_string(name, value, *, allow_empty=False):
    if type(value) is not str:
        raise TypeError(f"{name} must be a built-in string")
    if value.strip() != value or (not allow_empty and not value):
        raise ValueError(f"{name} must be a canonical string")
    return value


def _optional_canonical_string(name, value):
    if value is None:
        return None
    return _canonical_string(name, value)


def _finite_number(name, value):
    value = _native_scalar(value)
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a finite real number")
    return _finite_python_float(name, value)


def _optional_finite_number(name, value):
    return None if value is None else _finite_number(name, value)


def _optional_nonnegative_integer(name, value):
    return None if value is None else _strict_nonnegative_integer(name, value)


def _finite_vector(name, value):
    if type(value) is not list or not value:
        raise ValueError(f"{name} must be a nonempty built-in list")
    return [
        _finite_number(f"{name}[{index}]", item)
        for index, item in enumerate(value)
    ]


def _group_key(group):
    return (
        group["scene_id"],
        group["sample_token"],
        group["frame_id"],
        group["track_id"],
    )


def _group_fingerprint(group):
    return hashlib.sha256(canonical_json(group).encode("utf-8")).hexdigest()


def _metadata_from_group(group):
    return {
        "scene_id": group["scene_id"],
        "sample_token": group["sample_token"],
        "frame_id": group["frame_id"],
        "track_id": group["track_id"],
        "fingerprint": _group_fingerprint(group),
        "class_id": group["class_id"],
        "class_name": group["class_name"],
        "state_bucket": group["state_bucket"],
        "gt_target_status": group["gt_target_status"],
    }


def _derived_gt_target(track_gt_token, candidates):
    if not candidates:
        status = "no_match" if track_gt_token is not None else "unknown"
        return status, None, []
    if track_gt_token is None or any(
        candidate["det_gt_token"] is None for candidate in candidates
    ):
        return "unknown", None, [None] * len(candidates)

    matching_positions = [
        position
        for position, candidate in enumerate(candidates)
        if candidate["det_gt_token"] == track_gt_token
    ]
    if len(matching_positions) == 1:
        target_position = matching_positions[0]
        return (
            "matched",
            candidates[target_position]["det_index"],
            [int(position == target_position) for position in range(len(candidates))],
        )
    if not matching_positions:
        return "no_match", None, [0] * len(candidates)
    return "unknown", None, [None] * len(candidates)


def validate_candidate_group(group):
    """Strictly normalize one Task 2 group to independent built-in values."""
    _require_exact_fields("candidate group", group, _GROUP_FIELDS, {"track_embedding"})
    schema_version = _strict_nonnegative_integer(
        "schema_version", group["schema_version"]
    )
    if schema_version != INFERENCE_PAIR_CACHE_SCHEMA_VERSION:
        raise ValueError("candidate group has an unsupported schema_version")

    normalized = {
        "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
        "scene_id": _canonical_string("scene_id", group["scene_id"]),
        "sample_token": _canonical_string("sample_token", group["sample_token"]),
        "frame_id": _strict_nonnegative_integer("frame_id", group["frame_id"]),
        "track_id": _strict_nonnegative_integer("track_id", group["track_id"]),
        "class_id": _strict_nonnegative_integer("class_id", group["class_id"]),
        "class_name": _canonical_string("class_name", group["class_name"]),
        "state_bucket": _canonical_string("state_bucket", group["state_bucket"]),
        "track_gt_token": _optional_canonical_string(
            "track_gt_token", group["track_gt_token"]
        ),
        "production_det_index": _optional_nonnegative_integer(
            "production_det_index", group["production_det_index"]
        ),
        "production_match_cost": _optional_finite_number(
            "production_match_cost", group["production_match_cost"]
        ),
        "candidate_count": _strict_nonnegative_integer(
            "candidate_count", group["candidate_count"]
        ),
        "finite_candidate_count": _strict_nonnegative_integer(
            "finite_candidate_count", group["finite_candidate_count"]
        ),
        "gt_target_det_index": _optional_nonnegative_integer(
            "gt_target_det_index", group["gt_target_det_index"]
        ),
        "gt_target_status": _canonical_string(
            "gt_target_status", group["gt_target_status"]
        ),
    }
    if normalized["state_bucket"] not in TRACK_STATES:
        raise ValueError("state_bucket must be matched or unmatched")
    if normalized["gt_target_status"] not in _GROUP_STATUSES:
        raise ValueError("gt_target_status must be matched, no_match, or unknown")

    if "track_embedding" in group:
        normalized["track_embedding"] = _finite_vector(
            "track_embedding", group["track_embedding"]
        )

    candidates_value = group["candidates"]
    if type(candidates_value) is not list:
        raise TypeError("candidates must be a built-in list")
    candidates = []
    for position, candidate in enumerate(candidates_value):
        name = f"candidates[{position}]"
        _require_exact_fields(
            name,
            candidate,
            _CANDIDATE_FIELDS,
            {"det_embedding", "pair_vector"},
        )
        item = {
            "det_index": _strict_nonnegative_integer(
                f"{name}.det_index", candidate["det_index"]
            ),
            "geometric_cost": _optional_finite_number(
                f"{name}.geometric_cost", candidate["geometric_cost"]
            ),
            "geometric_threshold": _finite_number(
                f"{name}.geometric_threshold", candidate["geometric_threshold"]
            ),
            "finite": _strict_bool(f"{name}.finite", candidate["finite"]),
            "inside_geometric_gate": _strict_bool(
                f"{name}.inside_geometric_gate",
                candidate["inside_geometric_gate"],
            ),
            "candidate_rank": (
                None
                if candidate["candidate_rank"] is None
                else _strict_positive_integer(
                    f"{name}.candidate_rank", candidate["candidate_rank"]
                )
            ),
            "best_cost": _optional_finite_number(
                f"{name}.best_cost", candidate["best_cost"]
            ),
            "cost_gap": _optional_finite_number(
                f"{name}.cost_gap", candidate["cost_gap"]
            ),
            "association_score": _optional_finite_number(
                f"{name}.association_score", candidate["association_score"]
            ),
            "production_selected": _strict_bool(
                f"{name}.production_selected", candidate["production_selected"]
            ),
            "det_gt_token": _optional_canonical_string(
                f"{name}.det_gt_token", candidate["det_gt_token"]
            ),
        }
        label = candidate["gt_label"]
        if label is not None:
            label = _strict_nonnegative_integer(f"{name}.gt_label", label)
            if label not in (0, 1):
                raise ValueError(f"{name}.gt_label must be 0, 1, or None")
        item["gt_label"] = label
        for field in ("det_embedding", "pair_vector"):
            if field in candidate:
                item[field] = _finite_vector(f"{name}.{field}", candidate[field])
        candidates.append(item)

    candidates.sort(key=lambda item: item["det_index"])
    det_indices = [item["det_index"] for item in candidates]
    if len(set(det_indices)) != len(det_indices):
        raise ValueError("candidate det_index values must be unique")
    if normalized["candidate_count"] != len(candidates):
        raise ValueError("candidate_count does not match candidates")

    finite_candidates = [item for item in candidates if item["finite"]]
    if normalized["finite_candidate_count"] != len(finite_candidates):
        raise ValueError("finite_candidate_count does not match candidates")
    expected_best = (
        min(item["geometric_cost"] for item in finite_candidates)
        if finite_candidates
        else None
    )
    expected_ranks = {
        item["det_index"]: rank
        for rank, item in enumerate(
            sorted(
                finite_candidates,
                key=lambda entry: (entry["geometric_cost"], entry["det_index"]),
            ),
            start=1,
        )
    }
    for item in candidates:
        if item["best_cost"] != expected_best:
            raise ValueError("candidate best_cost is inconsistent")
        if item["finite"]:
            if item["geometric_cost"] is None:
                raise ValueError("finite candidate must have geometric_cost")
            if item["candidate_rank"] != expected_ranks[item["det_index"]]:
                raise ValueError("candidate_rank is inconsistent")
            if item["cost_gap"] != item["geometric_cost"] - expected_best:
                raise ValueError("candidate cost_gap is inconsistent")
            if item["inside_geometric_gate"] != (
                item["geometric_cost"] <= item["geometric_threshold"]
            ):
                raise ValueError("inside_geometric_gate is inconsistent")
        elif any(
            item[field] is not None
            for field in ("geometric_cost", "candidate_rank", "cost_gap")
        ) or item["inside_geometric_gate"]:
            raise ValueError("non-finite candidate fields are inconsistent")

    has_track_embedding = "track_embedding" in normalized
    det_embedding_flags = ["det_embedding" in item for item in candidates]
    if candidates and (
        any(det_embedding_flags) != all(det_embedding_flags)
        or all(det_embedding_flags) != has_track_embedding
    ):
        raise ValueError("track and detection embedding fields are inconsistent")
    if has_track_embedding:
        width = len(normalized["track_embedding"])
        if any(len(item["det_embedding"]) != width for item in candidates):
            raise ValueError("track and detection embedding dimensions differ")
    pair_widths = {
        len(item["pair_vector"])
        for item in candidates
        if "pair_vector" in item
    }
    if pair_widths and (
        len(pair_widths) != 1
        or not all("pair_vector" in item for item in candidates)
    ):
        raise ValueError("pair_vector fields or dimensions are inconsistent")

    production_index = normalized["production_det_index"]
    production_cost = normalized["production_match_cost"]
    selected = [item for item in candidates if item["production_selected"]]
    if (production_index is None) != (production_cost is None):
        raise ValueError("production detection and cost must both be null or present")
    if production_index is None:
        if selected:
            raise ValueError("production_selected requires a production detection")
    elif (
        len(selected) != 1
        or selected[0]["det_index"] != production_index
        or not selected[0]["finite"]
    ):
        raise ValueError("production selection is inconsistent")

    expected_status, expected_target_index, expected_labels = _derived_gt_target(
        normalized["track_gt_token"], candidates
    )
    if normalized["gt_target_status"] != expected_status:
        raise ValueError("gt_target_status is inconsistent with GT tokens")
    if normalized["gt_target_det_index"] != expected_target_index:
        raise ValueError("gt_target_det_index is inconsistent with GT tokens")
    if [item["gt_label"] for item in candidates] != expected_labels:
        raise ValueError("candidate gt_label values are inconsistent with GT tokens")

    normalized["candidates"] = candidates
    canonical_json(normalized)
    pickle.dumps(normalized, protocol=_PICKLE_PROTOCOL)
    return normalized


def _validate_group_ingestion_contract(group, runtime_manifest):
    thresholds = {
        candidate["geometric_threshold"] for candidate in group["candidates"]
    }
    if len(thresholds) > 1:
        raise ValueError(
            "all candidates in a group must use one geometric_threshold"
        )

    if "embedding_dim" in runtime_manifest:
        embedding_dim = runtime_manifest["embedding_dim"]
        embedding_dim = _strict_positive_integer("embedding_dim", embedding_dim)
        if "track_embedding" in group and (
            len(group["track_embedding"]) != embedding_dim
        ):
            raise ValueError("track_embedding width does not match embedding_dim")
        if any(
            "det_embedding" in candidate
            and len(candidate["det_embedding"]) != embedding_dim
            for candidate in group["candidates"]
        ):
            raise ValueError("det_embedding width does not match embedding_dim")

    if "pair_vector_dim" in runtime_manifest:
        pair_vector_dim = runtime_manifest["pair_vector_dim"]
        pair_vector_dim = _strict_positive_integer(
            "pair_vector_dim", pair_vector_dim
        )
        if any(
            "pair_vector" in candidate
            and len(candidate["pair_vector"]) != pair_vector_dim
            for candidate in group["candidates"]
        ):
            raise ValueError("pair_vector width does not match pair_vector_dim")

    if "export_options" in runtime_manifest:
        export_options = runtime_manifest["export_options"]
        if not isinstance(export_options, Mapping):
            raise TypeError("runtime manifest export_options must be a mapping")
        if "include_embeddings" in export_options:
            include_embeddings = _strict_bool(
                "export_options.include_embeddings",
                export_options["include_embeddings"],
            )
            has_embeddings = "track_embedding" in group and all(
                "det_embedding" in candidate
                for candidate in group["candidates"]
            )
            no_embeddings = "track_embedding" not in group and all(
                "det_embedding" not in candidate
                for candidate in group["candidates"]
            )
            if include_embeddings and not has_embeddings:
                raise ValueError("include_embeddings requires embedding fields")
            if not include_embeddings and not no_embeddings:
                raise ValueError("include_embeddings=false forbids embedding fields")
        if "include_pair_vectors" in export_options:
            include_pair_vectors = _strict_bool(
                "export_options.include_pair_vectors",
                export_options["include_pair_vectors"],
            )
            has_pair_vectors = all(
                "pair_vector" in candidate for candidate in group["candidates"]
            )
            no_pair_vectors = all(
                "pair_vector" not in candidate for candidate in group["candidates"]
            )
            if include_pair_vectors and not has_pair_vectors:
                raise ValueError("include_pair_vectors requires pair_vector fields")
            if not include_pair_vectors and not no_pair_vectors:
                raise ValueError(
                    "include_pair_vectors=false forbids pair_vector fields"
                )

    if "class_map" in runtime_manifest:
        class_map = runtime_manifest["class_map"]
        if not isinstance(class_map, Mapping):
            raise TypeError("runtime manifest class_map must be a mapping")
        if group["class_name"] not in class_map:
            raise ValueError("class_map does not contain group class_name")
        expected_class_id = _strict_nonnegative_integer(
            "class_map value", class_map[group["class_name"]]
        )
        if expected_class_id != group["class_id"]:
            raise ValueError("class_map class id does not match group class_id")


def validate_error_record(record):
    """Normalize one bounded audit error sample."""
    _require_exact_fields("error record", record, _ERROR_FIELDS)
    return {
        "scene_id": _canonical_string("scene_id", record["scene_id"]),
        "sample_token": _canonical_string("sample_token", record["sample_token"]),
        "frame_id": _strict_nonnegative_integer("frame_id", record["frame_id"]),
        "stage": _canonical_string("stage", record["stage"]),
        "error_type": _canonical_string("error_type", record["error_type"]),
        "message": _canonical_string("message", record["message"], allow_empty=True)[
            :1024
        ],
    }


def _canonical_mapping_key(value):
    value = _native_scalar(value)
    if value is None:
        return ["null", None]
    if type(value) is bool:
        return ["bool", value]
    if isinstance(value, Integral):
        return ["int", str(int(value))]
    if isinstance(value, Real):
        normalized = _finite_python_float("mapping key", value)
        return ["float", normalized.hex()]
    if type(value) is str:
        return ["str", value]
    raise TypeError(
        "canonical JSON mapping keys must be built-in strings or scalar values"
    )


def _canonical_json_value(value):
    value = _native_scalar(value)
    if isinstance(value, Mapping):
        items = []
        for key, item in value.items():
            canonical_key = _canonical_mapping_key(key)
            items.append((canonical_json(canonical_key), canonical_key, item))
        items.sort(key=lambda entry: entry[0])
        return {
            "__type__": "mapping",
            "items": [
                [canonical_key, _canonical_json_value(item)]
                for _, canonical_key, item in items
            ],
        }
    if isinstance(value, (list, tuple)):
        return [_canonical_json_value(item) for item in value]
    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return _finite_python_float("canonical JSON value", value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def canonical_json(value):
    """Return stable type-tagged JSON without ambiguous mapping keys."""
    return json.dumps(
        _canonical_json_value(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def _json_output_value(value):
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(
                    "JSON output mappings require exact built-in string mapping keys"
                )
            normalized[key] = _json_output_value(item)
        return normalized
    if isinstance(value, (list, tuple)):
        return [_json_output_value(item) for item in value]
    value = _native_scalar(value)
    if value is None or type(value) in (bool, str):
        return value
    if isinstance(value, Integral) and not isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, Real) and not isinstance(value, (bool, np.bool_)):
        return _finite_python_float("JSON output value", value)
    raise TypeError(f"value of type {type(value).__name__} is not JSON-safe")


def _json_output_copy(value):
    normalized = _json_output_value(value)
    json.dumps(normalized, ensure_ascii=True, allow_nan=False)
    return normalized


def _validate_observed_metadata(metadata):
    _require_exact_fields(
        "observed metadata", metadata, _OBSERVED_METADATA_FIELDS
    )
    normalized = {
        "scene_id": _canonical_string("scene_id", metadata["scene_id"]),
        "sample_token": _canonical_string(
            "sample_token", metadata["sample_token"]
        ),
        "frame_id": _strict_nonnegative_integer(
            "frame_id", metadata["frame_id"]
        ),
        "track_id": _strict_nonnegative_integer(
            "track_id", metadata["track_id"]
        ),
        "fingerprint": _canonical_string(
            "fingerprint", metadata["fingerprint"]
        ),
        "class_id": _strict_nonnegative_integer(
            "class_id", metadata["class_id"]
        ),
        "class_name": _canonical_string("class_name", metadata["class_name"]),
        "state_bucket": _canonical_string(
            "state_bucket", metadata["state_bucket"]
        ),
        "gt_target_status": _canonical_string(
            "gt_target_status", metadata["gt_target_status"]
        ),
    }
    fingerprint = normalized["fingerprint"]
    if len(fingerprint) != 64 or any(
        character not in "0123456789abcdef" for character in fingerprint
    ):
        raise ValueError("fingerprint must be a lowercase SHA-256 digest")
    if normalized["state_bucket"] not in TRACK_STATES:
        raise ValueError("state_bucket must be matched or unmatched")
    if normalized["gt_target_status"] not in _GROUP_STATUSES:
        raise ValueError("gt_target_status must be matched, no_match, or unknown")
    return normalized


def _validate_metadata_manifest_contract(metadata, runtime_manifest):
    if "class_map" not in runtime_manifest:
        return
    class_map = runtime_manifest["class_map"]
    if not isinstance(class_map, Mapping):
        raise TypeError("runtime manifest class_map must be a mapping")
    class_name = metadata["class_name"]
    if class_name not in class_map:
        raise ValueError("class_map does not contain observed class_name")
    expected_class_id = _strict_nonnegative_integer(
        "class_map value", class_map[class_name]
    )
    if expected_class_id != metadata["class_id"]:
        raise ValueError("class_map class id does not match observed class_id")


def _manifest_require_audit_gt(runtime_manifest):
    if "export_options" not in runtime_manifest:
        return None
    export_options = runtime_manifest["export_options"]
    if not isinstance(export_options, Mapping):
        raise TypeError("runtime manifest export_options must be a mapping")
    if "require_audit_gt" not in export_options:
        raise ValueError(
            "runtime manifest export_options must declare require_audit_gt"
        )
    return _strict_bool(
        "export_options.require_audit_gt",
        export_options["require_audit_gt"],
    )


def _observed_item_metadata(value):
    if isinstance(value, Mapping) and set(value.keys()) == _OBSERVED_METADATA_FIELDS:
        return _validate_observed_metadata(value)
    return _metadata_from_group(validate_candidate_group(value))


def validate_accumulator_state(state, max_error_samples=32):
    """Strictly validate and detach an exported worker accumulator state."""
    max_error_samples = _strict_nonnegative_integer(
        "max_error_samples", max_error_samples
    )
    _require_exact_fields("accumulator state", state, _STATE_FIELDS)
    schema_version = _strict_nonnegative_integer(
        "schema_version", state["schema_version"]
    )
    if schema_version != INFERENCE_PAIR_CACHE_SCHEMA_VERSION:
        raise ValueError("accumulator state has an unsupported schema_version")
    error_count = _strict_nonnegative_integer("error_count", state["error_count"])
    if type(state["error_samples"]) is not list:
        raise TypeError("error_samples must be a built-in list")
    if len(state["error_samples"]) > max_error_samples:
        raise ValueError(
            "error_samples exceeds max_error_samples "
            f"({len(state['error_samples'])} > {max_error_samples})"
        )
    if not isinstance(state["runtime_manifest"], Mapping):
        raise TypeError("runtime_manifest must be a mapping")
    runtime_manifest = _json_output_copy(state["runtime_manifest"])
    _manifest_require_audit_gt(runtime_manifest)
    if type(state["observed_metadata"]) is not list:
        raise TypeError("observed_metadata must be a built-in list")
    observed_metadata = [
        _validate_observed_metadata(item) for item in state["observed_metadata"]
    ]
    for metadata in observed_metadata:
        _validate_metadata_manifest_contract(metadata, runtime_manifest)
    observed_by_key = {}
    for metadata in observed_metadata:
        key = _group_key(metadata)
        if key in observed_by_key:
            raise ValueError(
                f"accumulator state contains duplicate observed key {key}"
            )
        observed_by_key[key] = metadata
    observed_metadata.sort(key=_group_key)
    if type(state["groups"]) is not list:
        raise TypeError("groups must be a built-in list")
    groups = [validate_candidate_group(group) for group in state["groups"]]
    seen_keys = set()
    for group in groups:
        key = _group_key(group)
        if key in seen_keys:
            raise ValueError(f"accumulator state contains duplicate group key {key}")
        if observed_by_key.get(key) != _metadata_from_group(group):
            raise ValueError(
                f"full group does not match observed metadata for key {key}"
            )
        seen_keys.add(key)
    groups.sort(key=_group_key)
    error_samples = [validate_error_record(item) for item in state["error_samples"]]
    if error_count < len(error_samples):
        raise ValueError("error_count must be at least len(error_samples)")
    return {
        "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
        "runtime_manifest": runtime_manifest,
        "observed_metadata": observed_metadata,
        "groups": groups,
        "error_count": error_count,
        "error_samples": error_samples,
    }


def _error_key(record):
    return (
        record["scene_id"],
        record["sample_token"],
        record["frame_id"],
        record["stage"],
        record["error_type"],
        record["message"],
    )


def _empty_counts():
    return {
        "observed_groups": 0,
        "accepted_groups": 0,
        "rejected_unknown_groups": 0,
        "capped_groups": 0,
        "matched_groups": 0,
        "no_match_groups": 0,
        "unknown_groups": 0,
        "candidate_count": 0,
        "finite_candidate_count": 0,
        "gated_candidate_count": 0,
        "production_selected_count": 0,
        "positive_count": 0,
        "negative_count": 0,
        "unknown_label_count": 0,
    }


def _retained_counts(groups):
    counts = _empty_counts()
    counts["accepted_groups"] = len(groups)
    for group in groups:
        counts[f"{group['gt_target_status']}_groups"] += 1
        counts["candidate_count"] += len(group["candidates"])
        for candidate in group["candidates"]:
            counts["finite_candidate_count"] += int(candidate["finite"])
            counts["gated_candidate_count"] += int(
                candidate["inside_geometric_gate"]
            )
            counts["production_selected_count"] += int(
                candidate["production_selected"]
            )
            if candidate["gt_label"] == 1:
                counts["positive_count"] += 1
            elif candidate["gt_label"] == 0:
                counts["negative_count"] += 1
            else:
                counts["unknown_label_count"] += 1
    return counts


def summarize_candidate_groups(
    observed_groups,
    retained_groups,
    error_count,
    error_samples,
    runtime_manifest,
    require_audit_gt,
):
    """Recompute deterministic summary counts from validated serialized groups.

    ``per_class_state`` uses canonical JSON keys of ``[class_id, state_bucket]``.
    """
    if type(observed_groups) is not list or type(retained_groups) is not list:
        raise TypeError("observed_groups and retained_groups must be built-in lists")
    require_audit_gt = _strict_bool("require_audit_gt", require_audit_gt)
    observed = [_observed_item_metadata(item) for item in observed_groups]
    retained = [validate_candidate_group(group) for group in retained_groups]
    observed_by_key = {_group_key(metadata): metadata for metadata in observed}
    if len(observed_by_key) != len(observed):
        raise ValueError("observed_groups contains duplicate stable keys")
    retained_by_key = {_group_key(group): group for group in retained}
    if len(retained_by_key) != len(retained):
        raise ValueError("duplicate retained stable keys are not allowed")
    eligible = [
        group
        for group in observed
        if not (
            require_audit_gt and group["gt_target_status"] == "unknown"
        )
    ]
    eligible_by_key = {_group_key(group): group for group in eligible}
    for group in retained:
        key = _group_key(group)
        if eligible_by_key.get(key) != _metadata_from_group(group):
            raise ValueError(
                "retained_groups must be an exact subset of eligible groups"
            )

    class_names = {}
    for group in observed + retained:
        class_id = group["class_id"]
        class_name = group["class_name"]
        if class_id in class_names and class_names[class_id] != class_name:
            raise ValueError(f"class_id {class_id} maps to inconsistent class names")
        if (
            class_name in class_names.values()
            and class_names.get(class_id) != class_name
        ):
            raise ValueError(f"class_name {class_name} maps to inconsistent class ids")
        class_names[class_id] = class_name

    def make_bucket(observed_bucket, retained_bucket):
        eligible_bucket = [
            group
            for group in observed_bucket
            if not (
                require_audit_gt and group["gt_target_status"] == "unknown"
            )
        ]
        counts = _retained_counts(retained_bucket)
        counts["observed_groups"] = len(observed_bucket)
        counts["rejected_unknown_groups"] = (
            len(observed_bucket) - len(eligible_bucket)
        )
        counts["capped_groups"] = len(eligible_bucket) - counts["accepted_groups"]
        if counts["capped_groups"] < 0:
            raise ValueError("summary group counts cannot be reconciled")
        return counts

    overall = make_bucket(observed, retained)
    overall["error_count"] = _strict_nonnegative_integer("error_count", error_count)

    per_class = {}
    per_class_state = {}
    for class_id, class_name in sorted(class_names.items()):
        observed_class = [group for group in observed if group["class_id"] == class_id]
        retained_class = [group for group in retained if group["class_id"] == class_id]
        per_class[class_name] = make_bucket(observed_class, retained_class)
        states = sorted(
            {
                group["state_bucket"]
                for group in observed_class + retained_class
            }
        )
        for state in states:
            bucket_key = canonical_json([class_id, state])
            per_class_state[bucket_key] = make_bucket(
                [group for group in observed_class if group["state_bucket"] == state],
                [group for group in retained_class if group["state_bucket"] == state],
            )

    normalized_errors = sorted(
        {
            _error_key(item): item
            for item in map(validate_error_record, error_samples)
        }.values(),
        key=_error_key,
    )
    if overall["error_count"] < len(normalized_errors):
        raise ValueError("error_count must be at least the number of error samples")
    return {
        "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
        "runtime_manifest": _json_output_copy(runtime_manifest or {}),
        "overall": overall,
        "per_class": per_class,
        "per_class_state": dict(sorted(per_class_state.items())),
        "error_samples": copy.deepcopy(normalized_errors),
    }


class InferencePairCacheAccumulator:
    """Order-independent accumulator for audit-only association groups."""

    def __init__(
        self,
        *,
        runtime_manifest=None,
        max_groups_per_class_state=0,
        require_audit_gt=False,
        max_error_samples=32,
    ):
        self.max_groups_per_class_state = _strict_nonnegative_integer(
            "max_groups_per_class_state", max_groups_per_class_state
        )
        self.require_audit_gt = _strict_bool("require_audit_gt", require_audit_gt)
        self.max_error_samples = _strict_nonnegative_integer(
            "max_error_samples", max_error_samples
        )
        if runtime_manifest is None:
            runtime_manifest = {}
        if not isinstance(runtime_manifest, Mapping):
            raise TypeError("runtime_manifest must be a mapping")
        self.runtime_manifest = _json_output_copy(runtime_manifest)
        manifest_require_audit_gt = _manifest_require_audit_gt(
            self.runtime_manifest
        )
        if (
            manifest_require_audit_gt is not None
            and manifest_require_audit_gt != self.require_audit_gt
        ):
            raise ValueError(
                "accumulator require_audit_gt does not match runtime manifest"
            )
        self._observed_metadata = {}
        self._groups = {}
        self._error_count = 0
        self._error_samples = {}

    def _eligible(self, metadata):
        return not (
            self.require_audit_gt
            and metadata["gt_target_status"] == "unknown"
        )

    def _consider_group(self, group):
        metadata = _metadata_from_group(group)
        if not self._eligible(metadata):
            return
        key = _group_key(group)
        if key in self._groups:
            return
        if not self.max_groups_per_class_state:
            self._groups[key] = group
            return

        bucket = (group["class_id"], group["state_bucket"])
        bucket_keys = [
            retained_key
            for retained_key, retained_group in self._groups.items()
            if (
                retained_group["class_id"],
                retained_group["state_bucket"],
            )
            == bucket
        ]
        if len(bucket_keys) < self.max_groups_per_class_state:
            self._groups[key] = group
            return
        largest_key = max(bucket_keys)
        if key < largest_key:
            del self._groups[largest_key]
            self._groups[key] = group

    def add_group(self, group):
        normalized = validate_candidate_group(group)
        _validate_group_ingestion_contract(normalized, self.runtime_manifest)
        key = _group_key(normalized)
        metadata = _metadata_from_group(normalized)
        _validate_metadata_manifest_contract(metadata, self.runtime_manifest)
        existing = self._observed_metadata.get(key)
        if existing is not None and existing["fingerprint"] != metadata["fingerprint"]:
            raise ValueError(f"conflicting inference pair group: {key}")
        if existing is None:
            self._observed_metadata[key] = metadata
        self._consider_group(normalized)
        return self._eligible(metadata)

    def _retain_error_sample(self, record):
        key = _error_key(record)
        self._error_samples[key] = copy.deepcopy(record)
        if len(self._error_samples) > self.max_error_samples:
            for dropped_key in sorted(self._error_samples)[self.max_error_samples :]:
                del self._error_samples[dropped_key]

    def add_error(self, record):
        normalized = validate_error_record(record)
        self._error_count += 1
        self._retain_error_sample(normalized)

    def _sorted_error_samples(self):
        return [
            copy.deepcopy(self._error_samples[key])
            for key in sorted(self._error_samples)
        ]

    def export_state(self):
        return {
            "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "runtime_manifest": copy.deepcopy(self.runtime_manifest),
            "observed_metadata": [
                copy.deepcopy(self._observed_metadata[key])
                for key in sorted(self._observed_metadata)
            ],
            "groups": [
                copy.deepcopy(self._groups[key]) for key in sorted(self._groups)
            ],
            "error_count": self._error_count,
            "error_samples": self._sorted_error_samples(),
        }

    def merge_state(self, state):
        normalized = validate_accumulator_state(
            state, max_error_samples=self.max_error_samples
        )
        if canonical_json(normalized["runtime_manifest"]) != canonical_json(
            self.runtime_manifest
        ):
            raise ValueError("inference pair cache runtime manifest mismatch")
        for group in normalized["groups"]:
            _validate_group_ingestion_contract(group, self.runtime_manifest)
        for metadata in normalized["observed_metadata"]:
            key = _group_key(metadata)
            existing = self._observed_metadata.get(key)
            if existing is not None and existing != metadata:
                raise ValueError(f"conflicting inference pair group: {key}")
        for metadata in normalized["observed_metadata"]:
            key = _group_key(metadata)
            if key not in self._observed_metadata:
                self._observed_metadata[key] = metadata
        for group in normalized["groups"]:
            self._consider_group(group)
        self._error_count += normalized["error_count"]
        for error in normalized["error_samples"]:
            self._retain_error_sample(error)

    def _retained_groups(self):
        return [self._groups[key] for key in sorted(self._groups)]

    def to_payload(self):
        return {
            "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "runtime_manifest": copy.deepcopy(self.runtime_manifest),
            "groups": copy.deepcopy(self._retained_groups()),
        }

    def to_summary(self):
        return summarize_candidate_groups(
            observed_groups=[
                self._observed_metadata[key]
                for key in sorted(self._observed_metadata)
            ],
            retained_groups=self._retained_groups(),
            error_count=self._error_count,
            error_samples=self._sorted_error_samples(),
            runtime_manifest=self.runtime_manifest,
            require_audit_gt=self.require_audit_gt,
        )


def sha256_file(path):
    """Compute an exact SHA-256 digest using bounded reads."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_section(cfg, name):
    value = cfg.get(name, {}) or {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    return value


def _git_identity_output(git_root, args, *, text):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=str(git_root),
            check=True,
            text=text,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            f"git identity command failed: git {' '.join(args)}"
        ) from exc
    output = result.stdout
    if text:
        if isinstance(output, bytes):
            output = output.decode("utf-8")
        if type(output) is not str:
            raise RuntimeError("git identity command returned invalid text output")
    else:
        if isinstance(output, str):
            output = output.encode("utf-8")
        if type(output) is not bytes:
            raise RuntimeError("git identity command returned invalid binary output")
    return output


def _configured_output_paths(cfg, git_root):
    cache_cfg = _config_section(cfg, "INFERENCE_PAIR_CACHE")
    paths = set()
    for key, value in cache_cfg.items():
        upper_key = str(key).upper()
        if type(value) is not str or not upper_key.endswith("PATH"):
            continue
        if not any(token in upper_key for token in ("OUTPUT", "CACHE", "SUMMARY")):
            continue
        path = Path(value).expanduser()
        if not path.is_absolute():
            path = git_root / path
        paths.add(path.resolve())
    return paths


def _build_git_source_identity(cfg, git_root):
    commit = _git_identity_output(git_root, ["rev-parse", "HEAD"], text=True).strip()
    if not commit:
        raise RuntimeError("git identity command returned an empty commit")
    status = _git_identity_output(
        git_root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
        text=True,
    )
    tracked_diff = _git_identity_output(
        git_root,
        ["diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        text=False,
    )
    untracked_output = _git_identity_output(
        git_root,
        [
            "ls-files",
            "--others",
            "--exclude-standard",
            "--",
            "*.py",
            "*.yaml",
            "*.yml",
            "*.json",
        ],
        text=True,
    )

    output_paths = _configured_output_paths(cfg, git_root)
    relevant_suffixes = {".py", ".yaml", ".yml", ".json"}
    identity_status_lines = []
    for line in status.splitlines():
        path_text = line[3:] if len(line) >= 3 else ""
        status_paths = [path_text]
        if " -> " in path_text:
            status_paths = path_text.split(" -> ", 1)
        resolved_paths = []
        for status_path in status_paths:
            candidate = Path(status_path)
            if not candidate.is_absolute():
                candidate = git_root / candidate
            resolved_paths.append(candidate.resolve())
        if any(path in output_paths for path in resolved_paths):
            continue
        if line.startswith("?? ") and not any(
            path.suffix.lower() in relevant_suffixes for path in resolved_paths
        ):
            continue
        identity_status_lines.append(line)
    identity_status = "\n".join(identity_status_lines)
    if identity_status_lines:
        identity_status += "\n"

    digest = hashlib.sha256()
    digest.update(b"git-worktree-v1\0status\0")
    digest.update(identity_status.encode("utf-8"))
    digest.update(b"\0tracked-diff\0")
    digest.update(tracked_diff)
    root_prefix = str(git_root) + os.sep
    for relative_name in sorted(set(untracked_output.splitlines())):
        if not relative_name:
            continue
        relative = Path(relative_name)
        if relative.suffix.lower() not in relevant_suffixes:
            continue
        source_path = (git_root / relative).resolve()
        if str(source_path) != str(git_root) and not str(source_path).startswith(
            root_prefix
        ):
            raise RuntimeError("git identity reported an out-of-root source path")
        if source_path in output_paths:
            continue
        digest.update(b"\0untracked\0")
        digest.update(relative.as_posix().encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(bytes.fromhex(sha256_file(source_path)))
        except (OSError, ValueError) as exc:
            raise RuntimeError(
                f"git identity could not hash untracked source {relative_name}"
            ) from exc
    return commit, bool(status.strip()), digest.hexdigest()


def _manifest_config_value(value, name):
    """Normalize a selected config value to an unambiguous JSON object."""
    if isinstance(value, Mapping):
        normalized = {}
        for key, item in value.items():
            key = _native_scalar(key)
            if type(key) is str:
                output_key = key
            elif isinstance(key, Integral) and not isinstance(
                key, (bool, np.bool_)
            ):
                output_key = str(int(key))
            else:
                raise TypeError(
                    f"{name} mapping keys must be strings or integers"
                )
            if output_key in normalized:
                raise ValueError(f"{name} contains colliding mapping keys")
            normalized[output_key] = _manifest_config_value(
                item, f"{name}.{output_key}"
            )
        return normalized
    if isinstance(value, (list, tuple)):
        return [
            _manifest_config_value(item, f"{name} item") for item in value
        ]
    return _json_output_value(value)


def _manifest_class_map(cfg):
    class_map = cfg.get("CATEGORY_MAP_TO_NUMBER", {})
    if not isinstance(class_map, Mapping):
        raise TypeError("class_map source CATEGORY_MAP_TO_NUMBER must be a mapping")
    normalized = {}
    for class_name, class_id in class_map.items():
        if type(class_name) is not str:
            raise TypeError("class_map keys must be exact built-in class-name strings")
        normalized_name = _canonical_string("class_map key", class_name)
        normalized[normalized_name] = _strict_nonnegative_integer(
            "class_map value", class_id
        )
    return normalized


def build_inference_pair_cache_manifest(
    *, cfg, config_path, process_count, git_root, pair_vector_dim, embedding_dim
):
    """Build the exact runtime identity used to reject incompatible merges."""
    if not isinstance(cfg, Mapping):
        raise TypeError("cfg must be a mapping")
    process_count = _strict_positive_integer("process_count", process_count)
    pair_vector_dim = _strict_positive_integer("pair_vector_dim", pair_vector_dim)
    embedding_dim = _strict_positive_integer("embedding_dim", embedding_dim)
    config_file = Path(config_path).expanduser().resolve(strict=True)
    normalized_git_root = Path(git_root).expanduser().resolve()

    effective_config_json = canonical_json(cfg)

    def resolve_checkpoint(section_name):
        section = _config_section(cfg, section_name)
        checkpoint_value = section.get("CHECKPOINT_PATH")
        if checkpoint_value is None or checkpoint_value == "":
            return None
        if type(checkpoint_value) is not str:
            raise TypeError(f"{section_name}.CHECKPOINT_PATH must be a string")
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = normalized_git_root / checkpoint
        return checkpoint.resolve(strict=True)

    mamba_checkpoint = resolve_checkpoint("MAMBA")
    association_checkpoint = resolve_checkpoint("MAMBA_ASSOCIATION_HEAD")
    git_commit, git_dirty, git_worktree_sha256 = _build_git_source_identity(
        cfg, normalized_git_root
    )

    cache_cfg = _config_section(cfg, "INFERENCE_PAIR_CACHE")
    export_options = {}
    for key, default in (
        ("INCLUDE_EMBEDDINGS", True),
        ("INCLUDE_PAIR_VECTORS", True),
        ("REQUIRE_AUDIT_GT", False),
    ):
        export_options[key.lower()] = _strict_bool(
            f"INFERENCE_PAIR_CACHE.{key}", cache_cfg.get(key, default)
        )
    threshold_bev = _config_section(_config_section(cfg, "THRESHOLD"), "BEV")
    matching_bev = _config_section(_config_section(cfg, "MATCHING"), "BEV")
    bev_cost_mode = threshold_bev.get(
        "COST_MODE", matching_bev.get("COST_MODE", "geometric")
    )

    manifest = {
        "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "effective_config_sha256": hashlib.sha256(
            effective_config_json.encode("utf-8")
        ).hexdigest(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "git_worktree_sha256": git_worktree_sha256,
        "mamba_checkpoint_path": (
            None if mamba_checkpoint is None else str(mamba_checkpoint)
        ),
        "mamba_checkpoint_sha256": (
            None if mamba_checkpoint is None else sha256_file(mamba_checkpoint)
        ),
        "association_checkpoint_path": (
            None if association_checkpoint is None else str(association_checkpoint)
        ),
        "association_checkpoint_sha256": (
            None
            if association_checkpoint is None
            else sha256_file(association_checkpoint)
        ),
        "filter_mode": _canonical_string(
            "FILTER_MODE", cfg.get("FILTER_MODE", ""), allow_empty=True
        ).lower(),
        "tracker_compat_mode": _canonical_string(
            "TRACKER_COMPAT_MODE", cfg.get("TRACKER_COMPAT_MODE", "default")
        ).lower(),
        "bev_cost_mode": _manifest_config_value(bev_cost_mode, "bev_cost_mode"),
        "pair_vector_dim": pair_vector_dim,
        "embedding_dim": embedding_dim,
        "process_count": process_count,
        "class_map": _manifest_class_map(cfg),
        "export_options": export_options,
    }
    return _json_output_copy(manifest)


def _fsync_directory(directory):
    descriptor = os.open(
        str(directory), os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _backup_destination(destination):
    descriptor, backup_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".bak",
    )
    backup = Path(backup_name)
    try:
        os.close(descriptor)
        descriptor = None
        backup.unlink()
        try:
            os.link(str(destination), str(backup))
        except OSError:
            with destination.open("rb") as source, backup.open("xb") as target:
                for chunk in iter(lambda: source.read(1024 * 1024), b""):
                    target.write(chunk)
                target.flush()
                os.fsync(target.fileno())
        return backup
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            backup.unlink()
        except FileNotFoundError:
            pass
        raise


def _atomic_write(path, write_value):
    """Write atomically, restoring the previous path if durability fails."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent),
        prefix=f".{destination.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    backup = None
    try:
        file_obj = os.fdopen(descriptor, "wb")
        descriptor = None
        with file_obj:
            write_value(file_obj)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        if destination.exists():
            backup = _backup_destination(destination)
        os.replace(str(temporary), str(destination))
        try:
            _fsync_directory(destination.parent)
        except BaseException:
            try:
                if backup is not None:
                    os.replace(str(backup), str(destination))
                    backup = None
                else:
                    destination.unlink()
            except OSError:
                pass
            try:
                _fsync_directory(destination.parent)
            except OSError:
                pass
            raise
        if backup is not None:
            committed_backup = backup
            backup = None
            try:
                committed_backup.unlink()
            except OSError:
                pass
    except BaseException:
        if descriptor is not None:
            try:
                os.close(descriptor)
            except OSError:
                pass
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        if backup is not None:
            try:
                backup.unlink()
            except FileNotFoundError:
                pass
        raise


def atomic_write_pickle(path, payload):
    _atomic_write(
        path,
        lambda file_obj: pickle.dump(payload, file_obj, protocol=_PICKLE_PROTOCOL),
    )


def atomic_write_json(path, payload):
    normalized = _json_output_copy(payload)

    def write_json(file_obj):
        serialized = json.dumps(
            normalized,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        ) + "\n"
        file_obj.write(serialized.encode("utf-8"))

    _atomic_write(path, write_json)
