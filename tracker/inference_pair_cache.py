"""Pure construction of inference-time association candidate groups."""

from __future__ import annotations

from collections.abc import Mapping
from numbers import Integral, Real

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
    return converted


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
