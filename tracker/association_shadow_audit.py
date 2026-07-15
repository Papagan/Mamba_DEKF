"""Pure helpers for comparing production and shadow assignments."""

from __future__ import annotations

import json
from collections.abc import Mapping

import numpy as np


_THRESHOLD_RTOL = 1e-6
_THRESHOLD_ATOL = 1e-8


def _canonical_gt_token(token):
    if not isinstance(token, str) or not token or token.strip() != token:
        return None
    return str(token)


def assignment_map(matches: np.ndarray, costs: np.ndarray) -> dict[int, dict]:
    """Map matched track rows to their selected detection and cost."""
    matches = np.asarray(matches)
    costs = np.asarray(costs)
    if matches.ndim != 2 or matches.shape[1] != 2:
        raise ValueError("matches must have shape (n_matches, 2)")
    if matches.size and not np.issubdtype(matches.dtype, np.integer):
        raise TypeError("matches must contain integer indices")
    if costs.ndim != 1 or costs.shape[0] != matches.shape[0]:
        raise ValueError("costs must have shape (n_matches,)")
    if np.issubdtype(costs.dtype, np.complexfloating):
        raise TypeError("costs must contain real-valued numeric values")
    if not np.issubdtype(costs.dtype, np.number):
        raise TypeError("costs must contain numeric values")
    if not np.all(np.isfinite(costs)):
        raise ValueError("selected assignment costs must be finite")

    result = {}
    seen_detections = set()
    for (track_index, det_index), cost in zip(matches, costs):
        track_index = int(track_index)
        det_index = int(det_index)
        if track_index < 0 or det_index < 0:
            raise ValueError("match indices must be non-negative")
        if track_index in result:
            raise ValueError(f"duplicate assignment for track index {track_index}")
        if det_index in seen_detections:
            raise ValueError(f"duplicate assignment for detection index {det_index}")
        result[track_index] = {"det_index": det_index, "cost": float(cost)}
        seen_detections.add(det_index)
    return result


def classify_assignment_outcome(
    track_gt_token,
    production_det_token,
    shadow_det_token,
    *,
    production_matched=None,
    shadow_matched=None,
) -> str:
    """Classify correctness while distinguishing unmatched from unknown GT."""
    track_gt_token = _canonical_gt_token(track_gt_token)
    if track_gt_token is None:
        return "unknown"

    if production_matched is None:
        production_matched = production_det_token is not None
    if shadow_matched is None:
        shadow_matched = shadow_det_token is not None
    for name, matched in (
        ("production_matched", production_matched),
        ("shadow_matched", shadow_matched),
    ):
        if not isinstance(matched, (bool, np.bool_)):
            raise TypeError(f"{name} must be a boolean or None")

    production_det_token = _canonical_gt_token(production_det_token)
    shadow_det_token = _canonical_gt_token(shadow_det_token)
    production_correct = (
        False
        if not production_matched
        else (
            None
            if production_det_token is None
            else production_det_token == track_gt_token
        )
    )
    shadow_correct = (
        False
        if not shadow_matched
        else (
            None if shadow_det_token is None else shadow_det_token == track_gt_token
        )
    )
    if production_correct is None or shadow_correct is None:
        return "unknown"
    if shadow_correct and not production_correct:
        return "improved"
    if production_correct and not shadow_correct:
        return "damaged"
    return "neutral"


def _matrix(name, value, expected_shape=None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if np.issubdtype(array.dtype, np.complexfloating):
        raise TypeError(f"{name} must contain real-valued numeric values")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    return array


def _native_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _metadata_values(name, values, track_count):
    try:
        array = np.asarray(values, dtype=object)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a one-dimensional sequence") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.shape[0] != track_count:
        raise ValueError(f"{name} must contain one value per track ({track_count})")
    return [_native_scalar(value) for value in array]


def _normalize_track_ids(values, track_count):
    track_ids = _metadata_values("track_ids", values, track_count)
    normalized = []
    for value in track_ids:
        if isinstance(value, bool) or not isinstance(value, (str, int)):
            raise ValueError("track_ids must contain string or integer identifiers")
        if isinstance(value, str):
            value = value.strip()
            if not value:
                raise ValueError("track_ids must not contain empty identifiers")
        if isinstance(value, int) and value < 0:
            raise ValueError("track_ids must contain non-negative identifiers")
        try:
            json.dumps(value, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("track_ids must be JSON serializable") from exc
        normalized.append(value)
    if len(set(normalized)) != len(normalized):
        raise ValueError("track_ids must contain unique identifiers")
    return normalized


def _normalize_class_ids(values, track_count):
    class_ids = _metadata_values("class_ids", values, track_count)
    normalized = []
    for value in class_ids:
        if isinstance(value, bool):
            raise ValueError("class_ids must contain integer-compatible values")
        try:
            numeric_value = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("class_ids must contain integer-compatible values") from exc
        if not np.isfinite(numeric_value) or not numeric_value.is_integer():
            raise ValueError("class_ids must contain integer-compatible values")
        normalized.append(int(numeric_value))
    return normalized


def _normalize_class_names(values, track_count):
    class_names = _metadata_values("class_names", values, track_count)
    normalized = []
    for value in class_names:
        if not isinstance(value, str) or not value.strip():
            raise ValueError("class_names must contain non-empty strings")
        normalized.append(value.strip())
    return normalized


def _normalize_state_buckets(values, track_count):
    state_buckets = _metadata_values("state_buckets", values, track_count)
    normalized = []
    for value in state_buckets:
        if not isinstance(value, str):
            raise ValueError("state_buckets must contain matched/unmatched strings")
        state = value.strip().lower()
        if state not in {"matched", "unmatched"}:
            raise ValueError("state_buckets values must be matched or unmatched")
        normalized.append(state)
    return normalized


def _normalize_gt_tokens(name, values, expected_count):
    if values is None:
        return None
    tokens = _metadata_values(name, values, expected_count)
    normalized = []
    for token in tokens:
        if token is None:
            normalized.append(None)
            continue
        canonical_token = _canonical_gt_token(token)
        if canonical_token is None:
            raise ValueError(
                f"{name} must contain canonical non-empty string tokens or None"
            )
        normalized.append(canonical_token)
    return normalized


def _selected_value(matrix: np.ndarray, track_index: int, det_index):
    if det_index is None:
        return None
    value = float(matrix[track_index, det_index])
    return value if np.isfinite(value) else None


def _candidate_rank(base_row: np.ndarray, det_index) -> int | None:
    """Return a 1-based competition rank among finite base-row costs."""
    if det_index is None:
        return None
    selected_cost = float(base_row[det_index])
    finite_costs = np.asarray(base_row)[np.isfinite(base_row)]
    if not np.isfinite(selected_cost):
        return None
    return int(np.count_nonzero(finite_costs < selected_cost)) + 1


def _pair_label(track_gt_token, det_gt_token, *, matched: bool):
    if not matched:
        return None
    track_gt_token = _canonical_gt_token(track_gt_token)
    det_gt_token = _canonical_gt_token(det_gt_token)
    if track_gt_token is None or det_gt_token is None:
        return None
    return int(track_gt_token == det_gt_token)


def _class_threshold(class_thresholds, class_id):
    if class_thresholds is None:
        return None
    if not isinstance(class_thresholds, Mapping):
        raise TypeError("class_thresholds must be a mapping or None")

    native_class_id = _native_scalar(class_id)
    candidates = [class_id, native_class_id, str(native_class_id)]
    try:
        numeric_class_id = float(native_class_id)
        if np.isfinite(numeric_class_id) and numeric_class_id.is_integer():
            candidates.extend((int(numeric_class_id), str(int(numeric_class_id))))
    except (TypeError, ValueError):
        pass
    threshold = None
    for key in candidates:
        try:
            if key in class_thresholds:
                threshold = class_thresholds[key]
                break
        except TypeError:
            continue
    if threshold is None:
        return None
    try:
        threshold = float(threshold)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"threshold for class {class_id!r} must be numeric") from exc
    if not np.isfinite(threshold):
        raise ValueError(f"threshold for class {class_id!r} must be finite")
    return threshold


def _threshold_crossing(
    *,
    track_index,
    production_det_index,
    shadow_det_index,
    threshold,
    base_cost_matrix,
    corrected_cost_matrix,
):
    if threshold is None:
        return None

    selected_det_indices = {
        det_index
        for det_index in (production_det_index, shadow_det_index)
        if det_index is not None
    }
    for det_index in selected_det_indices:
        base_cost = float(base_cost_matrix[track_index, det_index])
        corrected_cost = float(corrected_cost_matrix[track_index, det_index])
        base_accepted = _cost_within_threshold(base_cost, threshold)
        corrected_accepted = _cost_within_threshold(corrected_cost, threshold)
        if base_accepted != corrected_accepted:
            return True
    return False


def _cost_within_threshold(cost, threshold) -> bool:
    return bool(
        np.isfinite(cost)
        and (
            cost <= threshold
            or np.isclose(
                cost,
                threshold,
                rtol=_THRESHOLD_RTOL,
                atol=_THRESHOLD_ATOL,
            )
        )
    )


def _validate_selected_cost_threshold(
    label, assignment, source_matrix, track_index, threshold
) -> None:
    if assignment is None or threshold is None:
        return
    det_index = assignment["det_index"]
    selected_cost = float(source_matrix[track_index, det_index])
    if not _cost_within_threshold(selected_cost, threshold):
        raise ValueError(
            f"{label} selected cost {selected_cost} exceeds class threshold "
            f"{threshold} for match ({track_index}, {det_index})"
        )


def _assignment_gt_token(assignment, det_gt_tokens):
    if assignment is None:
        return None
    if det_gt_tokens is None:
        return None
    return _native_scalar(det_gt_tokens[assignment["det_index"]])


def _validate_assignment_costs(name, assignments, source_matrix) -> None:
    for track_index, assignment in assignments.items():
        det_index = assignment["det_index"]
        matrix_cost = float(source_matrix[track_index, det_index])
        if not np.isfinite(matrix_cost) or not np.isclose(
            assignment["cost"], matrix_cost, rtol=1e-6, atol=1e-8
        ):
            raise ValueError(
                f"{name} for match ({track_index}, {det_index}) does not agree "
                "with its source cost matrix"
            )


def compare_assignment_results(
    *,
    production_matches,
    production_costs,
    shadow_matches,
    shadow_costs,
    base_cost_matrix,
    corrected_cost_matrix,
    association_scores,
    track_ids,
    class_ids,
    class_names,
    state_buckets,
    track_gt_tokens=None,
    det_gt_tokens=None,
    context=None,
    class_thresholds=None,
) -> list[dict]:
    """Return one side-effect-free production/shadow decision record per track.

    Candidate ranks are 1-based competition ranks computed only from finite
    values in the original base-cost row. ``original_cost_gap`` is the shadow
    detection's base cost minus the production detection's base cost and is
    ``None`` unless both assignments select finite base-row candidates.
    """
    base_cost_matrix = _matrix("base_cost_matrix", base_cost_matrix)
    matrix_shape = base_cost_matrix.shape
    corrected_cost_matrix = _matrix(
        "corrected_cost_matrix", corrected_cost_matrix, matrix_shape
    )
    association_scores = _matrix(
        "association_scores", association_scores, matrix_shape
    )
    track_count, det_count = matrix_shape

    track_ids = _normalize_track_ids(track_ids, track_count)
    class_ids = _normalize_class_ids(class_ids, track_count)
    class_names = _normalize_class_names(class_names, track_count)
    state_buckets = _normalize_state_buckets(state_buckets, track_count)
    track_gt_tokens = _normalize_gt_tokens(
        "track_gt_tokens", track_gt_tokens, track_count
    )
    det_gt_tokens = _normalize_gt_tokens("det_gt_tokens", det_gt_tokens, det_count)
    if context is not None and not isinstance(context, Mapping):
        raise TypeError("context must be a mapping or None")
    context = context or {}

    production_by_track = assignment_map(production_matches, production_costs)
    shadow_by_track = assignment_map(shadow_matches, shadow_costs)
    for name, assignments in (
        ("production_matches", production_by_track),
        ("shadow_matches", shadow_by_track),
    ):
        for track_index, assignment in assignments.items():
            if track_index >= track_count:
                raise IndexError(
                    f"{name} track index {track_index} is outside [0, {track_count})"
                )
            det_index = assignment["det_index"]
            if det_index >= det_count:
                raise IndexError(
                    f"{name} detection index {det_index} is outside [0, {det_count})"
                )
    _validate_assignment_costs(
        "production_costs", production_by_track, base_cost_matrix
    )
    _validate_assignment_costs("shadow_costs", shadow_by_track, corrected_cost_matrix)

    records = []
    for track_index in range(track_count):
        production = production_by_track.get(track_index)
        shadow = shadow_by_track.get(track_index)
        production_det_index = None if production is None else production["det_index"]
        shadow_det_index = None if shadow is None else shadow["det_index"]
        track_gt_token = (
            None
            if track_gt_tokens is None
            else _native_scalar(track_gt_tokens[track_index])
        )
        production_det_gt_token = _assignment_gt_token(production, det_gt_tokens)
        shadow_det_gt_token = _assignment_gt_token(shadow, det_gt_tokens)

        production_pair_label = _pair_label(
            track_gt_token,
            production_det_gt_token,
            matched=production is not None,
        )
        shadow_pair_label = _pair_label(
            track_gt_token,
            shadow_det_gt_token,
            matched=shadow is not None,
        )
        outcome = classify_assignment_outcome(
            track_gt_token,
            production_det_gt_token,
            shadow_det_gt_token,
            production_matched=production is not None,
            shadow_matched=shadow is not None,
        )

        threshold = _class_threshold(class_thresholds, class_ids[track_index])
        _validate_selected_cost_threshold(
            "production/base",
            production,
            base_cost_matrix,
            track_index,
            threshold,
        )
        _validate_selected_cost_threshold(
            "shadow/corrected",
            shadow,
            corrected_cost_matrix,
            track_index,
            threshold,
        )
        production_base_cost = _selected_value(
            base_cost_matrix, track_index, production_det_index
        )
        shadow_base_cost = _selected_value(
            base_cost_matrix, track_index, shadow_det_index
        )
        original_cost_gap = (
            None
            if production_base_cost is None or shadow_base_cost is None
            else shadow_base_cost - production_base_cost
        )

        records.append(
            {
                "scene_id": context.get("scene_id"),
                "sample_token": context.get("sample_token"),
                "class_id": class_ids[track_index],
                "class_name": class_names[track_index],
                "track_id": track_ids[track_index],
                "track_state_bucket": state_buckets[track_index],
                "production_det_index": production_det_index,
                "shadow_det_index": shadow_det_index,
                "production_selected_cost": (
                    _selected_value(
                        base_cost_matrix, track_index, production_det_index
                    )
                ),
                "shadow_selected_cost": _selected_value(
                    corrected_cost_matrix, track_index, shadow_det_index
                ),
                "production_selected_association_score": _selected_value(
                    association_scores, track_index, production_det_index
                ),
                "shadow_selected_association_score": _selected_value(
                    association_scores, track_index, shadow_det_index
                ),
                "geometric_candidate_rank": _candidate_rank(
                    base_cost_matrix[track_index], production_det_index
                ),
                "shadow_candidate_rank": _candidate_rank(
                    base_cost_matrix[track_index], shadow_det_index
                ),
                "original_cost_gap": original_cost_gap,
                "assignment_changed": production_det_index != shadow_det_index,
                "threshold_crossing": _threshold_crossing(
                    track_index=track_index,
                    production_det_index=production_det_index,
                    shadow_det_index=shadow_det_index,
                    threshold=threshold,
                    base_cost_matrix=base_cost_matrix,
                    corrected_cost_matrix=corrected_cost_matrix,
                ),
                "track_gt_token": track_gt_token,
                "production_det_gt_token": production_det_gt_token,
                "shadow_det_gt_token": shadow_det_gt_token,
                "production_pair_label": production_pair_label,
                "shadow_pair_label": shadow_pair_label,
                "outcome": outcome,
            }
        )
    return records
