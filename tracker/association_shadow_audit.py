"""Pure helpers for comparing production and shadow assignments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


def _is_known_gt_token(token) -> bool:
    if token is None:
        return False
    if isinstance(token, str):
        return bool(token.strip())
    if isinstance(token, (list, tuple, set, dict, np.ndarray)):
        return False
    try:
        return not bool(np.isnan(token))
    except (TypeError, ValueError):
        return True


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
    track_gt_token, production_det_token, shadow_det_token
) -> str:
    """Classify whether a shadow choice changes GT-consistent correctness."""
    if not _is_known_gt_token(track_gt_token):
        return "unknown"
    if production_det_token is not None and not _is_known_gt_token(production_det_token):
        return "unknown"
    if shadow_det_token is not None and not _is_known_gt_token(shadow_det_token):
        return "unknown"

    production_correct = production_det_token == track_gt_token
    shadow_correct = shadow_det_token == track_gt_token
    if shadow_correct and not production_correct:
        return "improved"
    if production_correct and not shadow_correct:
        return "damaged"
    return "neutral"


def _matrix(name, value, expected_shape=None) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional array")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must contain numeric values")
    if expected_shape is not None and array.shape != expected_shape:
        raise ValueError(
            f"{name} must have shape {expected_shape}, got {array.shape}"
        )
    return array


def _validate_track_metadata(name: str, values: Sequence, track_count: int) -> None:
    if len(values) != track_count:
        raise ValueError(f"{name} must contain one value per track ({track_count})")


def _native_scalar(value):
    return value.item() if isinstance(value, np.generic) else value


def _selected_value(matrix: np.ndarray, track_index: int, det_index):
    if det_index is None:
        return None
    value = float(matrix[track_index, det_index])
    return value if np.isfinite(value) else None


def _candidate_rank(base_row: np.ndarray, det_index) -> int | None:
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
    if not _is_known_gt_token(track_gt_token) or not _is_known_gt_token(det_gt_token):
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


def _accepted_match(assignment, threshold) -> bool:
    if assignment is None:
        return False
    if threshold is None:
        return True
    return assignment["cost"] <= threshold


def _threshold_crossing(production, shadow, threshold) -> bool:
    if (production is None) != (shadow is None):
        return True
    if production is None:
        return False
    return _accepted_match(production, threshold) != _accepted_match(shadow, threshold)


def _assignment_gt_token(assignment, det_gt_tokens):
    if assignment is None:
        return None
    if det_gt_tokens is None:
        return None
    return _native_scalar(det_gt_tokens[assignment["det_index"]])


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
    """Return one side-effect-free production/shadow decision record per track."""
    base_cost_matrix = _matrix("base_cost_matrix", base_cost_matrix)
    matrix_shape = base_cost_matrix.shape
    corrected_cost_matrix = _matrix(
        "corrected_cost_matrix", corrected_cost_matrix, matrix_shape
    )
    association_scores = _matrix(
        "association_scores", association_scores, matrix_shape
    )
    track_count, det_count = matrix_shape

    for name, values in (
        ("track_ids", track_ids),
        ("class_ids", class_ids),
        ("class_names", class_names),
        ("state_buckets", state_buckets),
    ):
        _validate_track_metadata(name, values, track_count)
    if track_gt_tokens is not None:
        _validate_track_metadata("track_gt_tokens", track_gt_tokens, track_count)
    if det_gt_tokens is not None and len(det_gt_tokens) != det_count:
        raise ValueError(
            f"det_gt_tokens must contain one value per detection ({det_count})"
        )
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
        selected_gt_unknown = (
            (production is not None and production_pair_label is None)
            or (shadow is not None and shadow_pair_label is None)
        )
        if not _is_known_gt_token(track_gt_token) or selected_gt_unknown:
            outcome = "unknown"
        else:
            outcome = classify_assignment_outcome(
                track_gt_token,
                production_det_gt_token,
                shadow_det_gt_token,
            )

        threshold = _class_threshold(class_thresholds, class_ids[track_index])
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
                "class_id": _native_scalar(class_ids[track_index]),
                "class_name": str(class_names[track_index]),
                "track_id": _native_scalar(track_ids[track_index]),
                "track_state_bucket": str(state_buckets[track_index]),
                "production_det_index": production_det_index,
                "shadow_det_index": shadow_det_index,
                "production_selected_cost": (
                    None if production is None else production["cost"]
                ),
                "shadow_selected_cost": None if shadow is None else shadow["cost"],
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
                    production, shadow, threshold
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
