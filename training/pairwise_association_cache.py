from __future__ import annotations

import math
import hashlib
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np

from tracker.compat_utils import resolve_bev_cost_state


TRACKING_CLASS_TO_ID = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "motorcycle": 3,
    "bus": 4,
    "trailer": 5,
    "truck": 6,
}


def _tracking_category(category: str) -> str:
    if category in TRACKING_CLASS_TO_ID:
        return category
    suffix = str(category).split(".")[-1]
    if suffix in TRACKING_CLASS_TO_ID:
        return suffix
    if "pedestrian" in str(category):
        return "pedestrian"
    return str(category)


def _as_float_list(values: Any, length: int, default: float = 0.0) -> List[float]:
    if values is None:
        return [default] * length
    out = list(values)[:length]
    if len(out) < length:
        out.extend([default] * (length - len(out)))
    return [float(v) for v in out]


def _det_xyz(frame: Dict[str, Any]) -> List[float] | None:
    xyz = frame.get("det_global_xyz")
    if xyz is None:
        obs = frame.get("obs_feature_12")
        if obs is None:
            return None
        return _as_float_list(obs, 3)
    return _as_float_list(xyz, 3)


def _det_lwh(frame: Dict[str, Any]) -> List[float]:
    if frame.get("det_lwh") is not None:
        return _as_float_list(frame.get("det_lwh"), 3)
    obs = frame.get("obs_feature_12") or []
    return _as_float_list(obs[6:9], 3)


def _det_yaw(frame: Dict[str, Any]) -> float:
    if frame.get("det_yaw") is not None:
        return float(frame.get("det_yaw"))
    obs = frame.get("obs_feature_12") or []
    return float(obs[9]) if len(obs) > 9 else 0.0


def _det_score(frame: Dict[str, Any]) -> float:
    if frame.get("det_score") is not None:
        return float(frame.get("det_score"))
    obs = frame.get("obs_feature_12") or []
    return float(obs[11]) if len(obs) > 11 else 0.0


def _det_velocity_xy(frame: Dict[str, Any]) -> List[float]:
    if frame.get("det_velocity") is not None:
        return _as_float_list(frame.get("det_velocity"), 2)
    obs = frame.get("obs_feature_12") or []
    return _as_float_list(obs[3:5], 2)


def _anchor_frame(current_frame: Dict[str, Any], history_source: str) -> Dict[str, Any]:
    if history_source != "fusion" or not current_frame.get("fusion_valid", False):
        return current_frame
    fused = dict(current_frame)
    feature = _as_float_list(current_frame.get("fusion_feature_12"), 12)
    fused["det_global_xyz"] = _as_float_list(
        current_frame.get("fusion_global_xyz") or feature[:3], 3
    )
    fused["det_velocity"] = _as_float_list(
        current_frame.get("fusion_velocity") or feature[3:5], 2
    )
    fused["det_lwh"] = _as_float_list(current_frame.get("fusion_lwh") or feature[6:9], 3)
    fused["det_yaw"] = float(current_frame.get("fusion_yaw") if current_frame.get("fusion_yaw") is not None else feature[9])
    fused["obs_feature_12"] = feature
    return fused


def _predict_anchor_frame(current_frame: Dict[str, Any], future_frame: Dict[str, Any],
                          history_source: str = "det") -> Dict[str, Any]:
    current_frame = _anchor_frame(current_frame, history_source)
    predicted = dict(current_frame)
    xyz = _det_xyz(current_frame)
    if xyz is None:
        return predicted
    velocity = _det_velocity_xy(current_frame)
    dt = float(future_frame.get("timestamp", 0.0)) - float(current_frame.get("timestamp", 0.0))
    if dt <= 0.0:
        dt = 0.0
    predicted_xyz = [float(xyz[0] + velocity[0] * dt), float(xyz[1] + velocity[1] * dt), float(xyz[2])]
    predicted["det_global_xyz"] = predicted_xyz
    obs = _as_float_list(current_frame.get("obs_feature_12"), 12)
    obs[0] = predicted_xyz[0]
    obs[1] = predicted_xyz[1]
    obs[2] = predicted_xyz[2]
    predicted["obs_feature_12"] = obs
    return predicted


def _center_distance_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    xyz_a = _det_xyz(a)
    xyz_b = _det_xyz(b)
    if xyz_a is None or xyz_b is None:
        return float("inf")
    return float(math.hypot(xyz_a[0] - xyz_b[0], xyz_a[1] - xyz_b[1]))


def _lookup_class_value(mapping: Dict[Any, Any], class_id: int, default: Any) -> Any:
    if class_id in mapping:
        return mapping[class_id]
    if str(class_id) in mapping:
        return mapping[str(class_id)]
    return default


def _pose_xyz_lwh_yaw(frame: Dict[str, Any]) -> np.ndarray:
    xyz = _det_xyz(frame)
    if xyz is None:
        raise ValueError("Cannot calculate geometric cost for a frame without xyz")
    return np.asarray(xyz + _det_lwh(frame) + [_det_yaw(frame)], dtype=np.float64)


def _bev_corners(pose: np.ndarray) -> np.ndarray:
    x, y, _, length, width, _, yaw = pose.tolist()
    local = np.asarray([
        [length / 2.0, width / 2.0],
        [length / 2.0, -width / 2.0],
        [-length / 2.0, -width / 2.0],
        [-length / 2.0, width / 2.0],
    ], dtype=np.float64)
    cos_yaw = math.cos(yaw)
    sin_yaw = math.sin(yaw)
    rotation = np.asarray([[cos_yaw, -sin_yaw], [sin_yaw, cos_yaw]], dtype=np.float64)
    return local @ rotation.T + np.asarray([x, y], dtype=np.float64)


def _runtime_geometric_cost(
    anchor_frame: Dict[str, Any],
    candidate_frame: Dict[str, Any],
    category: str,
    runtime_cfg: Dict[str, Any],
) -> float:
    """Reproduce runtime ``1 - Ro_GDIoU`` for offline candidate mining."""
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - deployment dependency
        raise RuntimeError(
            "runtime_geometric inference-margin mining requires OpenCV (cv2)"
        ) from exc

    pose1 = _pose_xyz_lwh_yaw(anchor_frame)
    pose2 = _pose_xyz_lwh_yaw(candidate_frame)
    if np.any(~np.isfinite(pose1)) or np.any(~np.isfinite(pose2)):
        return float("inf")
    if np.any(pose1[3:6] <= 0.0) or np.any(pose2[3:6] <= 0.0):
        return float("inf")

    base_xy = pose1[:2].copy()
    rect1 = (
        tuple((pose1[:2] - base_xy).tolist()),
        (float(pose1[3]), float(pose1[4])),
        float(pose1[6] * 180.0 / math.pi),
    )
    rect2 = (
        tuple((pose2[:2] - base_xy).tolist()),
        (float(pose2[3]), float(pose2[4])),
        float(pose2[6] * 180.0 / math.pi),
    )
    intersection = cv2.rotatedRectangleIntersection(rect1, rect2)[1]
    if intersection is None:
        intersection_area = 0.0
    else:
        hull = cv2.convexHull(intersection, returnPoints=True)
        intersection_area = float(cv2.contourArea(hull))

    area1 = float(pose1[3] * pose1[4])
    area2 = float(pose2[3] * pose2[4])
    union_area = area1 + area2 - intersection_area
    iou = intersection_area / max(union_area, 1e-8)

    union_points = np.concatenate([_bev_corners(pose1), _bev_corners(pose2)], axis=0)
    union_points = (union_points - pose1[:2]).astype(np.float32)
    enclosing_rect = cv2.minAreaRect(union_points)
    universe_area = float(enclosing_rect[1][0] * enclosing_rect[1][1])
    universe_distance = float(np.linalg.norm(np.asarray(enclosing_rect[1], dtype=np.float64)) ** 2)
    extra_area = universe_area - union_area
    center_distance_sq = float(np.sum((pose1[:2] - pose2[:2]) ** 2))

    class_id = TRACKING_CLASS_TO_ID[category]
    weights = runtime_cfg.get("THRESHOLD", {}).get("BEV", {}).get("WEIGHT_RO_GDIOU", {})
    class_weights = _lookup_class_value(weights, class_id, {"w1": 1.0, "w2": 1.0})
    ro_gdiou = (
        iou
        - float(class_weights.get("w1", 1.0)) * extra_area / max(universe_area, 1e-8)
        - float(class_weights.get("w2", 1.0)) * center_distance_sq / max(universe_distance, 1e-8)
    )
    return float(1.0 - ro_gdiou)


def _backpredict_candidate_frame(candidate_frame: Dict[str, Any]) -> Dict[str, Any]:
    backpredicted = dict(candidate_frame)
    xyz = _det_xyz(candidate_frame)
    if xyz is None:
        return backpredicted
    velocity = _det_velocity_xy(candidate_frame)
    # tracker.BBox.backward_prediction uses the fixed nuScenes 2 Hz interval.
    backpredicted["det_global_xyz"] = [
        float(xyz[0] - velocity[0] * 0.5),
        float(xyz[1] - velocity[1] * 0.5),
        float(xyz[2]),
    ]
    return backpredicted


def _runtime_matching_cost(
    current_frame: Dict[str, Any],
    future_frame: Dict[str, Any],
    candidate_frame: Dict[str, Any],
    category: str,
    history_source: str,
    runtime_cfg: Dict[str, Any],
) -> float:
    class_id = TRACKING_CLASS_TO_ID[category]
    cost_states = runtime_cfg.get("MATCHING", {}).get("BEV", {}).get("COST_STATE", {})
    cost_state = resolve_bev_cost_state(runtime_cfg, class_id, _lookup_class_value(cost_states, class_id, "Predict")).lower()
    predicted_anchor = _predict_anchor_frame(current_frame, future_frame, history_source)
    pred_cost = _runtime_geometric_cost(predicted_anchor, candidate_frame, category, runtime_cfg)
    if cost_state == "predict":
        return pred_cost

    back_anchor = _anchor_frame(current_frame, history_source)
    back_candidate = _backpredict_candidate_frame(candidate_frame)
    back_cost = _runtime_geometric_cost(back_anchor, back_candidate, category, runtime_cfg)
    if cost_state == "backpredict":
        return back_cost
    if cost_state == "fusion":
        ratios = runtime_cfg.get("THRESHOLD", {}).get("COST_STATE_PREDICT_RATION", {})
        ratio = float(_lookup_class_value(ratios, class_id, 0.5))
        return float(ratio * pred_cost + (1.0 - ratio) * back_cost)
    raise ValueError(f"Unsupported runtime BEV COST_STATE={cost_state!r} for class {category}")


def _wrap_to_pi(value: float) -> float:
    return float(value - 2.0 * math.pi * round(value / (2.0 * math.pi)))


def _is_hard_negative_type(value: Any) -> bool:
    return str(value).strip().lower() in {"hard", "inference_margin", "inference_topk"}


def _state_bucket_from_history(frames: List[Dict[str, Any]], current_index: int) -> str:
    return "matched" if bool(frames[current_index].get("is_matched", False)) else "unmatched"


def _selected_history_frames(frames: List[Dict[str, Any]], end_index: int,
                             history_len: int, history_source: str) -> List[Dict[str, Any]]:
    selected = frames[max(0, end_index - history_len + 1):end_index + 1]
    if history_source == "fusion":
        # Offline tracklets retain frames across a pseudo-track death and
        # rebirth. The online tracker starts a new history after that gap.
        for index in range(len(selected) - 1, -1, -1):
            if not selected[index].get("fusion_valid", False):
                return selected[index + 1:]
            if selected[index].get("fusion_is_birth", False):
                return selected[index:]
    return selected


def _history_feature(
    frames: List[Dict[str, Any]],
    end_index: int,
    history_len: int,
    *,
    history_source: str = "det",
) -> List[List[float]]:
    source = str(history_source).strip().lower()
    if source not in {"det", "fusion"}:
        raise ValueError(f"Unsupported history_source={history_source!r}; expected 'det' or 'fusion'")
    selected = _selected_history_frames(frames, end_index, history_len, source)
    pad = [[0.0] * 12 for _ in range(max(0, history_len - len(selected)))]
    if selected:
        latest = selected[-1]
        reference = (
            (latest.get("fusion_global_xyz") or _as_float_list(latest.get("fusion_feature_12"), 12)[:3])
            if source == "fusion" and latest.get("fusion_valid", False)
            else _det_xyz(latest)
        )
    else:
        reference = None
    ref_xy = _as_float_list(reference, 3)[:2] if reference is not None else [0.0, 0.0]
    features = []
    for frame in selected:
        if source == "fusion":
            if bool(frame.get("fusion_valid", False)):
                feature = _as_float_list(frame.get("fusion_feature_12"), 12)
            else:
                feature = [0.0] * 12
        else:
            feature = _as_float_list(frame.get("obs_feature_12"), 12)
        if feature != [0.0] * 12:
            feature[0] -= ref_xy[0]
            feature[1] -= ref_xy[1]
        features.append(feature)
    if features:
        features[0][10] = 0.0 # no preceding yaw exists inside this window
    return pad + features


def _anchor_history_masks(frames: List[Dict[str, Any]], end_index: int, history_len: int,
                          history_source: str) -> Tuple[List[bool], List[bool]]:
    selected = _selected_history_frames(frames, end_index, history_len, history_source)
    padding = [False] * max(0, history_len - len(selected))
    if history_source == "fusion":
        valid = [bool(frame.get("fusion_valid", False)) for frame in selected]
        matched = [v and not bool(frame.get("fusion_is_fake", False))
                   for v, frame in zip(valid, selected)]
    else:
        valid = [True] * len(selected)
        matched = [bool(frame.get("is_matched", False)) for frame in selected]
    return padding + valid, padding + matched


def _inference_detection_history_feature(frame: Dict[str, Any], history_len: int) -> List[List[float]]:
    obs = _as_float_list(frame.get("obs_feature_12"), 12)
    token = [
        0.0,
        0.0,
        float(obs[2]),
        float(obs[3]),
        float(obs[4]),
        0.0,
        float(obs[6]),
        float(obs[7]),
        float(obs[8]),
        float(obs[9]),
        0.0,
        float(obs[11]),
    ]
    return [[0.0] * 12 for _ in range(max(0, history_len - 1))] + [token]


def _index_matched_frames(tracklets: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for trk in tracklets:
        category = _tracking_category(trk.get("category", ""))
        for frame_index, frame in enumerate(trk.get("frames", [])):
            key = (
                str(frame.get("scene_id", "")),
                str(frame.get("sample_token", "")),
                category,
            )
            if bool(frame.get("is_matched", False)):
                index[key].append({
                    "tracklet": trk,
                    "frame": frame,
                    "frame_index": frame_index,
                    "instance_token": trk.get("instance_token"),
                    "category": category,
                })
            for detection in frame.get("unmatched_detections", []):
                index[key].append({
                    "tracklet": None,
                    "frame": detection,
                    "frame_index": frame_index,
                    "instance_token": detection["candidate_id"],
                    "category": category,
                    "unmatched_gt": True,
                    "possible_gt_tokens": list(detection.get("possible_gt_tokens", [])),
                })
    return index


def _make_pair(
    *,
    anchor_tracklet: Dict[str, Any],
    current_frame: Dict[str, Any],
    future_frame: Dict[str, Any],
    candidate: Dict[str, Any],
    history_len: int,
    current_index: int,
    history_source: str,
    pair_geometry_source: str,
    label: int,
    negative_type: str,
    negative_mining_mode: str = "legacy",
    candidate_rank: int | None = None,
    best_candidate_distance: float | None = None,
    candidate_mining_cost: float | None = None,
    best_candidate_cost: float | None = None,
    inference_margin_metric: str = "center_distance",
    class_window_cfg: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    candidate_frame = candidate["frame"]
    category = _tracking_category(anchor_tracklet.get("category", ""))
    if pair_geometry_source == "predicted_track_candidate":
        geometry_anchor = _predict_anchor_frame(current_frame, future_frame, history_source)
    elif pair_geometry_source == "track_candidate":
        geometry_anchor = _anchor_frame(current_frame, history_source)
    else:
        geometry_anchor = future_frame
    anchor_lwh = _det_lwh(geometry_anchor)
    candidate_lwh = _det_lwh(candidate_frame)
    center_distance = _center_distance_xy(geometry_anchor, candidate_frame)
    frames = anchor_tracklet.get("frames", [])
    window_max = max(1, min(history_len, int(
        (class_window_cfg or {}).get(category, {}).get("MAX_HISTORY_LEN", history_len)
    )))
    history_frames = frames[max(0, current_index - window_max + 1):current_index + 1]
    history_end = len(history_frames) - 1
    anchor_history_mask, anchor_history_match_mask = _anchor_history_masks(
        history_frames, history_end, history_len, history_source
    )
    current_xyz = (
        (current_frame.get("fusion_global_xyz") or
         _as_float_list(current_frame.get("fusion_feature_12"), 12)[:3])
        if history_source == "fusion" and current_frame.get("fusion_valid", False)
        else _det_xyz(current_frame)
    )
    current_xyz = _as_float_list(current_xyz, 3)
    candidate_xyz = _det_xyz(candidate_frame)
    candidate_xyz = _as_float_list(candidate_xyz, 3)
    return {
        "feature_schema_version": 3,
        "category": category,
        "class_id": TRACKING_CLASS_TO_ID.get(category, -1),
        "state_bucket": _state_bucket_from_history(anchor_tracklet.get("frames", []), current_index),
        "anchor_instance_token": anchor_tracklet.get("instance_token"),
        "candidate_instance_token": candidate.get("instance_token"),
        "candidate_is_unmatched_gt": bool(candidate.get("unmatched_gt", False)),
        "current_sample_token": current_frame.get("sample_token"),
        "future_sample_token": future_frame.get("sample_token"),
        "scene_id": future_frame.get("scene_id"),
        "label": int(label),
        "negative_type": negative_type,
        "negative_mining_mode": str(negative_mining_mode).strip().lower(),
        "history_source": str(history_source).strip().lower(),
        "pair_geometry_source": str(pair_geometry_source).strip().lower(),
        "candidate_rank": int(candidate_rank) if candidate_rank is not None else None,
        "best_candidate_distance": (
            float(best_candidate_distance)
            if best_candidate_distance is not None and math.isfinite(float(best_candidate_distance))
            else None
        ),
        "inference_margin_metric": str(inference_margin_metric).strip().lower(),
        "candidate_mining_cost": (
            float(candidate_mining_cost)
            if candidate_mining_cost is not None and math.isfinite(float(candidate_mining_cost))
            else None
        ),
        "best_candidate_cost": (
            float(best_candidate_cost)
            if best_candidate_cost is not None and math.isfinite(float(best_candidate_cost))
            else None
        ),
        "cost_margin_gap": (
            float(candidate_mining_cost - best_candidate_cost)
            if candidate_mining_cost is not None
            and best_candidate_cost is not None
            and math.isfinite(float(candidate_mining_cost))
            and math.isfinite(float(best_candidate_cost))
            else None
        ),
        "anchor_history_12": _history_feature(
            history_frames,
            history_end,
            history_len,
            history_source=history_source,
        ),
        "positive_obs_feature_12": _as_float_list(future_frame.get("obs_feature_12"), 12),
        "candidate_obs_feature_12": _as_float_list(candidate_frame.get("obs_feature_12"), 12),
        "candidate_history_12": _inference_detection_history_feature(candidate_frame, history_len),
        "anchor_history_mask": anchor_history_mask,
        "anchor_history_match_mask": anchor_history_match_mask,
        "anchor_current_range": float(math.hypot(current_xyz[0], current_xyz[1])),
        "anchor_detection_driven": bool(current_frame.get("is_matched", False)),
        "candidate_current_range": float(math.hypot(candidate_xyz[0], candidate_xyz[1])),
        "center_distance": center_distance,
        "margin_gap": (
            float(center_distance - best_candidate_distance)
            if best_candidate_distance is not None and math.isfinite(float(best_candidate_distance))
            else None
        ),
        "yaw_diff": abs(_wrap_to_pi(_det_yaw(geometry_anchor) - _det_yaw(candidate_frame))),
        "size_l1": float(sum(abs(a - b) for a, b in zip(anchor_lwh, candidate_lwh))),
        "anchor_det_score": _det_score(current_frame),
        "candidate_det_score": _det_score(candidate_frame),
    }


def build_pairwise_association_samples(
    tracklets: List[Dict[str, Any]],
    *,
    history_len: int = 8,
    history_source: str = "det",
    pair_geometry_source: str = "predicted_track_candidate",
    future_step: int = 1,
    hard_negative_distance: float = 4.0,
    hard_negative_distance_by_class: Dict[str, float] | None = None,
    max_hard_negatives: int = 4,
    min_hard_negatives: int = 0,
    max_easy_negatives: int = 2,
    negative_mining_mode: str = "legacy",
    cost_margin_eps: float = 0.05,
    inference_margin_metric: str = "center_distance",
    runtime_cost_config: Dict[str, Any] | None = None,
    max_pairs_per_class: Dict[str, int] | None = None,
    require_current_match: bool = True,
    require_future_match: bool = True,
    sampling_seed: int = 0,
    class_window_cfg: Dict[str, Any] | None = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build same-frame same-class pairwise association samples.

    This is an offline cache builder. It does not alter tracker inference.
    Positives are the anchor tracklet's matched future detection. Negatives are
    other matched detections from the same scene/sample/category.
    """
    history_source = str(history_source).strip().lower()
    if history_source not in {"det", "fusion"}:
        raise ValueError(f"Unsupported history_source={history_source!r}; expected 'det' or 'fusion'")
    pair_geometry_source = str(pair_geometry_source).strip().lower()
    if pair_geometry_source not in {"predicted_track_candidate", "track_candidate", "future_candidate"}:
        raise ValueError(
            f"Unsupported pair_geometry_source={pair_geometry_source!r}; "
            "expected 'predicted_track_candidate', 'track_candidate', or 'future_candidate'"
        )
    negative_mining_mode = str(negative_mining_mode).strip().lower()
    if negative_mining_mode not in {"legacy", "inference_margin"}:
        raise ValueError(
            f"Unsupported negative_mining_mode={negative_mining_mode!r}; "
            "expected 'legacy' or 'inference_margin'"
        )
    inference_margin_metric = str(inference_margin_metric).strip().lower()
    if inference_margin_metric not in {"center_distance", "runtime_geometric"}:
        raise ValueError(
            f"Unsupported inference_margin_metric={inference_margin_metric!r}; "
            "expected 'center_distance' or 'runtime_geometric'"
        )
    ranking_metric = (
        inference_margin_metric if negative_mining_mode == "inference_margin" else "center_distance"
    )
    if ranking_metric == "runtime_geometric" and not runtime_cost_config:
        raise ValueError(
            "runtime_cost_config is required when inference_margin_metric='runtime_geometric'"
        )
    frame_index = _index_matched_frames(tracklets)
    samples: List[Dict[str, Any]] = []
    hard_distance_by_class = {
        str(key): float(value)
        for key, value in (hard_negative_distance_by_class or {}).items()
    }
    pair_caps = {
        str(key): int(value)
        for key, value in (max_pairs_per_class or {}).items()
        if int(value) > 0
    }
    pair_counts_by_class: Dict[str, int] = defaultdict(int)
    pending_group = []

    def _append_sample(sample: Dict[str, Any]) -> bool:
        pending_group.append(sample)
        return True

    def _retain_group():
        if not pending_group:
            return
        group = list(pending_group)
        first = group[0]
        category_name = str(first.get("category", ""))
        cap = pair_caps.get(category_name)
        if cap is None:
            samples.extend(group)
            return
        if pair_counts_by_class[category_name] + len(group) <= cap:
            samples.extend(group)
            pair_counts_by_class[category_name] += len(group)

    # Randomize ANCHORS before applying caps, not individual pairs afterwards.
    # Only store small (tracklet,index) references for ordering, not all vectors.
    anchor_windows = [
        (trk, index) for trk in tracklets if not trk.get("association_only", False)
        for index in range(max(0, len(trk.get("frames", [])) - future_step))
    ]
    if pair_caps:
        def priority(window):
            trk, index = window
            frame = trk["frames"][index]
            identity = repr((sampling_seed, trk.get("scene_id", frame.get("scene_id")),
                             trk.get("instance_token"), frame.get("sample_token")))
            return hashlib.sha256(identity.encode("utf-8")).digest()
        anchor_windows.sort(key=priority)

    for trk, selected_index in anchor_windows:
        category = _tracking_category(trk.get("category", ""))
        frames = trk.get("frames", [])
        if len(frames) <= future_step:
            continue
        current_index = selected_index
        pending_group.clear()
        current_frame = frames[current_index]
        future_frame = frames[current_index + future_step]
        if history_source == "fusion" and not all(
            bool(frame.get("fusion_valid", False))
            for frame in frames[current_index:current_index + future_step]
        ):
            # A dead pseudo-track cannot be associated to a later
            # detection even when both ends share a GT instance token.
            continue
        if history_source != "fusion" and not bool(current_frame.get("is_matched", False)):
            # Raw misses have no state to encode; only live fusion states
            # can represent coasted track anchors.
            continue
        lifecycle_id = current_frame.get("fusion_lifecycle_id")
        if history_source == "fusion" and lifecycle_id is not None and any(
            fr.get("fusion_lifecycle_id") != lifecycle_id
            for fr in frames[current_index:current_index + future_step + 1]
        ):
            continue
        if require_current_match and not bool(current_frame.get("is_matched", False)):
            continue
        if require_future_match and not bool(future_frame.get("is_matched", False)):
            continue

        key = (
            str(future_frame.get("scene_id", "")),
            str(future_frame.get("sample_token", "")),
            category,
        )
        candidates = frame_index.get(key, [])
        positive_candidates = [
            candidate for candidate in candidates
            if candidate.get("instance_token") == trk.get("instance_token")
        ]
        if not positive_candidates and require_future_match:
            continue

        if pair_geometry_source == "predicted_track_candidate":
            geometry_anchor = _predict_anchor_frame(current_frame, future_frame, history_source)
        elif pair_geometry_source == "track_candidate":
            geometry_anchor = _anchor_frame(current_frame, history_source)
        else:
            geometry_anchor = future_frame
        ranked_candidates = sorted(
            [
                (
                    rank_candidate,
                    (
                        _runtime_matching_cost(
                            current_frame,
                            future_frame,
                            rank_candidate["frame"],
                            category,
                            history_source,
                            runtime_cost_config,
                        )
                        if ranking_metric == "runtime_geometric"
                        else _center_distance_xy(geometry_anchor, rank_candidate["frame"])
                    ),
                )
                for rank_candidate in candidates
            ],
            key=lambda item: item[1],
        )
        candidate_rank = {
            candidate.get("instance_token"): rank
            for rank, (candidate, _) in enumerate(ranked_candidates, start=1)
        }
        candidate_cost = {
            candidate.get("instance_token"): cost
            for candidate, cost in ranked_candidates
        }
        candidate_center_distance = {
            candidate.get("instance_token"): _center_distance_xy(
                geometry_anchor, candidate["frame"]
            )
            for candidate, _ in ranked_candidates
        }
        best_candidate_cost = (
            float(ranked_candidates[0][1])
            if ranked_candidates and math.isfinite(float(ranked_candidates[0][1]))
            else None
        )
        best_candidate_center_distance = (
            min(candidate_center_distance.values())
            if candidate_center_distance
            else None
        )

        if positive_candidates:
            _append_sample(_make_pair(
                anchor_tracklet=trk,
                current_frame=current_frame,
                future_frame=future_frame,
                candidate=positive_candidates[0],
                history_len=history_len,
                current_index=current_index,
                history_source=history_source,
                pair_geometry_source=pair_geometry_source,
                label=1,
                negative_type="positive",
                negative_mining_mode=negative_mining_mode,
                inference_margin_metric=ranking_metric,
                class_window_cfg=class_window_cfg,
                candidate_rank=candidate_rank.get(positive_candidates[0].get("instance_token")),
                best_candidate_distance=best_candidate_center_distance,
                candidate_mining_cost=candidate_cost.get(positive_candidates[0].get("instance_token")),
                best_candidate_cost=best_candidate_cost,
            ))

        negatives = [
            candidate for candidate in candidates
            if candidate.get("instance_token") != trk.get("instance_token")
            and trk.get("instance_token") not in candidate.get("possible_gt_tokens", [])
        ]
        if negative_mining_mode == "inference_margin":
            negatives.sort(key=lambda candidate: candidate_cost.get(candidate.get("instance_token"), float("inf")))
            if best_candidate_cost is None:
                hard = []
            else:
                hard = [
                    candidate for candidate in negatives
                    if candidate_cost.get(candidate.get("instance_token"), float("inf"))
                    <= best_candidate_cost + float(cost_margin_eps)
                ][: max(0, int(max_hard_negatives))]
            hard_ids = {candidate.get("instance_token") for candidate in hard}
            inference_topk_ids = set()
            if len(hard) < int(min_hard_negatives):
                fallback = [
                    candidate for candidate in negatives
                    if candidate.get("instance_token") not in hard_ids
                ][: max(0, min(int(min_hard_negatives) - len(hard), int(max_hard_negatives) - len(hard)))]
                inference_topk_ids = {candidate.get("instance_token") for candidate in fallback}
                hard.extend(fallback)
            easy = []
        else:
            negative_anchor = future_frame if future_frame.get("is_matched", False) else geometry_anchor
            negatives.sort(key=lambda candidate: _center_distance_xy(negative_anchor, candidate["frame"]))
            hard_distance = float(hard_distance_by_class.get(category, hard_negative_distance))

            hard = [
                candidate for candidate in negatives
                if _center_distance_xy(negative_anchor, candidate["frame"]) <= hard_distance
            ][: max(0, int(max_hard_negatives))]
            hard_ids = {candidate.get("instance_token") for candidate in hard}
            inference_topk_ids = set()
            easy = [
                candidate for candidate in negatives
                if candidate.get("instance_token") not in hard_ids
            ][: max(0, int(max_easy_negatives))]

        for candidate in hard:
            _append_sample(_make_pair(
                anchor_tracklet=trk,
                current_frame=current_frame,
                future_frame=future_frame,
                candidate=candidate,
                history_len=history_len,
                current_index=current_index,
                history_source=history_source,
                pair_geometry_source=pair_geometry_source,
                label=0,
                negative_type=(
                    "inference_topk"
                    if negative_mining_mode == "inference_margin"
                    and candidate.get("instance_token") in inference_topk_ids
                    else ("inference_margin" if negative_mining_mode == "inference_margin" else "hard")
                ),
                negative_mining_mode=negative_mining_mode,
                inference_margin_metric=ranking_metric,
                class_window_cfg=class_window_cfg,
                candidate_rank=candidate_rank.get(candidate.get("instance_token")),
                best_candidate_distance=best_candidate_center_distance,
                candidate_mining_cost=candidate_cost.get(candidate.get("instance_token")),
                best_candidate_cost=best_candidate_cost,
            ))
        for candidate in easy:
            _append_sample(_make_pair(
                anchor_tracklet=trk,
                current_frame=current_frame,
                future_frame=future_frame,
                candidate=candidate,
                history_len=history_len,
                current_index=current_index,
                history_source=history_source,
                pair_geometry_source=pair_geometry_source,
                label=0,
                negative_type="easy",
                negative_mining_mode=negative_mining_mode,
                inference_margin_metric=ranking_metric,
                class_window_cfg=class_window_cfg,
                candidate_rank=candidate_rank.get(candidate.get("instance_token")),
                best_candidate_distance=best_candidate_center_distance,
                candidate_mining_cost=candidate_cost.get(candidate.get("instance_token")),
                best_candidate_cost=best_candidate_cost,
            ))

        _retain_group()

    return samples, summarize_pairwise_association_samples(samples)


def summarize_pairwise_association_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "pairs": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "unmatched_detection_negative_pairs": 0,
        "hard_negative_pairs": 0,
        "easy_negative_pairs": 0,
        "anchors": set(),
        "candidate_scores": [],
        "center_distances": [],
        "cost_margin_gaps": [],
        "state_buckets": defaultdict(int),
        "negative_types": defaultdict(int),
        "inference_margin_metrics": defaultdict(int),
    })
    overall = {
        "pairs": 0,
        "candidate_pairs": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "unmatched_detection_negative_pairs": 0,
        "hard_negative_pairs": 0,
        "easy_negative_pairs": 0,
        "state_buckets": defaultdict(int),
        "negative_types": defaultdict(int),
        "inference_margin_metrics": defaultdict(int),
    }
    for sample in samples:
        cls = str(sample.get("category", "unknown"))
        stat = per_class[cls]
        label = int(sample.get("label", 0))
        negative_type = str(sample.get("negative_type", ""))
        state_bucket = str(sample.get("state_bucket", "unknown"))
        margin_metric = str(sample.get("inference_margin_metric", "unknown"))
        stat["pairs"] += 1
        overall["pairs"] += 1
        overall["candidate_pairs"] += 1
        stat["anchors"].add((sample.get("anchor_instance_token"), sample.get("current_sample_token")))
        stat["state_buckets"][state_bucket] += 1
        overall["state_buckets"][state_bucket] += 1
        stat["negative_types"][negative_type] += 1
        overall["negative_types"][negative_type] += 1
        stat["inference_margin_metrics"][margin_metric] += 1
        overall["inference_margin_metrics"][margin_metric] += 1
        if label == 1:
            stat["positive_pairs"] += 1
            overall["positive_pairs"] += 1
        else:
            stat["negative_pairs"] += 1
            overall["negative_pairs"] += 1
            if bool(sample.get("candidate_is_unmatched_gt", False)):
                stat["unmatched_detection_negative_pairs"] += 1
                overall["unmatched_detection_negative_pairs"] += 1
            if _is_hard_negative_type(negative_type):
                stat["hard_negative_pairs"] += 1
                overall["hard_negative_pairs"] += 1
            elif negative_type == "easy":
                stat["easy_negative_pairs"] += 1
                overall["easy_negative_pairs"] += 1
        score = sample.get("candidate_det_score")
        if isinstance(score, (int, float)) and math.isfinite(float(score)):
            stat["candidate_scores"].append(float(score))
        dist = sample.get("center_distance")
        if isinstance(dist, (int, float)) and math.isfinite(float(dist)):
            stat["center_distances"].append(float(dist))
        cost_gap = sample.get("cost_margin_gap")
        if isinstance(cost_gap, (int, float)) and math.isfinite(float(cost_gap)):
            stat["cost_margin_gaps"].append(float(cost_gap))

    out_per_class = {}
    for cls, stat in sorted(per_class.items()):
        scores = stat["candidate_scores"]
        distances = stat["center_distances"]
        cost_gaps = stat["cost_margin_gaps"]
        out_per_class[cls] = {
            "pairs": stat["pairs"],
            "candidate_pairs": stat["pairs"],
            "anchors": len(stat["anchors"]),
            "positive_pairs": stat["positive_pairs"],
            "negative_pairs": stat["negative_pairs"],
            "unmatched_detection_negative_pairs": stat["unmatched_detection_negative_pairs"],
            "hard_negative_pairs": stat["hard_negative_pairs"],
            "easy_negative_pairs": stat["easy_negative_pairs"],
            "positive_ratio": (
                float(stat["positive_pairs"] / stat["pairs"])
                if stat["pairs"] else 0.0
            ),
            "negative_ratio": (
                float(stat["negative_pairs"] / stat["pairs"])
                if stat["pairs"] else 0.0
            ),
            "hard_negative_ratio": (
                float(stat["hard_negative_pairs"] / stat["pairs"])
                if stat["pairs"] else 0.0
            ),
            "negative_per_positive": (
                float(stat["negative_pairs"] / stat["positive_pairs"])
                if stat["positive_pairs"] else 0.0
            ),
            "candidate_score_mean": float(np.mean(scores)) if scores else 0.0,
            "center_distance_mean": float(np.mean(distances)) if distances else 0.0,
            "cost_margin_gap_mean": float(np.mean(cost_gaps)) if cost_gaps else 0.0,
            "state_buckets": dict(sorted(stat["state_buckets"].items())),
            "negative_types": dict(sorted(stat["negative_types"].items())),
            "inference_margin_metrics": dict(sorted(stat["inference_margin_metrics"].items())),
        }
    overall["classes"] = len(out_per_class)
    overall["positive_ratio"] = (
        float(overall["positive_pairs"] / overall["pairs"])
        if overall["pairs"] else 0.0
    )
    overall["negative_ratio"] = (
        float(overall["negative_pairs"] / overall["pairs"])
        if overall["pairs"] else 0.0
    )
    overall["hard_negative_ratio"] = (
        float(overall["hard_negative_pairs"] / overall["pairs"])
        if overall["pairs"] else 0.0
    )
    overall["negative_per_positive"] = (
        float(overall["negative_pairs"] / overall["positive_pairs"])
        if overall["positive_pairs"] else 0.0
    )
    overall["state_buckets"] = dict(sorted(overall["state_buckets"].items()))
    overall["negative_types"] = dict(sorted(overall["negative_types"].items()))
    overall["inference_margin_metrics"] = dict(sorted(overall["inference_margin_metrics"].items()))
    return {"overall": overall, "per_class": out_per_class}
