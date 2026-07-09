from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np


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


def _center_distance_xy(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    xyz_a = _det_xyz(a)
    xyz_b = _det_xyz(b)
    if xyz_a is None or xyz_b is None:
        return float("inf")
    return float(math.hypot(xyz_a[0] - xyz_b[0], xyz_a[1] - xyz_b[1]))


def _wrap_to_pi(value: float) -> float:
    return float(value - 2.0 * math.pi * round(value / (2.0 * math.pi)))


def _state_bucket_from_history(frames: List[Dict[str, Any]], current_index: int) -> str:
    return "matched" if bool(frames[current_index].get("is_matched", False)) else "unmatched"


def _history_feature(frames: List[Dict[str, Any]], end_index: int, history_len: int) -> List[List[float]]:
    start = max(0, end_index - history_len + 1)
    selected = frames[start : end_index + 1]
    pad = [[0.0] * 12 for _ in range(max(0, history_len - len(selected)))]
    features = [_as_float_list(frame.get("obs_feature_12"), 12) for frame in selected]
    return pad + features


def _index_matched_frames(tracklets: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str, str], List[Dict[str, Any]]]:
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for trk in tracklets:
        category = _tracking_category(trk.get("category", ""))
        for frame_index, frame in enumerate(trk.get("frames", [])):
            if not bool(frame.get("is_matched", False)):
                continue
            key = (
                str(frame.get("scene_id", "")),
                str(frame.get("sample_token", "")),
                category,
            )
            item = {
                "tracklet": trk,
                "frame": frame,
                "frame_index": frame_index,
                "instance_token": trk.get("instance_token"),
                "category": category,
            }
            index[key].append(item)
    return index


def _make_pair(
    *,
    anchor_tracklet: Dict[str, Any],
    current_frame: Dict[str, Any],
    future_frame: Dict[str, Any],
    candidate: Dict[str, Any],
    history_len: int,
    current_index: int,
    label: int,
    negative_type: str,
) -> Dict[str, Any]:
    candidate_frame = candidate["frame"]
    category = _tracking_category(anchor_tracklet.get("category", ""))
    anchor_lwh = _det_lwh(future_frame)
    candidate_lwh = _det_lwh(candidate_frame)
    return {
        "category": category,
        "class_id": TRACKING_CLASS_TO_ID.get(category, -1),
        "state_bucket": _state_bucket_from_history(anchor_tracklet.get("frames", []), current_index),
        "anchor_instance_token": anchor_tracklet.get("instance_token"),
        "candidate_instance_token": candidate.get("instance_token"),
        "current_sample_token": current_frame.get("sample_token"),
        "future_sample_token": future_frame.get("sample_token"),
        "scene_id": future_frame.get("scene_id"),
        "label": int(label),
        "negative_type": negative_type,
        "anchor_history_12": _history_feature(anchor_tracklet.get("frames", []), current_index, history_len),
        "positive_obs_feature_12": _as_float_list(future_frame.get("obs_feature_12"), 12),
        "candidate_obs_feature_12": _as_float_list(candidate_frame.get("obs_feature_12"), 12),
        "center_distance": _center_distance_xy(future_frame, candidate_frame),
        "yaw_diff": abs(_wrap_to_pi(_det_yaw(future_frame) - _det_yaw(candidate_frame))),
        "size_l1": float(sum(abs(a - b) for a, b in zip(anchor_lwh, candidate_lwh))),
        "anchor_det_score": _det_score(current_frame),
        "candidate_det_score": _det_score(candidate_frame),
    }


def build_pairwise_association_samples(
    tracklets: List[Dict[str, Any]],
    *,
    history_len: int = 8,
    future_step: int = 1,
    hard_negative_distance: float = 4.0,
    max_hard_negatives: int = 4,
    max_easy_negatives: int = 2,
    require_current_match: bool = True,
    require_future_match: bool = True,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Build same-frame same-class pairwise association samples.

    This is an offline cache builder. It does not alter tracker inference.
    Positives are the anchor tracklet's matched future detection. Negatives are
    other matched detections from the same scene/sample/category.
    """
    frame_index = _index_matched_frames(tracklets)
    samples: List[Dict[str, Any]] = []

    for trk in tracklets:
        category = _tracking_category(trk.get("category", ""))
        frames = trk.get("frames", [])
        if len(frames) <= future_step:
            continue
        for current_index in range(0, len(frames) - future_step):
            current_frame = frames[current_index]
            future_frame = frames[current_index + future_step]
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
            if not positive_candidates:
                continue

            samples.append(_make_pair(
                anchor_tracklet=trk,
                current_frame=current_frame,
                future_frame=future_frame,
                candidate=positive_candidates[0],
                history_len=history_len,
                current_index=current_index,
                label=1,
                negative_type="positive",
            ))

            negatives = [
                candidate for candidate in candidates
                if candidate.get("instance_token") != trk.get("instance_token")
            ]
            negatives.sort(key=lambda candidate: _center_distance_xy(future_frame, candidate["frame"]))

            hard = [
                candidate for candidate in negatives
                if _center_distance_xy(future_frame, candidate["frame"]) <= float(hard_negative_distance)
            ][: max(0, int(max_hard_negatives))]
            hard_ids = {candidate.get("instance_token") for candidate in hard}
            easy = [
                candidate for candidate in negatives
                if candidate.get("instance_token") not in hard_ids
            ][: max(0, int(max_easy_negatives))]

            for candidate in hard:
                samples.append(_make_pair(
                    anchor_tracklet=trk,
                    current_frame=current_frame,
                    future_frame=future_frame,
                    candidate=candidate,
                    history_len=history_len,
                    current_index=current_index,
                    label=0,
                    negative_type="hard",
                ))
            for candidate in easy:
                samples.append(_make_pair(
                    anchor_tracklet=trk,
                    current_frame=current_frame,
                    future_frame=future_frame,
                    candidate=candidate,
                    history_len=history_len,
                    current_index=current_index,
                    label=0,
                    negative_type="easy",
                ))

    return samples, summarize_pairwise_association_samples(samples)


def summarize_pairwise_association_samples(samples: List[Dict[str, Any]]) -> Dict[str, Any]:
    per_class: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "pairs": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "hard_negative_pairs": 0,
        "easy_negative_pairs": 0,
        "anchors": set(),
        "candidate_scores": [],
        "center_distances": [],
    })
    overall = {
        "pairs": 0,
        "candidate_pairs": 0,
        "positive_pairs": 0,
        "negative_pairs": 0,
        "hard_negative_pairs": 0,
        "easy_negative_pairs": 0,
    }
    for sample in samples:
        cls = str(sample.get("category", "unknown"))
        stat = per_class[cls]
        label = int(sample.get("label", 0))
        negative_type = str(sample.get("negative_type", ""))
        stat["pairs"] += 1
        overall["pairs"] += 1
        overall["candidate_pairs"] += 1
        stat["anchors"].add((sample.get("anchor_instance_token"), sample.get("current_sample_token")))
        if label == 1:
            stat["positive_pairs"] += 1
            overall["positive_pairs"] += 1
        else:
            stat["negative_pairs"] += 1
            overall["negative_pairs"] += 1
            if negative_type == "hard":
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

    out_per_class = {}
    for cls, stat in sorted(per_class.items()):
        scores = stat["candidate_scores"]
        distances = stat["center_distances"]
        out_per_class[cls] = {
            "pairs": stat["pairs"],
            "candidate_pairs": stat["pairs"],
            "anchors": len(stat["anchors"]),
            "positive_pairs": stat["positive_pairs"],
            "negative_pairs": stat["negative_pairs"],
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
    return {"overall": overall, "per_class": out_per_class}
