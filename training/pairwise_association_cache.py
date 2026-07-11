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


def _det_velocity_xy(frame: Dict[str, Any]) -> List[float]:
    if frame.get("det_velocity") is not None:
        return _as_float_list(frame.get("det_velocity"), 2)
    obs = frame.get("obs_feature_12") or []
    return _as_float_list(obs[3:5], 2)


def _predict_anchor_frame(current_frame: Dict[str, Any], future_frame: Dict[str, Any]) -> Dict[str, Any]:
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


def _wrap_to_pi(value: float) -> float:
    return float(value - 2.0 * math.pi * round(value / (2.0 * math.pi)))


def _is_hard_negative_type(value: Any) -> bool:
    return str(value).strip().lower() in {"hard", "inference_margin"}


def _state_bucket_from_history(frames: List[Dict[str, Any]], current_index: int) -> str:
    return "matched" if bool(frames[current_index].get("is_matched", False)) else "unmatched"


def _history_feature(
    frames: List[Dict[str, Any]],
    end_index: int,
    history_len: int,
    *,
    history_source: str = "det",
) -> List[List[float]]:
    start = max(0, end_index - history_len + 1)
    selected = frames[start : end_index + 1]
    pad = [[0.0] * 12 for _ in range(max(0, history_len - len(selected)))]
    source = str(history_source).strip().lower()
    if source not in {"det", "fusion"}:
        raise ValueError(f"Unsupported history_source={history_source!r}; expected 'det' or 'fusion'")
    features = []
    for frame in selected:
        if source == "fusion":
            if bool(frame.get("fusion_valid", False)):
                features.append(_as_float_list(frame.get("fusion_feature_12"), 12))
            else:
                features.append([0.0] * 12)
        else:
            features.append(_as_float_list(frame.get("obs_feature_12"), 12))
    return pad + features


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
    history_source: str,
    pair_geometry_source: str,
    label: int,
    negative_type: str,
    negative_mining_mode: str = "legacy",
    candidate_rank: int | None = None,
    best_candidate_distance: float | None = None,
) -> Dict[str, Any]:
    candidate_frame = candidate["frame"]
    category = _tracking_category(anchor_tracklet.get("category", ""))
    if pair_geometry_source == "predicted_track_candidate":
        geometry_anchor = _predict_anchor_frame(current_frame, future_frame)
    elif pair_geometry_source == "track_candidate":
        geometry_anchor = current_frame
    else:
        geometry_anchor = future_frame
    anchor_lwh = _det_lwh(geometry_anchor)
    candidate_lwh = _det_lwh(candidate_frame)
    center_distance = _center_distance_xy(geometry_anchor, candidate_frame)
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
        "negative_mining_mode": str(negative_mining_mode).strip().lower(),
        "history_source": str(history_source).strip().lower(),
        "pair_geometry_source": str(pair_geometry_source).strip().lower(),
        "candidate_rank": int(candidate_rank) if candidate_rank is not None else None,
        "best_candidate_distance": (
            float(best_candidate_distance)
            if best_candidate_distance is not None and math.isfinite(float(best_candidate_distance))
            else None
        ),
        "anchor_history_12": _history_feature(
            anchor_tracklet.get("frames", []),
            current_index,
            history_len,
            history_source=history_source,
        ),
        "positive_obs_feature_12": _as_float_list(future_frame.get("obs_feature_12"), 12),
        "candidate_obs_feature_12": _as_float_list(candidate_frame.get("obs_feature_12"), 12),
        "candidate_history_12": _inference_detection_history_feature(candidate_frame, history_len),
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
    max_easy_negatives: int = 2,
    negative_mining_mode: str = "legacy",
    cost_margin_eps: float = 0.05,
    max_pairs_per_class: Dict[str, int] | None = None,
    require_current_match: bool = True,
    require_future_match: bool = True,
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

    def _append_sample(sample: Dict[str, Any]) -> bool:
        category_name = str(sample.get("category", ""))
        cap = pair_caps.get(category_name)
        if cap is not None and pair_counts_by_class[category_name] >= cap:
            return False
        samples.append(sample)
        pair_counts_by_class[category_name] += 1
        return True

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

            if pair_geometry_source == "predicted_track_candidate":
                geometry_anchor = _predict_anchor_frame(current_frame, future_frame)
            elif pair_geometry_source == "track_candidate":
                geometry_anchor = current_frame
            else:
                geometry_anchor = future_frame
            ranked_candidates = sorted(
                [
                    (rank_candidate, _center_distance_xy(geometry_anchor, rank_candidate["frame"]))
                    for rank_candidate in candidates
                ],
                key=lambda item: item[1],
            )
            candidate_rank = {
                candidate.get("instance_token"): rank
                for rank, (candidate, _) in enumerate(ranked_candidates, start=1)
            }
            candidate_distance = {
                candidate.get("instance_token"): distance
                for candidate, distance in ranked_candidates
            }
            best_candidate_distance = (
                float(ranked_candidates[0][1])
                if ranked_candidates and math.isfinite(float(ranked_candidates[0][1]))
                else None
            )

            appended_positive = _append_sample(_make_pair(
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
                candidate_rank=candidate_rank.get(positive_candidates[0].get("instance_token")),
                best_candidate_distance=best_candidate_distance,
            ))
            if not appended_positive:
                continue

            negatives = [
                candidate for candidate in candidates
                if candidate.get("instance_token") != trk.get("instance_token")
            ]
            if negative_mining_mode == "inference_margin":
                negatives.sort(key=lambda candidate: candidate_distance.get(candidate.get("instance_token"), float("inf")))
                if best_candidate_distance is None:
                    hard = []
                else:
                    hard = [
                        candidate for candidate in negatives
                        if candidate_distance.get(candidate.get("instance_token"), float("inf"))
                        <= best_candidate_distance + float(cost_margin_eps)
                    ][: max(0, int(max_hard_negatives))]
                easy = []
            else:
                negatives.sort(key=lambda candidate: _center_distance_xy(future_frame, candidate["frame"]))
                hard_distance = float(hard_distance_by_class.get(category, hard_negative_distance))

                hard = [
                    candidate for candidate in negatives
                    if _center_distance_xy(future_frame, candidate["frame"]) <= hard_distance
                ][: max(0, int(max_hard_negatives))]
                hard_ids = {candidate.get("instance_token") for candidate in hard}
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
                    negative_type="inference_margin" if negative_mining_mode == "inference_margin" else "hard",
                    negative_mining_mode=negative_mining_mode,
                    candidate_rank=candidate_rank.get(candidate.get("instance_token")),
                    best_candidate_distance=best_candidate_distance,
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
                    candidate_rank=candidate_rank.get(candidate.get("instance_token")),
                    best_candidate_distance=best_candidate_distance,
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
