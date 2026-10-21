#!/usr/bin/env python3
"""
Audit a detection-driven training cache.

Focus:
  - per-class tracklet/frame/sample volume
  - matched vs miss ratio
  - detection score distribution
  - short-track / low-match-ratio concentration

Usage:
  python tools/audit_det_tracklet_cache.py \
    --input /home/alvin/datasets/track_data/training_cache/nuscenes/centerpoint_mini_train_from_val.pkl \
    --train-config config/train_nuscenes.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

import yaml


def category_to_tracking_name(category: str) -> str:
    if not isinstance(category, str):
        return "car"
    if category in {"car", "pedestrian", "bicycle", "motorcycle", "bus", "trailer", "truck"}:
        return category
    suffix = category.split(".")[-1]
    if suffix in {"car", "pedestrian", "bicycle", "motorcycle", "bus", "trailer", "truck"}:
        return suffix
    if "pedestrian" in category:
        return "pedestrian"
    return "car"


def load_pickle(path: str) -> List[dict]:
    with open(path, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"Expected list tracklets in {path}, got {type(obj).__name__}")
    return obj


def load_train_cfg(path: str | None) -> Dict[str, Any]:
    if not path:
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def safe_mean(vals: List[float]) -> float:
    return float(sum(vals) / len(vals)) if vals else 0.0


def safe_min(vals: List[float]) -> float:
    return float(min(vals)) if vals else 0.0


def safe_max(vals: List[float]) -> float:
    return float(max(vals)) if vals else 0.0


def percentile(vals: List[float], q: float) -> float:
    if not vals:
        return 0.0
    ordered = sorted(float(v) for v in vals)
    position = (len(ordered) - 1) * float(q)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    alpha = position - lower
    return ordered[lower] * (1.0 - alpha) + ordered[upper] * alpha


def wrapped_abs_angle_error(left: float, right: float) -> float:
    delta = float(left) - float(right)
    return abs(delta - 2.0 * math.pi * round(delta / (2.0 * math.pi)))


def estimate_samples(tracklet_len: int, history_len: int, rollout_steps: int) -> int:
    need = history_len + rollout_steps
    if tracklet_len < need:
        return 0
    return tracklet_len - need + 1


def resolve_class_min_window(
    category: str,
    history_len: int,
    rollout_steps: int,
    min_history_len: int,
    min_rollout_steps: int,
    class_window_cfg: Dict[str, Any],
    include_short_histories: bool = False,
) -> tuple[int, int]:
    track_name = category_to_tracking_name(category)
    cfg = (class_window_cfg or {}).get(track_name, {})
    hist_min = int(cfg.get("MIN_HISTORY_LEN", min_history_len))
    roll_min = int(cfg.get("MIN_ROLLOUT_STEPS", min_rollout_steps))
    hist_min = max(1, min(hist_min, history_len))
    if include_short_histories:
        hist_min = 1
    roll_min = max(1, min(roll_min, rollout_steps))
    return hist_min, roll_min


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit a detection-driven training cache.")
    parser.add_argument("--input", required=True, help="Tracklet cache .pkl file")
    parser.add_argument("--train-config", default=None, help="Optional training config yaml")
    parser.add_argument("--output", default=None, help="Optional summary json path")
    args = parser.parse_args()

    tracklets = load_pickle(args.input)
    train_cfg = load_train_cfg(args.train_config)
    model_cfg = train_cfg.get("MODEL", {})
    train_subcfg = train_cfg.get("TRAINING", {})
    data_cfg = train_cfg.get("DATA", {})
    history_len = int(model_cfg.get("HISTORY_LEN", 8))
    rollout_steps = int(train_subcfg.get("ROLLOUT_STEPS", 4))
    min_history_len = int(data_cfg.get("MIN_HISTORY_LEN", history_len))
    min_rollout_steps = int(data_cfg.get("MIN_ROLLOUT_STEPS", rollout_steps))
    adaptive_windows = bool(data_cfg.get("TRAIN_ADAPTIVE_WINDOWS", False))
    class_window_cfg = data_cfg.get("CLASS_WINDOW", {})

    overall = {
        "tracklets": len(tracklets),
        "frames": 0,
        "matched_frames": 0,
        "miss_frames": 0,
        "estimated_samples": 0,
        "estimated_samples_adaptive": 0,
        "history_len": history_len,
        "rollout_steps": rollout_steps,
        "min_history_len": min_history_len,
        "min_rollout_steps": min_rollout_steps,
        "adaptive_windows": adaptive_windows,
        "include_short_histories": bool(data_cfg.get("INCLUDE_SHORT_HISTORIES", False)),
        "fusion_valid_frames": 0,
        "fusion_fake_frames": 0,
        "unmatched_detection_candidates": 0,
        "association_only_tracklets": 0,
    }
    scene_ids = set()
    source_splits: Dict[str, int] = defaultdict(int)

    per_class: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "tracklets": 0,
        "frames": 0,
        "matched_frames": 0,
        "miss_frames": 0,
        "estimated_samples": 0,
        "estimated_samples_adaptive": 0,
        "short_tracklets_lt_need": 0,
        "short_tracklets_lt_adaptive_need": 0,
        "low_match_ratio_tracklets_lt_0_5": 0,
        "tracklet_lengths": [],
        "tracklet_match_ratios": [],
        "matched_scores": [],
        "all_scores": [],
        "match_distances": [],
        "position_errors_xy": [],
        "velocity_errors_xy": [],
        "yaw_errors": [],
        "miss_streaks": [],
        "fusion_valid_frames": 0,
        "fusion_fake_frames": 0,
        "unmatched_detection_candidates": 0,
    })

    need = history_len + rollout_steps
    for trk in tracklets:
        if trk.get("association_only", False):
            overall["association_only_tracklets"] += 1
            overall["unmatched_detection_candidates"] += sum(
                len(fr.get("unmatched_detections", []) or []) for fr in trk.get("frames", [])
            )
            continue
        cls = trk.get("category", "unknown")
        hist_min_cls, roll_min_cls = resolve_class_min_window(
            category=cls,
            history_len=history_len,
            rollout_steps=rollout_steps,
            min_history_len=min_history_len,
            min_rollout_steps=min_rollout_steps,
            class_window_cfg=class_window_cfg,
            include_short_histories=bool(data_cfg.get("INCLUDE_SHORT_HISTORIES", False)),
        )
        adaptive_need = hist_min_cls + roll_min_cls
        frames = trk.get("frames", [])
        cls_stat = per_class[cls]
        scene_ids.add(str(trk.get("scene_id", "")))
        source_splits[str(trk.get("source_split", "unknown"))] += 1
        cls_stat["tracklets"] += 1
        cls_stat["tracklet_lengths"].append(len(frames))

        matched_cnt = 0
        score_vals = []
        matched_score_vals = []
        current_miss_streak = 0
        for fr in frames:
            cls_stat["frames"] += 1
            overall["frames"] += 1
            is_matched = bool(fr.get("is_matched", False))
            if is_matched:
                cls_stat["matched_frames"] += 1
                overall["matched_frames"] += 1
                matched_cnt += 1
                if current_miss_streak:
                    cls_stat["miss_streaks"].append(current_miss_streak)
                    current_miss_streak = 0

                match_distance = fr.get("match_distance")
                if isinstance(match_distance, (int, float)) and math.isfinite(float(match_distance)):
                    cls_stat["match_distances"].append(float(match_distance))
                det_xyz = fr.get("det_global_xyz")
                gt_xyz = fr.get("gt_global_xyz")
                if det_xyz is not None and gt_xyz is not None:
                    cls_stat["position_errors_xy"].append(math.hypot(
                        float(det_xyz[0]) - float(gt_xyz[0]),
                        float(det_xyz[1]) - float(gt_xyz[1]),
                    ))
                det_velocity = fr.get("det_velocity")
                gt_velocity = fr.get("gt_velocity")
                if det_velocity is not None and gt_velocity is not None:
                    cls_stat["velocity_errors_xy"].append(math.hypot(
                        float(det_velocity[0]) - float(gt_velocity[0]),
                        float(det_velocity[1]) - float(gt_velocity[1]),
                    ))
                if fr.get("det_yaw") is not None and fr.get("gt_yaw") is not None:
                    cls_stat["yaw_errors"].append(wrapped_abs_angle_error(
                        float(fr["det_yaw"]), float(fr["gt_yaw"])
                    ))
            else:
                cls_stat["miss_frames"] += 1
                overall["miss_frames"] += 1
                current_miss_streak += 1

            if bool(fr.get("fusion_valid", False)):
                cls_stat["fusion_valid_frames"] += 1
                overall["fusion_valid_frames"] += 1
            if bool(fr.get("fusion_is_fake", False)):
                cls_stat["fusion_fake_frames"] += 1
                overall["fusion_fake_frames"] += 1
            candidate_count = len(fr.get("unmatched_detections", []) or [])
            cls_stat["unmatched_detection_candidates"] += candidate_count
            overall["unmatched_detection_candidates"] += candidate_count

            obs = fr.get("obs_feature_12", None)
            score = None
            if isinstance(obs, list) and len(obs) >= 12:
                score = obs[11]
            elif "det_score" in fr:
                score = fr["det_score"]
            if isinstance(score, (int, float)) and math.isfinite(score):
                score_vals.append(float(score))
                if is_matched:
                    matched_score_vals.append(float(score))

        if current_miss_streak:
            cls_stat["miss_streaks"].append(current_miss_streak)

        match_ratio = float(matched_cnt / len(frames)) if frames else 0.0
        cls_stat["tracklet_match_ratios"].append(match_ratio)
        if len(frames) < need:
            cls_stat["short_tracklets_lt_need"] += 1
        if len(frames) < adaptive_need:
            cls_stat["short_tracklets_lt_adaptive_need"] += 1
        if match_ratio < 0.5:
            cls_stat["low_match_ratio_tracklets_lt_0_5"] += 1

        samples = estimate_samples(len(frames), history_len, rollout_steps)
        adaptive_samples = estimate_samples(len(frames), hist_min_cls, roll_min_cls)
        cls_stat["estimated_samples"] += samples
        cls_stat["estimated_samples_adaptive"] += adaptive_samples
        overall["estimated_samples"] += samples
        overall["estimated_samples_adaptive"] += adaptive_samples

        cls_stat["all_scores"].extend(score_vals)
        cls_stat["matched_scores"].extend(matched_score_vals)

    summary = {
        "overall": {
            **overall,
            "matched_ratio": float(overall["matched_frames"] / overall["frames"]) if overall["frames"] else 0.0,
            "scenes": len(scene_ids),
            "source_splits": dict(sorted(source_splits.items())),
        },
        "per_class": {},
    }

    for cls, stat in sorted(per_class.items()):
        frames = stat["frames"]
        matched_frames = stat["matched_frames"]
        summary["per_class"][cls] = {
            "tracklets": stat["tracklets"],
            "frames": frames,
            "matched_frames": matched_frames,
            "miss_frames": stat["miss_frames"],
            "matched_ratio": float(matched_frames / frames) if frames else 0.0,
            "estimated_samples": stat["estimated_samples"],
            "estimated_samples_adaptive": stat["estimated_samples_adaptive"],
            "short_tracklets_lt_need": stat["short_tracklets_lt_need"],
            "short_tracklets_lt_adaptive_need": stat["short_tracklets_lt_adaptive_need"],
            "low_match_ratio_tracklets_lt_0_5": stat["low_match_ratio_tracklets_lt_0_5"],
            "tracklet_len_mean": safe_mean(stat["tracklet_lengths"]),
            "tracklet_len_min": safe_min(stat["tracklet_lengths"]),
            "tracklet_len_max": safe_max(stat["tracklet_lengths"]),
            "tracklet_match_ratio_mean": safe_mean(stat["tracklet_match_ratios"]),
            "score_all_min": safe_min(stat["all_scores"]),
            "score_all_mean": safe_mean(stat["all_scores"]),
            "score_all_max": safe_max(stat["all_scores"]),
            "score_matched_min": safe_min(stat["matched_scores"]),
            "score_matched_mean": safe_mean(stat["matched_scores"]),
            "score_matched_max": safe_max(stat["matched_scores"]),
            "match_distance_p50": percentile(stat["match_distances"], 0.50),
            "match_distance_p90": percentile(stat["match_distances"], 0.90),
            "match_distance_p95": percentile(stat["match_distances"], 0.95),
            "match_distance_p99": percentile(stat["match_distances"], 0.99),
            "position_error_xy_mean": safe_mean(stat["position_errors_xy"]),
            "velocity_error_xy_mean": safe_mean(stat["velocity_errors_xy"]),
            "yaw_error_mean": safe_mean(stat["yaw_errors"]),
            "miss_streak_mean": safe_mean(stat["miss_streaks"]),
            "miss_streak_max": safe_max(stat["miss_streaks"]),
            "fusion_valid_frames": stat["fusion_valid_frames"],
            "fusion_fake_frames": stat["fusion_fake_frames"],
            "unmatched_detection_candidates": stat["unmatched_detection_candidates"],
        }

    out_text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(out_text)

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(out_text + "\n", encoding="utf-8")
        print(f"[SAVE] {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
