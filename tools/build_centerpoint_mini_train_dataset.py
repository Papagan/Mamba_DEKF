#!/usr/bin/env python3
"""
Build a mini detection-driven training cache from CenterPoint BaseVersion detections.

This script aligns detections in a BaseVersion nuScenes JSON with nuScenes GT
annotations and writes a mini tracklet cache under a target directory.

The output is intentionally close to the existing GT tracklet cache format, but
each frame stores both:
  1. GT labels (`gt_feature_12`, `gt_*`)
  2. Detector observations (`obs_feature_12`, `det_*`, `is_matched`)

This makes the cache suitable for a later detection-driven training dataset
without breaking the current GT-only training pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import yaml
from scipy.optimize import linear_sum_assignment


TRACKING_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
]


def quat_to_yaw(rotation: List[float]) -> float:
    """Convert quaternion [w, x, y, z] to yaw."""
    w, x, y, z = rotation
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return float(math.atan2(siny_cosp, cosy_cosp))


def wrap_to_pi(angle: float) -> float:
    return float(angle - 2.0 * math.pi * round(angle / (2.0 * math.pi)))


def map_nuscenes_category(category_name: str) -> Optional[str]:
    """Map nuScenes fine-grained category names to tracking classes."""
    if category_name == "vehicle.car":
        return "car"
    if category_name == "vehicle.truck":
        return "truck"
    if category_name == "vehicle.trailer":
        return "trailer"
    if category_name in {"vehicle.bus.bendy", "vehicle.bus.rigid"}:
        return "bus"
    if category_name == "vehicle.motorcycle":
        return "motorcycle"
    if category_name == "vehicle.bicycle":
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


def center_distance_2d(a_xyz: List[float], b_xyz: List[float]) -> float:
    return float(np.linalg.norm(np.asarray(a_xyz[:2], dtype=np.float32) - np.asarray(b_xyz[:2], dtype=np.float32)))


def load_baseversion_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_gt_feature(ann: dict, nusc) -> Tuple[List[float], Dict]:
    xyz = ann["translation"]
    size_wlh = ann["size"]
    lwh = [size_wlh[1], size_wlh[0], size_wlh[2]]
    yaw = quat_to_yaw(ann["rotation"])
    try:
        vel_3d = nusc.box_velocity(ann["token"])
        if np.any(np.isnan(vel_3d)):
            vel_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    except Exception:
        vel_3d = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    vx, vy = float(vel_3d[0]), float(vel_3d[1])
    feature = [
        float(xyz[0]), float(xyz[1]), float(xyz[2]),
        vx, vy, 0.0,
        float(lwh[0]), float(lwh[1]), float(lwh[2]),
        yaw, 0.0, 1.0,
    ]
    meta = {
        "global_xyz": [float(v) for v in xyz],
        "lwh": [float(v) for v in lwh],
        "yaw": yaw,
        "velocity": [vx, vy],
    }
    return feature, meta


def build_det_feature(det: dict) -> Tuple[List[float], Dict]:
    xyz = det["global_xyz"]
    lwh = det["lwh"]
    yaw = float(det["global_yaw"])
    vel = det.get("global_velocity", [0.0, 0.0])
    if vel is None:
        vel = [0.0, 0.0]
    feature = [
        float(xyz[0]), float(xyz[1]), float(xyz[2]),
        float(vel[0]), float(vel[1]), 0.0,
        float(lwh[0]), float(lwh[1]), float(lwh[2]),
        yaw, 0.0, float(det["detection_score"]),
    ]
    meta = {
        "global_xyz": [float(v) for v in xyz],
        "lwh": [float(v) for v in lwh],
        "yaw": yaw,
        "velocity": [float(vel[0]), float(vel[1])],
        "score": float(det["detection_score"]),
    }
    return feature, meta


def match_classwise(gt_items: List[dict], det_items: List[dict], dist_th: float) -> Tuple[List[Tuple[int, int, float]], List[int], List[int]]:
    if not gt_items or not det_items:
        return [], list(range(len(gt_items))), list(range(len(det_items)))

    cost = np.full((len(gt_items), len(det_items)), fill_value=1e6, dtype=np.float32)
    for i, gt in enumerate(gt_items):
        for j, det in enumerate(det_items):
            cost[i, j] = center_distance_2d(gt["global_xyz"], det["global_xyz"])

    row_ind, col_ind = linear_sum_assignment(cost)

    matches = []
    matched_gt = set()
    matched_det = set()
    for r, c in zip(row_ind, col_ind):
        dist = float(cost[r, c])
        if dist <= dist_th:
            matches.append((r, c, dist))
            matched_gt.add(r)
            matched_det.add(c)

    unmatched_gt = [i for i in range(len(gt_items)) if i not in matched_gt]
    unmatched_det = [j for j in range(len(det_items)) if j not in matched_det]
    return matches, unmatched_gt, unmatched_det


def finalize_tracklet_frames(tracklet: dict) -> None:
    """Sort frames and fill per-track omega for GT and detection observations."""
    frames = sorted(tracklet["frames"], key=lambda x: (x["timestamp"], x["frame_id"]))

    prev_gt_yaw = None
    prev_gt_ts = None
    prev_det_yaw = None
    prev_det_ts = None

    for frame in frames:
        if prev_gt_yaw is not None and prev_gt_ts is not None:
            dt = frame["timestamp"] - prev_gt_ts
            if dt > 0:
                frame["gt_feature_12"][10] = wrap_to_pi(frame["gt_yaw"] - prev_gt_yaw) / dt
        prev_gt_yaw = frame["gt_yaw"]
        prev_gt_ts = frame["timestamp"]

        if frame["is_matched"]:
            if prev_det_yaw is not None and prev_det_ts is not None:
                dt = frame["timestamp"] - prev_det_ts
                if dt > 0:
                    frame["obs_feature_12"][10] = wrap_to_pi(frame["det_yaw"] - prev_det_yaw) / dt
            prev_det_yaw = frame["det_yaw"]
            prev_det_ts = frame["timestamp"]

    tracklet["frames"] = frames


def load_training_config(train_config_path: Optional[str]) -> Dict:
    if not train_config_path:
        return {}
    cfg_path = Path(train_config_path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Training config not found: {train_config_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def estimate_training_samples(tracklets: List[dict], history_len: int, rollout_steps: int) -> int:
    total = 0
    min_len = history_len + rollout_steps
    for trk in tracklets:
        n = len(trk["frames"])
        if n < min_len:
            continue
        total += n - min_len + 1
    return total


def resolve_class_min_window(
    category: str,
    history_len: int,
    rollout_steps: int,
    min_history_len: int,
    min_rollout_steps: int,
    class_window_cfg: Dict,
) -> Tuple[int, int]:
    cfg = (class_window_cfg or {}).get(category, {})
    hist_min = int(cfg.get("MIN_HISTORY_LEN", min_history_len))
    roll_min = int(cfg.get("MIN_ROLLOUT_STEPS", min_rollout_steps))
    hist_min = max(1, min(hist_min, history_len))
    roll_min = max(1, min(roll_min, rollout_steps))
    return hist_min, roll_min


def estimate_training_samples_adaptive(
    tracklets: List[dict],
    history_len: int,
    rollout_steps: int,
    min_history_len: int,
    min_rollout_steps: int,
    class_window_cfg: Dict,
) -> int:
    total = 0
    for trk in tracklets:
        hist_min, roll_min = resolve_class_min_window(
            category=trk.get("category", "car"),
            history_len=history_len,
            rollout_steps=rollout_steps,
            min_history_len=min_history_len,
            min_rollout_steps=min_rollout_steps,
            class_window_cfg=class_window_cfg,
        )
        total += estimate_training_samples(len(trk["frames"]), hist_min, roll_min)
    return total


def build_mini_dataset(
    det_json_path: str,
    nusc_version: str,
    nusc_dataroot: str,
    output_path: str,
    max_scenes: int,
    dist_th: float,
    min_frames: int,
    min_matched_frames: int,
    include_misses: bool,
    train_config_path: Optional[str],
) -> Dict[str, int]:
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"[mini-train] loading detections: {det_json_path}")
    scenes_data = load_baseversion_json(det_json_path)
    print(f"[mini-train] detected scenes in json: {len(scenes_data)}")

    print(f"[mini-train] loading nuScenes: version={nusc_version} dataroot={nusc_dataroot}")
    nusc = NuScenes(version=nusc_version, dataroot=nusc_dataroot, verbose=False)
    val_scene_names = set(create_splits_scenes().get("val", []))

    selected_scene_ids = []
    for scene_id, frames in scenes_data.items():
        if not frames:
            continue
        sample_token = frames[0]["cur_sample_token"]
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        scene_name = scene["name"]
        if scene_name not in val_scene_names:
            continue
        selected_scene_ids.append(scene_id)
        if max_scenes > 0 and len(selected_scene_ids) >= max_scenes:
            break

    print(f"[mini-train] selected scenes: {len(selected_scene_ids)}")

    tracklets = {}
    stats = {
        "scenes": len(selected_scene_ids),
        "frames": 0,
        "gt_boxes": 0,
        "det_boxes": 0,
        "matches": 0,
        "misses": 0,
        "ignored_dets": 0,
        "timestamp_diff_max": 0.0,
        "timestamp_diff_mean_abs_accum": 0.0,
    }

    for scene_rank, scene_id in enumerate(selected_scene_ids, start=1):
        frames = scenes_data[scene_id]
        print(f"[mini-train] scene {scene_rank}/{len(selected_scene_ids)}: {scene_id} frames={len(frames)}")
        for frame in frames:
            stats["frames"] += 1
            sample_token = frame["cur_sample_token"]
            sample = nusc.get("sample", sample_token)
            timestamp_json = float(frame["timestamp"])
            timestamp_nusc = float(sample["timestamp"]) / 1e6
            timestamp = timestamp_nusc
            frame_id = int(frame["frame_id"])
            ts_diff = abs(timestamp_json - timestamp_nusc)
            stats["timestamp_diff_max"] = max(stats["timestamp_diff_max"], ts_diff)
            stats["timestamp_diff_mean_abs_accum"] += ts_diff

            gt_by_cls = defaultdict(list)
            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                category = map_nuscenes_category(ann["category_name"])
                if category not in TRACKING_CLASSES:
                    continue
                gt_feature, gt_meta = build_gt_feature(ann, nusc)
                gt_item = {
                    "instance_token": ann["instance_token"],
                    "ann_token": ann["token"],
                    "category": category,
                    "global_xyz": gt_meta["global_xyz"],
                    "gt_feature_12": gt_feature,
                    **gt_meta,
                }
                gt_by_cls[category].append(gt_item)
                stats["gt_boxes"] += 1

            det_by_cls = defaultdict(list)
            for det in frame["bboxes"]:
                category = det["category"]
                if category not in TRACKING_CLASSES:
                    continue
                det_feature, det_meta = build_det_feature(det)
                det_item = {
                    "category": category,
                    "global_xyz": det_meta["global_xyz"],
                    "obs_feature_12": det_feature,
                    **det_meta,
                }
                det_by_cls[category].append(det_item)
                stats["det_boxes"] += 1

            for category in TRACKING_CLASSES:
                gt_items = gt_by_cls.get(category, [])
                det_items = det_by_cls.get(category, [])
                matches, unmatched_gt, unmatched_det = match_classwise(gt_items, det_items, dist_th)

                for gt_idx, det_idx, match_dist in matches:
                    gt = gt_items[gt_idx]
                    det = det_items[det_idx]
                    instance_token = gt["instance_token"]
                    if instance_token not in tracklets:
                        tracklets[instance_token] = {
                            "instance_token": instance_token,
                            "category": category,
                            "source_split": "val",
                            "source_detector": "centerpoint",
                            "scene_id": scene_id,
                            "frames": [],
                        }
                    tracklets[instance_token]["frames"].append({
                        "sample_token": sample_token,
                        "timestamp": timestamp,
                        "frame_id": frame_id,
                        "scene_id": scene_id,
                        "is_matched": True,
                        "match_distance": float(match_dist),
                        "det_score": det["score"],
                        "gt_feature_12": list(gt["gt_feature_12"]),
                        "obs_feature_12": list(det["obs_feature_12"]),
                        "gt_global_xyz": list(gt["global_xyz"]),
                        "gt_lwh": list(gt["lwh"]),
                        "gt_yaw": float(gt["yaw"]),
                        "gt_velocity": list(gt["velocity"]),
                        "det_global_xyz": list(det["global_xyz"]),
                        "det_lwh": list(det["lwh"]),
                        "det_yaw": float(det["yaw"]),
                        "det_velocity": list(det["velocity"]),
                    })
                    stats["matches"] += 1

                if include_misses:
                    for gt_idx in unmatched_gt:
                        gt = gt_items[gt_idx]
                        instance_token = gt["instance_token"]
                        if instance_token not in tracklets:
                            tracklets[instance_token] = {
                                "instance_token": instance_token,
                                "category": category,
                                "source_split": "val",
                                "source_detector": "centerpoint",
                                "scene_id": scene_id,
                                "frames": [],
                            }
                        tracklets[instance_token]["frames"].append({
                            "sample_token": sample_token,
                            "timestamp": timestamp,
                            "frame_id": frame_id,
                            "scene_id": scene_id,
                            "is_matched": False,
                            "match_distance": None,
                            "det_score": 0.0,
                            "gt_feature_12": list(gt["gt_feature_12"]),
                            "obs_feature_12": [0.0] * 12,
                            "gt_global_xyz": list(gt["global_xyz"]),
                            "gt_lwh": list(gt["lwh"]),
                            "gt_yaw": float(gt["yaw"]),
                            "gt_velocity": list(gt["velocity"]),
                            "det_global_xyz": None,
                            "det_lwh": None,
                            "det_yaw": None,
                            "det_velocity": None,
                        })
                        stats["misses"] += 1

                stats["ignored_dets"] += len(unmatched_det)

    result = []
    for tracklet in tracklets.values():
        finalize_tracklet_frames(tracklet)
        n_frames = len(tracklet["frames"])
        n_matched = sum(1 for fr in tracklet["frames"] if fr["is_matched"])
        if n_frames < min_frames or n_matched < min_matched_frames:
            continue
        result.append(tracklet)

    result.sort(key=lambda x: (x["category"], x["instance_token"]))

    with open(output_path, "wb") as f:
        pickle.dump(result, f)

    train_cfg = load_training_config(train_config_path)
    model_cfg = train_cfg.get("MODEL", {})
    runtime_cfg = train_cfg.get("TRAINING", {})
    data_cfg = train_cfg.get("DATA", {})
    history_len = int(model_cfg.get("HISTORY_LEN", 8))
    rollout_steps = int(runtime_cfg.get("ROLLOUT_STEPS", 1))
    min_history_len = int(data_cfg.get("MIN_HISTORY_LEN", history_len))
    min_rollout_steps = int(data_cfg.get("MIN_ROLLOUT_STEPS", rollout_steps))
    adaptive_windows = bool(data_cfg.get("TRAIN_ADAPTIVE_WINDOWS", False))
    class_window_cfg = data_cfg.get("CLASS_WINDOW", {})
    batch_size = int(runtime_cfg.get("BATCH_SIZE", 1))
    estimated_samples = estimate_training_samples(result, history_len, rollout_steps)
    estimated_samples_adaptive = estimate_training_samples_adaptive(
        result,
        history_len=history_len,
        rollout_steps=rollout_steps,
        min_history_len=min_history_len,
        min_rollout_steps=min_rollout_steps,
        class_window_cfg=class_window_cfg,
    )
    estimated_batches = int(math.ceil(estimated_samples / float(batch_size))) if batch_size > 0 else 0
    estimated_batches_adaptive = int(math.ceil(estimated_samples_adaptive / float(batch_size))) if batch_size > 0 else 0

    summary_path = output_path.replace(".pkl", "_summary.json")
    category_counts = defaultdict(int)
    category_matched = defaultdict(int)
    for trk in result:
        category_counts[trk["category"]] += 1
        category_matched[trk["category"]] += sum(1 for fr in trk["frames"] if fr["is_matched"])

    summary = {
        **stats,
        "output_path": output_path,
        "tracklets": len(result),
        "estimated_samples": estimated_samples,
        "estimated_samples_adaptive": estimated_samples_adaptive,
        "training_batch_size": batch_size,
        "estimated_batches_per_epoch": estimated_batches,
        "estimated_batches_per_epoch_adaptive": estimated_batches_adaptive,
        "history_len": history_len,
        "rollout_steps": rollout_steps,
        "min_history_len": min_history_len,
        "min_rollout_steps": min_rollout_steps,
        "adaptive_windows": adaptive_windows,
        "category_tracklets": dict(sorted(category_counts.items())),
        "category_matched_frames": dict(sorted(category_matched.items())),
        "config": {
            "nusc_version": nusc_version,
            "nusc_dataroot": nusc_dataroot,
            "det_json_path": det_json_path,
            "max_scenes": max_scenes,
            "dist_th": dist_th,
            "min_frames": min_frames,
            "min_matched_frames": min_matched_frames,
            "include_misses": include_misses,
            "train_config_path": train_config_path,
        },
    }
    if stats["frames"] > 0:
        summary["timestamp_diff_mean_abs"] = stats["timestamp_diff_mean_abs_accum"] / stats["frames"]
    else:
        summary["timestamp_diff_mean_abs"] = 0.0
    summary.pop("timestamp_diff_mean_abs_accum", None)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[mini-train] saved tracklets: {len(result)} -> {output_path}")
    print(f"[mini-train] saved summary: {summary_path}")
    print(
        "[mini-train] training alignment: "
        f"history_len={history_len} rollout_steps={rollout_steps} "
        f"min_history_len={min_history_len} min_rollout_steps={min_rollout_steps} "
        f"adaptive_windows={adaptive_windows} batch_size={batch_size} "
        f"estimated_samples={estimated_samples} estimated_batches={estimated_batches} "
        f"estimated_samples_adaptive={estimated_samples_adaptive} "
        f"estimated_batches_adaptive={estimated_batches_adaptive}"
    )
    print(f"[mini-train] stats: {json.dumps(summary, indent=2, ensure_ascii=False)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a mini detection-driven training cache from CenterPoint val detections.",
    )
    parser.add_argument(
        "--det-json",
        default="/root/autodl-tmp/data/base_version/nuscenes/centerpoint/val.json",
        help="Path to BaseVersion detection json.",
    )
    parser.add_argument(
        "--nusc-dataroot",
        default="/root/autodl-tmp/data/nuscenes/datasets/",
        help="nuScenes dataset root.",
    )
    parser.add_argument(
        "--nusc-version",
        default="v1.0-trainval",
        help="nuScenes metadata version.",
    )
    parser.add_argument(
        "--output",
        default="/root/autodl-tmp/data/training_cache/nuscenes/centerpoint_mini_train_from_val.pkl",
        help="Output pickle path.",
    )
    parser.add_argument(
        "--max-scenes",
        type=int,
        default=10,
        help="Maximum number of val scenes to use. Use 0 for all available scenes.",
    )
    parser.add_argument(
        "--dist-th",
        type=float,
        default=2.0,
        help="Center-distance threshold for GT/detection alignment in meters.",
    )
    parser.add_argument(
        "--min-frames",
        type=int,
        default=2,
        help="Minimum total frames per saved tracklet.",
    )
    parser.add_argument(
        "--min-matched-frames",
        type=int,
        default=2,
        help="Minimum matched detection frames per saved tracklet.",
    )
    parser.add_argument(
        "--no-misses",
        action="store_true",
        help="Do not keep GT miss frames in the saved tracklets.",
    )
    parser.add_argument(
        "--train-config",
        default="config/train_nuscenes.yaml",
        help="Training config used to estimate HISTORY_LEN / ROLLOUT_STEPS / BATCH_SIZE.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_mini_dataset(
        det_json_path=args.det_json,
        nusc_version=args.nusc_version,
        nusc_dataroot=args.nusc_dataroot,
        output_path=args.output,
        max_scenes=args.max_scenes,
        dist_th=args.dist_th,
        min_frames=args.min_frames,
        min_matched_frames=args.min_matched_frames,
        include_misses=not args.no_misses,
        train_config_path=args.train_config,
    )


if __name__ == "__main__":
    main()
