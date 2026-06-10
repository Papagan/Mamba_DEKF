#!/usr/bin/env python3
import argparse
import json
import math
import os
from collections import Counter, defaultdict

import numpy as np


TRACKING_CLASS_MAP = {
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "human.pedestrian.police_officer": "pedestrian",
    "vehicle.trailer": "trailer",
    "vehicle.truck": "truck",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export per-track quality features and GT-aligned labels from nuScenes results.json."
    )
    parser.add_argument("--results", required=True, help="Path to results.json")
    parser.add_argument("--nusc-dataroot", required=True, help="nuScenes dataroot")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--output", required=True, help="Output feature JSON path")
    parser.add_argument(
        "--dist-th",
        type=float,
        default=2.0,
        help="Center-distance threshold for GT alignment",
    )
    return parser.parse_args()


def xy_dist(a, b):
    dx = float(a[0]) - float(b[0])
    dy = float(a[1]) - float(b[1])
    return math.hypot(dx, dy)


def safe_mean(values, default=0.0):
    return float(np.mean(values)) if values else float(default)


def safe_std(values, default=0.0):
    return float(np.std(values)) if values else float(default)


def safe_percentile(values, q, default=0.0):
    return float(np.percentile(values, q)) if values else float(default)


def load_results(path):
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("results", {})


def build_sample_meta(nusc, sample_tokens):
    sample_meta = {}
    for sample_token in sample_tokens:
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        sample_meta[sample_token] = {
            "scene_name": scene["name"],
            "timestamp": float(sample["timestamp"]) / 1e6,
            "anns": sample["anns"],
        }
    return sample_meta


def build_gt_by_sample(nusc, sample_meta):
    gt_by_sample = defaultdict(lambda: defaultdict(list))
    gt_track_counts = Counter()

    for sample_token, meta in sample_meta.items():
        scene_name = meta["scene_name"]
        for ann_token in meta["anns"]:
            ann = nusc.get("sample_annotation", ann_token)
            cls_name = TRACKING_CLASS_MAP.get(ann["category_name"])
            if cls_name is None:
                continue
            gt_item = {
                "instance_token": ann["instance_token"],
                "translation": ann["translation"],
                "sample_token": sample_token,
            }
            gt_by_sample[sample_token][cls_name].append(gt_item)
            gt_track_counts[(scene_name, cls_name, ann["instance_token"])] += 1

    return gt_by_sample, gt_track_counts


def collect_predictions(results, sample_meta):
    tracks = defaultdict(list)
    pred_entries_by_uid = {}
    preds_by_sample = defaultdict(lambda: defaultdict(list))

    for sample_token, boxes in results.items():
        meta = sample_meta[sample_token]
        scene_name = meta["scene_name"]
        timestamp = meta["timestamp"]
        for idx, box in enumerate(boxes):
            cls_name = box["tracking_name"]
            track_key = f"{scene_name}|{cls_name}|{box['tracking_id']}"
            uid = f"{sample_token}:{idx}"
            velocity = box.get("velocity", [0.0, 0.0])
            entry = {
                "uid": uid,
                "track_key": track_key,
                "scene_name": scene_name,
                "class_name": cls_name,
                "tracking_id": str(box["tracking_id"]),
                "sample_token": sample_token,
                "timestamp": timestamp,
                "translation": box["translation"],
                "tracking_score": float(box.get("tracking_score", 0.0)),
                "velocity": velocity,
                "matched_gt": None,
                "match_dist": None,
            }
            pred_entries_by_uid[uid] = entry
            preds_by_sample[sample_token][cls_name].append(uid)
            tracks[track_key].append(entry)

    for track_entries in tracks.values():
        track_entries.sort(key=lambda x: x["timestamp"])

    return tracks, pred_entries_by_uid, preds_by_sample


def greedy_match_predictions(preds_by_sample, pred_entries_by_uid, gt_by_sample, dist_th):
    for sample_token, class_map in preds_by_sample.items():
        for cls_name, pred_uids in class_map.items():
            gt_items = gt_by_sample[sample_token].get(cls_name, [])
            if not pred_uids or not gt_items:
                continue

            pairs = []
            for pred_uid in pred_uids:
                pred = pred_entries_by_uid[pred_uid]
                pred_xy = pred["translation"][:2]
                for gt_idx, gt_item in enumerate(gt_items):
                    dist = xy_dist(pred_xy, gt_item["translation"][:2])
                    if dist <= dist_th:
                        pairs.append((dist, pred_uid, gt_idx))

            pairs.sort(key=lambda x: x[0])
            used_pred = set()
            used_gt = set()
            for dist, pred_uid, gt_idx in pairs:
                if pred_uid in used_pred or gt_idx in used_gt:
                    continue
                pred_entries_by_uid[pred_uid]["matched_gt"] = gt_items[gt_idx]["instance_token"]
                pred_entries_by_uid[pred_uid]["match_dist"] = float(dist)
                used_pred.add(pred_uid)
                used_gt.add(gt_idx)


def compute_track_features(track_key, entries, gt_track_counts):
    scene_name, cls_name, tracking_id = track_key.split("|", 2)
    timestamps = [e["timestamp"] for e in entries]
    scores = [e["tracking_score"] for e in entries]
    translations = [e["translation"] for e in entries]
    velocity_mag = [
        math.hypot(float(e["velocity"][0]), float(e["velocity"][1]))
        for e in entries
        if isinstance(e.get("velocity"), list) and len(e["velocity"]) >= 2
    ]

    dts = [
        max(0.0, timestamps[i] - timestamps[i - 1])
        for i in range(1, len(timestamps))
    ]
    path_steps = [
        xy_dist(translations[i][:2], translations[i - 1][:2])
        for i in range(1, len(translations))
    ]
    path_length = float(sum(path_steps))
    displacement = float(xy_dist(translations[-1][:2], translations[0][:2])) if len(translations) > 1 else 0.0
    straightness = float(displacement / max(path_length, 1e-6)) if path_length > 0 else 1.0

    matched_entries = [e for e in entries if e["matched_gt"] is not None]
    tp_count = len(matched_entries)
    fp_count = len(entries) - tp_count
    tp_ratio = float(tp_count / max(len(entries), 1))

    gt_counter = Counter(e["matched_gt"] for e in matched_entries if e["matched_gt"] is not None)
    dominant_gt = None
    dominant_count = 0
    if gt_counter:
        dominant_gt, dominant_count = gt_counter.most_common(1)[0]

    purity = float(dominant_count / max(tp_count, 1)) if tp_count > 0 else 0.0
    gt_total = gt_track_counts.get((scene_name, cls_name, dominant_gt), 0) if dominant_gt else 0
    dominant_recall = float(dominant_count / max(gt_total, 1)) if gt_total > 0 else 0.0

    match_dists = [
        float(e["match_dist"]) for e in matched_entries if e["match_dist"] is not None
    ]
    gap_count = int(sum(1 for dt in dts if dt > 0.75))

    quality_target = float(np.clip(tp_ratio * purity * dominant_recall, 0.0, 1.0))
    is_good_track = int(tp_ratio >= 0.6 and purity >= 0.8 and dominant_recall >= 0.3)

    return {
        "track_key": track_key,
        "scene_name": scene_name,
        "class_name": cls_name,
        "tracking_id": tracking_id,
        "num_frames": len(entries),
        "duration_sec": float(max(timestamps[-1] - timestamps[0], 0.0)) if len(timestamps) > 1 else 0.0,
        "mean_dt": safe_mean(dts, 0.5),
        "max_dt": max(dts) if dts else 0.0,
        "gap_count": gap_count,
        "score_mean": safe_mean(scores),
        "score_std": safe_std(scores),
        "score_min": min(scores) if scores else 0.0,
        "score_max": max(scores) if scores else 0.0,
        "score_last": scores[-1] if scores else 0.0,
        "speed_mean": safe_mean(velocity_mag),
        "speed_max": max(velocity_mag) if velocity_mag else 0.0,
        "path_length_xy": path_length,
        "displacement_xy": displacement,
        "straightness": straightness,
        "tp_count": tp_count,
        "fp_count": fp_count,
        "tp_ratio": tp_ratio,
        "dominant_gt": dominant_gt,
        "dominant_gt_count": dominant_count,
        "dominant_gt_total": gt_total,
        "purity": purity,
        "dominant_recall": dominant_recall,
        "mean_match_dist": safe_mean(match_dists, 2.0),
        "p90_match_dist": safe_percentile(match_dists, 90, 2.0),
        "quality_target": quality_target,
        "is_good_track": is_good_track,
    }


def build_summary(records):
    per_class = defaultdict(lambda: {"count": 0, "quality_mean": 0.0, "good_ratio": 0.0})
    for rec in records:
        cls = rec["class_name"]
        per_class[cls]["count"] += 1
        per_class[cls]["quality_mean"] += rec["quality_target"]
        per_class[cls]["good_ratio"] += rec["is_good_track"]

    for stats in per_class.values():
        n = max(stats["count"], 1)
        stats["quality_mean"] /= n
        stats["good_ratio"] /= n

    return {
        "num_tracks": len(records),
        "per_class": dict(per_class),
    }


def main():
    args = parse_args()
    from nuscenes.nuscenes import NuScenes

    results = load_results(args.results)
    sample_tokens = sorted(results.keys())

    nusc = NuScenes(version=args.version, dataroot=args.nusc_dataroot, verbose=False)
    sample_meta = build_sample_meta(nusc, sample_tokens)
    gt_by_sample, gt_track_counts = build_gt_by_sample(nusc, sample_meta)
    tracks, pred_entries_by_uid, preds_by_sample = collect_predictions(results, sample_meta)
    greedy_match_predictions(preds_by_sample, pred_entries_by_uid, gt_by_sample, args.dist_th)

    records = [
        compute_track_features(track_key, entries, gt_track_counts)
        for track_key, entries in sorted(tracks.items())
    ]
    payload = {
        "meta": {
            "results": os.path.abspath(args.results),
            "nusc_dataroot": os.path.abspath(args.nusc_dataroot),
            "version": args.version,
            "dist_th": args.dist_th,
        },
        "summary": build_summary(records),
        "records": records,
    }
    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"[track-quality-export] tracks={len(records)} output={args.output}")


if __name__ == "__main__":
    main()
