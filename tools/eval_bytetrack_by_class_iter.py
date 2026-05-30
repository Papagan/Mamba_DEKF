#!/usr/bin/env python3
"""
Per-class iterator-based ByteTrack-style MOT evaluation for nuScenes.

Design:
1) Iterate classes.
2) For each class, iterate scenes -> frames in temporal order.
3) Frame-wise update of motmetrics accumulator.

This script evaluates tracking outputs in nuScenes `results.json` format.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np

try:
    import motmetrics as mm
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "motmetrics is required. Install with: pip install motmetrics==1.1.3"
    ) from exc

try:
    from nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "nuscenes-devkit is required. Install with: pip install nuscenes-devkit"
    ) from exc


DEFAULT_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
]

DEFAULT_CLASS_RANGE = {
    "car": 50.0,
    "truck": 50.0,
    "bus": 50.0,
    "trailer": 50.0,
    "pedestrian": 40.0,
    "motorcycle": 40.0,
    "bicycle": 40.0,
}

MAX_EVAL_WORKERS = 8


@dataclass
class Obj:
    obj_id: str
    xy: np.ndarray


def category_to_tracking_name(category_name: str) -> str | None:
    """Map nuScenes category to tracking class."""
    if category_name.startswith("vehicle.car"):
        return "car"
    if category_name.startswith("vehicle.truck"):
        return "truck"
    if category_name.startswith("vehicle.bus"):
        return "bus"
    if category_name.startswith("vehicle.trailer"):
        return "trailer"
    if category_name.startswith("vehicle.motorcycle"):
        return "motorcycle"
    if category_name.startswith("vehicle.bicycle"):
        return "bicycle"
    if category_name.startswith("human.pedestrian"):
        return "pedestrian"
    return None


def load_results(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "results" not in data:
        raise ValueError(f"Invalid result file: missing `results` in {path}")
    return data


def get_eval_scene_names(eval_set: str) -> List[str]:
    splits = create_splits_scenes()
    if eval_set not in splits:
        raise ValueError(f"Unsupported eval_set={eval_set}. Available: {list(splits.keys())}")
    return splits[eval_set]


def load_scene_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        lines = [x.strip() for x in f.readlines()]
    scenes = [x for x in lines if x and not x.startswith("#")]
    if not scenes:
        raise ValueError(f"No valid scenes found in scene-list file: {path}")
    return scenes


def iter_sample_tokens_by_scene(nusc: NuScenes, eval_scene_names: List[str]) -> Iterable[Tuple[str, List[str]]]:
    scene_name_set = set(eval_scene_names)
    for scene in nusc.scene:
        if scene["name"] not in scene_name_set:
            continue
        sample_tokens: List[str] = []
        token = scene["first_sample_token"]
        while token:
            sample_tokens.append(token)
            sample = nusc.get("sample", token)
            token = sample["next"]
        yield scene["name"], sample_tokens


def ego_xy_of_sample(nusc: NuScenes, sample: Dict) -> np.ndarray:
    lidar_token = sample["data"]["LIDAR_TOP"]
    sd = nusc.get("sample_data", lidar_token)
    ego_pose = nusc.get("ego_pose", sd["ego_pose_token"])
    return np.asarray(ego_pose["translation"][:2], dtype=np.float64)


def point_dist_xy(a_xy: np.ndarray, b_xy: np.ndarray) -> float:
    return float(np.linalg.norm(a_xy - b_xy))


def collect_gt_objects(
    nusc: NuScenes,
    sample: Dict,
    cls_name: str,
    class_range: Dict[str, float],
) -> List[Obj]:
    ego_xy = ego_xy_of_sample(nusc, sample)
    max_range = class_range.get(cls_name, 50.0)
    out: List[Obj] = []

    for ann_token in sample["anns"]:
        ann = nusc.get("sample_annotation", ann_token)
        tracking_name = category_to_tracking_name(ann["category_name"])
        if tracking_name != cls_name:
            continue
        num_pts = ann["num_lidar_pts"] + ann["num_radar_pts"]
        if num_pts <= 0:
            continue
        ann_xy = np.asarray(ann["translation"][:2], dtype=np.float64)
        if point_dist_xy(ann_xy, ego_xy) > max_range:
            continue
        out.append(Obj(obj_id=ann["instance_token"], xy=ann_xy))
    return out


def collect_pred_objects(
    pred_by_sample: Dict[str, List[Dict]],
    sample_token: str,
    scene_name: str,
    cls_name: str,
    ego_xy: np.ndarray,
    class_range: Dict[str, float],
    score_thr: float,
) -> List[Obj]:
    out: List[Obj] = []
    max_range = class_range.get(cls_name, 50.0)
    pred_list = pred_by_sample.get(sample_token, [])
    for box in pred_list:
        if box.get("tracking_name") != cls_name:
            continue
        score = float(box.get("tracking_score", 0.0))
        if score < score_thr:
            continue
        t = box.get("translation", [math.nan, math.nan, math.nan])
        xy = np.asarray(t[:2], dtype=np.float64)
        if not np.isfinite(xy).all():
            continue
        if point_dist_xy(xy, ego_xy) > max_range:
            continue
        # Prefix scene to avoid cross-scene ID collisions.
        pid = f"{scene_name}:{box.get('tracking_id', 'unknown')}"
        out.append(Obj(obj_id=pid, xy=xy))
    return out


def build_distance_matrix(gt_objs: List[Obj], pred_objs: List[Obj], dist_th: float) -> np.ndarray:
    if len(gt_objs) == 0 or len(pred_objs) == 0:
        return np.empty((len(gt_objs), len(pred_objs)))
    gt_xy = np.asarray([o.xy for o in gt_objs], dtype=np.float64)
    pr_xy = np.asarray([o.xy for o in pred_objs], dtype=np.float64)
    dist_mat = np.linalg.norm(gt_xy[:, None, :] - pr_xy[None, :, :], axis=2)
    dist_mat[dist_mat > dist_th] = np.nan
    return dist_mat


def eval_one_class_iterative(
    nusc: NuScenes,
    eval_scene_names: List[str],
    pred_by_sample: Dict[str, List[Dict]],
    cls_name: str,
    class_range: Dict[str, float],
    dist_th: float,
    score_thr: float,
    verbose_every: int,
) -> Tuple[mm.MOTAccumulator, Dict[str, int]]:
    acc = mm.MOTAccumulator(auto_id=True)
    frame_count = 0
    gt_total = 0
    pred_total = 0
    # motmetrics (with some pandas versions) expects numeric OId/HId.
    # Keep stable per-class integer ids for GT and prediction objects.
    gt_id_map: Dict[str, int] = {}
    pred_id_map: Dict[str, int] = {}
    gt_next_id = 0
    pred_next_id = 0

    for scene_name, sample_tokens in iter_sample_tokens_by_scene(nusc, eval_scene_names):
        for sample_token in sample_tokens:
            sample = nusc.get("sample", sample_token)
            ego_xy = ego_xy_of_sample(nusc, sample)
            gt_objs = collect_gt_objects(nusc, sample, cls_name, class_range)
            pred_objs = collect_pred_objects(
                pred_by_sample=pred_by_sample,
                sample_token=sample_token,
                scene_name=scene_name,
                cls_name=cls_name,
                ego_xy=ego_xy,
                class_range=class_range,
                score_thr=score_thr,
            )
            dists = build_distance_matrix(gt_objs, pred_objs, dist_th)

            gt_ids: List[int] = []
            for o in gt_objs:
                if o.obj_id not in gt_id_map:
                    gt_id_map[o.obj_id] = gt_next_id
                    gt_next_id += 1
                gt_ids.append(gt_id_map[o.obj_id])

            pred_ids: List[int] = []
            for o in pred_objs:
                if o.obj_id not in pred_id_map:
                    pred_id_map[o.obj_id] = pred_next_id
                    pred_next_id += 1
                pred_ids.append(pred_id_map[o.obj_id])

            acc.update(gt_ids, pred_ids, dists)

            frame_count += 1
            gt_total += len(gt_objs)
            pred_total += len(pred_objs)
            if verbose_every > 0 and (frame_count % verbose_every == 0):
                finite = int(np.isfinite(dists).sum()) if dists.size > 0 else 0
                print(
                    f"[ITER][{cls_name}] frame={frame_count} gt={len(gt_objs)} "
                    f"pred={len(pred_objs)} feasible_pairs={finite}",
                    flush=True,
                )

    return acc, {"frames": frame_count, "gt_boxes": gt_total, "pred_boxes": pred_total}


def summarize_acc(acc: mm.MOTAccumulator, cls_name: str, extras: Dict[str, int]) -> Dict:
    mh = mm.metrics.create()
    metric_names = [
        "num_frames",
        "num_objects",
        "num_predictions",
        "num_matches",
        "num_switches",
        "num_false_positives",
        "num_misses",
        "mota",
        "motp",
        "precision",
        "recall",
        "idf1",
        "idp",
        "idr",
    ]
    summary = mh.compute(acc, metrics=metric_names, name=cls_name)
    row = summary.loc[cls_name].to_dict()
    out = {k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
           for k, v in row.items()}
    out.update(extras)
    return out


def parse_classes(raw: str) -> List[str]:
    vals = [x.strip() for x in raw.split(",") if x.strip()]
    if not vals:
        raise ValueError("`--classes` is empty.")
    return vals


def clamp_num_workers(n: int) -> int:
    if n < 1:
        return 1
    return min(n, MAX_EVAL_WORKERS)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Iterative per-class ByteTrack-style MOT metrics on nuScenes tracking results."
    )
    parser.add_argument("-r", "--result", required=True, help="Path to results.json")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataset root")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--eval-set", default="val", help="Split name: val/train/mini_val/...")
    parser.add_argument(
        "--classes",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class list",
    )
    parser.add_argument("--dist-th", type=float, default=2.0, help="Center-distance match threshold (meters)")
    parser.add_argument("--score-thr", type=float, default=0.0, help="Prediction score threshold")
    parser.add_argument(
        "--scene-list",
        default="",
        help="Optional text file (one scene name per line). Intersect with eval-set scenes.",
    )
    parser.add_argument("--verbose-every", type=int, default=0, help="Print iterator log every N frames (0=off)")
    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help=f"Parallel worker threads across classes (1..{MAX_EVAL_WORKERS}).",
    )
    parser.add_argument("-o", "--output", default="", help="Optional output JSON path")
    args = parser.parse_args()

    if not os.path.exists(args.result):
        raise FileNotFoundError(args.result)

    class_list = parse_classes(args.classes)
    result_data = load_results(args.result)
    pred_by_sample = result_data.get("results", {})
    num_workers = clamp_num_workers(args.num_workers)
    if num_workers != args.num_workers:
        print(
            f"[INFO] clamp --num-workers from {args.num_workers} to {num_workers} "
            f"(max={MAX_EVAL_WORKERS})",
            flush=True,
        )

    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=False)
    eval_scene_names = get_eval_scene_names(args.eval_set)
    if args.scene_list:
        if not os.path.exists(args.scene_list):
            raise FileNotFoundError(args.scene_list)
        selected_scenes = set(load_scene_list(args.scene_list))
        eval_scene_names = [s for s in eval_scene_names if s in selected_scenes]
        if not eval_scene_names:
            raise ValueError(
                f"No scenes left after intersecting eval_set={args.eval_set} with scene-list={args.scene_list}"
            )

    all_metrics: Dict[str, Dict] = {}
    all_accs = []
    all_names = []

    print(
        f"[INFO] eval_set={args.eval_set} scenes={len(eval_scene_names)} "
        f"classes={class_list} dist_th={args.dist_th} score_thr={args.score_thr} "
        f"scene_list={'on' if args.scene_list else 'off'} workers={num_workers}",
        flush=True,
    )

    def _eval_one_class(cls_name: str):
        acc, extras = eval_one_class_iterative(
            nusc=nusc,
            eval_scene_names=eval_scene_names,
            pred_by_sample=pred_by_sample,
            cls_name=cls_name,
            class_range=DEFAULT_CLASS_RANGE,
            dist_th=args.dist_th,
            score_thr=args.score_thr,
            verbose_every=args.verbose_every,
        )
        cls_metrics = summarize_acc(acc, cls_name, extras)
        return cls_name, cls_metrics, acc

    acc_by_class: Dict[str, mm.MOTAccumulator] = {}
    if num_workers == 1:
        for cls_name in class_list:
            cls_name, cls_metrics, acc = _eval_one_class(cls_name)
            all_metrics[cls_name] = cls_metrics
            acc_by_class[cls_name] = acc
            print(
                f"[CLASS] {cls_name:>10s} "
                f"MOTA={cls_metrics['mota']:.4f} IDF1={cls_metrics['idf1']:.4f} "
                f"R={cls_metrics['recall']:.4f} P={cls_metrics['precision']:.4f} "
                f"TP={int(cls_metrics['num_matches'])} FP={int(cls_metrics['num_false_positives'])} "
                f"FN={int(cls_metrics['num_misses'])} IDSW={int(cls_metrics['num_switches'])}",
                flush=True,
            )
    else:
        pool_workers = min(num_workers, len(class_list))
        with ThreadPoolExecutor(max_workers=pool_workers) as ex:
            future_map = {ex.submit(_eval_one_class, cls_name): cls_name for cls_name in class_list}
            for fut in as_completed(future_map):
                cls_name, cls_metrics, acc = fut.result()
                all_metrics[cls_name] = cls_metrics
                acc_by_class[cls_name] = acc
                print(
                    f"[CLASS] {cls_name:>10s} "
                    f"MOTA={cls_metrics['mota']:.4f} IDF1={cls_metrics['idf1']:.4f} "
                    f"R={cls_metrics['recall']:.4f} P={cls_metrics['precision']:.4f} "
                    f"TP={int(cls_metrics['num_matches'])} FP={int(cls_metrics['num_false_positives'])} "
                    f"FN={int(cls_metrics['num_misses'])} IDSW={int(cls_metrics['num_switches'])}",
                    flush=True,
                )

    for cls_name in class_list:
        all_accs.append(acc_by_class[cls_name])
        all_names.append(cls_name)

    mh = mm.metrics.create()
    overall = mh.compute_many(
        all_accs,
        names=all_names,
        metrics=["mota", "motp", "precision", "recall", "idf1", "num_matches", "num_false_positives", "num_misses", "num_switches"],
        generate_overall=True,
    )
    overall_row = overall.loc["OVERALL"].to_dict()
    overall_metrics = {
        k: (float(v) if isinstance(v, (np.floating, float)) else int(v) if isinstance(v, (np.integer, int)) else v)
        for k, v in overall_row.items()
    }

    print(
        f"[OVERALL] MOTA={overall_metrics['mota']:.4f} IDF1={overall_metrics['idf1']:.4f} "
        f"R={overall_metrics['recall']:.4f} P={overall_metrics['precision']:.4f} "
        f"TP={int(overall_metrics['num_matches'])} FP={int(overall_metrics['num_false_positives'])} "
        f"FN={int(overall_metrics['num_misses'])} IDSW={int(overall_metrics['num_switches'])}",
        flush=True,
    )

    out_obj = {
        "config": {
            "result": args.result,
            "dataroot": args.dataroot,
            "version": args.version,
            "eval_set": args.eval_set,
            "scene_count": len(eval_scene_names),
            "scene_list": args.scene_list,
            "classes": class_list,
            "dist_th": args.dist_th,
            "score_thr": args.score_thr,
            "num_workers": num_workers,
            "class_range": DEFAULT_CLASS_RANGE,
        },
        "per_class": all_metrics,
        "overall": overall_metrics,
        "meta": result_data.get("meta", {}),
    }

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2, ensure_ascii=False)
        print(f"[SAVE] {args.output}", flush=True)
    else:
        print(json.dumps(out_obj, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
