# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json, yaml
import logging
import copy
import argparse
import os
import time
import multiprocessing
import math
from tqdm import tqdm
from datetime import datetime
from functools import partial
from kalmanfilter.noise_audit import NoiseAuditAccumulator
from tracker.base_tracker import Base3DTracker
from dataset.baseversion_dataset import BaseVersionTrackingDataset
from evaluation.static_evaluation.kitti.evaluation_HOTA.scripts.run_kitti import (
    eval_kitti,
)
from evaluation.static_evaluation.nuscenes.eval import eval_nusc
from evaluation.static_evaluation.waymo.eval import eval_waymo
from utils.kitti_utils import save_results_kitti
from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion
from utils.waymo_utils.convert_result import save_results_waymo


def _print_nuscenes_result_diagnostics(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    results = saved.get("results", {})
    sample_count = len(results)
    nonempty_samples = sum(1 for boxes in results.values() if boxes)
    total_boxes = sum(len(boxes) for boxes in results.values())
    print(
        f"[SAVE DIAG] samples={sample_count} nonempty_samples={nonempty_samples} "
        f"boxes={total_boxes}"
    )

    class_counts = {}
    invalid_counts = {
        "score_nonfinite": 0,
        "translation_nonfinite": 0,
        "size_nonfinite": 0,
        "rotation_nonfinite": 0,
        "velocity_nonfinite": 0,
    }
    score_stats = {}
    first_examples = []

    for sample_token, boxes in results.items():
        for box in boxes:
            cls = box.get("tracking_name", "UNKNOWN")
            class_counts[cls] = class_counts.get(cls, 0) + 1

            score = box.get("tracking_score", None)
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                invalid_counts["score_nonfinite"] += 1
            else:
                stat = score_stats.setdefault(cls, {"min": score, "max": score})
                stat["min"] = min(stat["min"], score)
                stat["max"] = max(stat["max"], score)

            for key, invalid_key in [
                ("translation", "translation_nonfinite"),
                ("size", "size_nonfinite"),
                ("rotation", "rotation_nonfinite"),
                ("velocity", "velocity_nonfinite"),
            ]:
                values = box.get(key, [])
                if (
                    not isinstance(values, list)
                    or any((not isinstance(v, (int, float)) or not math.isfinite(v)) for v in values)
                ):
                    invalid_counts[invalid_key] += 1

            if len(first_examples) < 5:
                first_examples.append(
                    {
                        "sample_token": sample_token,
                        "tracking_name": cls,
                        "tracking_id": box.get("tracking_id"),
                        "tracking_score": box.get("tracking_score"),
                        "translation": box.get("translation"),
                        "size": box.get("size"),
                    }
                )

    print(f"[SAVE DIAG] class_counts={class_counts}")
    print(f"[SAVE DIAG] invalid_counts={invalid_counts}")
    print(f"[SAVE DIAG] score_stats={score_stats}")
    for idx, example in enumerate(first_examples):
        print(f"[SAVE DIAG] example_{idx}={example}")


def _build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


def _collect_scene_inference_audit_state(scene_id, tracker, cfg, scene_audit_states):
    audit_cfg = _build_noise_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return
    state = tracker.export_noise_audit_state()
    if state is not None:
        scene_audit_states[scene_id] = state


def _write_merged_infer_noise_audit(cfg, scene_audit_states):
    audit_cfg = _build_noise_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return

    merged = NoiseAuditAccumulator()
    for scene_id in sorted(scene_audit_states.keys()):
        merged.merge_state(scene_audit_states[scene_id])

    output_path = audit_cfg.get("INFER_OUTPUT_PATH", "debug/infer_noise_audit.json")
    try:
        merged.write_json(output_path)
    except Exception as exc:
        if audit_cfg.get("STRICT", False):
            raise
        print(f"[main] WARNING: failed to write merged inference noise audit to {output_path}: {exc}")


def run(scene_id, scenes_data, cfg, args, tracking_results, scene_audit_states):
    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
            scene_audit_states: Dictionary to store per-scene audit state.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    scene_data = scenes_data[scene_id]
    dataset = BaseVersionTrackingDataset(scene_id, scene_data, cfg=cfg)
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}

    for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        cur_sample_token = frame_info.cur_sample_token
        all_traj = tracker.track_single_frame(frame_info)
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": copy.deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = copy.deepcopy(result_info)
    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (
                        bbox.frame_id == frame_id
                        and bbox.is_interpolation
                        and track_id not in all_trajs[frame_id]["trajs"].keys()
                    ):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (
                    track_id in all_trajs[frame_id]["trajs"].keys()
                    and det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]
                ):
                    del all_trajs[frame_id]["trajs"][track_id]

    _collect_scene_inference_audit_state(scene_id, tracker, cfg, scene_audit_states)
    tracking_results[scene_id] = all_trajs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTrack")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional config path. If set, overrides dataset default config path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="Which Dataset: kitti/nuscenes/waymo",
    )
    parser.add_argument("--eval", "-e", action="store_true", help="evaluation")
    parser.add_argument("--load_image", "-lm", action="store_true", help="load_image")
    parser.add_argument("--load_point", "-lp", action="store_true", help="load_point")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--mode", "-m", action="store_true", help="online or offline")
    parser.add_argument("--process", "-p", type=int, default=1, help="multi-process!")
    args = parser.parse_args()

    if args.config:
        cfg_path = args.config
    else:
        if args.dataset == "kitti":
            cfg_path = "./config/kitti.yaml"
        elif args.dataset == "nuscenes":
            cfg_path = "./config/nuscenes.yaml"
        elif args.dataset == "waymo":
            cfg_path = "./config/waymo.yaml"
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        if args.mode:
            cfg_path = cfg_path.replace(".yaml", "_offline.yaml")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)

    save_path = os.path.join(
        os.path.dirname(cfg["SAVE_PATH"]),
        cfg["DATASET"],
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(save_path, exist_ok=True)
    cfg["SAVE_PATH"] = save_path

    start_time = time.time()

    detections_root = os.path.join(
        cfg["DETECTIONS_ROOT"], cfg["DETECTOR"], cfg["SPLIT"] + ".json"
    )
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        print("Data loaded successfully.")

    if args.debug:
        if args.dataset == "kitti":
            scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
        elif args.dataset == "nuscenes":
            scene_lists = [scene_id for scene_id in data.keys()][:2]
        else:
            scene_lists = [scene_id for scene_id in data.keys()][:2]
    else:
        scene_lists = [scene_id for scene_id in data.keys()]

    manager = multiprocessing.Manager()
    tracking_results = manager.dict()
    scene_audit_states = manager.dict()
    if args.process > 1:
        pool = multiprocessing.Pool(args.process)
        func = partial(
            run,
            scenes_data=data,
            cfg=cfg,
            args=args,
            tracking_results=tracking_results,
            scene_audit_states=scene_audit_states,
        )
        pool.map(func, scene_lists)
        pool.close()
        pool.join()
    else:
        for scene_id in tqdm(scene_lists, desc="Running scenes"):
            run(scene_id, data, cfg, args, tracking_results, scene_audit_states)
    tracking_results = dict(tracking_results)
    scene_audit_states = dict(scene_audit_states)
    _write_merged_infer_noise_audit(cfg, scene_audit_states)

    if args.dataset == "kitti":
        save_results_kitti(tracking_results, cfg)
        if args.eval:
            eval_kitti(cfg)
    if args.dataset == "nuscenes":
        save_results_nuscenes(tracking_results, save_path, cfg=cfg)
        # --- quick sanity: count saved boxes ---
        _res_file = os.path.join(save_path, "results.json")
        with open(_res_file) as _f:
            _saved = json.load(_f)
        _n_samples = len(_saved.get("results", {}))
        _n_boxes = sum(len(v) for v in _saved.get("results", {}).values())
        print(f"[SAVE] samples={_n_samples} boxes={_n_boxes} path={_res_file}")
        if _n_boxes == 0:
            # dump first scene's first frame for debugging
            _scenes = list(tracking_results.keys())
            if _scenes:
                _s0 = _scenes[0]
                _frames = list(tracking_results[_s0].keys())
                if _frames:
                    _f0 = _frames[0]
                    _trajs = tracking_results[_s0][_f0].get("trajs", {})
                    print(f"[SAVE DEBUG] scene={_s0} frame={_f0} n_trajs={len(_trajs)}")
                    for _tid, _bb in list(_trajs.items())[:3]:
                        print(f"[SAVE DEBUG]   track={_tid} cat={_bb.category} "
                              f"score={_bb.det_score:.3f} "
                              f"xyz=[{_bb.global_xyz_lwh_yaw_fusion[0]:.1f},"
                              f"{_bb.global_xyz_lwh_yaw_fusion[1]:.1f},"
                              f"{_bb.global_xyz_lwh_yaw_fusion[2]:.1f}] "
                              f"yaw_fusion={getattr(_bb, 'global_yaw_fusion', 'MISSING')}")
        _print_nuscenes_result_diagnostics(_res_file)
        save_results_nuscenes_for_motion(tracking_results, save_path)
        if args.eval:
            eval_nusc(cfg)
    elif args.dataset == "waymo":
        save_results_waymo(tracking_results, save_path)
        if args.eval:
            eval_waymo(cfg, save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
