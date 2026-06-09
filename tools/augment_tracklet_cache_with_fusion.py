#!/usr/bin/env python3
"""
Augment an aligned detection-driven tracklet cache with MCTrack-compatible
fusion-history fields.

The script runs a per-tracklet pure-DEKF rollout:
  - before the first matched detection: no fusion state exists
  - on matched frames: predict + KF update, then store fused state
  - on missed frames after birth: predict only, then store fake fusion state
  - after long miss gaps (> PREDICT_BBOX_LENGTH): the pseudo-track dies until
    the next matched frame re-initializes it

This is designed to align training inputs with TRACKER_COMPAT_MODE=mctrack
without requiring a full online tracker replay.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import pickle
from pathlib import Path
from typing import Dict, List, Tuple, TYPE_CHECKING

import yaml

if TYPE_CHECKING:
    import torch


def wrap_to_pi(angle: float) -> float:
    return float(angle - 2.0 * math.pi * round(angle / (2.0 * math.pi)))


def load_pickle(path: str) -> List[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def save_pickle(path: str, data: List[dict]) -> None:
    with open(path, "wb") as f:
        pickle.dump(data, f)


def build_feature12(
    xyz: List[float],
    vel_xy: List[float],
    lwh: List[float],
    yaw: float,
    omega: float,
    score: float,
) -> List[float]:
    return [
        float(xyz[0]), float(xyz[1]), float(xyz[2]),
        float(vel_xy[0]), float(vel_xy[1]), 0.0,
        float(lwh[0]), float(lwh[1]), float(lwh[2]),
        float(yaw), float(omega), float(score),
    ]


def init_kf_from_detection(frame: dict, device) -> DecoupledAdaptiveKF:
    import torch
    from kalmanfilter.mamba_adaptive_kf import DecoupledAdaptiveKF

    kf = DecoupledAdaptiveKF(batch_size=1, device=device)
    vel = frame["det_velocity"] or [0.0, 0.0]
    pos_x0 = torch.tensor([
        frame["det_global_xyz"][0], frame["det_global_xyz"][1], frame["det_global_xyz"][2],
        vel[0], vel[1], 0.0,
    ], dtype=torch.float32, device=device).view(1, 6, 1)
    pos_P0 = torch.eye(6, dtype=torch.float32, device=device).unsqueeze(0)
    pos_P0[:, 3, 3] = 10.0
    pos_P0[:, 4, 4] = 10.0
    siz_x0 = torch.tensor(frame["det_lwh"], dtype=torch.float32, device=device).view(1, 3, 1)
    siz_P0 = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0) * 0.1
    ori_x0 = torch.tensor([frame["det_yaw"], frame["obs_feature_12"][10]], dtype=torch.float32, device=device).view(1, 2, 1)
    ori_P0 = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0) * 0.1
    kf.init_states(pos_x0, pos_P0, siz_x0, siz_P0, ori_x0, ori_P0)
    return kf


def get_base_covariances(base_noise_cfg: dict, category: str, device) -> Dict[str, "torch.Tensor"]:
    import torch
    from kalmanfilter.noise_priors import (
        build_base_covariances,
        categories_to_class_ids,
    )

    class_ids = categories_to_class_ids([category], device=device)
    return build_base_covariances(
        base_noise_cfg=base_noise_cfg,
        class_ids=class_ids,
        dtype=torch.float32,
        device=device,
        track_history=None,
    )


def finalize_fusion_omegas(tracklet: dict) -> None:
    prev_yaw = None
    prev_ts = None
    for frame in tracklet["frames"]:
        if frame.get("fusion_valid", False):
            if prev_yaw is not None and prev_ts is not None:
                dt = float(frame["timestamp"] - prev_ts)
                if dt > 0:
                    omega = wrap_to_pi(float(frame["fusion_yaw"]) - prev_yaw) / dt
                    frame["fusion_feature_12"][10] = float(omega)
            prev_yaw = float(frame["fusion_yaw"])
            prev_ts = float(frame["timestamp"])


def augment_tracklet(tracklet: dict, base_noise_cfg: dict, device) -> Tuple[dict, Dict[str, int]]:
    import torch

    trk = copy.deepcopy(tracklet)
    frames = sorted(trk["frames"], key=lambda x: (x["timestamp"], x["frame_id"]))
    stats = {
        "fusion_valid_frames": 0,
        "fusion_fake_frames": 0,
        "rebirths": 0,
        "prebirth_invalid_frames": 0,
    }

    for frame in frames:
        frame["fusion_valid"] = False
        frame["fusion_is_fake"] = False
        frame["fusion_global_xyz"] = None
        frame["fusion_lwh"] = None
        frame["fusion_yaw"] = None
        frame["fusion_velocity"] = None
        frame["fusion_feature_12"] = [0.0] * 12

    if not frames:
        trk["frames"] = frames
        return trk, stats

    cov = get_base_covariances(base_noise_cfg, trk["category"], device=device)
    Q_pos = cov["Q_pos_base"]
    Q_siz = cov["Q_siz_base"]
    Q_ori = cov["Q_ori_base"]
    R_pos = cov["R_pos_base"]
    R_siz = cov["R_siz_base"]
    R_ori = cov["R_ori_base"]
    max_predict_len = int(base_noise_cfg.get("TRACKER_LIMITS", {}).get("PREDICT_BBOX_LENGTH", 3))

    kf = None
    active = False
    last_ts = None
    last_fused_xyz = None
    last_fused_lwh = None
    last_fused_yaw = None
    consecutive_miss = 0

    for frame in frames:
        is_matched = bool(frame["is_matched"]) and frame["det_global_xyz"] is not None

        if not active:
            if not is_matched:
                stats["prebirth_invalid_frames"] += 1
                last_ts = float(frame["timestamp"])
                continue

            kf = init_kf_from_detection(frame, device)
            active = True
            consecutive_miss = 0
            stats["rebirths"] += 1

            fused_xyz = [float(v) for v in frame["det_global_xyz"]]
            fused_lwh = [float(v) for v in frame["det_lwh"]]
            fused_yaw = float(frame["det_yaw"])
            fused_vel = [float(v) for v in (frame["det_velocity"] or [0.0, 0.0])]
            frame["fusion_valid"] = True
            frame["fusion_is_fake"] = False
            frame["fusion_global_xyz"] = fused_xyz
            frame["fusion_lwh"] = fused_lwh
            frame["fusion_yaw"] = fused_yaw
            frame["fusion_velocity"] = fused_vel
            frame["fusion_feature_12"] = build_feature12(
                fused_xyz, fused_vel, fused_lwh, fused_yaw, 0.0, float(frame["det_score"])
            )
            stats["fusion_valid_frames"] += 1
            last_fused_xyz = fused_xyz
            last_fused_lwh = fused_lwh
            last_fused_yaw = fused_yaw
            last_ts = float(frame["timestamp"])
            continue

        dt = float(frame["timestamp"] - last_ts) if last_ts is not None else 0.1
        dt = max(dt, 1e-6)
        pos_x_pred, pos_P_pred, siz_x_pred, siz_P_pred, ori_x_pred, ori_P_pred = kf.predict(
            torch.tensor([dt], dtype=torch.float32, device=device),
            Q_pos=Q_pos, Q_siz=Q_siz, Q_ori=Q_ori,
        )

        if is_matched:
            z_vel = frame["det_velocity"] or [0.0, 0.0]
            z_pos = torch.tensor([
                frame["det_global_xyz"][0], frame["det_global_xyz"][1], frame["det_global_xyz"][2],
                z_vel[0], z_vel[1],
            ], dtype=torch.float32, device=device).view(1, 5, 1)
            z_siz = torch.tensor(frame["det_lwh"], dtype=torch.float32, device=device).view(1, 3, 1)
            z_ori = torch.tensor([[frame["det_yaw"]]], dtype=torch.float32, device=device).view(1, 1, 1)
            pos_x_upd, pos_P_upd, siz_x_upd, siz_P_upd, ori_x_upd, ori_P_upd = kf.update(
                z_pos, z_siz, z_ori, R_pos=R_pos, R_siz=R_siz, R_ori=R_ori
            )
            fused_xyz = [
                float(pos_x_upd[0, 0, 0].item()),
                float(pos_x_upd[0, 1, 0].item()),
                float(frame["det_global_xyz"][2]),
            ]
            fused_lwh = [float(v) for v in siz_x_upd[0, :, 0].tolist()]
            fused_yaw = float(ori_x_upd[0, 0, 0].item())
            fused_vel = [
                float(pos_x_upd[0, 3, 0].item()),
                float(pos_x_upd[0, 4, 0].item()),
            ]
            frame_score = float(frame["det_score"])
            frame_fake = False
            consecutive_miss = 0
        else:
            fused_xyz = [
                float(pos_x_pred[0, 0, 0].item()),
                float(pos_x_pred[0, 1, 0].item()),
                float(last_fused_xyz[2]),
            ]
            fused_lwh = list(last_fused_lwh)
            fused_yaw = float(last_fused_yaw)
            fused_vel = [
                float(pos_x_pred[0, 3, 0].item()),
                float(pos_x_pred[0, 4, 0].item()),
            ]
            frame_score = 0.0
            frame_fake = True
            consecutive_miss += 1
            if consecutive_miss > max_predict_len:
                active = False

        if active or is_matched:
            frame["fusion_valid"] = True
            frame["fusion_is_fake"] = frame_fake
            frame["fusion_global_xyz"] = fused_xyz
            frame["fusion_lwh"] = fused_lwh
            frame["fusion_yaw"] = fused_yaw
            frame["fusion_velocity"] = fused_vel
            frame["fusion_feature_12"] = build_feature12(
                fused_xyz, fused_vel, fused_lwh, fused_yaw, 0.0, frame_score
            )
            stats["fusion_valid_frames"] += 1
            if frame_fake:
                stats["fusion_fake_frames"] += 1
            last_fused_xyz = fused_xyz
            last_fused_lwh = fused_lwh
            last_fused_yaw = fused_yaw
        else:
            last_fused_xyz = None
            last_fused_lwh = None
            last_fused_yaw = None
            kf = None
            consecutive_miss = 0

        last_ts = float(frame["timestamp"])

    trk["frames"] = frames
    finalize_fusion_omegas(trk)
    return trk, stats


def build_tracker_limits(train_cfg: dict) -> Dict[str, int]:
    data_cfg = train_cfg.get("DATA", {})
    return {
        "PREDICT_BBOX_LENGTH": int(data_cfg.get("FUSION_MAX_PREDICT_LEN", 3))
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Augment aligned tracklet cache with fusion history.")
    parser.add_argument("--input", required=True, help="Input aligned detection-driven cache (.pkl)")
    parser.add_argument("--output", required=True, help="Output augmented cache (.pkl)")
    parser.add_argument("--train-config", default="config/train_nuscenes.yaml", help="Training config YAML")
    args = parser.parse_args()

    import torch

    with open(args.train_config, "r", encoding="utf-8") as f:
        train_cfg = yaml.safe_load(f) or {}

    tracklets = load_pickle(args.input)
    device = torch.device("cpu")
    base_noise_cfg = copy.deepcopy(train_cfg.get("BASE_NOISE", {}))
    base_noise_cfg["TRACKER_LIMITS"] = build_tracker_limits(train_cfg)

    out_tracklets = []
    total_stats = {
        "tracklets": len(tracklets),
        "fusion_valid_frames": 0,
        "fusion_fake_frames": 0,
        "prebirth_invalid_frames": 0,
        "rebirths": 0,
    }
    for trk in tracklets:
        aug_trk, stats = augment_tracklet(trk, base_noise_cfg=base_noise_cfg, device=device)
        out_tracklets.append(aug_trk)
        for k, v in stats.items():
            total_stats[k] = total_stats.get(k, 0) + int(v)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_pickle(str(output_path), out_tracklets)

    summary = {
        **total_stats,
        "input": args.input,
        "output": str(output_path),
        "train_config": args.train_config,
        "history_semantics": "mctrack_compat_pure_dekf_local_rollout",
    }
    summary_path = output_path.with_name(output_path.stem + "_summary.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[fusion-cache] saved augmented cache -> {output_path}")
    print(f"[fusion-cache] saved summary -> {summary_path}")
    print(
        "[fusion-cache] "
        f"tracklets={total_stats['tracklets']} "
        f"fusion_valid_frames={total_stats['fusion_valid_frames']} "
        f"fusion_fake_frames={total_stats['fusion_fake_frames']} "
        f"prebirth_invalid_frames={total_stats['prebirth_invalid_frames']} "
        f"rebirths={total_stats['rebirths']}"
    )


if __name__ == "__main__":
    main()
