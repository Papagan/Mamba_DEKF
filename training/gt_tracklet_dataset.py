# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — Training Dataset
#
# Extracts GT tracklets from nuScenes annotations and provides
# training samples for TemporalMamba.
#
# Each sample:
#   Input  : track_history [T, 13]  — past T frames of a GT tracklet
#   Target : gt_next_pos [3], gt_next_size [3], gt_next_ori [1]
#   Meta   : delta_t, instance_token (for contrastive pairing)
# ------------------------------------------------------------------------

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple


def _quat_to_yaw(rotation: list) -> float:
    """Convert nuScenes quaternion [w, x, y, z] to yaw angle."""
    w, x, y, z = rotation
    # yaw = atan2(2*(w*z + x*y), 1 - 2*(y^2 + z^2))
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _wrap_to_pi(angle: float) -> float:
    """Wrap angle to [-pi, pi]."""
    return angle - 2.0 * np.pi * round(angle / (2.0 * np.pi))


def extract_gt_tracklets_nuscenes(
    nusc_version: str = "v1.0-trainval",
    nusc_dataroot: str = "data/nuscenes/datasets/",
    split: str = "train",
    output_dir: str = "data/training_cache/nuscenes/",
    category_filter: Optional[List[str]] = None,
) -> str:
    """
    Extract GT tracklets from nuScenes using nuscenes-devkit.

    Each tracklet is a sequence of frames for one instance_token,
    with 13-dim features matching _extract_track_history() format:
        [x, y, z, vx, vy, vz, ax, ay, l, w, h, theta, omega]

    Args:
        nusc_version  : e.g. "v1.0-trainval" or "v1.0-mini"
        nusc_dataroot : path to nuScenes dataset root
        split         : "train" or "val"
        output_dir    : where to save the pickle cache
        category_filter : if set, only keep these categories

    Returns:
        Path to the saved pickle file.
    """
    from nuscenes.nuscenes import NuScenes
    from nuscenes.utils.splits import create_splits_scenes

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"gt_tracklets_{split}.pkl")

    if os.path.exists(output_path):
        print(f"[extract] Cache exists: {output_path}, skipping.")
        return output_path

    print(f"[extract] Loading nuScenes {nusc_version} from {nusc_dataroot} ...")
    nusc = NuScenes(version=nusc_version, dataroot=nusc_dataroot, verbose=True)

    # Get scene names for this split
    splits = create_splits_scenes()
    scene_names = set(splits.get(split, []))
    if not scene_names:
        # fallback: "train" might be "train_detect" or similar
        for k in splits:
            if split in k:
                scene_names.update(splits[k])
        if not scene_names:
            raise ValueError(f"Split '{split}' not found. Available: {list(splits.keys())}")

    print(f"[extract] Processing {len(scene_names)} scenes for split '{split}' ...")

    # Collect all annotations grouped by instance_token
    # instance_token -> list of (timestamp_sec, annotation_record)
    instance_frames: Dict[str, List[Tuple[float, dict]]] = {}

    for scene in nusc.scene:
        if scene["name"] not in scene_names:
            continue

        sample_token = scene["first_sample_token"]
        while sample_token:
            sample = nusc.get("sample", sample_token)
            timestamp_sec = sample["timestamp"] / 1e6  # microseconds -> seconds

            for ann_token in sample["anns"]:
                ann = nusc.get("sample_annotation", ann_token)
                inst_token = ann["instance_token"]

                if category_filter:
                    # nuScenes categories like "vehicle.car" -> check prefix
                    cat = ann["category_name"]
                    if not any(c in cat for c in category_filter):
                        continue

                if inst_token not in instance_frames:
                    instance_frames[inst_token] = []

                instance_frames[inst_token].append((timestamp_sec, ann))

            sample_token = sample["next"] if sample["next"] else None

    print(f"[extract] Found {len(instance_frames)} unique instances.")

    # Build tracklets
    tracklets = []
    for inst_token, frame_list in instance_frames.items():
        # Sort by timestamp
        frame_list.sort(key=lambda x: x[0])

        if len(frame_list) < 2:
            continue  # need at least 2 frames for a training sample

        category = frame_list[0][1]["category_name"]

        frames = []
        prev_vel = None
        prev_yaw = None
        prev_ts = None

        for ts, ann in frame_list:
            # Position (global)
            xyz = ann["translation"]  # [x, y, z]

            # Size: nuScenes uses [w, l, h], project convention is [l, w, h]
            size_wlh = ann["size"]
            lwh = [size_wlh[1], size_wlh[0], size_wlh[2]]

            # Yaw from quaternion
            yaw = _quat_to_yaw(ann["rotation"])

            # Velocity
            try:
                vel_3d = nusc.box_velocity(ann["token"])  # [vx, vy, vz]
                if np.any(np.isnan(vel_3d)):
                    vel_3d = np.array([0.0, 0.0, 0.0])
            except Exception:
                vel_3d = np.array([0.0, 0.0, 0.0])
            vx, vy = float(vel_3d[0]), float(vel_3d[1])

            # Acceleration (finite-diff of velocity)
            ax, ay = 0.0, 0.0
            if prev_vel is not None and prev_ts is not None:
                dt = ts - prev_ts
                if dt > 0:
                    ax = (vx - prev_vel[0]) / dt
                    ay = (vy - prev_vel[1]) / dt

            # Omega (finite-diff of yaw)
            omega = 0.0
            if prev_yaw is not None and prev_ts is not None:
                dt = ts - prev_ts
                if dt > 0:
                    dyaw = _wrap_to_pi(yaw - prev_yaw)
                    omega = dyaw / dt

            feature_13 = [
                xyz[0], xyz[1], xyz[2],
                vx, vy, 0.0,
                ax, ay,
                lwh[0], lwh[1], lwh[2],
                yaw, omega,
            ]

            frames.append({
                "feature_13": feature_13,
                "timestamp": ts,
                "global_xyz": xyz,
                "lwh": lwh,
                "yaw": yaw,
                "velocity": [vx, vy],
            })

            prev_vel = [vx, vy]
            prev_yaw = yaw
            prev_ts = ts

        tracklets.append({
            "instance_token": inst_token,
            "category": category,
            "frames": frames,
        })

    print(f"[extract] Built {len(tracklets)} tracklets (>= 2 frames each).")

    with open(output_path, "wb") as f:
        pickle.dump(tracklets, f)
    print(f"[extract] Saved to {output_path}")

    return output_path


class TrackletDataset(Dataset):
    """
    PyTorch Dataset for training TemporalMamba.

    Produces sliding-window samples from GT tracklets:
      - Input:  track_history [T, 13]
      - Target: GT state at frame T+1
      - Meta:   delta_t, instance_token

    The 13-dim feature format matches base_tracker._extract_track_history():
        [x, y, z, vx, vy, vz, ax, ay, l, w, h, theta, omega]
    """

    def __init__(
        self,
        tracklet_pkl_path: str,
        history_len: int = 10,
        min_track_len: int = 3,
    ) -> None:
        with open(tracklet_pkl_path, "rb") as f:
            tracklets = pickle.load(f)

        self.history_len = history_len
        self.samples: List[Dict] = []

        for trk in tracklets:
            frames = trk["frames"]
            n = len(frames)
            if n < min_track_len:
                continue

            inst_token = trk["instance_token"]
            category = trk["category"]

            # Sliding window: for each valid index i, use frames[max(0,i-T+1):i+1]
            # as history, frames[i+1] as target
            for i in range(min_track_len - 1, n - 1):
                start = max(0, i - history_len + 1)
                history_frames = frames[start:i + 1]
                target_frame = frames[i + 1]
                current_frame = frames[i]

                # delta_t
                dt = target_frame["timestamp"] - current_frame["timestamp"]
                if dt <= 0:
                    continue

                self.samples.append({
                    "history_frames": history_frames,
                    "target_frame": target_frame,
                    "current_frame": current_frame,
                    "delta_t": dt,
                    "instance_token": inst_token,
                    "category": category,
                })

        print(f"[TrackletDataset] {len(self.samples)} samples from "
              f"{len(tracklets)} tracklets (T={history_len})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        T = self.history_len

        # ---- Build history [T, 13] (right-aligned, zero-padded) ----
        history = np.zeros((T, 13), dtype=np.float32)
        mask = np.zeros(T, dtype=np.bool_)
        n_frames = len(s["history_frames"])
        for t_idx, frame in enumerate(s["history_frames"]):
            offset = T - n_frames + t_idx
            history[offset] = frame["feature_13"]
            mask[offset] = True

        # ---- Current state (for KF init at frame T) ----
        cur = s["current_frame"]
        # pos state: [x, y, z, vx, vy, vz=0, ax, ay]
        gt_current_pos = np.array([
            cur["global_xyz"][0], cur["global_xyz"][1], cur["global_xyz"][2],
            cur["velocity"][0], cur["velocity"][1], 0.0,
            cur["feature_13"][6], cur["feature_13"][7],  # ax, ay
        ], dtype=np.float32)
        gt_current_siz = np.array(cur["lwh"], dtype=np.float32)
        gt_current_ori = np.array([
            cur["yaw"], cur["feature_13"][12],  # theta, omega
        ], dtype=np.float32)

        # ---- Target state at frame T+1 ----
        tgt = s["target_frame"]
        gt_next_pos = np.array(tgt["global_xyz"], dtype=np.float32)
        gt_next_siz = np.array(tgt["lwh"], dtype=np.float32)
        gt_next_ori = np.array([tgt["yaw"]], dtype=np.float32)

        return {
            "track_history": torch.from_numpy(history),           # [T, 13]
            "history_mask": torch.from_numpy(mask),               # [T]
            "gt_current_state_pos": torch.from_numpy(gt_current_pos),  # [8]
            "gt_current_state_siz": torch.from_numpy(gt_current_siz),  # [3]
            "gt_current_state_ori": torch.from_numpy(gt_current_ori),  # [2]
            "gt_next_pos": torch.from_numpy(gt_next_pos),         # [3]
            "gt_next_size": torch.from_numpy(gt_next_siz),        # [3]
            "gt_next_ori": torch.from_numpy(gt_next_ori),         # [1]
            "delta_t": torch.tensor(s["delta_t"], dtype=torch.float32),
            "instance_token": s["instance_token"],
            "category": s["category"],
        }


def tracklet_collate_fn(batch: List[Dict]) -> Dict:
    """
    Custom collate that handles string fields (instance_token, category).
    """
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result
