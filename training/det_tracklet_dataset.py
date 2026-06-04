from __future__ import annotations

import pickle
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class DetectionTrackletDataset(Dataset):
    """
    Detection-driven dataset backed by aligned detection/GT tracklet caches.

    The cache format is produced by tools/build_centerpoint_mini_train_dataset.py.
    History uses detector observations (`obs_feature_12`) with explicit miss steps.
    Targets stay on GT, preserving supervised rollout training.
    """

    def __init__(
        self,
        tracklet_pkl_path: str,
        history_len: int = 10,
        min_track_len: int = 3,
        rollout_steps: int = 1,
        require_current_match: bool = True,
        min_history_match_ratio: float = 0.25,
    ) -> None:
        with open(tracklet_pkl_path, "rb") as f:
            tracklets = pickle.load(f)

        self.history_len = history_len
        self.rollout_steps = rollout_steps
        self.samples: List[Dict] = []

        min_len = max(min_track_len, history_len + rollout_steps)

        for trk in tracklets:
            frames = trk["frames"]
            n = len(frames)
            if n < min_len:
                continue

            inst_token = trk["instance_token"]
            category = trk["category"]

            for i in range(history_len - 1, n - rollout_steps):
                start = max(0, i - history_len + 1)
                history_frames = frames[start : i + 1]
                current_frame = frames[i]
                future_frames = frames[i + 1 : i + 1 + rollout_steps]

                if require_current_match and not bool(current_frame["is_matched"]):
                    continue

                matched_hist = sum(1 for fr in history_frames if bool(fr["is_matched"]))
                hist_match_ratio = matched_hist / max(len(history_frames), 1)
                if hist_match_ratio < min_history_match_ratio:
                    continue

                dt = future_frames[0]["timestamp"] - current_frame["timestamp"]
                if dt <= 0:
                    continue

                self.samples.append({
                    "history_frames": history_frames,
                    "current_frame": current_frame,
                    "future_frames": future_frames,
                    "delta_t": dt,
                    "instance_token": inst_token,
                    "category": category,
                })

        print(
            f"[DetectionTrackletDataset] {len(self.samples)} samples from "
            f"{len(tracklets)} tracklets (T={history_len}, K={rollout_steps})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        T = self.history_len
        K = self.rollout_steps

        history = np.zeros((T, 12), dtype=np.float32)
        history_mask = np.zeros(T, dtype=np.bool_)
        history_match_mask = np.zeros(T, dtype=np.bool_)

        cur = s["current_frame"]
        ref_xyz = np.asarray(
            cur["det_global_xyz"] if cur["is_matched"] and cur["det_global_xyz"] is not None else cur["gt_global_xyz"],
            dtype=np.float32,
        )

        n_frames = len(s["history_frames"])
        for t_idx, frame in enumerate(s["history_frames"]):
            offset = T - n_frames + t_idx
            feat = np.asarray(frame["obs_feature_12"], dtype=np.float32)
            if bool(frame["is_matched"]):
                feat = feat.copy()
                feat[0] -= ref_xyz[0]
                feat[1] -= ref_xyz[1]
                history[offset] = feat
                history_match_mask[offset] = True
            history_mask[offset] = True

        gt_current_pos = np.array([
            cur["gt_global_xyz"][0], cur["gt_global_xyz"][1], cur["gt_global_xyz"][2],
            cur["gt_velocity"][0], cur["gt_velocity"][1], 0.0,
        ], dtype=np.float32)
        gt_current_siz = np.array(cur["gt_lwh"], dtype=np.float32)
        gt_current_ori = np.array([cur["gt_yaw"], cur["gt_feature_12"][10]], dtype=np.float32)

        if cur["is_matched"] and cur["det_global_xyz"] is not None:
            obs_current_pos = np.array([
                cur["det_global_xyz"][0], cur["det_global_xyz"][1], cur["det_global_xyz"][2],
                cur["det_velocity"][0], cur["det_velocity"][1], 0.0,
            ], dtype=np.float32)
            obs_current_siz = np.array(cur["det_lwh"], dtype=np.float32)
            obs_current_ori = np.array([cur["det_yaw"], cur["obs_feature_12"][10]], dtype=np.float32)
        else:
            obs_current_pos = gt_current_pos.copy()
            obs_current_siz = gt_current_siz.copy()
            obs_current_ori = gt_current_ori.copy()

        gt_future_pos = np.zeros((K, 3), dtype=np.float32)
        gt_future_siz = np.zeros((K, 3), dtype=np.float32)
        gt_future_ori = np.zeros((K, 1), dtype=np.float32)
        obs_future_pos = np.zeros((K, 5), dtype=np.float32)
        obs_future_siz = np.zeros((K, 3), dtype=np.float32)
        obs_future_ori = np.zeros((K, 1), dtype=np.float32)
        obs_future_match = np.zeros(K, dtype=np.bool_)
        delta_ts_future = np.zeros(K, dtype=np.float32)

        prev_ts = cur["timestamp"]
        for k, frm in enumerate(s["future_frames"]):
            gt_future_pos[k] = np.asarray(frm["gt_global_xyz"], dtype=np.float32)
            gt_future_siz[k] = np.asarray(frm["gt_lwh"], dtype=np.float32)
            gt_future_ori[k, 0] = float(frm["gt_yaw"])
            if bool(frm["is_matched"]) and frm["det_global_xyz"] is not None:
                obs_future_pos[k] = np.asarray([
                    frm["det_global_xyz"][0], frm["det_global_xyz"][1], frm["det_global_xyz"][2],
                    frm["det_velocity"][0], frm["det_velocity"][1],
                ], dtype=np.float32)
                obs_future_siz[k] = np.asarray(frm["det_lwh"], dtype=np.float32)
                obs_future_ori[k, 0] = float(frm["det_yaw"])
                obs_future_match[k] = True
            dt_k = float(frm["timestamp"] - prev_ts)
            delta_ts_future[k] = max(dt_k, 1e-6)
            prev_ts = frm["timestamp"]

        matched_hist = history_match_mask.sum()
        current_range = float(np.linalg.norm(np.asarray(cur["gt_global_xyz"][:2], dtype=np.float32)))
        current_speed = float(np.linalg.norm(obs_current_pos[3:5]))
        current_det_score = float(cur["det_score"])

        return {
            "track_history": torch.from_numpy(history),
            "history_mask": torch.from_numpy(history_mask),
            "history_match_mask": torch.from_numpy(history_match_mask),
            "gt_current_state_pos": torch.from_numpy(gt_current_pos),
            "gt_current_state_siz": torch.from_numpy(gt_current_siz),
            "gt_current_state_ori": torch.from_numpy(gt_current_ori),
            "obs_current_state_pos": torch.from_numpy(obs_current_pos),
            "obs_current_state_siz": torch.from_numpy(obs_current_siz),
            "obs_current_state_ori": torch.from_numpy(obs_current_ori),
            "gt_future_pos": torch.from_numpy(gt_future_pos),
            "gt_future_siz": torch.from_numpy(gt_future_siz),
            "gt_future_ori": torch.from_numpy(gt_future_ori),
            "obs_future_pos": torch.from_numpy(obs_future_pos),
            "obs_future_siz": torch.from_numpy(obs_future_siz),
            "obs_future_ori": torch.from_numpy(obs_future_ori),
            "obs_future_match": torch.from_numpy(obs_future_match),
            "delta_ts_future": torch.from_numpy(delta_ts_future),
            "delta_t": torch.tensor(s["delta_t"], dtype=torch.float32),
            "instance_token": s["instance_token"],
            "category": s["category"],
            "current_det_score": torch.tensor(current_det_score, dtype=torch.float32),
            "history_match_ratio": torch.tensor(matched_hist / max(n_frames, 1), dtype=torch.float32),
            "current_range": torch.tensor(current_range, dtype=torch.float32),
            "current_speed": torch.tensor(current_speed, dtype=torch.float32),
            "is_detection_driven": torch.tensor(True, dtype=torch.bool),
        }


def detection_tracklet_collate_fn(batch: List[Dict]) -> Dict:
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result
