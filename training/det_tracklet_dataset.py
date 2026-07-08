from __future__ import annotations

import pickle
import random
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from kalmanfilter.noise_priors import category_to_tracking_name


def _resolve_class_window(
    category: str,
    history_len: int,
    rollout_steps: int,
    min_history_len: int,
    min_rollout_steps: int,
    class_window_cfg: Dict | None,
) -> tuple[str, int, int, int, int]:
    track_name = category_to_tracking_name(category)
    cfg = (class_window_cfg or {}).get(track_name, {})
    hist_min = int(cfg.get("MIN_HISTORY_LEN", min_history_len))
    hist_max = int(cfg.get("MAX_HISTORY_LEN", history_len))
    roll_min = int(cfg.get("MIN_ROLLOUT_STEPS", min_rollout_steps))
    roll_max = int(cfg.get("MAX_ROLLOUT_STEPS", rollout_steps))

    hist_max = max(1, min(history_len, hist_max))
    roll_max = max(1, min(rollout_steps, roll_max))
    hist_min = max(1, min(hist_min, hist_max))
    roll_min = max(1, min(roll_min, roll_max))
    return track_name, hist_min, hist_max, roll_min, roll_max


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
        adaptive_windows: bool = False,
        min_history_len: int | None = None,
        min_rollout_steps: int | None = None,
        class_window_cfg: Dict | None = None,
        history_source: str = "det",
        init_state_source: str = "det",
    ) -> None:
        with open(tracklet_pkl_path, "rb") as f:
            tracklets = pickle.load(f)

        self.history_source = str(history_source).strip().lower()
        self.init_state_source = str(init_state_source).strip().lower()
        valid_sources = {"det", "fusion"}
        if self.history_source not in valid_sources:
            raise ValueError(f"Unsupported history_source={history_source!r}; expected one of {sorted(valid_sources)}")
        if self.init_state_source not in valid_sources:
            raise ValueError(f"Unsupported init_state_source={init_state_source!r}; expected one of {sorted(valid_sources)}")

        self.history_len = history_len
        self.rollout_steps = rollout_steps
        self.adaptive_windows = bool(adaptive_windows)
        self.min_history_len = int(min_history_len or history_len)
        self.min_rollout_steps = int(min_rollout_steps or rollout_steps)
        self.class_window_cfg = class_window_cfg or {}
        self.samples: List[Dict] = []

        if tracklets and (self.history_source == "fusion" or self.init_state_source == "fusion"):
            has_fusion = False
            for trk in tracklets[: min(len(tracklets), 32)]:
                for fr in trk.get("frames", [])[: min(len(trk.get("frames", [])), 32)]:
                    if "fusion_feature_12" in fr and "fusion_valid" in fr:
                        has_fusion = True
                        break
                if has_fusion:
                    break
            if not has_fusion:
                raise ValueError(
                    f"{tracklet_pkl_path} does not contain fusion history fields, "
                    "but HISTORY_SOURCE/INIT_STATE_SOURCE requests fusion. "
                    "Run tools/augment_tracklet_cache_with_fusion.py first."
                )

        for trk in tracklets:
            frames = trk["frames"]
            n = len(frames)

            inst_token = trk["instance_token"]
            category = trk["category"]
            track_name, hist_min, hist_max, roll_min, roll_max = _resolve_class_window(
                category=category,
                history_len=history_len,
                rollout_steps=rollout_steps,
                min_history_len=self.min_history_len,
                min_rollout_steps=self.min_rollout_steps,
                class_window_cfg=self.class_window_cfg,
            )
            class_min_len = max(min_track_len, hist_min + roll_min)
            if n < class_min_len:
                continue

            for i in range(hist_min - 1, n - roll_min):
                start = max(0, i - hist_max + 1)
                history_frames = frames[start : i + 1]
                current_frame = frames[i]
                future_frames = frames[i + 1 : i + 1 + roll_max]

                if require_current_match and not bool(current_frame["is_matched"]):
                    continue

                available_history = len(history_frames)
                available_future = len(future_frames)
                if available_history < hist_min or available_future < roll_min:
                    continue

                valid_history_lengths = []
                valid_history_upper = min(available_history, hist_max)
                for eff_h in range(hist_min, valid_history_upper + 1):
                    hist_slice = history_frames[-eff_h:]
                    if self.history_source == "fusion":
                        matched_hist = sum(
                            1 for fr in hist_slice
                            if bool(fr.get("fusion_valid", False)) and not bool(fr.get("fusion_is_fake", False))
                        )
                    else:
                        matched_hist = sum(1 for fr in hist_slice if bool(fr["is_matched"]))
                    hist_match_ratio = matched_hist / max(len(hist_slice), 1)
                    if hist_match_ratio >= min_history_match_ratio:
                        valid_history_lengths.append(eff_h)

                if not valid_history_lengths:
                    continue

                dt = future_frames[0]["timestamp"] - current_frame["timestamp"]
                if dt <= 0:
                    continue

                self.samples.append({
                    "history_frames": history_frames,
                    "current_frame": current_frame,
                    "future_frames": future_frames,
                    "valid_history_lengths": valid_history_lengths,
                    "history_min": hist_min,
                    "history_max": hist_max,
                    "rollout_min": roll_min,
                    "rollout_max": roll_max,
                    "delta_t": dt,
                    "instance_token": inst_token,
                    "category": category,
                    "tracking_category": track_name,
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
        valid_history_lengths = s.get("valid_history_lengths", [len(s["history_frames"])])
        rollout_min = int(s.get("rollout_min", self.min_rollout_steps))
        rollout_max = int(s.get("rollout_max", self.rollout_steps))
        if self.adaptive_windows:
            eff_history_len = random.choice(valid_history_lengths)
            eff_rollout_steps = random.randint(
                rollout_min,
                min(K, rollout_max, len(s["future_frames"])),
            )
        else:
            eff_history_len = min(T, int(s.get("history_max", T)), len(s["history_frames"]))
            eff_rollout_steps = min(K, rollout_max, len(s["future_frames"]))

        history = np.zeros((T, 12), dtype=np.float32)
        history_mask = np.zeros(T, dtype=np.bool_)
        history_match_mask = np.zeros(T, dtype=np.bool_)

        cur = s["current_frame"]
        cur_fusion_valid = bool(cur.get("fusion_valid", False))
        if self.init_state_source == "fusion" and cur_fusion_valid:
            ref_xyz = np.asarray(cur["fusion_global_xyz"], dtype=np.float32)
        else:
            ref_xyz = np.asarray(
                cur["det_global_xyz"] if cur["is_matched"] and cur["det_global_xyz"] is not None else cur["gt_global_xyz"],
                dtype=np.float32,
            )

        selected_history_frames = s["history_frames"][-eff_history_len:]
        n_frames = len(selected_history_frames)
        for t_idx, frame in enumerate(selected_history_frames):
            offset = T - n_frames + t_idx
            if self.history_source == "fusion":
                if bool(frame.get("fusion_valid", False)):
                    feat = np.asarray(frame["fusion_feature_12"], dtype=np.float32).copy()
                    feat[0] -= ref_xyz[0]
                    feat[1] -= ref_xyz[1]
                    history[offset] = feat
                    history_mask[offset] = True
                    history_match_mask[offset] = not bool(frame.get("fusion_is_fake", False))
            else:
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

        if self.init_state_source == "fusion" and cur_fusion_valid:
            fusion_feat = np.asarray(cur["fusion_feature_12"], dtype=np.float32)
            obs_current_pos = np.array([
                cur["fusion_global_xyz"][0], cur["fusion_global_xyz"][1], cur["fusion_global_xyz"][2],
                cur["fusion_velocity"][0], cur["fusion_velocity"][1], 0.0,
            ], dtype=np.float32)
            obs_current_siz = np.array(cur["fusion_lwh"], dtype=np.float32)
            obs_current_ori = np.array([cur["fusion_yaw"], fusion_feat[10]], dtype=np.float32)
        elif cur["is_matched"] and cur["det_global_xyz"] is not None:
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
        obs_future_score = np.zeros(K, dtype=np.float32)
        obs_future_match = np.zeros(K, dtype=np.bool_)
        future_mask = np.zeros(K, dtype=np.bool_)
        delta_ts_future = np.zeros(K, dtype=np.float32)

        prev_ts = cur["timestamp"]
        for k, frm in enumerate(s["future_frames"][:eff_rollout_steps]):
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
                obs_future_score[k] = float(frm.get("det_score", 0.0))
                obs_future_match[k] = True
            future_mask[k] = True
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
            "obs_future_score": torch.from_numpy(obs_future_score),
            "obs_future_match": torch.from_numpy(obs_future_match),
            "future_mask": torch.from_numpy(future_mask),
            "delta_ts_future": torch.from_numpy(delta_ts_future),
            "delta_t": torch.tensor(s["delta_t"], dtype=torch.float32),
            "instance_token": s["instance_token"],
            "category": s["category"],
            "current_det_score": torch.tensor(current_det_score, dtype=torch.float32),
            "history_match_ratio": torch.tensor(matched_hist / max(n_frames, 1), dtype=torch.float32),
            "current_range": torch.tensor(current_range, dtype=torch.float32),
            "current_speed": torch.tensor(current_speed, dtype=torch.float32),
            "effective_history_len": torch.tensor(eff_history_len, dtype=torch.int64),
            "effective_rollout_steps": torch.tensor(eff_rollout_steps, dtype=torch.int64),
            "is_detection_driven": torch.tensor(True, dtype=torch.bool),
            "history_source": self.history_source,
            "init_state_source": self.init_state_source,
        }


def detection_tracklet_collate_fn(batch: List[Dict]) -> Dict:
    result = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            result[key] = torch.stack([b[key] for b in batch], dim=0)
        else:
            result[key] = [b[key] for b in batch]
    return result
