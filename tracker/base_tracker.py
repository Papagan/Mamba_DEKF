# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track
# Base3DTracker: Main tracking loop with MambaDecoupledEKF integration
#
# Data flow per frame:
#   1. Extract track_history [B, T, 12] from active trajectories
#   2. Compute real delta_t from timestamps
#   3. MambaDecoupledEKF.predict_with_mamba → Q/R/embedding + predicted state
#   4. Write predicted state back into Trajectory bboxes
#   5. Uncertainty-aware association (Module C) using embeddings + P trace
#   6. MambaDecoupledEKF.update → fused state for matched tracklets
#   7. Write fused state back into Trajectory bboxes
# ------------------------------------------------------------------------

import os
import copy
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from tracker.matching import (
    match_trajs_and_dets_uncertainty_aware,
)
from tracker.trajectory import Trajectory
from tracker.bbox import BBox
from kalmanfilter.mamba_adaptive_kf import MambaDecoupledEKF


# ---- Mamba config defaults (can be overridden via cfg) ----
_DEFAULT_MAMBA_CFG = {
    "D_MODEL": 64,
    "D_STATE": 16,
    "D_CONV": 4,
    "EXPAND": 2,
    "N_MAMBA_LAYERS": 3,
    "EMBED_DIM": 32,
    "HISTORY_LEN": 5,       # T: temporal window for Mamba input
    "MAX_BATCH_SIZE": 256,   # pre-allocated batch size
    "MIN_DIAG_Q": 0.1,       # Q head Cholesky floor
    "MIN_DIAG_R": 0.1,       # R head Cholesky floor
    "NUM_CLASSES": 10,        # number of object categories for size embeddings
}


class Base3DTracker:
    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg
        self.current_frame_id: Optional[int] = None
        self.all_trajs: Dict[int, Trajectory] = {}
        self.all_dead_trajs: Dict[int, Trajectory] = {}
        self.id_seed: int = 0
        self.cache_size: int = 3
        self.track_id_counter: int = 0

        # ---- timestamp tracking for real delta_t ----
        self.last_timestamp: Optional[float] = None
        self.frame_rate: float = cfg.get("FRAME_RATE", 10)

        # ---- MambaDecoupledEKF (Module A + B) ----
        mamba_cfg = cfg.get("MAMBA", _DEFAULT_MAMBA_CFG)
        for k, v in _DEFAULT_MAMBA_CFG.items():
            mamba_cfg.setdefault(k, v)

        self.history_len: int = mamba_cfg["HISTORY_LEN"]
        self.max_batch: int = mamba_cfg["MAX_BATCH_SIZE"]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.mamba_ekf = MambaDecoupledEKF(
            batch_size=self.max_batch,
            d_model=mamba_cfg["D_MODEL"],
            d_state=mamba_cfg["D_STATE"],
            d_conv=mamba_cfg["D_CONV"],
            expand=mamba_cfg["EXPAND"],
            n_mamba_layers=mamba_cfg["N_MAMBA_LAYERS"],
            embed_dim=mamba_cfg["EMBED_DIM"],
            min_diag_q=mamba_cfg.get("MIN_DIAG_Q", 0.1),
            min_diag_r=mamba_cfg.get("MIN_DIAG_R", 0.1),
            num_classes=mamba_cfg.get("NUM_CLASSES", 10),
            device=self.device,
        ).to(self.device)

        # ---- Load trained weights if checkpoint path is provided ----
        ckpt_path = mamba_cfg.get("CHECKPOINT_PATH", None)
        if ckpt_path:
            if os.path.exists(ckpt_path):
                ckpt = torch.load(ckpt_path, map_location=self.device)
                # checkpoints saved by training/train.py contain "model_state_dict"
                state_dict = ckpt.get("model_state_dict", ckpt)
                # strip stale _tril_rows/_tril_cols keys from old checkpoints
                # (now lazily created in CholeskyHead.forward())
                stale_keys = [k for k in state_dict if "_tril_rows" in k or "_tril_cols" in k]
                for k in stale_keys:
                    del state_dict[k]
                missing, unexpected = self.mamba_ekf.mamba.load_state_dict(
                    state_dict, strict=False,
                )
                if missing:
                    print(f"[Base3DTracker] WARNING: missing keys in checkpoint: {missing}")
                if unexpected:
                    print(f"[Base3DTracker] WARNING: unexpected keys in checkpoint: {unexpected}")
                print(f"[Base3DTracker] Loaded Mamba weights from {ckpt_path}")
            else:
                print(f"[Base3DTracker] WARNING: CHECKPOINT_PATH={ckpt_path} not found. "
                      f"Running with RANDOM Mamba weights — results will be poor.")
        else:
            print("[Base3DTracker] WARNING: cfg['MAMBA']['CHECKPOINT_PATH'] not set. "
                  "Running with RANDOM Mamba weights — set CHECKPOINT_PATH to load trained weights.")

        self.mamba_ekf.eval()

        self.embed_dim: int = mamba_cfg["EMBED_DIM"]

        # ---- per-track KF state storage ----
        # Keyed by track_id → dict of tensors on device
        self.kf_states: Dict[int, Dict[str, torch.Tensor]] = {}

    # ==================================================================
    # History extraction: Trajectory bboxes → [B, T, 12] tensor
    # ==================================================================

    def _extract_track_history(
        self, trajs: List[Trajectory],
    ) -> torch.Tensor:
        """
        Extract joint historical state from each trajectory's bbox history,
        padded/truncated to self.history_len frames.

        # Per-frame feature (dim=12):
        #   [Δx, Δy, z, vx, vy, vz,  l, w, h,  theta, omega,  det_score]
        #    ├── pos state (6) ──┤  ├size(3)┤  ├ orient(2) ┤  ├qual┤
        #
        # Δx, Δy are relative to the latest frame's position to keep
        # feature magnitudes small and avoid gradient saturation in Mamba.
        # For missing velocity/acceleration, use zeros or finite-difference.
        # det_score = 0.0 for coasted (is_fake) frames to signal low-quality
        # observations to the Mamba soft-coupler.

        Args:
            trajs : list of N active Trajectory objects

        Returns:
            history : [N, T, 12] tensor on self.device
        """
        B = len(trajs)
        T = self.history_len
        history = torch.zeros(B, T, 12, device=self.device)

        for i, traj in enumerate(trajs):
            bboxes = traj.bboxes
            n_frames = min(len(bboxes), T)

            # take the last T frames (most recent at index T-1)
            recent = bboxes[-n_frames:]

            # reference x, y from the latest frame → relative coords
            # avoids large absolute global coordinates causing gradient saturation
            ref_xyz = recent[-1].global_xyz

            for t_idx, bbox in enumerate(recent):
                offset = T - n_frames + t_idx  # right-aligned padding

                xyz = bbox.global_xyz           # [x, y, z]
                vel = bbox.global_velocity      # [vx, vy] (2D)
                lwh = bbox.lwh                  # [l, w, h]
                yaw = bbox.global_yaw           # scalar

                # omega: finite-difference of yaw if possible
                omega = 0.0
                if t_idx > 0:
                    prev_bbox = recent[t_idx - 1]
                    dt_frames = bbox.frame_id - prev_bbox.frame_id
                    if dt_frames > 0:
                        dy = yaw - prev_bbox.global_yaw
                        # wrap to [-pi, pi]
                        dy = dy - 2.0 * np.pi * round(dy / (2.0 * np.pi))
                        omega = dy / (dt_frames / self.frame_rate)

                # vz = 0 (flat-world assumption consistent with PositionFilter)
                # x, y are relative to the latest frame in this tracklet
                # det_score: 0.0 for coasted/fake bboxes → Mamba sees low quality
                det_score = 0.0 if getattr(bbox, "is_fake", False) else bbox.det_score
                history[i, offset, :] = torch.tensor([
                    xyz[0] - ref_xyz[0], xyz[1] - ref_xyz[1], xyz[2],
                    vel[0], vel[1], 0.0,
                    lwh[0], lwh[1], lwh[2],
                    yaw, omega, det_score,
                ], device=self.device)

        return history

    # ==================================================================
    # KF state management per track
    # ==================================================================

    def _init_kf_state(self, track_id: int, bbox: BBox) -> None:
        """Initialise decoupled KF state for a new track from its first detection."""
        dev = self.device
        # Position: [x, y, z, vx, vy, vz]  (6D CV model)
        # Velocity initialised from detection; zero-velocity detectors
        # (e.g. CenterPoint) get a larger velocity variance so the KF
        # can infer speed from consecutive position observations.
        vel = bbox.global_velocity  # [vx, vy]
        pos_x = torch.tensor([
            bbox.global_xyz[0], bbox.global_xyz[1], bbox.global_xyz[2],
            vel[0], vel[1], 0.0,
        ], device=dev).reshape(1, 6, 1)
        pos_P = torch.eye(6, device=dev).unsqueeze(0)
        pos_P[:, 3, 3] = 10.0  # vx variance: high → K can learn vx from position
        pos_P[:, 4, 4] = 10.0  # vy variance: high → K can learn vy from position

        # Size: [l, w, h]
        siz_x = torch.tensor(bbox.lwh, device=dev).reshape(1, 3, 1)
        siz_P = torch.eye(3, device=dev).unsqueeze(0) * 0.1

        # Orientation: [theta, omega]
        ori_x = torch.tensor([bbox.global_yaw, 0.0], device=dev).reshape(1, 2, 1)
        ori_P = torch.eye(2, device=dev).unsqueeze(0) * 0.1

        self.kf_states[track_id] = {
            "pos_x": pos_x, "pos_P": pos_P,
            "siz_x": siz_x, "siz_P": siz_P,
            "ori_x": ori_x, "ori_P": ori_P,
        }

    def _batch_kf_states(
        self, track_ids: List[int],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Stack per-track KF states into batched tensors for the filters.

        Returns:
            pos_x [B,8,1], pos_P [B,8,8],
            siz_x [B,3,1], siz_P [B,3,3],
            ori_x [B,2,1], ori_P [B,2,2]
        """
        pos_x = torch.cat([self.kf_states[tid]["pos_x"] for tid in track_ids], dim=0)
        pos_P = torch.cat([self.kf_states[tid]["pos_P"] for tid in track_ids], dim=0)
        siz_x = torch.cat([self.kf_states[tid]["siz_x"] for tid in track_ids], dim=0)
        siz_P = torch.cat([self.kf_states[tid]["siz_P"] for tid in track_ids], dim=0)
        ori_x = torch.cat([self.kf_states[tid]["ori_x"] for tid in track_ids], dim=0)
        ori_P = torch.cat([self.kf_states[tid]["ori_P"] for tid in track_ids], dim=0)
        return pos_x, pos_P, siz_x, siz_P, ori_x, ori_P

    def _unbatch_kf_states(
        self,
        track_ids: List[int],
        pos_x: torch.Tensor, pos_P: torch.Tensor,
        siz_x: torch.Tensor, siz_P: torch.Tensor,
        ori_x: torch.Tensor, ori_P: torch.Tensor,
    ) -> None:
        """Write batched KF results back into per-track storage."""
        for i, tid in enumerate(track_ids):
            self.kf_states[tid] = {
                "pos_x": pos_x[i:i+1], "pos_P": pos_P[i:i+1],
                "siz_x": siz_x[i:i+1], "siz_P": siz_P[i:i+1],
                "ori_x": ori_x[i:i+1], "ori_P": ori_P[i:i+1],
            }

    def _write_predicted_state_to_bbox(
        self, traj: Trajectory,
        pos_x: torch.Tensor, siz_x: torch.Tensor, ori_x: torch.Tensor,
    ) -> None:
        """
        Write KF-predicted state back into the trajectory's latest bbox.

        A sanity check discards predictions that jump >10 m from the
        previous frame — unstable Q matrices during early training can
        cause the position filter to produce kilometre-scale displacements.

        Args:
            pos_x : [1, 6, 1] — [x, y, z, vx, vy, vz]
            siz_x : [1, 3, 1] — [l, w, h]
            ori_x : [1, 2, 1] — [theta, omega]
        """
        bbox = traj.bboxes[-1]
        px = pos_x.squeeze().cpu().numpy()   # [6]
        sx = siz_x.squeeze().cpu().numpy()   # [3]
        ox = ori_x.squeeze().cpu().numpy()   # [2]

        # ---- Sanity check: reject anomalous position jumps (>10 m) ----
        if len(traj.bboxes) > 1:
            prev_bbox = traj.bboxes[-2]
            if (hasattr(prev_bbox, "global_xyz_lwh_yaw_predict")
                    and prev_bbox.global_xyz_lwh_yaw_predict is not None):
                prev_pos = prev_bbox.global_xyz_lwh_yaw_predict[:2]
            else:
                prev_pos = prev_bbox.global_xyz[:2]
            displacement = np.sqrt(
                (px[0] - prev_pos[0]) ** 2 + (px[1] - prev_pos[1]) ** 2
            )
            if displacement > 10.0:
                return  # discard prediction, keep previous-frame coordinates

        predict_xyz = [px[0], px[1], px[2]]
        predict_lwh = [sx[0], sx[1], sx[2]]
        predict_yaw = float(ox[0])

        bbox.global_xyz_lwh_yaw_predict = predict_xyz + predict_lwh + [predict_yaw]
        bbox.global_yaw_fusion = predict_yaw
        bbox.lwh_fusion = predict_lwh

    def _write_updated_state_to_bbox(
        self, bbox: BBox,
        pos_x: torch.Tensor, siz_x: torch.Tensor, ori_x: torch.Tensor,
    ) -> None:
        """
        Write KF-updated (fused) state back into a bbox after matching.

        Args:
            pos_x : [1, 8, 1]
            siz_x : [1, 3, 1]
            ori_x : [1, 2, 1]
        """
        px = pos_x.squeeze().cpu().numpy()   # [6]
        sx = siz_x.squeeze().cpu().numpy()   # [3]
        ox = ori_x.squeeze().cpu().numpy()   # [2]

        bbox.global_velocity_fusion = [px[3], px[4]]
        bbox.global_yaw_fusion = float(ox[0])
        bbox.lwh_fusion = [sx[0], sx[1], sx[2]]
        bbox.global_xyz_lwh_yaw_fusion = np.array([
            px[0], px[1], px[2],
            sx[0], sx[1], sx[2],
            ox[0],
        ])

    # ==================================================================
    # Helpers
    # ==================================================================

    def get_trajectory_bbox(self, all_trajs: dict) -> List[Trajectory]:
        track_ids = sorted(all_trajs.keys())
        return [all_trajs[i] for i in track_ids]

    def _compute_delta_t(self, timestamp: Optional[float]) -> float:
        """
        Compute real delta_t in seconds from timestamps.
        Falls back to 1/FRAME_RATE if timestamps are unavailable.
        """
        if timestamp is not None and self.last_timestamp is not None:
            dt = timestamp - self.last_timestamp
            if dt > 0:
                return dt
        return 1.0 / self.frame_rate

    # ==================================================================
    # Core: predict all active trajectories via MambaDecoupledEKF
    # ==================================================================

    def predict_before_associate(
        self, trajs: List[Trajectory], delta_t: float,
    ) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
        """
        Batch-predict all active trajectories through TemporalMamba + DecoupledKF.

        Returns:
            mamba_out      : dict with Q/R/embedding (or None if no trajs)
            trk_embeddings : [N_trk, embed_dim] numpy array (or None)
        """
        if len(trajs) == 0:
            return None, None

        track_ids = [t.track_id for t in trajs]
        B = len(trajs)

        # ---- 1. Extract history [B, T, 12] ----
        history = self._extract_track_history(trajs)

        # ---- 2. Load per-track KF states into batch ----
        pos_x, pos_P, siz_x, siz_P, ori_x, ori_P = self._batch_kf_states(track_ids)

        # ---- 3. Reinitialise the KF module with current batch ----
        self.mamba_ekf.kf.B = B
        self.mamba_ekf.kf.pos_filter.B = B
        self.mamba_ekf.kf.siz_filter.B = B
        self.mamba_ekf.kf.ori_filter.B = B
        self.mamba_ekf.kf.init_states(pos_x, pos_P, siz_x, siz_P, ori_x, ori_P)

        # ---- 4. Mamba predict: history → Q/R/embedding, then KF predict ----
        class_ids = torch.tensor([t.category_num for t in trajs],
                                 dtype=torch.long, device=self.device)
        with torch.no_grad():
            mamba_out, px, pP, sx, sP, ox, oP = self.mamba_ekf.predict_with_mamba(
                history, delta_t, class_ids=class_ids,
            )

        # ---- 5. Write predicted states back to per-track storage ----
        self._unbatch_kf_states(track_ids, px, pP, sx, sP, ox, oP)

        # ---- 6. Write predicted state into each trajectory's bbox ----
        for i, traj in enumerate(trajs):
            self._write_predicted_state_to_bbox(
                traj, px[i:i+1], sx[i:i+1], ox[i:i+1],
            )
            # also trigger the (now no-op) predict on trajectory for interface compat
            traj.predict()

        # ---- 7. Extract embeddings for Module C ----
        trk_embeddings = mamba_out["embedding"].cpu().numpy()  # [B, embed_dim]

        return mamba_out, trk_embeddings

    # ==================================================================
    # Core: update matched tracks via DecoupledKF
    # ==================================================================

    def _update_matched_tracks(
        self,
        matched_track_ids: List[int],
        matched_bboxes: List[BBox],
        mamba_out: Dict,
        traj_index_map: Dict[int, int],
    ) -> None:
        """
        Run KF update for all matched (track, detection) pairs in batch.

        Args:
            matched_track_ids : track IDs that were matched
            matched_bboxes    : corresponding detection BBox objects
            mamba_out         : Mamba output containing R matrices
            traj_index_map    : map from track_id → index in the predict batch
        """
        if len(matched_track_ids) == 0:
            return

        B_m = len(matched_track_ids)
        dev = self.device

        # ---- build observation tensors from matched detections ----
        z_pos_list, z_siz_list, z_ori_list = [], [], []
        for bbox in matched_bboxes:
            z_pos_list.append([bbox.global_xyz[0], bbox.global_xyz[1], bbox.global_xyz[2]])
            z_siz_list.append(bbox.lwh)
            z_ori_list.append([bbox.global_yaw])

        z_pos = torch.tensor(z_pos_list, device=dev, dtype=torch.float32).unsqueeze(-1)  # [B_m, 3, 1]
        z_siz = torch.tensor(z_siz_list, device=dev, dtype=torch.float32).unsqueeze(-1)  # [B_m, 3, 1]
        z_ori = torch.tensor(z_ori_list, device=dev, dtype=torch.float32).unsqueeze(-1)  # [B_m, 1, 1]

        # ---- extract R matrices for matched subset from mamba_out ----
        batch_indices = [traj_index_map[tid] for tid in matched_track_ids]
        R_pos = mamba_out["R_pos"][batch_indices]  # [B_m, 3, 3]
        R_siz = mamba_out["R_siz"][batch_indices]  # [B_m, 3, 3]
        R_ori = mamba_out["R_ori"][batch_indices]  # [B_m, 1, 1]

        # ---- load predicted KF states for matched tracks ----
        pos_x, pos_P, siz_x, siz_P, ori_x, ori_P = self._batch_kf_states(matched_track_ids)

        # ---- set filter batch size and states ----
        self.mamba_ekf.kf.B = B_m
        self.mamba_ekf.kf.pos_filter.B = B_m
        self.mamba_ekf.kf.siz_filter.B = B_m
        self.mamba_ekf.kf.ori_filter.B = B_m
        self.mamba_ekf.kf.init_states(pos_x, pos_P, siz_x, siz_P, ori_x, ori_P)

        # ---- KF update ----
        with torch.no_grad():
            ux, uP, usx, usP, uox, uoP = self.mamba_ekf.kf.update(
                z_pos, z_siz, z_ori,
                R_pos=R_pos, R_siz=R_siz, R_ori=R_ori,
            )

        # ---- write back ----
        self._unbatch_kf_states(matched_track_ids, ux, uP, usx, usP, uox, uoP)

        for i, (tid, bbox) in enumerate(zip(matched_track_ids, matched_bboxes)):
            self._write_updated_state_to_bbox(bbox, ux[i:i+1], usx[i:i+1], uox[i:i+1])

    # ==================================================================
    # Main entry: track a single frame
    # ==================================================================

    def track_single_frame(self, frame_info) -> Dict:
        """
        Process one frame: predict → associate → update → manage lifecycle.

        Data flow:
            1. Compute delta_t from real timestamps
            2. Batch-predict all active tracks via MambaDecoupledEKF
            3. Uncertainty-aware association (Module C)
            4. Batch-update matched tracks via DecoupledKF
            5. Handle unmatched tracks (coast) and new detections (birth)
            6. Lifecycle management (death)

        Args:
            frame_info : Frame object with .bboxes, .frame_id, .timestamp, etc.

        Returns:
            output_trajs : dict of track_id → BBox for confirmed tracks
        """
        # ---- delta_t from real timestamps ----
        delta_t = self._compute_delta_t(frame_info.timestamp)
        self.last_timestamp = frame_info.timestamp

        # ---- get active trajectories ----
        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt = len(trajs)
        dets_cnt = len(frame_info.bboxes)

        # ---- predict all active tracks (Module A + B) ----
        mamba_out, trk_embeddings = self.predict_before_associate(trajs, delta_t)

        # ---- build index maps ----
        # traj_index_map: track_id → position in the trajs list (= batch index)
        traj_index_map = {t.track_id: i for i, t in enumerate(trajs)}

        # ---- extract uncertainty for Module C ----
        trk_pos_P = None
        trk_ori_P = None
        if trajs_cnt > 0:
            trk_pos_P = [
                self.kf_states[t.track_id]["pos_P"].squeeze(0).cpu().numpy()
                for t in trajs
            ]
            trk_ori_P = [
                self.kf_states[t.track_id]["ori_P"].squeeze(0).cpu().numpy()
                for t in trajs
            ]

        # ---- detection split: ByteTrack two-stage matching paradigm ----
        # High-score dets (≥0.4) → strict matching + can birth new tracks
        # Low-score dets  (0.1–0.4) → relaxed matching only (no birth)
        # Score < 0.1 → discarded
        high_det_indices: List[int] = []   # original indices in frame_info.bboxes
        low_det_indices: List[int] = []
        high_dets: List[BBox] = []
        low_dets: List[BBox] = []
        for i, det in enumerate(frame_info.bboxes):
            score = det.det_score
            if score >= 0.4:
                high_det_indices.append(i)
                high_dets.append(det)
            elif score >= 0.1:
                low_det_indices.append(i)
                low_dets.append(det)

        # ---- detection embeddings (zeros for new detections with no history) ----
        det_embeddings_all = np.zeros((dets_cnt, self.embed_dim), dtype=np.float32) \
            if dets_cnt > 0 else None

        # ---- bookkeeping ----
        matched_track_ids: List[int] = []
        matched_bboxes: List[BBox] = []
        matched_traj_full_set: set = set()  # indices into full trajs list

        # ================================================================
        # Stage 1: all active trajs  vs  high-score dets  (strict)
        # ================================================================
        if trajs_cnt > 0 and len(high_dets) > 0:
            det_emb_high = det_embeddings_all[high_det_indices] \
                if det_embeddings_all is not None else None
            match_res_1, _ = match_trajs_and_dets_uncertainty_aware(
                trajs, high_dets, self.cfg,
                trk_embeddings=trk_embeddings,
                det_embeddings=det_emb_high,
                trk_pos_P=trk_pos_P,
                trk_ori_P=trk_ori_P,
            )
        else:
            match_res_1 = np.empty((0, 2), dtype=int)

        matched_high_sub_set: set = set()
        for row in (match_res_1 if len(match_res_1) > 0 else []):
            traj_idx = int(row[0])
            high_sub_idx = int(row[1])
            det_global_idx = high_det_indices[high_sub_idx]
            track_id = trajs[traj_idx].track_id
            det_bbox = frame_info.bboxes[det_global_idx]
            self.all_trajs[track_id].update(det_bbox, 0.0)
            matched_track_ids.append(track_id)
            matched_bboxes.append(det_bbox)
            matched_traj_full_set.add(traj_idx)
            matched_high_sub_set.add(high_sub_idx)

        # ================================================================
        # Stage 2: unmatched trajs  vs  low-score dets  (relaxed threshold)
        # ================================================================
        unmatched_traj_full_indices = [
            i for i in range(trajs_cnt) if i not in matched_traj_full_set
        ]

        if len(unmatched_traj_full_indices) > 0 and len(low_dets) > 0:
            unmatched_trajs = [trajs[i] for i in unmatched_traj_full_indices]

            # slice per-trajectory data to unmatched subset
            um_trk_emb = trk_embeddings[unmatched_traj_full_indices] \
                if trk_embeddings is not None else None
            um_trk_pos_P = [trk_pos_P[i] for i in unmatched_traj_full_indices] \
                if trk_pos_P is not None else None
            um_trk_ori_P = [trk_ori_P[i] for i in unmatched_traj_full_indices] \
                if trk_ori_P is not None else None
            det_emb_low = det_embeddings_all[low_det_indices] \
                if det_embeddings_all is not None else None

            # relaxed cost threshold: 2× wider gate for low-quality detections
            cfg_relaxed = copy.deepcopy(self.cfg)
            orig_thre = cfg_relaxed["THRESHOLD"]["BEV"]["COST_THRE"]
            cfg_relaxed["THRESHOLD"]["BEV"]["COST_THRE"] = [
                t * 2.0 for t in orig_thre
            ]

            match_res_2, _ = match_trajs_and_dets_uncertainty_aware(
                unmatched_trajs, low_dets, cfg_relaxed,
                trk_embeddings=um_trk_emb,
                det_embeddings=det_emb_low,
                trk_pos_P=um_trk_pos_P,
                trk_ori_P=um_trk_ori_P,
            )
        else:
            match_res_2 = np.empty((0, 2), dtype=int)

        for row in (match_res_2 if len(match_res_2) > 0 else []):
            um_sub_idx = int(row[0])    # index into unmatched_traj_full_indices
            low_sub_idx = int(row[1])   # index into low_dets
            traj_full_idx = unmatched_traj_full_indices[um_sub_idx]
            det_global_idx = low_det_indices[low_sub_idx]
            track_id = trajs[traj_full_idx].track_id
            det_bbox = frame_info.bboxes[det_global_idx]
            # ---- low-score isolation: prevent noisy det_score from
            #      degrading trajectory quality & confirmation status ----
            det_bbox.det_score = max(det_bbox.det_score, 0.5)
            self.all_trajs[track_id].update(det_bbox, 0.0)
            matched_track_ids.append(track_id)
            matched_bboxes.append(det_bbox)
            matched_traj_full_set.add(traj_full_idx)

        # ---- coast: trajectories that missed both stages ----
        for i in range(trajs_cnt):
            if i not in matched_traj_full_set:
                track_id = trajs[i].track_id
                self.all_trajs[track_id].unmatch_update(frame_info.frame_id)

        # ---- batch KF update for all matched tracks (both stages) ----
        if mamba_out is not None and len(matched_track_ids) > 0:
            self._update_matched_tracks(
                matched_track_ids, matched_bboxes, mamba_out, traj_index_map,
            )

        # ---- Strict Birth: ONLY from unmatched HIGH-score dets ----
        # LOW-SCORE DETS (0.1–0.4) ARE NEVER ALLOWED TO CREATE NEW TRAJECTORIES.
        # They only serve as rescue candidates for existing tracks in Stage 2.
        # This is the single most effective FP suppression mechanism.
        for sub_idx in range(len(high_dets)):
            if sub_idx not in matched_high_sub_set:
                det_bbox = high_dets[sub_idx]
                new_traj = Trajectory(
                    track_id=self.track_id_counter,
                    init_bbox=det_bbox,
                    cfg=self.cfg,
                )
                self.all_trajs[self.track_id_counter] = new_traj
                self._init_kf_state(self.track_id_counter, det_bbox)
                self.track_id_counter += 1

        # ---- death: remove dead tracks ----
        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]
                # clean up KF state
                self.kf_states.pop(track_id, None)

        output_trajs = self.get_output_trajs(frame_info.frame_id)
        return output_trajs

    # ==================================================================
    # Output & post-processing
    # ==================================================================

    def get_output_trajs(self, frame_id: int) -> Dict:
        output_trajs = {}
        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 1 or frame_id < 3:
                bbox = self.all_trajs[track_id].bboxes[-1]
                if bbox.det_score == self.all_trajs[track_id]._is_filter_predict_box:
                    continue
                # Quality-gated scoring: mirrors filtering() in trajectory.py.
                # Only high-confidence real detections (score >= 0.4) determine
                # the output score. Stage 2 ByteTrack rescue dets (0.1-0.4) are
                # excluded — they exist to maintain track continuity, not to
                # represent object existence probability.
                # This runs in both ONLINE and GLOBAL tracking modes.
                traj = self.all_trajs[track_id]
                real_scores = [
                    b.det_score for b in traj.bboxes
                    if not getattr(b, "is_fake", False)
                ]
                quality_scores = [s for s in real_scores if s >= 0.4]
                if quality_scores:
                    bbox.det_score = sum(quality_scores) / len(quality_scores)
                elif real_scores:
                    bbox.det_score = max(real_scores)
                output_trajs[track_id] = bbox
                self.all_trajs[track_id].is_output = True
        return output_trajs

    def post_processing(self) -> Dict:
        trajs = {}
        for track_id in self.all_dead_trajs.keys():
            traj = self.all_dead_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        for track_id in self.all_trajs.keys():
            traj = self.all_trajs[track_id]
            traj.filtering()
            trajs[track_id] = traj
        return trajs
