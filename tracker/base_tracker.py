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
    match_trajs_and_dets,
    match_trajs_and_dets_uncertainty_aware,
)
from tracker.compat_utils import (
    apply_dirty_track_suppressor_to_output,
    allow_single_stage_birth_under_mode,
    classify_single_stage_birth,
    extract_bbox_history_fields,
    normalize_tracker_compat_mode,
    select_output_tracking_score,
    sync_bbox_fields_from_state,
    use_mctrack_exact_matched_update,
    use_mctrack_exact_unmatch_update,
)
from tracker.dirty_suppressor_audit import DirtySuppressorAuditAccumulator
from tracker.mctrack_motion import MCTrackOriginalPoseMotion
from tracker.trajectory import Trajectory
from tracker.bbox import BBox
from kalmanfilter.checkpoint_compat import adapt_num_class_state_dict
from kalmanfilter.mamba_adaptive_kf import MambaDecoupledEKF, build_noise_audit_samples
from kalmanfilter.bounded_residual import infer_state_bucket
from kalmanfilter.state_residual import apply_bounded_state_residuals
from kalmanfilter.noise_audit import NoiseAuditAccumulator
from utils.debug_log import emit_debug_line
from utils.utils import norm_realative_radian


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
    "MIN_DIAG_SIZ": 0.05,   # size noise floor (prevents NLL→-∞)
    "MIN_KAPPA": 0.1,        # kappa floor (prevents R_ori→∞)
    "NUM_CLASSES": 10,        # number of object categories for size embeddings
}


def _build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


def _noise_audit_enabled(cfg):
    return bool(_build_noise_audit_cfg(cfg).get("ENABLED", False))


def parse_residual_history_window_cfg(
    history_len: int,
    cfg: Optional[Dict],
) -> Tuple[Dict[str, int], Dict[str, Dict[str, int]]]:
    cfg = cfg or {}
    data_cfg = (cfg.get("DATA", {}) or {})
    residual_window_cfg = (cfg.get("RESIDUAL_HISTORY", {}) or {})

    merged_window_cfg = {
        "MIN_HISTORY_LEN": data_cfg.get("MIN_HISTORY_LEN", min(history_len, 4)),
        "MAX_HISTORY_LEN": data_cfg.get("MAX_HISTORY_LEN", history_len),
    }
    merged_window_cfg["CLASS_WINDOW"] = dict(data_cfg.get("CLASS_WINDOW", {}) or {})

    if residual_window_cfg:
        if "MIN_HISTORY_LEN" in residual_window_cfg:
            merged_window_cfg["MIN_HISTORY_LEN"] = residual_window_cfg["MIN_HISTORY_LEN"]
        if "MAX_HISTORY_LEN" in residual_window_cfg:
            merged_window_cfg["MAX_HISTORY_LEN"] = residual_window_cfg["MAX_HISTORY_LEN"]
        override_class_window = residual_window_cfg.get("CLASS_WINDOW", {}) or {}
        if override_class_window:
            merged_window_cfg["CLASS_WINDOW"].update(override_class_window)

    default_min_history = int(
        merged_window_cfg.get("MIN_HISTORY_LEN", min(history_len, 4))
    )
    default_max_history = int(
        merged_window_cfg.get("MAX_HISTORY_LEN", history_len)
    )
    default_window_cfg = {
        "MIN_HISTORY_LEN": max(1, min(default_min_history, history_len)),
        "MAX_HISTORY_LEN": max(1, min(default_max_history, history_len)),
    }

    runtime_window_cfg: Dict[str, Dict[str, int]] = {}
    for class_name, class_cfg in (merged_window_cfg.get("CLASS_WINDOW", {}) or {}).items():
        if not isinstance(class_cfg, dict):
            continue
        min_history = int(
            class_cfg.get("MIN_HISTORY_LEN", default_window_cfg["MIN_HISTORY_LEN"])
        )
        max_history = int(
            class_cfg.get("MAX_HISTORY_LEN", default_window_cfg["MAX_HISTORY_LEN"])
        )
        runtime_window_cfg[str(class_name).strip().lower()] = {
            "MIN_HISTORY_LEN": max(1, min(min_history, history_len)),
            "MAX_HISTORY_LEN": max(1, min(max_history, history_len)),
        }
    return default_window_cfg, runtime_window_cfg


def build_runtime_contract_warnings(
    runtime_contract,
    tracker_compat_mode,
    filter_mode,
    current_cost_mode,
    current_history_source=None,
    current_init_state_source=None,
    current_closure_cfg=None,
):
    def _normalize_state_list(values):
        return [str(value).strip().lower() for value in (values or [])]

    def _normalize_class_state_map(values):
        out = {}
        for key, states in (values or {}).items():
            class_key = str(int(key)) if str(key).strip().lstrip("-").isdigit() else str(key).strip()
            out[class_key] = _normalize_state_list(states)
        return out

    tracker_compat_mode = (
        "default" if tracker_compat_mode is None else str(tracker_compat_mode).strip().lower()
    )
    filter_mode = "mamba" if filter_mode is None else str(filter_mode).strip().lower()
    current_cost_mode = (
        "unknown" if current_cost_mode is None else str(current_cost_mode).strip().lower()
    )
    current_history_source = (
        None if current_history_source is None else str(current_history_source).strip().lower()
    )
    current_init_state_source = (
        None if current_init_state_source is None else str(current_init_state_source).strip().lower()
    )

    expected_compat = str(
        runtime_contract.get("tracker_compat_mode", tracker_compat_mode)
    ).strip().lower()
    expected_cost_mode = str(
        runtime_contract.get("expected_bev_cost_mode", current_cost_mode)
    ).strip().lower()
    expected_history_source = runtime_contract.get("history_source", None)
    if expected_history_source is not None:
        expected_history_source = str(expected_history_source).strip().lower()
    expected_init_state_source = runtime_contract.get("init_state_source", None)
    if expected_init_state_source is not None:
        expected_init_state_source = str(expected_init_state_source).strip().lower()
    expected_filter_mode = str(
        runtime_contract.get("filter_mode", filter_mode)
    ).strip().lower()
    current_closure_cfg = current_closure_cfg or {}

    warnings = []
    if expected_compat != tracker_compat_mode:
        warnings.append(
            "[Base3DTracker] WARNING: checkpoint runtime_contract expects "
            f"TRACKER_COMPAT_MODE={expected_compat}, but current config uses "
            f"{tracker_compat_mode}. Results may degrade due to history semantics mismatch."
        )
    if expected_cost_mode != current_cost_mode:
        warnings.append(
            "[Base3DTracker] WARNING: checkpoint runtime_contract expects "
            f"BEV COST_MODE={expected_cost_mode}, but current config uses "
            f"{current_cost_mode}. Results may degrade due to mismatched "
            "training/inference matching behavior."
        )
    if (
        expected_history_source is not None
        and current_history_source is not None
        and expected_history_source != current_history_source
    ):
        warnings.append(
            "[Base3DTracker] WARNING: checkpoint runtime_contract expects "
            f"history_source={expected_history_source}, but current runtime uses "
            f"{current_history_source}. Results may degrade due to mismatched "
            "history feature semantics."
        )
    if (
        expected_init_state_source is not None
        and current_init_state_source is not None
        and expected_init_state_source != current_init_state_source
    ):
        warnings.append(
            "[Base3DTracker] WARNING: checkpoint runtime_contract expects "
            f"init_state_source={expected_init_state_source}, but current runtime uses "
            f"{current_init_state_source}. Results may degrade due to mismatched "
            "KF initialization semantics."
        )
    if expected_filter_mode != filter_mode:
        warnings.append(
            "[Base3DTracker] WARNING: checkpoint runtime_contract expects "
            f"FILTER_MODE={expected_filter_mode}, but current config uses "
            f"{filter_mode}. Results may degrade due to mismatched "
            "Mamba/KF inference behavior."
        )
    if "closure_use_conditional_prior" in runtime_contract:
        expected = bool(runtime_contract.get("closure_use_conditional_prior"))
        current = bool(current_closure_cfg.get("USE_CONDITIONAL_PRIOR", True))
        if expected != current:
            warnings.append(
                "[Base3DTracker] WARNING: checkpoint runtime_contract "
                f"closure_use_conditional_prior={expected}, but current config uses {current}."
            )
    if "closure_force_prior_states" in runtime_contract:
        expected = _normalize_state_list(runtime_contract.get("closure_force_prior_states"))
        current = _normalize_state_list(current_closure_cfg.get("FORCE_PRIOR_STATES", ["matched"]))
        if expected != current:
            warnings.append(
                "[Base3DTracker] WARNING: checkpoint runtime_contract "
                f"closure_force_prior_states={expected}, but current config uses {current}."
            )
    if "closure_active_class_states" in runtime_contract:
        expected = _normalize_class_state_map(runtime_contract.get("closure_active_class_states"))
        current = _normalize_class_state_map(current_closure_cfg.get("ACTIVE_CLASS_STATES", {}) or {})
        if expected != current:
            warnings.append(
                "[Base3DTracker] WARNING: checkpoint runtime_contract "
                f"closure_active_class_states={expected}, but current config uses {current}."
            )
    return warnings


def run_mctrack_exact_unmatch_kf_step(
    *,
    pos_x: torch.Tensor,
    pos_P: torch.Tensor,
    siz_x: torch.Tensor,
    siz_P: torch.Tensor,
    ori_x: torch.Tensor,
    ori_P: torch.Tensor,
    R_pos: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply the MCTrack-style fake update on already-predicted KF states.

    In original MCTrack, unmatch_update() calls predict() to get a predicted
    observation, then calls update(predicted_observation). Because the original
    KF object still holds the last matched state internally, that update becomes
    a zero-innovation fake update on the one-step predicted state.

    In this project, the outer tracker has already advanced kf_states to the
    predicted state before association. Therefore the numerically equivalent
    operation is:
      - keep size/orientation at the predicted state
      - run a zero-innovation update on the predicted position state only
    """
    kf = MambaDecoupledEKF(batch_size=pos_x.shape[0], device=device)
    kf.kf.B = pos_x.shape[0]
    kf.kf.pos_filter.B = pos_x.shape[0]
    kf.kf.siz_filter.B = pos_x.shape[0]
    kf.kf.ori_filter.B = pos_x.shape[0]
    kf.kf.init_states(pos_x, pos_P, siz_x, siz_P, ori_x, ori_P)

    z_pos = torch.cat([pos_x[:, 0:3, :], pos_x[:, 3:5, :]], dim=1)
    upd_pos_x, upd_pos_P = kf.kf.pos_filter.update(z_pos, R_pos)
    return upd_pos_x, upd_pos_P, siz_x, siz_P, ori_x, ori_P


def run_mctrack_exact_matched_kf_step(
    *,
    pos_x: torch.Tensor,
    pos_P: torch.Tensor,
    siz_x: torch.Tensor,
    siz_P: torch.Tensor,
    ori_x: torch.Tensor,
    ori_P: torch.Tensor,
    z_pos: torch.Tensor,
    z_siz: torch.Tensor,
    z_ori: torch.Tensor,
    R_pos: torch.Tensor,
    R_siz: torch.Tensor,
    R_ori: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Apply a stricter MCTrack-style matched update on already-predicted states.

    Original MCTrack pose/size filters do not consume the same state/measurement
    dimensions as the current decoupled KF:
      - pose filter does not directly observe z
      - size filter does not directly observe h

    To approximate that behavior while keeping the current filter implementation,
    we run the standard batched update and then restore the dimensions that are
    not directly constrained in MCTrack to their predicted-state values.
    """
    kf = MambaDecoupledEKF(batch_size=pos_x.shape[0], device=device)
    kf.kf.B = pos_x.shape[0]
    kf.kf.pos_filter.B = pos_x.shape[0]
    kf.kf.siz_filter.B = pos_x.shape[0]
    kf.kf.ori_filter.B = pos_x.shape[0]
    kf.kf.init_states(pos_x, pos_P, siz_x, siz_P, ori_x, ori_P)

    upd_pos_x, upd_pos_P, upd_siz_x, upd_siz_P, upd_ori_x, upd_ori_P = kf.kf.update(
        z_pos, z_siz, z_ori,
        R_pos=R_pos, R_siz=R_siz, R_ori=R_ori,
    )

    upd_pos_x = upd_pos_x.clone()
    upd_pos_P = upd_pos_P.clone()
    upd_siz_x = upd_siz_x.clone()
    upd_siz_P = upd_siz_P.clone()

    # MCTrack pose filter is 2D/velocity-centric. Restore z and vz to the
    # predicted state, and keep their covariance block from the predicted P.
    for idx in (2, 5):
        upd_pos_x[:, idx, :] = pos_x[:, idx, :]
        upd_pos_P[:, idx, :] = pos_P[:, idx, :]
        upd_pos_P[:, :, idx] = pos_P[:, :, idx]

    # MCTrack size filter only consumes l/w. Restore h to the predicted state.
    upd_siz_x[:, 2, :] = siz_x[:, 2, :]
    upd_siz_P[:, 2, :] = siz_P[:, 2, :]
    upd_siz_P[:, :, 2] = siz_P[:, :, 2]

    return upd_pos_x, upd_pos_P, upd_siz_x, upd_siz_P, upd_ori_x, upd_ori_P


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

        # ---- Filter Mode (mamba, pure_dekf, fusion) ----
        self.filter_mode = str(cfg.get("FILTER_MODE", "mamba")).lower()
        self.tracker_compat_mode = normalize_tracker_compat_mode(
            cfg.get("TRACKER_COMPAT_MODE", "default")
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ---- Peek at checkpoint to decide GRU vs Mamba backbone ----
        # If the checkpoint was trained without mamba-ssm (uses fallback_gru),
        # we must force GRU mode even when mamba-ssm is installed. Otherwise
        # Mamba layers would be randomly initialized and the trained GRU weights
        # would be skipped (strict=False silently ignores the mismatch).
        ckpt_path = mamba_cfg.get("CHECKPOINT_PATH", None)
        force_gru = False
        if ckpt_path and os.path.exists(ckpt_path):
            ckpt_peek = torch.load(ckpt_path, map_location="cpu")
            state_peek = ckpt_peek.get("model_state_dict", ckpt_peek)
            force_gru = any("fallback_gru" in k for k in state_peek.keys())
            if force_gru:
                print("[Base3DTracker] Detected GRU-trained checkpoint → forcing GRU fallback")
            del ckpt_peek, state_peek

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
            min_diag_siz=mamba_cfg.get("MIN_DIAG_SIZ", 0.05),
            min_kappa=mamba_cfg.get("MIN_KAPPA", 0.1),
            device=self.device,
            base_noise_cfg=cfg.get("DEKF_BASE_NOISE", None),
            force_gru=force_gru,
        ).to(self.device)
        self.mamba_input_dim: int = int(getattr(self.mamba_ekf.mamba, "INPUT_DIM", 12))
        self.default_window_cfg, self.runtime_window_cfg = parse_residual_history_window_cfg(
            history_len=self.history_len,
            cfg=cfg,
        )
        self._dirty_pos_trace_priors = self._build_dirty_pos_trace_priors()
        self.original_motion = MCTrackOriginalPoseMotion(
            cfg=cfg,
            frame_rate=self.frame_rate,
            filter_mode=self.filter_mode,
        )
        if self.original_motion.active():
            print(
                "[Base3DTracker] MCTRACK_ORIGINAL_MOTION enabled "
                f"for FILTER_MODE={self.filter_mode}",
                flush=True,
            )

        # ---- Load trained weights if checkpoint path is provided ----
        if ckpt_path:
            if os.path.exists(ckpt_path):
                resolved_ckpt_path = os.path.abspath(ckpt_path)
                ckpt = torch.load(ckpt_path, map_location=self.device)
                runtime_contract = ckpt.get("runtime_contract", None)
                # checkpoints saved by training/train.py contain "model_state_dict"
                state_dict = ckpt.get("model_state_dict", ckpt)
                head_bank_key_count = sum(
                    1 for key in state_dict.keys() if str(key).startswith("head_bank.")
                )
                print(
                    f"[Base3DTracker] CHECKPOINT_PATH resolved to {resolved_ckpt_path} "
                    f"(head_bank_keys={head_bank_key_count})"
                )
                if runtime_contract:
                    print(
                        "[Base3DTracker] Checkpoint runtime_contract: "
                        f"{runtime_contract}"
                    )
                    expected_backbone_type = str(runtime_contract.get("backbone_type", "")).strip().lower()
                    if expected_backbone_type and expected_backbone_type != ("fallback_gru" if force_gru else "mamba_ssm"):
                        print(
                            "[Base3DTracker] WARNING: checkpoint runtime_contract backbone_type="
                            f"{expected_backbone_type} but runtime backbone="
                            f"{'fallback_gru' if force_gru else 'mamba_ssm'}"
                        )
                # strip stale _tril_rows/_tril_cols keys from old checkpoints
                # (now lazily created in CholeskyHead.forward())
                stale_keys = [k for k in state_dict if "_tril_rows" in k or "_tril_cols" in k]
                for k in stale_keys:
                    del state_dict[k]
                state_dict, adapted_keys = adapt_num_class_state_dict(
                    state_dict,
                    self.mamba_ekf.mamba.state_dict(),
                )
                missing, unexpected = self.mamba_ekf.mamba.load_state_dict(
                    state_dict, strict=False,
                )
                if adapted_keys:
                    print(
                        "[Base3DTracker] Adapted class-count-dependent checkpoint tensors: "
                        + ", ".join(adapted_keys)
                    )
                if missing:
                    print(f"[Base3DTracker] WARNING: missing keys in checkpoint: {missing}")
                if unexpected:
                    print(f"[Base3DTracker] WARNING: unexpected keys in checkpoint: {unexpected}")
                print(f"[Base3DTracker] Loaded Mamba weights from {ckpt_path}")
                if runtime_contract:
                    current_cost_mode = str(
                        self.cfg.get("THRESHOLD", {}).get("BEV", {}).get("COST_MODE", "unknown")
                    ).strip().lower()
                    for warning in build_runtime_contract_warnings(
                        runtime_contract=runtime_contract,
                        tracker_compat_mode=self.tracker_compat_mode,
                        filter_mode=self.filter_mode,
                        current_cost_mode=current_cost_mode,
                        current_history_source=self.cfg.get("HISTORY_SOURCE", None),
                        current_init_state_source=self.cfg.get("INIT_STATE_SOURCE", None),
                        current_closure_cfg=(self.cfg.get("DEKF_BASE_NOISE", {}) or {}).get("MAMBA_CLOSURE", {}),
                    ):
                        print(warning)
            else:
                print(f"[Base3DTracker] WARNING: CHECKPOINT_PATH={ckpt_path} not found. "
                      f"Running with RANDOM Mamba weights — results will be poor.")
        else:
            print("[Base3DTracker] WARNING: cfg['MAMBA']['CHECKPOINT_PATH'] not set. "
                  "Running with RANDOM Mamba weights — set CHECKPOINT_PATH to load trained weights.")

        self.mamba_ekf.eval()

        self.embed_dim: int = mamba_cfg["EMBED_DIM"]
        print(
            f"[Base3DTracker] FILTER_MODE={self.filter_mode} "
            f"BEV_COST_MODE={self.cfg.get('THRESHOLD', {}).get('BEV', {}).get('COST_MODE', 'unknown')}",
            flush=True,
        )

        # ---- per-track KF state storage ----
        # Keyed by track_id → dict of tensors on device
        self.kf_states: Dict[int, Dict[str, torch.Tensor]] = {}
        self.noise_audit_cfg = _build_noise_audit_cfg(cfg)
        self.noise_audit = (
            NoiseAuditAccumulator() if _noise_audit_enabled(cfg) else None
        )
        self._noise_audit_pending: Optional[Dict] = None
        self.dirty_suppressor_cfg = (cfg.get("DIRTY_TRACK_SUPPRESSOR", {}) or {})
        self.dirty_suppressor_audit_cfg = (
            (self.dirty_suppressor_cfg.get("AUDIT", {}) or {})
        )
        self.dirty_suppressor_audit = (
            DirtySuppressorAuditAccumulator()
            if bool(self.dirty_suppressor_audit_cfg.get("ENABLED", False))
            else None
        )

    def _normalize_delta_t(self, dt_raw: float) -> float:
        """
        Normalize raw timestamp delta to seconds with unit auto-detection.

        NuScenes sample timestamps are in microseconds. If used directly,
        delta_t becomes ~5e5 instead of ~0.5, causing KF covariance blow-up
        and uncertainty-aware association collapse.
        """
        if dt_raw <= 0 or not np.isfinite(dt_raw):
            return 1.0 / self.frame_rate

        # Heuristic unit detection:
        #   >1e3   : likely microseconds (nuscenes/waymo style)
        #   >10    : likely milliseconds
        #   else   : already seconds
        if dt_raw > 1e3:
            dt_sec = dt_raw / 1e6
        elif dt_raw > 10:
            dt_sec = dt_raw / 1e3
        else:
            dt_sec = dt_raw

        # Safety clamp: avoid extreme dt spikes from broken timestamps.
        dt_sec = float(np.clip(dt_sec, 1e-3, 5.0))
        return dt_sec

    def _record_noise_audit_sample(
        self,
        *,
        mode: str,
        class_id: int,
        class_name: str,
        state: str,
        history_len: Optional[int],
        families: Dict[str, float],
        prior_families: Dict[str, Optional[float]],
    ) -> None:
        if self.noise_audit is None:
            return
        self.noise_audit.add_sample(
            split="infer",
            mode=mode,
            class_id=class_id,
            class_name=class_name,
            state=state,
            history_len=history_len,
            families=families,
            prior_families=prior_families,
        )

    def _stage_noise_audit_samples(
        self,
        *,
        track_ids: List[int],
        trajs: List[Trajectory],
        class_ids: torch.Tensor,
        history_mask: torch.Tensor,
        mamba_out: Dict,
    ) -> None:
        if self.noise_audit is None:
            self._noise_audit_pending = None
            return

        audit_values = mamba_out.get("noise_audit_values")
        audit_priors = mamba_out.get("noise_audit_priors")
        if audit_values is None or audit_priors is None:
            self._noise_audit_pending = None
            return

        def _to_cpu_list(values):
            if values is None:
                return None
            return values.detach().cpu().tolist()

        self._noise_audit_pending = {
            "mode": self.filter_mode,
            "track_ids": list(track_ids),
            "class_ids": class_ids.detach().cpu().tolist(),
            "class_names": [traj.bboxes[-1].category for traj in trajs],
            "history_lens": history_mask.to(dtype=torch.int64).sum(dim=1).cpu().tolist(),
            "values": {name: _to_cpu_list(values) for name, values in audit_values.items()},
            "priors": {name: _to_cpu_list(values) for name, values in audit_priors.items()},
        }

    def _flush_noise_audit_samples(self) -> None:
        if self.noise_audit is None or not self._noise_audit_pending:
            return

        pending = self._noise_audit_pending
        matched_mask = []
        for track_id in pending["track_ids"]:
            traj = self.all_trajs.get(track_id)
            if traj is None:
                traj = self.all_dead_trajs.get(track_id)
            bbox = traj.bboxes[-1] if traj is not None and traj.bboxes else None
            matched_mask.append(bool(bbox is not None and not getattr(bbox, "is_fake", False)))

        samples = build_noise_audit_samples(
            mode=pending["mode"],
            traj_labels=pending["class_ids"],
            matched_mask=matched_mask,
            history_lens=pending["history_lens"],
            q_pos=pending["values"]["q_pos"],
            r_pos=pending["values"]["r_pos"],
            r_siz=pending["values"]["r_siz"],
            r_ori=pending["values"]["r_ori"],
            prior_q_pos=pending["priors"]["q_pos"],
            prior_r_pos=pending["priors"]["r_pos"],
            prior_r_siz=pending["priors"]["r_siz"],
            prior_r_ori=pending["priors"]["r_ori"],
        )

        for class_name, sample in zip(pending["class_names"], samples):
            self._record_noise_audit_sample(
                mode=sample["mode"],
                class_id=sample["class_id"],
                class_name=class_name,
                state=sample["state"],
                history_len=sample["history_len"],
                families=sample["families"],
                prior_families=sample["prior_families"],
            )

        self._noise_audit_pending = None

    def dump_noise_audit_if_needed(self) -> None:
        if self.noise_audit is None:
            return
        output_path = self.noise_audit_cfg.get("INFER_OUTPUT_PATH", "debug/infer_noise_audit.json")
        try:
            self.noise_audit.write_json(output_path)
        except Exception as exc:
            if self.noise_audit_cfg.get("STRICT", False):
                raise
            print(f"[Base3DTracker] WARNING: failed to write noise audit to {output_path}: {exc}")

    def export_noise_audit_state(self) -> Optional[Dict]:
        if self.noise_audit is None:
            return None
        return self.noise_audit.export_state()

    def _record_dirty_suppressor_audit_sample(
        self,
        *,
        class_id: int,
        class_name: str,
        profile_name: str | None,
        penalty: float,
        hard_reject: bool,
        triggered_reasons: list,
        features: Dict,
    ) -> None:
        if self.dirty_suppressor_audit is None:
            return
        self.dirty_suppressor_audit.add_sample(
            class_id=class_id,
            class_name=class_name,
            profile_name=profile_name,
            penalty=penalty,
            hard_reject=hard_reject,
            triggered_reasons=triggered_reasons,
            features=features,
        )

    def dump_dirty_suppressor_audit_if_needed(self) -> None:
        if self.dirty_suppressor_audit is None:
            return
        output_path = self.dirty_suppressor_audit_cfg.get(
            "INFER_OUTPUT_PATH",
            "debug/dirty_track_suppressor_audit.json",
        )
        try:
            self.dirty_suppressor_audit.write_json(output_path)
        except Exception as exc:
            if self.dirty_suppressor_audit_cfg.get("STRICT", False):
                raise
            print(
                "[Base3DTracker] WARNING: failed to write dirty suppressor audit "
                f"to {output_path}: {exc}"
            )

    def export_dirty_suppressor_audit_state(self) -> Optional[Dict]:
        if self.dirty_suppressor_audit is None:
            return None
        return self.dirty_suppressor_audit.export_state()

    # ==================================================================
    # History extraction: Trajectory bboxes → [B, T, 12] tensor
    # ==================================================================

    def _extract_track_history(
        self, trajs: List[Trajectory],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
            history            : [N, T, 12] tensor on self.device
            history_mask       : [N, T] bool tensor
            history_match_mask : [N, T] bool tensor
        """
        B = len(trajs)
        T = self.history_len
        history = torch.zeros(B, T, 12, device=self.device)
        history_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        history_match_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)

        for i, traj in enumerate(trajs):
            bboxes = traj.bboxes
            n_frames = min(len(bboxes), T)

            # take the last T frames (most recent at index T-1)
            recent = bboxes[-n_frames:]

            # reference x, y from the latest frame → relative coords
            # avoids large absolute global coordinates causing gradient saturation
            ref_xyz, _, _, _ = extract_bbox_history_fields(
                recent[-1], self.tracker_compat_mode
            )

            for t_idx, bbox in enumerate(recent):
                offset = T - n_frames + t_idx  # right-aligned padding

                xyz, vel, lwh, yaw = extract_bbox_history_fields(
                    bbox, self.tracker_compat_mode
                )

                # omega: finite-difference of yaw if possible
                omega = 0.0
                if t_idx > 0:
                    prev_bbox = recent[t_idx - 1]
                    _, _, _, prev_yaw = extract_bbox_history_fields(
                        prev_bbox, self.tracker_compat_mode
                    )
                    dt = self._normalize_delta_t(bbox.timestamp - prev_bbox.timestamp)
                    if dt > 0:
                        dy = yaw - prev_yaw
                        # wrap to [-pi, pi]
                        dy = dy - 2.0 * np.pi * round(dy / (2.0 * np.pi))
                        omega = dy / dt
                    else:
                        dt_frames = bbox.frame_id - prev_bbox.frame_id
                        if dt_frames > 0:
                            dy = yaw - prev_yaw
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
                history_mask[i, offset] = True
                history_match_mask[i, offset] = not getattr(bbox, "is_fake", False)

        return history, history_mask, history_match_mask

    def _class_window_cfg(self, class_name: str) -> Dict[str, int]:
        normalized = "" if class_name is None else str(class_name).strip().lower()
        cfg = dict(self.default_window_cfg)
        cfg.update(self.runtime_window_cfg.get(normalized, {}))
        min_history = max(1, int(cfg.get("MIN_HISTORY_LEN", 1)))
        max_history = max(min_history, int(cfg.get("MAX_HISTORY_LEN", self.history_len)))
        return {
            "MIN_HISTORY_LEN": min_history,
            "MAX_HISTORY_LEN": min(max_history, self.history_len),
        }

    def _resolve_effective_history_len(self, traj, valid_history_len: int) -> int:
        if valid_history_len <= 0:
            return 0

        class_name = getattr(traj.bboxes[-1], "category", None) if getattr(traj, "bboxes", None) else None
        window_cfg = self._class_window_cfg(class_name)
        effective_len = min(
            int(valid_history_len),
            int(window_cfg["MAX_HISTORY_LEN"]),
            int(self.history_len),
        )
        unmatch_length = int(getattr(traj, "unmatch_length", 0) or 0)
        if unmatch_length > 0:
            effective_len = max(int(window_cfg["MIN_HISTORY_LEN"]), effective_len - 2)
        if unmatch_length > 1:
            effective_len = min(effective_len, 3)
        effective_len = min(effective_len, int(valid_history_len))
        return max(1, effective_len)

    def _encode_residual_token(
        self,
        traj: Trajectory,
        residual_entry: Dict,
        *,
        age_index: int,
        effective_len: int,
    ) -> torch.Tensor:
        del traj  # reserved for future branch-specific token features
        pos_residual = [float(value) for value in residual_entry.get("pos_residual", [0.0] * 5)]
        siz_residual = [float(value) for value in residual_entry.get("siz_residual", [0.0] * 3)]
        ori_residual = float(residual_entry.get("ori_residual", 0.0))
        det_score = float(residual_entry.get("det_score", 0.0))
        match_flag = 1.0 if bool(residual_entry.get("is_matched", False)) else 0.0
        age_feature = 0.0
        if effective_len > 1:
            age_feature = float(age_index) / float(effective_len - 1)
        return torch.tensor(
            pos_residual + siz_residual + [ori_residual, det_score, match_flag, age_feature],
            device=self.device,
        )

    def _extract_residual_token_history(
        self, trajs: List[Trajectory],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B = len(trajs)
        T = self.history_len
        D = self.mamba_input_dim
        history = torch.zeros(B, T, D, device=self.device)
        history_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
        history_match_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)

        for i, traj in enumerate(trajs):
            residual_history = list(getattr(traj, "residual_history", []) or [])
            effective_len = self._resolve_effective_history_len(traj, len(residual_history))
            if effective_len <= 0:
                continue

            recent = residual_history[-effective_len:]
            start = T - len(recent)
            for local_idx, residual_entry in enumerate(recent):
                offset = start + local_idx
                history[i, offset, :] = self._encode_residual_token(
                    traj,
                    residual_entry,
                    age_index=local_idx,
                    effective_len=len(recent),
                )
                history_mask[i, offset] = True
                history_match_mask[i, offset] = bool(residual_entry.get("is_matched", False))

        return history, history_mask, history_match_mask

    def _build_dirty_pos_trace_priors(self) -> Dict[int, float]:
        fallback_trace = 23.0
        class_ids = torch.arange(7, device=self.device, dtype=torch.long)
        try:
            q_pos, _, _, _, _, _ = self.mamba_ekf._get_base_noise(
                bsize=class_ids.shape[0],
                dtype=torch.float32,
                class_ids=class_ids,
            )
            traces = q_pos.diagonal(dim1=-2, dim2=-1).sum(-1).detach().cpu().tolist()
            return {
                int(class_id): max(float(trace), 1e-6)
                for class_id, trace in zip(class_ids.tolist(), traces)
            }
        except Exception:
            return {int(class_id): fallback_trace for class_id in class_ids.tolist()}

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
        self.original_motion.init_track(track_id, bbox, self.cfg["CATEGORY_MAP_TO_NUMBER"][bbox.category])

    def _batch_kf_states(
        self, track_ids: List[int],
    ) -> Tuple[torch.Tensor, ...]:
        """
        Stack per-track KF states into batched tensors for the filters.

        Returns:
            pos_x [B,6,1], pos_P [B,6,6],
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
        dt = getattr(self, "_cur_delta_t", 1.0 / self.frame_rate)

        def _safe_vec2(vec):
            arr = np.asarray(vec, dtype=np.float32).reshape(-1)
            if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
                return np.array([0.0, 0.0], dtype=np.float32)
            return arr[:2]

        def _choose_stable_velocity() -> np.ndarray:
            kf_vel = _safe_vec2([px[3], px[4]])
            diff_vel = _safe_vec2(getattr(bbox, "global_velocity_diff", [0.0, 0.0]))
            curve_vel = _safe_vec2(getattr(bbox, "global_velocity_curve", [0.0, 0.0]))
            det_vel = _safe_vec2(getattr(bbox, "global_velocity", [0.0, 0.0]))

            geom_vel = curve_vel if np.linalg.norm(curve_vel) > 1e-3 else diff_vel
            if np.linalg.norm(geom_vel) <= 1e-3:
                geom_vel = det_vel

            cls_name = traj.bboxes[-1].category
            agile_cls = {"pedestrian", "bicycle", "motorcycle"}
            disagreement = float(np.linalg.norm(kf_vel - geom_vel))
            kf_speed = float(np.linalg.norm(kf_vel))
            geom_speed = float(np.linalg.norm(geom_vel))

            if cls_name in agile_cls and traj.track_length >= 3:
                if geom_speed > 1e-3 and (disagreement > 2.5 or kf_speed > 25.0):
                    return geom_vel
                if geom_speed > 1e-3:
                    return 0.5 * kf_vel + 0.5 * geom_vel

            if geom_speed > 1e-3 and (disagreement > 4.0 or kf_speed > 35.0):
                return geom_vel
            return kf_vel

        # ---- Sanity check: anomalous position jumps (>10 m) ----
        # When the KF produces an extreme jump (divergent Q), fall back to
        # the detection position rather than silently retaining stale bbox
        # fields.  This keeps bbox ↔ KF-state consistent: the next frame's
        # history extraction reads realistic positions and Mamba sees a
        # coherent tracklet.
        kf_jump_ok = True
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
            kf_jump_ok = displacement <= 10.0

        stable_vel = _choose_stable_velocity()
        prev_bbox = traj.bboxes[-2] if len(traj.bboxes) > 1 else bbox
        prev_pos_src = (
            prev_bbox.global_xyz_lwh_yaw_fusion[:3]
            if hasattr(prev_bbox, "global_xyz_lwh_yaw_fusion")
            and prev_bbox.global_xyz_lwh_yaw_fusion is not None
            else prev_bbox.global_xyz
        )
        motion_xyz = [
            float(prev_pos_src[0] + stable_vel[0] * dt),
            float(prev_pos_src[1] + stable_vel[1] * dt),
            float(px[2]),
        ]

        if self.tracker_compat_mode == "mctrack":
            last_fused = getattr(bbox, "global_xyz_lwh_yaw_fusion", None)
            if last_fused is None:
                last_fused = bbox.global_xyz_lwh_yaw
            predict_xyz = [float(px[0]), float(px[1]), float(last_fused[2])]
            predict_lwh = [float(last_fused[3]), float(last_fused[4]), float(last_fused[5])]
            predict_yaw = float(last_fused[6])
            predict_state = predict_xyz + predict_lwh + [predict_yaw]
            sync_bbox_fields_from_state(
                bbox,
                predict_state,
                update_fusion=False,
                update_predict=True,
            )
            bbox.global_velocity_fusion = [float(stable_vel[0]), float(stable_vel[1])]
            bbox.global_yaw_fusion = float(ox[0])
            bbox.lwh_fusion = predict_lwh
            return

        # Use KF prediction for mature stable tracks; otherwise fall back to a
        # MCTrack-style CV extrapolation from recent geometric velocity.
        if traj.track_length <= 1:
            predict_xyz = bbox.global_xyz
            vel_x, vel_y = float(stable_vel[0]), float(stable_vel[1])
        elif not kf_jump_ok:
            predict_xyz = motion_xyz
            vel_x, vel_y = float(stable_vel[0]), float(stable_vel[1])
        else:
            predict_xyz = [px[0], px[1], px[2]]
            vel_x, vel_y = float(stable_vel[0]), float(stable_vel[1])
        predict_lwh = [sx[0], sx[1], sx[2]]
        predict_yaw = float(ox[0])

        bbox.global_xyz_lwh_yaw_predict = predict_xyz + predict_lwh + [predict_yaw]
        bbox.global_velocity_fusion = [vel_x, vel_y]
        bbox.global_yaw_fusion = predict_yaw
        bbox.lwh_fusion = predict_lwh

    def _write_updated_state_to_bbox(
        self, bbox: BBox,
        pos_x: torch.Tensor, siz_x: torch.Tensor, ori_x: torch.Tensor,
    ) -> None:
        """
        Write KF-updated (fused) state back into a bbox after matching.

        Args:
            pos_x : [1, 6, 1]
            siz_x : [1, 3, 1]
            ori_x : [1, 2, 1]
        """
        px = pos_x.squeeze().cpu().numpy()   # [6]
        sx = siz_x.squeeze().cpu().numpy()   # [3]
        ox = ori_x.squeeze().cpu().numpy()   # [2]

        bbox.global_velocity_fusion = [px[3], px[4]]
        bbox.global_yaw_fusion = float(ox[0])
        bbox.lwh_fusion = [sx[0], sx[1], sx[2]]
        if self.tracker_compat_mode == "mctrack":
            sync_bbox_fields_from_state(
                bbox,
                [
                    px[0], px[1], bbox.global_xyz_lwh_yaw[2],
                    bbox.global_xyz_lwh_yaw[3], bbox.global_xyz_lwh_yaw[4],
                    bbox.global_xyz_lwh_yaw[5], bbox.global_xyz_lwh_yaw[6],
                ],
                update_fusion=True,
                update_predict=False,
            )
        else:
            bbox.global_xyz_lwh_yaw_fusion = np.array([
                px[0], px[1], px[2],
                sx[0], sx[1], sx[2],
                ox[0],
            ])

    def _apply_mctrack_exact_unmatch_update(
        self,
        track_ids: List[int],
        mamba_out: Optional[Dict],
        traj_index_map: Dict[int, int],
    ) -> None:
        if not track_ids or mamba_out is None:
            return

        eligible_ids = [
            tid for tid in track_ids
            if tid in self.all_trajs
            and use_mctrack_exact_unmatch_update(self.cfg, self.all_trajs[tid].category_num)
            and tid in traj_index_map
        ]
        if not eligible_ids:
            return

        pos_x, pos_P, siz_x, siz_P, ori_x, ori_P = self._batch_kf_states(eligible_ids)
        batch_indices = [traj_index_map[tid] for tid in eligible_ids]
        R_pos = mamba_out["R_pos"][batch_indices]

        upd_pos_x, upd_pos_P, upd_siz_x, upd_siz_P, upd_ori_x, upd_ori_P = run_mctrack_exact_unmatch_kf_step(
            pos_x=pos_x,
            pos_P=pos_P,
            siz_x=siz_x,
            siz_P=siz_P,
            ori_x=ori_x,
            ori_P=ori_P,
            R_pos=R_pos,
            device=self.device,
        )

        self._unbatch_kf_states(
            eligible_ids,
            upd_pos_x, upd_pos_P,
            upd_siz_x, upd_siz_P,
            upd_ori_x, upd_ori_P,
        )

        for i, tid in enumerate(eligible_ids):
            traj = self.all_trajs[tid]
            bbox = traj.bboxes[-1]
            pred_state = list(getattr(bbox, "global_xyz_lwh_yaw_predict", bbox.global_xyz_lwh_yaw))
            fake_state = [
                float(upd_pos_x[i, 0, 0]),
                float(upd_pos_x[i, 1, 0]),
                float(pred_state[2]),
                float(pred_state[3]),
                float(pred_state[4]),
                float(pred_state[5]),
                float(pred_state[6]),
            ]
            bbox.global_xyz_lwh_yaw_fake_update = fake_state
            bbox.global_velocity_fake_update = [
                float(upd_pos_x[i, 3, 0]),
                float(upd_pos_x[i, 4, 0]),
            ]

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
            dt_raw = timestamp - self.last_timestamp
            if dt_raw > 0:
                return self._normalize_delta_t(dt_raw)
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
            self._noise_audit_pending = None
            return None, None

        track_ids = [t.track_id for t in trajs]
        B = len(trajs)

        # ---- 1. Extract histories [B, T, 12] ----
        if self.filter_mode == "mamba_multihead_closure":
            history, history_mask, history_match_mask = self._extract_residual_token_history(trajs)
            prior_track_history, prior_history_mask, prior_history_match_mask = self._extract_track_history(trajs)
        else:
            history, history_mask, history_match_mask = self._extract_track_history(trajs)
            prior_track_history = history
            prior_history_mask = history_mask
            prior_history_match_mask = history_match_mask

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
        current_range = torch.tensor(
            [float(np.linalg.norm(np.asarray(t.bboxes[-1].global_xyz[:2], dtype=np.float32))) for t in trajs],
            dtype=torch.float32,
            device=self.device,
        )
        detection_driven_mask = torch.tensor(
            [
                bool(getattr(traj, "unmatch_length", 0) == 0)
                and not bool(getattr(traj.bboxes[-1], "is_fake", False))
                for traj in trajs
            ],
            dtype=torch.bool,
            device=self.device,
        )
        state_buckets = [
            infer_state_bucket(getattr(traj, "unmatch_length", 0))
            for traj in trajs
        ]
        with torch.no_grad():
            mamba_out, px, pP, sx, sP, ox, oP = self.mamba_ekf.predict_with_mamba(
                history,
                delta_t,
                class_ids=class_ids,
                mode=self.filter_mode,
                current_range=current_range,
                detection_driven_mask=detection_driven_mask,
                history_mask=history_mask,
                history_match_mask=history_match_mask,
                prior_track_history=prior_track_history,
                prior_history_mask=prior_history_mask,
                prior_history_match_mask=prior_history_match_mask,
                state_buckets=state_buckets,
            )

            px, sx, ox, state_residual_mask = self._apply_state_residual_to_prediction(
                px, sx, ox, mamba_out, class_ids, state_buckets
            )
            mamba_out["state_residual_active_mask"] = state_residual_mask

            px = self._apply_original_motion_predict_to_batch(track_ids, px, delta_t)

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
        self._stage_noise_audit_samples(
            track_ids=track_ids,
            trajs=trajs,
            class_ids=class_ids,
            history_mask=history_mask,
            mamba_out=mamba_out,
        )

        return mamba_out, trk_embeddings

    def _apply_original_motion_predict_to_batch(
        self,
        track_ids: List[int],
        pos_x: torch.Tensor,
        delta_t: float,
    ) -> torch.Tensor:
        if not self.original_motion.active():
            return pos_x

        out = pos_x.clone()
        for i, track_id in enumerate(track_ids):
            state = self.original_motion.predict(track_id, delta_t)
            if state is None:
                continue
            filt = self.original_motion.filters.get(int(track_id))
            mode = self.original_motion._mode_from_filter(filt) if filt is not None else "CV"
            out[i:i+1] = self.original_motion.apply_vector_to_pos_tensor(
                state,
                mode,
                out[i:i+1],
            )
        return out

    def _apply_original_motion_update_to_pos_batch(
        self,
        track_ids: List[int],
        bboxes: List[BBox],
        pos_x: torch.Tensor,
    ) -> torch.Tensor:
        if not self.original_motion.active():
            return pos_x

        out = pos_x.clone()
        dt = getattr(self, "_cur_delta_t", 1.0 / self.frame_rate)
        for i, (track_id, bbox) in enumerate(zip(track_ids, bboxes)):
            state = self.original_motion.update(track_id, bbox, dt)
            if state is None:
                continue
            filt = self.original_motion.filters.get(int(track_id))
            mode = self.original_motion._mode_from_filter(filt) if filt is not None else "CV"
            out[i:i+1] = self.original_motion.apply_vector_to_pos_tensor(
                state,
                mode,
                out[i:i+1],
            )
        return out

    def _apply_original_motion_fake_update(self, track_id: int) -> None:
        if not self.original_motion.active() or track_id not in self.all_trajs:
            return

        dt = getattr(self, "_cur_delta_t", 1.0 / self.frame_rate)
        state = self.original_motion.fake_update(track_id, dt)
        if state is None:
            return

        traj = self.all_trajs[track_id]
        bbox = traj.bboxes[-1]
        pos_x = self.kf_states[track_id]["pos_x"]
        filt = self.original_motion.filters.get(int(track_id))
        mode = self.original_motion._mode_from_filter(filt) if filt is not None else "CV"
        updated_pos = self.original_motion.apply_vector_to_pos_tensor(state, mode, pos_x)
        self.kf_states[track_id]["pos_x"] = updated_pos

        pred_state = list(getattr(bbox, "global_xyz_lwh_yaw_predict", bbox.global_xyz_lwh_yaw))
        pred_state[0] = float(updated_pos[0, 0, 0])
        pred_state[1] = float(updated_pos[0, 1, 0])
        bbox.global_xyz_lwh_yaw_fake_update = pred_state
        bbox.global_velocity_fake_update = [
            float(updated_pos[0, 3, 0]),
            float(updated_pos[0, 4, 0]),
        ]

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

        dev = self.device

        exact_pairs = []
        standard_pairs = []
        for tid, bbox in zip(matched_track_ids, matched_bboxes):
            traj = self.all_trajs.get(tid)
            if traj is not None and use_mctrack_exact_matched_update(self.cfg, traj.category_num):
                exact_pairs.append((tid, bbox))
            else:
                standard_pairs.append((tid, bbox))

        def _run_subset(pairs, *, exact_mode: bool) -> None:
            if not pairs:
                return

            sub_track_ids = [tid for tid, _ in pairs]
            sub_bboxes = [bbox for _, bbox in pairs]
            B_m = len(sub_track_ids)

            z_pos_list, z_siz_list, z_ori_list = [], [], []
            for bbox in sub_bboxes:
                vel = bbox.global_velocity
                z_pos_list.append([
                    bbox.global_xyz[0], bbox.global_xyz[1], bbox.global_xyz[2],
                    vel[0], vel[1],
                ])
                z_siz_list.append(bbox.lwh)
                z_ori_list.append([bbox.global_yaw])

            z_pos = torch.tensor(z_pos_list, device=dev, dtype=torch.float32).unsqueeze(-1)
            z_siz = torch.tensor(z_siz_list, device=dev, dtype=torch.float32).unsqueeze(-1)
            z_ori = torch.tensor(z_ori_list, device=dev, dtype=torch.float32).unsqueeze(-1)

            batch_indices = [traj_index_map[tid] for tid in sub_track_ids]
            R_pos = mamba_out["R_pos"][batch_indices]
            R_siz = mamba_out["R_siz"][batch_indices]
            R_ori = mamba_out["R_ori"][batch_indices]

            pos_x, pos_P, siz_x, siz_P, ori_x, ori_P = self._batch_kf_states(sub_track_ids)

            if exact_mode:
                with torch.no_grad():
                    ux, uP, usx, usP, uox, uoP = run_mctrack_exact_matched_kf_step(
                        pos_x=pos_x,
                        pos_P=pos_P,
                        siz_x=siz_x,
                        siz_P=siz_P,
                        ori_x=ori_x,
                        ori_P=ori_P,
                        z_pos=z_pos,
                        z_siz=z_siz,
                        z_ori=z_ori,
                        R_pos=R_pos,
                        R_siz=R_siz,
                        R_ori=R_ori,
                        device=dev,
                    )
            else:
                self.mamba_ekf.kf.B = B_m
                self.mamba_ekf.kf.pos_filter.B = B_m
                self.mamba_ekf.kf.siz_filter.B = B_m
                self.mamba_ekf.kf.ori_filter.B = B_m
                self.mamba_ekf.kf.init_states(pos_x, pos_P, siz_x, siz_P, ori_x, ori_P)
                with torch.no_grad():
                    ux, uP, usx, usP, uox, uoP = self.mamba_ekf.kf.update(
                        z_pos, z_siz, z_ori,
                        R_pos=R_pos, R_siz=R_siz, R_ori=R_ori,
                    )

            ux = self._apply_original_motion_update_to_pos_batch(
                sub_track_ids,
                sub_bboxes,
                ux,
            )

            self._unbatch_kf_states(sub_track_ids, ux, uP, usx, usP, uox, uoP)

            for i, bbox in enumerate(sub_bboxes):
                self._write_updated_state_to_bbox(bbox, ux[i:i+1], usx[i:i+1], uox[i:i+1])

        _run_subset(standard_pairs, exact_mode=False)
        _run_subset(exact_pairs, exact_mode=True)

    def _state_residual_cfg(self) -> Dict:
        return (
            (self.cfg.get("DEKF_BASE_NOISE", {}) or {}).get("MAMBA_STATE_RESIDUAL", {})
            or {}
        )

    def _mamba_association_prior_enabled(self) -> bool:
        return bool((self.cfg.get("MAMBA_ASSOCIATION_PRIOR", {}) or {}).get("ENABLED", False))

    def _apply_state_residual_to_prediction(
        self,
        pos_x: torch.Tensor,
        siz_x: torch.Tensor,
        ori_x: torch.Tensor,
        mamba_out: Dict,
        class_ids: torch.Tensor,
        state_buckets: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return apply_bounded_state_residuals(
            pos_x,
            siz_x,
            ori_x,
            mamba_out.get("delta_pos"),
            class_ids=class_ids,
            state_buckets=state_buckets,
            cfg=self._state_residual_cfg(),
        )

    def _summarize_small_class_pending_reasons(self) -> Dict[str, Dict[str, int]]:
        summary: Dict[str, Dict[str, int]] = {}
        target_classes = {"bicycle", "motorcycle"}
        for traj in self.all_trajs.values():
            cls_name = traj.bboxes[-1].category
            if cls_name not in target_classes or traj.status_flag != 0:
                continue
            info = summary.setdefault(
                cls_name,
                {
                    "pending": 0,
                    "need_more_hits": 0,
                    "low_det_score": 0,
                    "low_match_score": 0,
                    "has_real_det": 0,
                    "no_real_det_yet": 0,
                },
            )
            info["pending"] += 1
            real_bboxes = [b for b in traj.bboxes if not getattr(b, "is_fake", False)]
            if real_bboxes:
                info["has_real_det"] += 1
            else:
                info["no_real_det_yet"] += 1
            if traj.track_length < traj._confirmed_track_length:
                info["need_more_hits"] += 1
            last_real_score = max((b.det_score for b in real_bboxes), default=0.0)
            if last_real_score <= traj._confirmed_det_score:
                info["low_det_score"] += 1
            last_match_score = traj.matched_scores[-1] if traj.matched_scores else 0.0
            if last_match_score > traj._confirmed_match_score:
                info["low_match_score"] += 1
        return summary

    def _print_small_class_debug(
        self,
        frame_id: int,
        det_counts: Dict[str, int],
        matched_counts: Dict[str, int],
        birth_counts: Dict[str, int],
        coast_counts: Dict[str, int],
        dead_counts: Dict[str, int],
        output_trajs: Dict,
    ) -> None:
        target_classes = ["bicycle", "motorcycle"]
        alive_by_cls = {
            cls_name: {"init": 0, "confirmed": 0, "obscured": 0}
            for cls_name in target_classes
        }
        for traj in self.all_trajs.values():
            cls_name = traj.bboxes[-1].category
            if cls_name not in alive_by_cls:
                continue
            if traj.status_flag == 0:
                alive_by_cls[cls_name]["init"] += 1
            elif traj.status_flag == 1:
                alive_by_cls[cls_name]["confirmed"] += 1
            elif traj.status_flag == 2:
                alive_by_cls[cls_name]["obscured"] += 1

        output_counts = {cls_name: 0 for cls_name in target_classes}
        for bbox in output_trajs.values():
            if bbox.category in output_counts:
                output_counts[bbox.category] += 1

        pending_summary = self._summarize_small_class_pending_reasons()
        for cls_name in target_classes:
            alive_info = alive_by_cls[cls_name]
            pending_info = pending_summary.get(cls_name, {})
            emit_debug_line(
                f"[TRK-SMALL] frame={frame_id} cls={cls_name} "
                f"dets={det_counts.get(cls_name, 0)} matched={matched_counts.get(cls_name, 0)} "
                f"births={birth_counts.get(cls_name, 0)} coasts={coast_counts.get(cls_name, 0)} "
                f"dead={dead_counts.get(cls_name, 0)} output={output_counts.get(cls_name, 0)} "
                f"alive_init={alive_info['init']} alive_confirmed={alive_info['confirmed']} "
                f"alive_obscured={alive_info['obscured']}"
            )
            if pending_info:
                emit_debug_line(
                    f"[TRK-SMALL] frame={frame_id} cls={cls_name} pending={pending_info.get('pending', 0)} "
                    f"need_more_hits={pending_info.get('need_more_hits', 0)} "
                    f"low_det_score={pending_info.get('low_det_score', 0)} "
                    f"low_match_score={pending_info.get('low_match_score', 0)} "
                    f"has_real_det={pending_info.get('has_real_det', 0)} "
                    f"no_real_det_yet={pending_info.get('no_real_det_yet', 0)}"
                )

    def _rv_rescue_match(
        self,
        frame_info,
        unmatched_trajs: List[Trajectory],
        unmatched_det_indices: List[int],
    ) -> List[Tuple[int, int, BBox, float]]:
        if not self.cfg.get("IS_RV_MATCHING", False):
            return []
        if len(unmatched_trajs) == 0 or len(unmatched_det_indices) == 0:
            return []

        unmatched_dets = [frame_info.bboxes[i] for i in unmatched_det_indices]
        match_res_rv, costs_rv = match_trajs_and_dets(
            unmatched_trajs,
            unmatched_dets,
            self.cfg,
            frame_info.transform_matrix,
            is_rv=True,
        )

        rescued: List[Tuple[int, int, BBox, float]] = []
        for match_idx, row in enumerate(match_res_rv if len(match_res_rv) > 0 else []):
            traj_sub_idx = int(row[0])
            det_sub_idx = int(row[1])
            track_id = unmatched_trajs[traj_sub_idx].track_id
            trk_bbox = self.all_trajs[track_id].bboxes[-1]
            det_bbox = unmatched_dets[det_sub_idx]
            diff_rot = (
                abs(norm_realative_radian(trk_bbox.global_yaw - det_bbox.global_yaw))
                * 180.0
                / np.pi
            )
            dist = np.linalg.norm(
                np.array(trk_bbox.global_xyz) - np.array(det_bbox.global_xyz)
            )
            if diff_rot > 90 or dist > 5:
                continue
            global_det_idx = unmatched_det_indices[det_sub_idx]
            match_cost = float(costs_rv[match_idx]) if match_idx < len(costs_rv) else float("inf")
            rescued.append((traj_sub_idx, global_det_idx, det_bbox, match_cost))
        return rescued

    def _track_single_stage_mctrack_compat(
        self,
        frame_info,
        trajs: List[Trajectory],
        mamba_out: Optional[Dict],
        traj_index_map: Dict[int, int],
        trk_embeddings: Optional[np.ndarray],
        det_embeddings_all: Optional[np.ndarray],
        det_counts: Dict[str, int],
        matched_counts: Dict[str, int],
        birth_counts: Dict[str, int],
        coast_counts: Dict[str, int],
        _dbg: str,
        _dbg_small: str,
    ) -> None:
        trajs_cnt = len(trajs)
        dets_cnt = len(frame_info.bboxes)
        matched_track_ids: List[int] = []
        matched_bboxes: List[BBox] = []

        if trajs_cnt > 0 and dets_cnt > 0:
            match_res, costs = match_trajs_and_dets(
                trajs,
                frame_info.bboxes,
                self.cfg,
                trk_embeddings=trk_embeddings,
                det_embeddings=det_embeddings_all,
            )
        else:
            match_res = np.empty((0, 2), dtype=int)
            costs = np.empty((0,), dtype=float)

        matched_det_indices = set(int(i) for i in match_res[:, 1]) if len(match_res) > 0 else set()
        unmatched_det_indices = np.array(
            [i for i in range(dets_cnt) if i not in matched_det_indices],
            dtype=int,
        )

        unmatched_trajs: Dict[int, Trajectory] = {}
        matched_traj_indices: set = set()
        for match_idx, row in enumerate(match_res if len(match_res) > 0 else []):
            traj_idx = int(row[0])
            det_idx = int(row[1])
            track_id = trajs[traj_idx].track_id
            det_bbox = frame_info.bboxes[det_idx]
            match_cost = float(costs[match_idx]) if match_idx < len(costs) else float("inf")
            self.all_trajs[track_id].update(det_bbox, match_cost)
            matched_track_ids.append(track_id)
            matched_bboxes.append(det_bbox)
            matched_traj_indices.add(traj_idx)
            if _dbg_small and det_bbox.category in matched_counts:
                matched_counts[det_bbox.category] += 1
                emit_debug_line(
                    f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                    f"stage=single_mc event=match tid={track_id} det_score={det_bbox.det_score:.4f} "
                    f"assoc_cost={match_cost:.4f} track_len={self.all_trajs[track_id].track_length} "
                    f"status={self.all_trajs[track_id].status_flag}"
                )

        if _dbg:
            emit_debug_line(
                f"[TRK] frame={frame_info.frame_id} stage_single_mc "
                f"dets={dets_cnt} trajs={trajs_cnt} matched={len(match_res)}"
            )

        for i in range(trajs_cnt):
            track_id = trajs[i].track_id
            if i not in matched_traj_indices:
                unmatched_trajs[track_id] = self.all_trajs[track_id]
                if not self.cfg.get("IS_RV_MATCHING", False):
                    self._apply_mctrack_exact_unmatch_update(
                        [track_id],
                        mamba_out,
                        traj_index_map,
                    )
                    self._apply_original_motion_fake_update(track_id)
                    self.all_trajs[track_id].unmatch_update(
                        frame_info.frame_id, timestamp=frame_info.timestamp
                    )
                    if _dbg_small and trajs[i].bboxes[-1].category in coast_counts:
                        coast_counts[trajs[i].bboxes[-1].category] += 1

        init_bboxes = frame_info.bboxes
        rv_matched = 0
        if self.cfg.get("IS_RV_MATCHING", False):
            unmatched_trajs_inbev = self.get_trajectory_bbox(unmatched_trajs)
            dets_cnt_inbev = len(unmatched_det_indices)
            unmatched_dets_inbev = (
                np.array(frame_info.bboxes, dtype=object)[unmatched_det_indices].tolist()
                if dets_cnt_inbev > 0
                else []
            )

            if len(unmatched_trajs_inbev) > 0 and len(unmatched_dets_inbev) > 0:
                match_res_inbev, cost_matrix_inbev = match_trajs_and_dets(
                    unmatched_trajs_inbev,
                    unmatched_dets_inbev,
                    self.cfg,
                    frame_info.transform_matrix,
                    is_rv=True,
                )
            else:
                match_res_inbev = np.empty((0, 2), dtype=int)
                cost_matrix_inbev = np.empty((0,), dtype=float)

            matched_det_indices_rv = set(int(i) for i in match_res_inbev[:, 1]) if len(match_res_inbev) > 0 else set()
            for i in range(len(unmatched_trajs_inbev)):
                track_id = unmatched_trajs_inbev[i].track_id
                if i in match_res_inbev[:, 0]:
                    indexes = np.where(match_res_inbev[:, 0] == i)[0]
                    det_bbox = unmatched_dets_inbev[int(match_res_inbev[indexes, 1][0])]
                    trk_bbox = self.all_trajs[track_id].bboxes[-1]
                    diff_rot = (
                        abs(norm_realative_radian(trk_bbox.global_yaw - det_bbox.global_yaw))
                        * 180
                        / np.pi
                    )
                    dist = np.linalg.norm(
                        np.array(trk_bbox.global_xyz) - np.array(det_bbox.global_xyz)
                    )
                    if diff_rot > 90 or dist > 5:
                        self._apply_original_motion_fake_update(track_id)
                        self.all_trajs[track_id].unmatch_update(
                            frame_info.frame_id, timestamp=frame_info.timestamp
                        )
                        if _dbg_small and trk_bbox.category in coast_counts:
                            coast_counts[trk_bbox.category] += 1
                        continue
                    match_cost = float(cost_matrix_inbev[indexes][0])
                    self.all_trajs[track_id].update(det_bbox, match_cost)
                    matched_track_ids.append(track_id)
                    matched_bboxes.append(det_bbox)
                    rv_matched += 1
                    if _dbg_small and det_bbox.category in matched_counts:
                        matched_counts[det_bbox.category] += 1
                        emit_debug_line(
                            f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                            f"stage=rv_mc event=match tid={track_id} det_score={det_bbox.det_score:.4f} "
                            f"assoc_cost={match_cost:.4f} track_len={self.all_trajs[track_id].track_length} "
                            f"status={self.all_trajs[track_id].status_flag}"
                        )
                else:
                    self._apply_original_motion_fake_update(track_id)
                    self.all_trajs[track_id].unmatch_update(
                        frame_info.frame_id, timestamp=frame_info.timestamp
                    )
                    if _dbg_small and self.all_trajs[track_id].bboxes[-1].category in coast_counts:
                        coast_counts[self.all_trajs[track_id].bboxes[-1].category] += 1

            unmatched_det_indices = np.array(
                [i for i in range(len(unmatched_dets_inbev)) if i not in matched_det_indices_rv],
                dtype=int,
            )
            init_bboxes = unmatched_dets_inbev
            if _dbg:
                emit_debug_line(
                    f"[TRK] frame={frame_info.frame_id} stage_rv_mc "
                    f"candidate_trajs={len(unmatched_trajs_inbev)} "
                    f"candidate_dets={len(unmatched_dets_inbev)} matched={rv_matched}"
                )

        if mamba_out is not None and len(matched_track_ids) > 0:
            self._update_matched_tracks(
                matched_track_ids, matched_bboxes, mamba_out, traj_index_map,
            )

        for i in unmatched_det_indices:
            det_bbox = init_bboxes[int(i)]
            self.all_trajs[self.track_id_counter] = Trajectory(
                track_id=self.track_id_counter,
                init_bbox=det_bbox,
                cfg=self.cfg,
            )
            self._init_kf_state(self.track_id_counter, det_bbox)
            self.track_id_counter += 1
            if _dbg_small and det_bbox.category in birth_counts:
                birth_counts[det_bbox.category] += 1
                emit_debug_line(
                    f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                    f"event=birth_mc tid={self.track_id_counter - 1} det_score={det_bbox.det_score:.4f}"
                )

        if _dbg:
            emit_debug_line(
                f"[TRK] frame={frame_info.frame_id} births={len(unmatched_det_indices)} "
                f"coasts={len(unmatched_trajs)} matched_total={len(matched_track_ids)}"
            )

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
        self._cur_delta_t = delta_t
        self.last_timestamp = frame_info.timestamp

        for det in frame_info.bboxes:
            det.timestamp = frame_info.timestamp

        # ---- get active trajectories ----
        trajs = self.get_trajectory_bbox(self.all_trajs)
        trajs_cnt = len(trajs)
        dets_cnt = len(frame_info.bboxes)

        _dbg = os.environ.get("DEBUG_TRACKER", "")
        _dbg_small = os.environ.get("DEBUG_SMALL_CLASSES", "")
        if _dbg:
            _statuses = [t.status_flag for t in trajs]
            emit_debug_line(
                f"[TRK] frame={frame_info.frame_id} dets={dets_cnt} "
                f"trajs={trajs_cnt} status_0={_statuses.count(0)} "
                f"status_1={_statuses.count(1)} status_2={_statuses.count(2)} "
                f"dt={delta_t:.4f}s"
            )
        debug_target_classes = {"bicycle", "motorcycle"}
        det_counts = {cls_name: 0 for cls_name in debug_target_classes}
        matched_counts = {cls_name: 0 for cls_name in debug_target_classes}
        birth_counts = {cls_name: 0 for cls_name in debug_target_classes}
        coast_counts = {cls_name: 0 for cls_name in debug_target_classes}
        dead_counts = {cls_name: 0 for cls_name in debug_target_classes}
        if _dbg_small:
            for det in frame_info.bboxes:
                if det.category in det_counts:
                    det_counts[det.category] += 1

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

        # ---- detection embeddings (shared by both paths) ----
        cost_mode = self.cfg.get("THRESHOLD", {}).get("BEV", {}).get("COST_MODE", "geometric")
        use_mamba_assoc_prior = self._mamba_association_prior_enabled()
        if (
            dets_cnt > 0
            and (
                (cost_mode == "full" and self.filter_mode in ["mamba", "fusion"])
                or (use_mamba_assoc_prior and self.filter_mode in ["mamba", "fusion", "mamba_multihead_closure"])
            )
        ):
            det_history = torch.zeros(dets_cnt, self.history_len, 12, device=self.device)
            cat_map = self.cfg["CATEGORY_MAP_TO_NUMBER"]
            det_class_ids = torch.tensor(
                [cat_map.get(det.category, 0) for det in frame_info.bboxes],
                dtype=torch.long, device=self.device)
            det_history_mask = torch.zeros(dets_cnt, self.history_len, dtype=torch.bool, device=self.device)
            det_history_match_mask = torch.zeros(dets_cnt, self.history_len, dtype=torch.bool, device=self.device)
            det_current_range = torch.zeros(dets_cnt, dtype=torch.float32, device=self.device)
            with torch.no_grad():
                for idx, det in enumerate(frame_info.bboxes):
                    det_history[idx, -1, :] = torch.tensor([
                        0.0, 0.0, det.global_xyz[2],
                        det.global_velocity[0], det.global_velocity[1], 0.0,
                        det.lwh[0], det.lwh[1], det.lwh[2],
                        det.global_yaw, 0.0, det.det_score
                    ], device=self.device)
                    det_history_mask[idx, -1] = True
                    det_history_match_mask[idx, -1] = True
                    det_current_range[idx] = float(np.linalg.norm(np.asarray(det.global_xyz[:2], dtype=np.float32)))
                det_mamba_out = self.mamba_ekf.mamba(
                    det_history,
                    class_ids=det_class_ids,
                    current_range=det_current_range,
                    detection_driven_mask=torch.ones(dets_cnt, dtype=torch.bool, device=self.device),
                    history_mask=det_history_mask,
                    history_match_mask=det_history_match_mask,
                )
            det_embeddings_all = det_mamba_out["embedding"].cpu().numpy()
        else:
            det_embeddings_all = np.zeros((dets_cnt, self.embed_dim), dtype=np.float32) if dets_cnt > 0 else None

        # ---- per-trajectory cal_flag from COST_STATE config (shared) ----
        if trajs_cnt > 0:
            cost_state_cfg = self.cfg["MATCHING"]["BEV"]["COST_STATE"]
            cal_flags = [
                cost_state_cfg.get(t.category_num, "Predict") for t in trajs
            ]
        else:
            cal_flags = []

        if self.tracker_compat_mode == "mctrack":
            self._track_single_stage_mctrack_compat(
                frame_info=frame_info,
                trajs=trajs,
                mamba_out=mamba_out,
                traj_index_map=traj_index_map,
                trk_embeddings=trk_embeddings,
                det_embeddings_all=det_embeddings_all,
                det_counts=det_counts,
                matched_counts=matched_counts,
                birth_counts=birth_counts,
                coast_counts=coast_counts,
                _dbg=_dbg,
                _dbg_small=_dbg_small,
            )
        else:
            # ================================================================
            # Default single-stage matching with uncertainty-aware association.
            #   All dets participate in one matching round; all unmatched
            #   dets can birth new tracks.
            # ================================================================
            matched_track_ids: List[int] = []
            matched_bboxes: List[BBox] = []
            matched_traj_indices: set = set()
            matched_det_indices: set = set()

            if trajs_cnt > 0 and dets_cnt > 0:
                match_res, costs = match_trajs_and_dets_uncertainty_aware(
                    trajs, frame_info.bboxes, self.cfg,
                    trk_embeddings=trk_embeddings,
                    det_embeddings=det_embeddings_all,
                    trk_pos_P=trk_pos_P,
                    trk_ori_P=trk_ori_P,
                    cal_flag=cal_flags,
                )
            else:
                match_res = np.empty((0, 2), dtype=int)
                costs = np.empty((0,), dtype=float)

            for match_idx, row in enumerate(match_res if len(match_res) > 0 else []):
                traj_idx = int(row[0])
                det_idx = int(row[1])
                track_id = trajs[traj_idx].track_id
                det_bbox = frame_info.bboxes[det_idx]
                match_cost = float(costs[match_idx]) if match_idx < len(costs) else float("inf")
                self.all_trajs[track_id].update(det_bbox, match_cost)
                matched_track_ids.append(track_id)
                matched_bboxes.append(det_bbox)
                matched_traj_indices.add(traj_idx)
                matched_det_indices.add(det_idx)
                if _dbg_small and det_bbox.category in matched_counts:
                    matched_counts[det_bbox.category] += 1
                    assoc_cost = float(costs[match_idx]) if match_idx < len(costs) else float("nan")
                    emit_debug_line(
                        f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                        f"stage=single event=match tid={track_id} det_score={det_bbox.det_score:.4f} "
                        f"assoc_cost={assoc_cost:.4f} track_len={self.all_trajs[track_id].track_length} "
                        f"status={self.all_trajs[track_id].status_flag}"
                    )

            # ---- Optional MCTrack-style RV rescue on remaining unmatched pairs ----
            rv_matched = 0
            remaining_traj_indices = [
                i for i in range(trajs_cnt) if i not in matched_traj_indices
            ]
            remaining_det_indices = [
                i for i in range(dets_cnt) if i not in matched_det_indices
            ]
            if self.cfg.get("IS_RV_MATCHING", False):
                rescued_matches = self._rv_rescue_match(
                    frame_info,
                    [trajs[i] for i in remaining_traj_indices],
                    remaining_det_indices,
                )
                for traj_sub_idx, det_global_idx, det_bbox, match_cost in rescued_matches:
                    traj_idx = remaining_traj_indices[traj_sub_idx]
                    track_id = trajs[traj_idx].track_id
                    self.all_trajs[track_id].update(det_bbox, match_cost)
                    matched_track_ids.append(track_id)
                    matched_bboxes.append(det_bbox)
                    matched_traj_indices.add(traj_idx)
                    matched_det_indices.add(det_global_idx)
                    rv_matched += 1
                    if _dbg_small and det_bbox.category in matched_counts:
                        matched_counts[det_bbox.category] += 1
                        emit_debug_line(
                            f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                            f"stage=rv event=match tid={track_id} det_score={det_bbox.det_score:.4f} "
                            f"assoc_cost={match_cost:.4f} track_len={self.all_trajs[track_id].track_length} "
                            f"status={self.all_trajs[track_id].status_flag}"
                        )
                if _dbg and (len(remaining_traj_indices) > 0 or len(remaining_det_indices) > 0):
                    emit_debug_line(
                        f"[TRK] frame={frame_info.frame_id} stage_rv "
                        f"candidate_trajs={len(remaining_traj_indices)} "
                        f"candidate_dets={len(remaining_det_indices)} matched={rv_matched}"
                    )

            # ---- coast: unmatched trajectories ----
            for i in range(trajs_cnt):
                if i not in matched_traj_indices:
                    track_id = trajs[i].track_id
                    self._apply_original_motion_fake_update(track_id)
                    self.all_trajs[track_id].unmatch_update(frame_info.frame_id, timestamp=frame_info.timestamp)
                    if _dbg_small and trajs[i].bboxes[-1].category in coast_counts:
                        coast_counts[trajs[i].bboxes[-1].category] += 1

            # ---- batch KF update for all matched tracks ----
            if mamba_out is not None and len(matched_track_ids) > 0:
                self._update_matched_tracks(
                    matched_track_ids, matched_bboxes, mamba_out, traj_index_map,
                )

            # ---- Birth: unmatched dets create new tracks.
            # In mixed single-stage mode, classes listed in
            # SINGLE_STAGE_BIRTH_SCORE must pass their per-class score gate;
            # classes without a configured gate keep legacy MCTrack behavior.
            single_stage_birth_cfg = self.cfg["THRESHOLD"]["TRAJECTORY_THRE"].get(
                "SINGLE_STAGE_BIRTH_SCORE", {}
            )
            for det_idx in range(dets_cnt):
                if det_idx not in matched_det_indices:
                    det_bbox = frame_info.bboxes[det_idx]
                    gate_allowed = classify_single_stage_birth(
                        category=det_bbox.category,
                        score=det_bbox.det_score,
                        category_map=self.cfg["CATEGORY_MAP_TO_NUMBER"],
                        birth_gate_cfg=single_stage_birth_cfg,
                    )
                    if not allow_single_stage_birth_under_mode(
                        compat_mode=self.tracker_compat_mode,
                        gate_allowed=gate_allowed,
                    ):
                        if _dbg_small and det_bbox.category in birth_counts:
                            emit_debug_line(
                                f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                                f"event=birth_filtered det_score={det_bbox.det_score:.4f}"
                            )
                        continue
                    new_traj = Trajectory(
                        track_id=self.track_id_counter,
                        init_bbox=det_bbox,
                        cfg=self.cfg,
                    )
                    self.all_trajs[self.track_id_counter] = new_traj
                    self._init_kf_state(self.track_id_counter, det_bbox)
                    self.track_id_counter += 1
                    if _dbg_small and det_bbox.category in birth_counts:
                        birth_counts[det_bbox.category] += 1
                        emit_debug_line(
                            f"[TRK-SMALL] frame={frame_info.frame_id} cls={det_bbox.category} "
                            f"event=birth tid={self.track_id_counter - 1} det_score={det_bbox.det_score:.4f}"
                        )

        # ---- death: remove dead tracks ----
        _n_dead = 0
        for track_id in list(self.all_trajs.keys()):
            if self.all_trajs[track_id].status_flag == 4:
                if _dbg_small and self.all_trajs[track_id].bboxes[-1].category in dead_counts:
                    dead_counts[self.all_trajs[track_id].bboxes[-1].category] += 1
                self.all_dead_trajs[track_id] = self.all_trajs[track_id]
                del self.all_trajs[track_id]
                self.kf_states.pop(track_id, None)
                self.original_motion.remove_track(track_id)
                _n_dead += 1

        output_trajs = self.get_output_trajs(frame_info.frame_id)
        if _dbg_small:
            self._print_small_class_debug(
                frame_info.frame_id,
                det_counts,
                matched_counts,
                birth_counts,
                coast_counts,
                dead_counts,
                output_trajs,
            )
        if _dbg:
            emit_debug_line(
                f"[TRK] frame={frame_info.frame_id} dead={_n_dead} "
                f"alive={len(self.all_trajs)} output={len(output_trajs)}"
            )
        self._flush_noise_audit_samples()
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
                # Only real detections above the configured per-class OUTPUT_SCORE
                # threshold determine the output score. Low-score rescue dets are
                # excluded — they exist to maintain track continuity, not to
                # represent object existence probability.
                # This runs in both ONLINE and GLOBAL tracking modes.
                traj = self.all_trajs[track_id]
                output_score_thre = getattr(traj, "_output_score", 0.4)
                real_scores = [
                    float(getattr(b, "raw_det_score", b.det_score)) for b in traj.bboxes
                    if not getattr(b, "is_fake", False)
                ]
                quality_scores = [
                    float(getattr(b, "raw_det_score", b.det_score)) for b in traj.bboxes
                    if (
                        not getattr(b, "is_fake", False)
                        and not getattr(b, "is_low_score_match", False)
                        and float(getattr(b, "raw_det_score", b.det_score)) >= output_score_thre
                    )
                ]
                quality_score = select_output_tracking_score(
                    current_score=bbox.det_score,
                    real_scores=real_scores,
                    quality_scores=quality_scores,
                    compat_mode=self.tracker_compat_mode,
                )
                bbox.det_score = quality_score
                suppressor_cfg = self.dirty_suppressor_cfg
                base_score = (
                    bbox.det_score if getattr(bbox, "det_score", None) is not None
                    else getattr(bbox, "score", 0.0)
                )
                pos_trace = 0.0
                kf_state = self.kf_states.get(track_id, None)
                if kf_state is not None and kf_state.get("pos_P", None) is not None:
                    pos_trace = float(
                        kf_state["pos_P"].diagonal(dim1=-2, dim2=-1).sum().item()
                    )
                suppress_result = apply_dirty_track_suppressor_to_output(
                    base_score=base_score,
                    class_id=traj.category_num,
                    traj=traj,
                    suppressor_cfg=suppressor_cfg,
                    pos_trace=pos_trace,
                    pos_trace_prior=self._dirty_pos_trace_priors.get(
                        int(traj.category_num),
                        23.0,
                    ),
                )
                self._record_dirty_suppressor_audit_sample(
                    class_id=int(traj.category_num),
                    class_name=str(getattr(bbox, "category", "unknown")),
                    profile_name=suppress_result.get("profile_name"),
                    penalty=float(suppress_result["penalty"]),
                    hard_reject=bool(suppress_result["hard_reject"]),
                    triggered_reasons=list(suppress_result.get("triggered_reasons", [])),
                    features=suppress_result.get("features", {}),
                )
                if suppress_result["hard_reject"]:
                    continue
                bbox.det_score = suppress_result["final_score"]
                if hasattr(bbox, "score"):
                    bbox.score = suppress_result["final_score"]
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
