import math
from typing import Dict, Optional

import numpy as np
import torch

from MCkalman.extend_kalman import EKF_CA, EKF_CTRA, EKF_CV


_DEFAULT_MOTION_MODE = {
    0: "CV",
    1: "CV",
    2: "CV",
    3: "CV",
    4: "CV",
    5: "CV",
    6: "CV",
}


def _class_map(values) -> Dict[int, str]:
    out = {}
    for key, value in (values or {}).items():
        out[int(key)] = str(value).strip().upper()
    return out


def _class_set(values):
    if values is None:
        return None
    return {int(value) for value in values}


def _diag(values):
    return np.matrix(np.diag([float(value) for value in values]))


def _finite_xy(values):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
        return np.array([0.0, 0.0], dtype=np.float64)
    return arr[:2]


def _norm_yaw(yaw: float) -> float:
    return float(yaw - 2.0 * math.pi * round(yaw / (2.0 * math.pi)))


def original_mctrack_motion_enabled(cfg: Optional[dict], filter_mode: str) -> bool:
    motion_cfg = ((cfg or {}).get("MCTRACK_ORIGINAL_MOTION") or {})
    if not bool(motion_cfg.get("ENABLED", False)):
        return False
    allowed = [
        str(value).strip().lower()
        for value in motion_cfg.get("APPLY_IN_FILTER_MODES", ["pure_dekf"])
    ]
    return str(filter_mode).strip().lower() in allowed


class MCTrackOriginalPoseMotion:
    """Pose-only wrapper around the original MCTrack CV/CA/CTRA filters."""

    def __init__(self, cfg: dict, frame_rate: float, filter_mode: str):
        self.cfg = cfg or {}
        self.frame_rate = float(frame_rate)
        self.filter_mode = str(filter_mode).strip().lower()
        self.motion_cfg = self.cfg.get("MCTRACK_ORIGINAL_MOTION", {}) or {}
        self.enabled = original_mctrack_motion_enabled(self.cfg, self.filter_mode)
        pose_cfg = self.motion_cfg.get("KALMAN_FILTER_POSE") or self.cfg.get("KALMAN_FILTER_POSE")
        self.pose_cfg = pose_cfg or {}
        self.motion_mode = dict(_DEFAULT_MOTION_MODE)
        self.motion_mode.update(_class_map(self.pose_cfg.get("MOTION_MODE")))
        self.motion_mode.update(_class_map(self.motion_cfg.get("MOTION_MODE")))
        self.active_classes = _class_set(self.motion_cfg.get("ACTIVE_CLASSES"))
        self.ctra_eps = float(self.motion_cfg.get("CTRA_YAW_RATE_EPS", 1e-3))
        self.filters: Dict[int, object] = {}

    def active(self) -> bool:
        return bool(self.enabled and self.pose_cfg)

    def remove_track(self, track_id: int) -> None:
        self.filters.pop(int(track_id), None)

    def init_track(self, track_id: int, bbox, class_id: int) -> None:
        if not self.active():
            return
        if not self.class_enabled(class_id):
            self.remove_track(track_id)
            return
        mode = self.mode_for_class(class_id)
        spec = (self.pose_cfg.get(mode) or {})
        noise = ((spec.get("NOISE") or {}).get(int(class_id))
                 or (spec.get("NOISE") or {}).get(str(int(class_id))))
        if noise is None:
            raise KeyError(f"Missing KALMAN_FILTER_POSE.{mode}.NOISE for class {class_id}")

        dt = 1.0 / max(self.frame_rate, 1e-6)
        n = int(spec.get("N", len(noise["P"])))
        m = int(spec.get("M", len(noise["R"])))
        init_x = self._initial_state(mode, bbox)
        klass = {"CV": EKF_CV, "CA": EKF_CA, "CTRA": EKF_CTRA}[mode]
        self.filters[int(track_id)] = klass(
            n=n,
            m=m,
            dt=dt,
            P=_diag(noise["P"]),
            Q=_diag(noise["Q"]),
            R=_diag(noise["R"]),
            init_x=init_x,
        )

    def mode_for_class(self, class_id: int) -> str:
        mode = self.motion_mode.get(int(class_id), "CV")
        if mode not in {"CV", "CA", "CTRA"}:
            raise ValueError(f"Unsupported MCTrack motion mode: {mode}")
        return mode

    def class_enabled(self, class_id: int) -> bool:
        return self.active_classes is None or int(class_id) in self.active_classes

    def predict(self, track_id: int, delta_t: float):
        filt = self.filters.get(int(track_id))
        if filt is None:
            return None
        self._set_dt(filt, delta_t)
        self._guard_ctra(filt)
        return self._safe_call(filt.predict)

    def update(self, track_id: int, bbox, delta_t: float):
        filt = self.filters.get(int(track_id))
        if filt is None:
            return None
        self._set_dt(filt, delta_t)
        self._guard_ctra(filt)
        z = self._measurement(filt, bbox)
        return self._safe_call(lambda: filt.update(z))

    def fake_update(self, track_id: int, delta_t: float):
        filt = self.filters.get(int(track_id))
        if filt is None:
            return None
        self._set_dt(filt, delta_t)
        self._guard_ctra(filt)
        pred = self._safe_call(filt.predict)
        if pred is None:
            return None
        z = pred[: int(filt.m)]
        self._guard_ctra(filt)
        return self._safe_call(lambda: filt.update(z))

    def apply_state_to_pos_tensor(self, track_id: int, pos_x: torch.Tensor) -> torch.Tensor:
        filt = self.filters.get(int(track_id))
        if filt is None:
            return pos_x
        state = np.asarray(filt.x, dtype=np.float64).reshape(-1)
        return self.apply_vector_to_pos_tensor(state, self._mode_from_filter(filt), pos_x)

    def apply_vector_to_pos_tensor(self, state, mode: str, pos_x: torch.Tensor) -> torch.Tensor:
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        if state.shape[0] < 2 or not np.all(np.isfinite(state[:2])):
            return pos_x

        out = pos_x.clone()
        out[0, 0, 0] = float(state[0])
        out[0, 1, 0] = float(state[1])
        vel = self._velocity_from_state(mode, state)
        out[0, 3, 0] = float(vel[0])
        out[0, 4, 0] = float(vel[1])
        return out

    def state_xy_velocity(self, track_id: int):
        filt = self.filters.get(int(track_id))
        if filt is None:
            return None
        state = np.asarray(filt.x, dtype=np.float64).reshape(-1)
        if state.shape[0] < 2 or not np.all(np.isfinite(state[:2])):
            return None
        return state[:2].tolist(), self._velocity_from_state(self._mode_from_filter(filt), state).tolist()

    def _initial_state(self, mode: str, bbox):
        xy = _finite_xy(bbox.global_xyz)
        vel = _finite_xy(getattr(bbox, "global_velocity", [0.0, 0.0]))
        acc = _finite_xy(getattr(bbox, "global_acceleration", [0.0, 0.0]))
        if mode == "CV":
            return [xy[0], xy[1], vel[0], vel[1]]
        if mode == "CA":
            return [xy[0], xy[1], vel[0], vel[1], acc[0], acc[1]]

        yaw = _norm_yaw(float(getattr(bbox, "global_yaw", 0.0)))
        speed = float(np.linalg.norm(vel))
        signed_speed = speed
        if speed > 1e-6:
            heading = np.array([math.cos(yaw), math.sin(yaw)], dtype=np.float64)
            signed_speed = float(np.dot(vel, heading))
        accel = float(np.linalg.norm(acc))
        return [xy[0], xy[1], yaw, signed_speed, self.ctra_eps, accel]

    def _measurement(self, filt, bbox):
        xy = _finite_xy(bbox.global_xyz)
        vel = _finite_xy(getattr(bbox, "global_velocity", [0.0, 0.0]))
        measure = np.array([xy[0], xy[1], vel[0], vel[1]], dtype=np.float64)
        return measure[: int(filt.m)]

    def _mode_from_filter(self, filt) -> str:
        if isinstance(filt, EKF_CTRA):
            return "CTRA"
        if isinstance(filt, EKF_CA):
            return "CA"
        return "CV"

    def _velocity_from_state(self, mode: str, state: np.ndarray) -> np.ndarray:
        if mode == "CTRA":
            yaw = float(state[2]) if state.shape[0] > 2 else 0.0
            speed = float(state[3]) if state.shape[0] > 3 else 0.0
            return np.array([speed * math.cos(yaw), speed * math.sin(yaw)], dtype=np.float64)
        if state.shape[0] >= 4:
            return state[2:4].astype(np.float64)
        return np.array([0.0, 0.0], dtype=np.float64)

    def _set_dt(self, filt, delta_t: float) -> None:
        dt = float(delta_t)
        if not np.isfinite(dt) or dt <= 0:
            dt = 1.0 / max(self.frame_rate, 1e-6)
        filt.dt = dt

    def _guard_ctra(self, filt) -> None:
        if not isinstance(filt, EKF_CTRA):
            return
        yaw_rate = float(np.asarray(filt.x).reshape(-1)[4])
        if not np.isfinite(yaw_rate) or abs(yaw_rate) < self.ctra_eps:
            sign = -1.0 if yaw_rate < 0 else 1.0
            filt.x[4, 0] = sign * self.ctra_eps

    def _safe_call(self, fn):
        try:
            result = np.asarray(fn(), dtype=np.float64).reshape(-1)
        except Exception:
            return None
        if not np.all(np.isfinite(result)):
            return None
        return result
