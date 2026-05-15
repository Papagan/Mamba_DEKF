# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track
# Module A: Decoupled Adaptive Kalman Filters (Kinematics)
# Module B: Mamba Soft-Coupler & Multi-Head Noise Prediction
#
# Three independent filters prevent "dimension confusion" and error propagation:
#   1. PositionFilter   : CV model, state [x, y, z, vx, vy, vz]  dim=6
#   2. SizeFilter       : Constant model, state [l, w, h]                 dim=3
#   3. OrientationFilter: CV model, state [theta, omega]                   dim=2
#
# All filters:
#   - Operate in batched PyTorch (torch.bmm) — NO for-loops over tracklets
#   - Accept dynamic Q/R matrices from Module B (Mamba Soft-Coupler)
#   - Integrate delta_t explicitly for asynchronous inputs / missing detections
#   - Enforce PSD via Softplus + epsilon(1e-5) on Cholesky diagonals (safety lock)
# ------------------------------------------------------------------------

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Tuple, Dict

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None  # graceful fallback for environments without mamba-ssm


# ============================================================
# Utility: PSD Safety Lock via Cholesky
# ============================================================

def cholesky_to_psd(L_raw: Tensor, eps: float = 1e-5, min_diag: float = 0.0) -> Tensor:
    """
    Reconstruct a strictly positive-definite matrix from a raw lower-triangular
    Cholesky factor by applying Softplus + min_diag + eps on the diagonal.

    # PSD reconstruction formula:
    #   L_diag = Softplus(L_raw_diag) + min_diag + eps
    #   L      = tril(L_raw) with corrected diagonal
    #   M      = L @ L^T                       (strictly PD by construction)

    min_diag sets a floor on the Cholesky diagonal, preventing the model
    from collapsing to near-zero eigenvalues (a degenerate solution where
    logdet(S) → -∞ drives the NLL artificially negative).

    Typical values:
      R heads (measurement noise): min_diag = 0.1  → min eigenvalue ≈ 0.01
      Q heads (process noise):     min_diag = 0.1  → min eigenvalue ≈ 0.01

    Args:
        L_raw:    Raw lower-triangular factor. Shape: [B, D, D]
        eps:      Small numerical epsilon (kept for numerical safety).
        min_diag: Minimum value added to Softplus on diagonal.

    Returns:
        PSD matrix. Shape: [B, D, D]
    """
    # [B, D, D] — extract lower triangular part
    L = torch.tril(L_raw)
    diag_idx = torch.arange(L.shape[-1], device=L.device)
    L = L.clone()
    # Softplus + min_diag + eps — smooth, strictly positive, gradient-safe
    L[:, diag_idx, diag_idx] = F.softplus(L_raw[:, diag_idx, diag_idx]) + min_diag + eps
    return torch.bmm(L, L.transpose(-1, -2))


def wrap_to_pi(angles: Tensor) -> Tensor:
    """
    Wrap angles to [-pi, pi].

    # wrap(θ) = θ - 2π * round(θ / 2π)

    Args:
        angles: Arbitrary radian values. Shape: any

    Returns:
        Wrapped angles in [-pi, pi]. Shape: same as input
    """
    return angles - 2.0 * math.pi * torch.round(angles / (2.0 * math.pi))


# ============================================================
# Filter 1: Position Filter  (Constant Velocity model)
# State: [x, y, z, vx, vy, vz]  dim=6
# Obs  : [x, y, z]              dim=3
# ============================================================

class PositionFilter(nn.Module):
    """
    Batched EKF for 3-D position under a Constant Velocity (CV) model.

    State vector (dim=6): [x, y, z, vx, vy, vz]
      - Acceleration is NOT modelled (avoids CA inference shock/catapult).

    Observation vector (dim=3): [x, y, z]
    """

    STATE_DIM: int = 6
    OBS_DIM: int = 3

    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> None:
        """
        Args:
            batch_size : Number of tracklets processed in parallel.
            device     : Torch device.

        Initialised buffers
        -------------------
        x : [B, 6, 1]   — state mean
        P : [B, 6, 6]   — state covariance
        H : [3, 6]       — observation matrix (constant)
        """
        super().__init__()
        self.B = batch_size
        self.device = device


        self.x: Tensor = torch.zeros(batch_size, self.STATE_DIM, 1, device=device)
        self.P: Tensor = torch.eye(self.STATE_DIM, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # H selects [x, y, z] from the 6-dim state — constant across batch
        # H = [[1,0,0,0,0,0],
        #      [0,1,0,0,0,0],
        #      [0,0,1,0,0,0]]   shape: [3, 6]
        self.register_buffer("H", torch.zeros(self.OBS_DIM, self.STATE_DIM, device=device))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # default Q/R when Module B is not yet active (explicit .to(device) avoids CPU/GPU mismatch)
        self.register_buffer("_default_Q", (0.01 * torch.eye(self.STATE_DIM)).to(device))
        self.register_buffer("_default_R", (0.1 * torch.eye(self.OBS_DIM)).to(device))

    def init_state(self, x0: Tensor, P0: Tensor) -> None:
        """
        Args:
            x0 : [B, 6, 1]
            P0 : [B, 6, 6]
        """
        self.x = x0.to(self.device)
        self.P = P0.to(self.device)
        self.B = x0.shape[0]

    def build_F(self, delta_t: Tensor) -> Tensor:
        """
        Construct batched CV state-transition matrices.

        # F[b, i, j] for each sample with its own dt[b].

        Args:
            delta_t : Seconds. Shape: [B] or [B, 1] or scalar.

        Returns:
            F : [B, 6, 6]
        """
        if not isinstance(delta_t, Tensor):
            delta_t = torch.tensor(delta_t, device=self.device, dtype=torch.float32)
        dt = delta_t.view(-1, 1, 1)                        # [B, 1, 1]
        B = dt.shape[0]
        F = torch.eye(self.STATE_DIM, device=dt.device, dtype=dt.dtype)
        F = F.unsqueeze(0).expand(B, -1, -1).clone()       # [B, 6, 6]
        F[:, 0, 3] = dt.squeeze(-1).squeeze(-1)            # x  ← vx*dt
        F[:, 1, 4] = dt.squeeze(-1).squeeze(-1)            # y  ← vy*dt
        F[:, 2, 5] = dt.squeeze(-1).squeeze(-1)            # z  ← vz*dt
        return F

    def predict(
        self,
        delta_t: Tensor,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Kalman predict step (batched, per-sample delta_t).

        # x_pred = F @ x                      [B,6,6] @ [B,6,1] -> [B,6,1]
        # P_pred = F @ P @ F^T + Q             [B,6,6]

        Args:
            delta_t : Elapsed seconds. Shape: [B], [B, 1], or scalar.
            Q       : [B, 6, 6] — process noise from Module B. If None, uses default.

        Returns:
            x_pred : [B, 6, 1]
            P_pred : [B, 6, 6]
        """
        F_batch = self.build_F(delta_t)                     # [B_dt, 6, 6]
        B_state = self.x.shape[0]
        # When delta_t is a scalar, build_F returns [1,6,6] — broadcast to state batch
        if F_batch.shape[0] == 1 and B_state > 1:
            F_batch = F_batch.expand(B_state, -1, -1)
        B = F_batch.shape[0]

        if Q is None:
            Q = self._default_Q.unsqueeze(0).expand(B, -1, -1)  # [B, 6, 6]

        # x_pred = F @ x   →  [B, 6, 1]
        self.x = torch.bmm(F_batch, self.x)

        # P_pred = F @ P @ F^T + Q   →  [B, 6, 6]
        self.P = torch.bmm(torch.bmm(F_batch, self.P), F_batch.transpose(-1, -2)) + Q

        return self.x, self.P

    def update(
        self,
        z: Tensor,
        R: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Kalman update step (batched, no for-loop).

        # H       : [3, 6] (constant)
        # S       = H @ P @ H^T + R            [B,3,3]
        # K       = P @ H^T @ S^{-1}           [B,6,3]
        # innov   = z - H @ x                  [B,3,1]
        # x_upd   = x + K @ innov              [B,6,1]
        # P_upd   = (I - K @ H) @ P            [B,6,6]

        Args:
            z : [B, 3, 1] — observation [x, y, z]
            R : [B, 3, 3] — measurement noise from Module B. If None, uses default.

        Returns:
            x_upd : [B, 6, 1]
            P_upd : [B, 6, 6]
        """
        if R is None:
            R = self._default_R.unsqueeze(0).expand(self.B, -1, -1)  # [B, 3, 3]

        H = self.H.unsqueeze(0).expand(self.B, -1, -1)      # [B, 3, 6]
        Ht = H.transpose(-1, -2)                             # [B, 6, 3]

        # S = H @ P @ H^T + R   →  [B, 3, 3]
        S = torch.bmm(torch.bmm(H, self.P), Ht) + R

        # K = P @ H^T @ S^{-1}  →  [B, 6, 3]
        K = torch.bmm(torch.bmm(self.P, Ht), torch.linalg.inv(S))

        # innovation = z - H @ x   →  [B, 3, 1]
        innov = z - torch.bmm(H, self.x)

        # x_upd = x + K @ innovation   →  [B, 6, 1]
        self.x = self.x + torch.bmm(K, innov)

        # P_upd = (I - K @ H) @ P   →  [B, 6, 6]
        I = torch.eye(self.STATE_DIM, device=self.device).unsqueeze(0).expand(self.B, -1, -1)
        self.P = torch.bmm(I - torch.bmm(K, H), self.P)

        return self.x, self.P

    def get_state(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            x : [B, 6, 1]
            P : [B, 6, 6]
        """
        return self.x, self.P


# ============================================================
# Filter 2: Size Filter  (Constant model)
# State: [l, w, h]   dim=3
# Obs  : [l, w, h]   dim=3
# ============================================================

class SizeFilter(nn.Module):
    """
    Batched KF for object dimensions under a Constant (static) model.

    State vector (dim=3): [l, w, h]
    Observation vector (dim=3): [l, w, h]
    F = I_3, H = I_3

    Rationale: object size rarely changes; isolating it prevents motion noise
    from corrupting dimension estimates.
    """

    STATE_DIM: int = 3
    OBS_DIM: int = 3

    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> None:
        """
        Args:
            batch_size : Number of tracklets processed in parallel.
            device     : Torch device.

        Initialised buffers
        -------------------
        x : [B, 3, 1]
        P : [B, 3, 3]
        """
        super().__init__()
        self.B = batch_size
        self.device = device


        self.x: Tensor = torch.zeros(batch_size, self.STATE_DIM, 1, device=device)
        self.P: Tensor = torch.eye(self.STATE_DIM, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        self.register_buffer("_default_Q", (0.001 * torch.eye(self.STATE_DIM)).to(device))
        self.register_buffer("_default_R", (0.05 * torch.eye(self.OBS_DIM)).to(device))

    def init_state(self, x0: Tensor, P0: Tensor) -> None:
        """
        Args:
            x0 : [B, 3, 1]
            P0 : [B, 3, 3]
        """
        self.x = x0.to(self.device)
        self.P = P0.to(self.device)
        self.B = x0.shape[0]

    def predict(
        self,
        delta_t,          # unused (F = I constant), kept for API consistency
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict step under constant model (F = I_3, H = I_3).

        # x_pred = x          (size unchanged)
        # P_pred = I @ P @ I^T + Q = P + Q

        delta_t is ignored — size is assumed constant (rigid-body prior).
        Kept in the signature for drop-in interchangeability with the
        Position / Orientation filters.

        Args:
            delta_t : Elapsed time (unused; kept for API consistency).
            Q       : [B, 3, 3] — process noise from Module B.

        Returns:
            x_pred : [B, 3, 1]
            P_pred : [B, 3, 3]
        """
        if Q is None:
            Q = self._default_Q.unsqueeze(0).expand(self.B, -1, -1)

        # F = I, so x_pred = x (no change)
        # P_pred = P + Q
        self.P = self.P + Q

        return self.x, self.P

    def update(
        self,
        z: Tensor,
        R: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Update step (H = I_3, batched, no for-loop).

        # S     = H @ P @ H^T + R = P + R     [B, 3, 3]
        # K     = P @ (P + R)^{-1}             [B, 3, 3]
        # innov = z - x                        [B, 3, 1]
        # x_upd = x + K @ innov                [B, 3, 1]
        # P_upd = (I - K) @ P                  [B, 3, 3]

        Args:
            z : [B, 3, 1] — observed [l, w, h]
            R : [B, 3, 3] — measurement noise from Module B.

        Returns:
            x_upd : [B, 3, 1]
            P_upd : [B, 3, 3]
        """
        if R is None:
            R = self._default_R.unsqueeze(0).expand(self.B, -1, -1)

        # S = P + R (since H = I)   →  [B, 3, 3]
        S = self.P + R

        # K = P @ S^{-1}   →  [B, 3, 3]
        K = torch.bmm(self.P, torch.linalg.inv(S))

        # innov = z - x   →  [B, 3, 1]
        innov = z - self.x

        # x_upd = x + K @ innov   →  [B, 3, 1]
        self.x = self.x + torch.bmm(K, innov)

        # P_upd = (I - K) @ P   →  [B, 3, 3]
        I = torch.eye(self.STATE_DIM, device=self.device).unsqueeze(0).expand(self.B, -1, -1)
        self.P = torch.bmm(I - K, self.P)

        return self.x, self.P

    def get_state(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            x : [B, 3, 1]
            P : [B, 3, 3]
        """
        return self.x, self.P


# ============================================================
# Filter 3: Orientation Filter  (Constant Velocity model)
# State: [theta, omega]   dim=2
# Obs  : [theta]          dim=1
# ============================================================

class OrientationFilter(nn.Module):
    """
    Batched EKF for heading angle under a Constant Angular Velocity (CV) model.

    State vector (dim=2): [theta, omega]
    Observation vector (dim=1): [theta]

    Critical: angle wrapping applied on innovation and post-update theta.
    """

    STATE_DIM: int = 2
    OBS_DIM: int = 1

    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> None:
        """
        Args:
            batch_size : Number of tracklets processed in parallel.
            device     : Torch device.

        Initialised buffers
        -------------------
        x : [B, 2, 1]
        P : [B, 2, 2]
        H : [1, 2]  — selects theta from state
        """
        super().__init__()
        self.B = batch_size
        self.device = device


        self.x: Tensor = torch.zeros(batch_size, self.STATE_DIM, 1, device=device)
        self.P: Tensor = torch.eye(self.STATE_DIM, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # H = [[1, 0]]  — observe only theta
        self.register_buffer("H", torch.tensor([[1.0, 0.0]], device=device))

        self.register_buffer("_default_Q", (0.01 * torch.eye(self.STATE_DIM)).to(device))
        self.register_buffer("_default_R", (0.1 * torch.eye(self.OBS_DIM)).to(device))

    def init_state(self, x0: Tensor, P0: Tensor) -> None:
        """
        Args:
            x0 : [B, 2, 1]
            P0 : [B, 2, 2]
        """
        self.x = x0.to(self.device)
        self.P = P0.to(self.device)
        self.B = x0.shape[0]

    def build_F(self, delta_t: Tensor) -> Tensor:
        """
        Construct batched CV state-transition matrices for orientation.

        Args:
            delta_t : Seconds. Shape: [B] or [B, 1] or scalar.

        Returns:
            F : [B, 2, 2]
        """
        if not isinstance(delta_t, Tensor):
            delta_t = torch.tensor(delta_t, device=self.device, dtype=torch.float32)
        dt = delta_t.view(-1, 1, 1)                        # [B, 1, 1]
        B = dt.shape[0]
        F = torch.eye(self.STATE_DIM, device=dt.device, dtype=dt.dtype)
        F = F.unsqueeze(0).expand(B, -1, -1).clone()       # [B, 2, 2]
        F[:, 0, 1] = dt.squeeze(-1).squeeze(-1)            # theta ← omega*dt
        return F

    def predict(
        self,
        delta_t: Tensor,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        EKF predict step with angle wrapping (batched, per-sample delta_t).

        # x_pred = F(dt) @ x                  [B,2,2] @ [B,2,1] -> [B,2,1]
        # theta_pred = wrap_to_pi(theta_pred)
        # P_pred = F @ P @ F^T + Q            [B,2,2]

        Args:
            delta_t : Elapsed seconds. Shape: [B], [B, 1], or scalar.
            Q       : [B, 2, 2] — process noise from Module B.

        Returns:
            x_pred : [B, 2, 1]
            P_pred : [B, 2, 2]
        """
        F_batch = self.build_F(delta_t)                     # [B_dt, 2, 2]
        B_state = self.x.shape[0]
        # When delta_t is a scalar, build_F returns [1,2,2] — broadcast to state batch
        if F_batch.shape[0] == 1 and B_state > 1:
            F_batch = F_batch.expand(B_state, -1, -1)
        B = F_batch.shape[0]

        if Q is None:
            Q = self._default_Q.unsqueeze(0).expand(B, -1, -1)

        # x_pred = F @ x   →  [B, 2, 1]
        self.x = torch.bmm(F_batch, self.x)

        # wrap theta to [-pi, pi]
        self.x[:, 0, :] = wrap_to_pi(self.x[:, 0, :])

        # P_pred = F @ P @ F^T + Q   →  [B, 2, 2]
        self.P = torch.bmm(torch.bmm(F_batch, self.P), F_batch.transpose(-1, -2)) + Q

        return self.x, self.P

    def update(
        self,
        z: Tensor,
        R: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        EKF update step with angle-wrapping on the innovation (batched).

        # H       = [[1, 0]]                   [1, 2]
        # S       = H @ P @ H^T + R            [B, 1, 1]
        # K       = P @ H^T @ S^{-1}           [B, 2, 1]
        # innov   = wrap_to_pi(z - H @ x)      [B, 1, 1]  ← critical for continuity
        # x_upd   = x + K @ innov              [B, 2, 1]
        # theta   = wrap_to_pi(theta)
        # P_upd   = (I - K @ H) @ P            [B, 2, 2]

        Args:
            z : [B, 1, 1] — observed heading angle (radians)
            R : [B, 1, 1] — measurement noise from Module B.

        Returns:
            x_upd : [B, 2, 1]
            P_upd : [B, 2, 2]
        """
        if R is None:
            R = self._default_R.unsqueeze(0).expand(self.B, -1, -1)  # [B, 1, 1]

        H = self.H.unsqueeze(0).expand(self.B, -1, -1)       # [B, 1, 2]
        Ht = H.transpose(-1, -2)                              # [B, 2, 1]

        # S = H @ P @ H^T + R   →  [B, 1, 1]
        S = torch.bmm(torch.bmm(H, self.P), Ht) + R

        # K = P @ H^T @ S^{-1}  →  [B, 2, 1]
        K = torch.bmm(torch.bmm(self.P, Ht), torch.linalg.inv(S))

        # innovation with angle wrapping
        # innov = wrap_to_pi(z - H @ x)   →  [B, 1, 1]
        innov = wrap_to_pi(z - torch.bmm(H, self.x))

        # x_upd = x + K @ innov   →  [B, 2, 1]
        self.x = self.x + torch.bmm(K, innov)

        # wrap theta again after update
        self.x[:, 0, :] = wrap_to_pi(self.x[:, 0, :])

        # P_upd = (I - K @ H) @ P   →  [B, 2, 2]
        I = torch.eye(self.STATE_DIM, device=self.device).unsqueeze(0).expand(self.B, -1, -1)
        self.P = torch.bmm(I - torch.bmm(K, H), self.P)

        return self.x, self.P

    def get_state(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            x : [B, 2, 1]
            P : [B, 2, 2]
        """
        return self.x, self.P


# ============================================================
# Master class: DecoupledAdaptiveKF
# Orchestrates the three independent filters.
# ============================================================

class DecoupledAdaptiveKF(nn.Module):
    """
    Orchestrates PositionFilter, SizeFilter, and OrientationFilter for a batch
    of tracklets. Accepts dynamic Q/R tensors from Module B (Mamba Soft-Coupler).

    PSD Safety Lock
    ---------------
    Module B predicts Cholesky factors L for each Q/R matrix.
    Diagonals are processed as:  L_diag = Softplus(raw_diag) + 1e-5
    so that Q = L @ L^T and R = L @ L^T are strictly positive definite.

    Tensor shape conventions (B = number of active tracklets):
        pos_x  : [B, 6, 1]    pos_P  : [B, 6, 6]
        siz_x  : [B, 3, 1]    siz_P  : [B, 3, 3]
        ori_x  : [B, 2, 1]    ori_P  : [B, 2, 2]
    """

    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> None:
        super().__init__()
        self.B = batch_size
        self.device = device


        self.pos_filter = PositionFilter(batch_size, device)
        self.siz_filter = SizeFilter(batch_size, device)
        self.ori_filter = OrientationFilter(batch_size, device)

    def init_states(
        self,
        pos_x0: Tensor, pos_P0: Tensor,
        siz_x0: Tensor, siz_P0: Tensor,
        ori_x0: Tensor, ori_P0: Tensor,
    ) -> None:
        """
        Args:
            pos_x0 : [B, 6, 1]   pos_P0 : [B, 6, 6]
            siz_x0 : [B, 3, 1]   siz_P0 : [B, 3, 3]
            ori_x0 : [B, 2, 1]   ori_P0 : [B, 2, 2]
        """
        self.pos_filter.init_state(pos_x0, pos_P0)
        self.siz_filter.init_state(siz_x0, siz_P0)
        self.ori_filter.init_state(ori_x0, ori_P0)

    def predict(
        self,
        delta_t,          # float (scalar) or [B] Tensor — per-sample when batched
        Q_pos: Optional[Tensor] = None,
        Q_siz: Optional[Tensor] = None,
        Q_ori: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run predict step on all three filters independently.

        Each active track can have its own dt when delta_t is a [B] Tensor.
        The Position and Orientation filters build per-sample state-transition
        matrices F[b] from dt[b]; the Size filter ignores dt (F = I constant).

        Args:
            delta_t : Elapsed seconds.  Scalar float (inference: all tracks share
                      the same inter-frame dt) or [B] Tensor (training: per-sample
                      rollout dt preserves individual object dynamics).
            Q_pos   : [B, 6, 6]  — from Module B (PSD-guaranteed)
            Q_siz   : [B, 3, 3]  — from Module B (PSD-guaranteed)
            Q_ori   : [B, 2, 2]  — from Module B (PSD-guaranteed)

        Returns:
            pos_x_pred, pos_P_pred : [B,6,1], [B,6,6]
            siz_x_pred, siz_P_pred : [B,3,1], [B,3,3]
            ori_x_pred, ori_P_pred : [B,2,1], [B,2,2]
        """
        pos_x, pos_P = self.pos_filter.predict(delta_t, Q_pos)
        siz_x, siz_P = self.siz_filter.predict(delta_t, Q_siz)
        ori_x, ori_P = self.ori_filter.predict(delta_t, Q_ori)
        return pos_x, pos_P, siz_x, siz_P, ori_x, ori_P

    def update(
        self,
        z_pos: Tensor,
        z_siz: Tensor,
        z_ori: Tensor,
        R_pos: Optional[Tensor] = None,
        R_siz: Optional[Tensor] = None,
        R_ori: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run update step on all three filters independently.

        Args:
            z_pos : [B, 3, 1]    R_pos : [B, 3, 3]
            z_siz : [B, 3, 1]    R_siz : [B, 3, 3]
            z_ori : [B, 1, 1]    R_ori : [B, 1, 1]

        Returns:
            pos_x_upd, pos_P_upd : [B,6,1], [B,6,6]
            siz_x_upd, siz_P_upd : [B,3,1], [B,3,3]
            ori_x_upd, ori_P_upd : [B,2,1], [B,2,2]
        """
        pos_x, pos_P = self.pos_filter.update(z_pos, R_pos)
        siz_x, siz_P = self.siz_filter.update(z_siz, R_siz)
        ori_x, ori_P = self.ori_filter.update(z_ori, R_ori)
        return pos_x, pos_P, siz_x, siz_P, ori_x, ori_P

    def get_states(self) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Returns:
            pos_x, pos_P, siz_x, siz_P, ori_x, ori_P
        """
        pos_x, pos_P = self.pos_filter.get_state()
        siz_x, siz_P = self.siz_filter.get_state()
        ori_x, ori_P = self.ori_filter.get_state()
        return pos_x, pos_P, siz_x, siz_P, ori_x, ori_P


# ============================================================
# Module B: Mamba Soft-Coupler & Multi-Head Noise Prediction
# ============================================================

class CholeskyHead(nn.Module):
    """
    Shallow MLP head that predicts the lower-triangular Cholesky factor L
    for one covariance matrix, then reconstructs PSD via L @ L^T with
    the Softplus + min_diag + eps safety lock on the diagonal.

    Input  : Mamba temporal embedding   [B, d_model]
    Output : PSD covariance matrix      [B, D, D]
    """

    def __init__(self, d_model: int, mat_dim: int, hidden_dim: int = 64,
                 min_diag: float = 0.0) -> None:
        """
        Args:
            d_model   : Mamba output feature dimension.
            mat_dim   : Dimension of the target covariance matrix (D).
            hidden_dim: Hidden layer width.
            min_diag  : Floor on Cholesky diagonal. 0.1 for R, 0.03 for Q.

        The MLP predicts D*(D+1)/2 free parameters — the lower-triangular entries.
        tril indices are created lazily in forward() so they always live on the
        correct device, avoiding CPU/GPU mismatch regardless of how the module
        is moved via .to(device).
        """
        super().__init__()
        self.mat_dim = mat_dim
        self.min_diag = min_diag
        # number of free parameters in a lower-triangular DxD matrix
        n_tril = mat_dim * (mat_dim + 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_tril),
        )

        # ---- Zero-init last layer: near-zero output → Softplus(0) ≈ 0.69 ----
        # This is critical for gradient health at Epoch 1. Without it, large
        # initial weights produce saturated Softplus outputs that immediately
        # kill downstream gradients through the covariance matrices.
        nn.init.uniform_(self.mlp[-1].weight, -1e-4, 1e-4)
        nn.init.constant_(self.mlp[-1].bias, 0.0)

        # pre-allocated container for lazily-created tril indices
        # (created on correct device in forward(); avoids register_buffer device issues)
        self._tril_rows: Optional[Tensor] = None
        self._tril_cols: Optional[Tensor] = None

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Args:
            embedding : [B, d_model] — Mamba temporal embedding

        Returns:
            PSD matrix : [B, D, D] — strictly positive-definite

        Internal steps:
            1. MLP predicts D*(D+1)/2 raw values                  [B, n_tril]
            2. Fill lower-triangular matrix L_raw                  [B, D, D]
            3. cholesky_to_psd(L_raw, min_diag=self.min_diag) applies floor
            4. Return L @ L^T                                     [B, D, D]
        """
        B = embedding.shape[0]
        dev = embedding.device
        raw = self.mlp(embedding)                                    # [B, n_tril]

        # lazily create tril indices on the correct device
        if self._tril_rows is None or self._tril_rows.device != dev:
            tril = torch.tril_indices(self.mat_dim, self.mat_dim, device=dev)
            self._tril_rows = tril[0]
            self._tril_cols = tril[1]

        # scatter into lower-triangular matrix
        L_raw = torch.zeros(B, self.mat_dim, self.mat_dim,
                            device=dev, dtype=embedding.dtype)
        L_raw[:, self._tril_rows, self._tril_cols] = raw

        return cholesky_to_psd(L_raw, eps=1e-5, min_diag=self.min_diag)


class TemporalMamba(nn.Module):
    """
    Module B: Mamba Soft-Coupler for temporal tracklet modeling.

    Processes joint historical states of each tracklet over the past T frames
    to capture latent correlations between orientation, velocity, and position.

    Architecture:
        Input projection → LayerNorm → Mamba SSM backbone → Output projection
        → 2 CholeskyHeads (Q_pos[6x6], R_pos[3x3]) — position only
        → 1 kappa head (orientation concentration, Von Mises)
        → Static params: Q_siz, R_siz (size), Q_ori (orientation)
        → 1 Temporal Embedding head (for semantic data association)

    Complexity: O(1) per tracklet (no cross-attention between tracklets).

    Input feature per frame (dim = 6 + 3 + 2 + 1 = 12):
        [x, y, z, vx, vy, vz,  l, w, h,  theta, omega,  det_score]
         ├── pos state (6) ─┤  ├size(3)┤  ├ orient(2) ┤  ├qual┤
    """

    # Joint input = pos(6) + size(3) + orientation(2) + det_score(1) = 12
    INPUT_DIM: int = 12

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_mamba_layers: int = 3,
        embed_dim: int = 32,
        min_diag_q: float = 0.1,
        min_diag_r: float = 0.1,
        num_classes: int = 10,
    ) -> None:
        """
        Args:
            d_model       : Internal Mamba feature dimension.
            d_state       : SSM state dimension (N in Mamba paper).
            d_conv        : Local convolution width.
            expand         : Expansion factor for inner dimension.
            n_mamba_layers : Number of stacked Mamba layers.
            embed_dim      : Output temporal embedding dimension for association.
            min_diag_q     : Floor on Cholesky diagonal for Q heads (process noise).
            min_diag_r     : Floor on Cholesky diagonal for R heads (measurement noise).
            num_classes    : Number of object categories for size embeddings.
        """
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        # input projection: 13 → d_model
        self.input_proj = nn.Linear(self.INPUT_DIM, d_model)

        # LayerNorm right after input projection — forces zero-mean unit-variance
        # before Mamba blocks to prevent gradient saturation from disparate input scales
        self.input_norm = nn.LayerNorm(d_model)

        # stacked Mamba layers
        if Mamba is not None:
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                )
                for _ in range(n_mamba_layers)
            ])
        else:
            # fallback: simple GRU for testing without mamba-ssm installed
            self.mamba_layers = None
            self.fallback_gru = nn.GRU(
                input_size=d_model, hidden_size=d_model,
                num_layers=n_mamba_layers, batch_first=True,
            )

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(d_model) for _ in range(n_mamba_layers)
        ])

        # ---- Position Noise (Cholesky heads — only Mamba-adaptive module) ----
        self.head_Q_pos = CholeskyHead(d_model, mat_dim=6, min_diag=min_diag_q)   # Q for Pos  [B,6,6]
        self.head_R_pos = CholeskyHead(d_model, mat_dim=3, min_diag=min_diag_r)   # R for Pos  [B,3,3]

        # ---- Size Noise (category-aware embeddings — rigid-body prior) ----
        # Initialised from CenterPoint nuScenes detection statistics
        # (noise.log) so each class starts near its observed noise level.
        # Classes beyond the 7 known ones default to car-like values.
        self.raw_q_siz = nn.Embedding(num_classes, 3)
        self.raw_r_siz = nn.Embedding(num_classes, 3)

        # Class 0: car       R=[0.058, 0.0064]  Q=R/5
        # Class 1: pedestrian R=[0.0032, 0.0018]
        # Class 2: bicycle    R=[0.0125, 0.0044]
        # Class 3: motorcycle R=[0.0348, 0.0074]
        # Class 4: bus        R=[1.46, 0.0215]
        # Class 5: trailer    R=[3.74, 0.0318]
        # Class 6: truck      R=[0.59, 0.0246]
        # Classes 7+: neutral (car-like) default
        import math as _math
        _r_init = torch.tensor([
            [-2.84, -5.05, _math.log(0.02)],   # 0: car
            [-5.74, -6.32, _math.log(0.02)],   # 1: pedestrian
            [-4.38, -5.43, _math.log(0.02)],   # 2: bicycle
            [-3.36, -4.91, _math.log(0.02)],   # 3: motorcycle
            [ 0.84, -3.84, _math.log(0.02)],   # 4: bus
            [ 3.72, -3.45, _math.log(0.02)],   # 5: trailer
            [-0.22, -3.70, _math.log(0.02)],   # 6: truck
        ])
        # Q ≈ R / 5 (size changes slowly under rigid-body assumption)
        _q_init = _r_init.clone()
        _q_init[_q_init > -1.0] -= _math.log(5.0)  # scale down larger values
        # Pad remaining classes with car-like defaults
        if num_classes > 7:
            _pad_r = _r_init[0:1].repeat(num_classes - 7, 1)
            _pad_q = _q_init[0:1].repeat(num_classes - 7, 1)
            _r_init = torch.cat([_r_init, _pad_r], dim=0)
            _q_init = torch.cat([_q_init, _pad_q], dim=0)
        with torch.no_grad():
            self.raw_q_siz.weight.copy_(_q_init)
            self.raw_r_siz.weight.copy_(_r_init)

        # ---- Orientation: Von Mises kappa head + static process noise ----
        self.head_kappa_ori = nn.Linear(d_model, 1)
        nn.init.uniform_(self.head_kappa_ori.weight, -1e-4, 1e-4)
        nn.init.constant_(self.head_kappa_ori.bias, 0.0)

        self.raw_q_ori = nn.Parameter(torch.full((1,), -4.0))

        # ---- Temporal Embedding head (for semantic association in Module C) ----
        self.embed_head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        track_history: Tensor,
        class_ids: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Process joint historical states and produce noise matrices + embedding.

        Args:
            track_history : [B, T, 12] — past T frames of joint state per tracklet
                            B = number of active tracklets
                            T = temporal window length
                            12 = [x,y,z,vx,vy,vz, l,w,h, theta,omega, det_score]
            class_ids     : [B] — integer category IDs for per-class size noise.
                            If None, uses class 0 (car) for all samples.

        Returns:
            dict with keys:
                "Q_pos"     : [B, 6, 6]  — position process noise     (PSD)
                "Q_siz"     : [B, 3, 3]  — size process noise          (static)
                "Q_ori"     : [B, 2, 2]  — orientation process noise   (static)
                "R_pos"     : [B, 3, 3]  — position measurement noise  (PSD)
                "R_siz"     : [B, 3, 3]  — size measurement noise      (static)
                "R_ori"     : [B, 1, 1]  — orientation measurement noise (1/kappa)
                "kappa_ori" : [B, 1]     — Von Mises concentration for loss
                "embedding" : [B, embed_dim] — temporal embedding for association
        """
        # input projection: [B, T, 12] → [B, T, d_model]
        h = self.input_proj(track_history)

        # LayerNorm before Mamba blocks — stabilises training by normalising
        # across the d_model dimension after the linear projection
        h = self.input_norm(h)

        # pass through Mamba layers with residual + LayerNorm
        if self.mamba_layers is not None:
            for mamba_layer, ln in zip(self.mamba_layers, self.layer_norms):
                # Mamba: [B, T, d_model] → [B, T, d_model]
                h = h + mamba_layer(ln(h))
        else:
            # fallback GRU path
            h_normed = self.layer_norms[0](h)
            h_gru, _ = self.fallback_gru(h_normed)
            h = h + h_gru

        # take last time-step's feature as the summary   [B, d_model]
        h_last = h[:, -1, :]

        # NaN stability check — clamp if Mamba produces extreme values
        if torch.isnan(h_last).any():
            h_last = torch.nan_to_num(h_last, nan=0.0)

        B = h_last.shape[0]
        dev = h_last.device

        # ---- Position Noise (Cholesky heads) ----
        Q_pos = self.head_Q_pos(h_last)    # [B, 6, 6]
        R_pos = self.head_R_pos(h_last)    # [B, 3, 3]

        # ---- Size Noise (category-aware embeddings) ----
        # Each category learns its own Q/R diagonal in log-space.
        # Pedestrians (~0.06m noise) vs trailers (~1.93m noise) differ by 30×.
        if class_ids is None:
            class_ids = torch.zeros(B, dtype=torch.long, device=dev)
        q_siz_diag = F.softplus(self.raw_q_siz(class_ids)) + 1e-4    # [B, 3]
        r_siz_diag = F.softplus(self.raw_r_siz(class_ids)) + 1e-4    # [B, 3]
        Q_siz = torch.diag_embed(q_siz_diag)                           # [B, 3, 3]
        R_siz = torch.diag_embed(r_siz_diag)                           # [B, 3, 3]

        # ---- Orientation: Von Mises kappa + static Q, derived R ----
        # kappa: concentration parameter (higher = more confident)
        kappa_ori = F.softplus(self.head_kappa_ori(h_last)) + 1e-3   # [B, 1]

        # R_ori = 1 / kappa (measurement noise derived from concentration)
        R_ori = (1.0 / kappa_ori).unsqueeze(-1)                       # [B, 1, 1]

        # Q_ori: static learnable process noise shared across batch
        q_val = F.softplus(self.raw_q_ori) + 1e-4                     # scalar
        Q_ori = torch.eye(2, device=dev, dtype=h_last.dtype) * q_val  # [2, 2]
        Q_ori = Q_ori.unsqueeze(0).expand(B, -1, -1)                  # [B, 2, 2]

        # ---- Temporal Embedding for semantic association ----
        embedding = self.embed_head(h_last)  # [B, embed_dim]

        return {
            "Q_pos": Q_pos, "Q_siz": Q_siz, "Q_ori": Q_ori,
            "R_pos": R_pos, "R_siz": R_siz, "R_ori": R_ori,
            "kappa_ori": kappa_ori,
            "embedding": embedding,
        }


# ============================================================
# Full Pipeline: MambaDecoupledEKF
# Combines Module A (DecoupledAdaptiveKF) and Module B (TemporalMamba)
# ============================================================

class MambaDecoupledEKF(nn.Module):
    """
    End-to-end module combining:
      - Module B (TemporalMamba):      history → Q/R/embedding
      - Module A (DecoupledAdaptiveKF): Q/R + observations → state update

    This is the single nn.Module to instantiate in the tracker pipeline.
    """

    def __init__(
        self,
        batch_size: int,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_mamba_layers: int = 3,
        embed_dim: int = 32,
        min_diag_q: float = 0.1,
        min_diag_r: float = 0.1,
        num_classes: int = 10,
        device: torch.device = torch.device("cpu"),
        base_noise_cfg: dict = None,
    ) -> None:
        super().__init__()
        self.mamba = TemporalMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            embed_dim=embed_dim,
            min_diag_q=min_diag_q,
            min_diag_r=min_diag_r,
            num_classes=num_classes,
        )
        self.kf = DecoupledAdaptiveKF(batch_size, device)
        self.base_noise_cfg = base_noise_cfg
        self.device = device


    def _get_base_noise(self, bsize: int, dtype: torch.dtype, class_ids: Tensor = None) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Produce baseline static decoupling Q/R tensors for Fallback/IMM mode based on DEKF_BASE_NOISE config.
        """
        if self.base_noise_cfg is None:
            # Fallback exact defaults
            q_pos = [0.1, 0.1, 0.1, 1.5, 1.5, 1.5]
            q_siz = [0.05, 0.05, 0.05]
            q_ori = [0.1]
            
            r_pos = torch.full((bsize, 3,), 0.2, device=self.device, dtype=dtype)
            r_siz = torch.full((bsize, 3,), 0.3, device=self.device, dtype=dtype)
            r_ori = torch.full((bsize, 1,), 0.2, device=self.device, dtype=dtype)
        else:
            Q_cfg = self.base_noise_cfg["Q"]
            q_pos = Q_cfg["POS"]
            q_siz = Q_cfg["SIZ"]
            q_ori = Q_cfg["ORI"]
            
            R_cfg = self.base_noise_cfg["R"]
            mul = R_cfg.get("MEAS_MULTIPLIER", 1.0)
            
            p_std = torch.tensor(R_cfg["POS_STD"], device=self.device, dtype=dtype) * mul
            s_std = torch.tensor(R_cfg["SIZ_STD"], device=self.device, dtype=dtype) * mul
            o_std = torch.tensor(R_cfg["ORI_STD"], device=self.device, dtype=dtype) * mul
            
            if class_ids is None:
                r_pos = p_std[0].expand(bsize, 3) ** 2
                r_siz = s_std[0].expand(bsize, 3) ** 2
                r_ori = o_std[0].expand(bsize, 1) ** 2
            else:
                c_ids = torch.clamp(class_ids, 0, len(p_std) - 1)
                r_pos = (p_std[c_ids] ** 2).unsqueeze(1).expand(bsize, 3)
                r_siz = (s_std[c_ids] ** 2).unsqueeze(1).expand(bsize, 3)
                r_ori = (o_std[c_ids] ** 2).unsqueeze(1).expand(bsize, 1)

        # Process Noise Q (Constant)
        q_pos_diag = torch.tensor(q_pos, device=self.device, dtype=dtype)
        Q_pos_base = torch.diag(q_pos_diag).unsqueeze(0).expand(bsize, -1, -1)
        
        q_siz_diag = torch.tensor(q_siz, device=self.device, dtype=dtype)
        Q_siz_base = torch.diag(q_siz_diag).unsqueeze(0).expand(bsize, -1, -1)
        
        # Ori process noise
        q_ori_full = torch.tensor([q_ori[0], 0.5], device=self.device, dtype=dtype)
        Q_ori_base = torch.diag(q_ori_full).unsqueeze(0).expand(bsize, -1, -1)

        # Measurement Noise R (Batched diagonals)
        R_pos_base = torch.diag_embed(r_pos)
        R_siz_base = torch.diag_embed(r_siz)
        R_ori_base = torch.diag_embed(r_ori)

        return Q_pos_base, R_pos_base, Q_siz_base, R_siz_base, Q_ori_base, R_ori_base

    def predict_with_mamba(
        self,
        track_history: Tensor,
        delta_t,          # float (scalar) or [B] Tensor — per-sample when batched
        class_ids: Optional[Tensor] = None,
        mode: str = "mamba",
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run Mamba to get adaptive Q/R, then run KF predict.

        delta_t is forwarded to DecoupledAdaptiveKF.predict():
          - Inference: scalar float (all tracks share the inter-frame dt).
          - Training:   [B] Tensor (each sample has its own rollout step dt).

        Args:
            track_history : [B, T, 12] — joint historical states
            delta_t       : Elapsed seconds (float or [B] Tensor).
            class_ids     : [B] — integer category IDs for size noise
            mode          : "mamba", "pure_dekf", or "fusion"

        Returns:
            mamba_out : dict with Q_pos/Q_siz/Q_ori/R_pos/R_siz/R_ori/embedding
            pos_x, pos_P, siz_x, siz_P, ori_x, ori_P  (predicted states)
        """
        # Run Mamba network (O(N) operation)
        mamba_out = self.mamba(track_history, class_ids=class_ids)

        if mode == "pure_dekf":
            # Bypass Mamba noise predictions, inject static DEKF physical priors
            bsize = track_history.size(0)
            dtype = track_history.dtype
            Q_p, R_p, Q_s, R_s, Q_o, R_o = self._get_base_noise(bsize, dtype, class_ids)
            mamba_out.update({
                "Q_pos": Q_p, "R_pos": R_p,
                "Q_siz": Q_s, "R_siz": R_s,
                "Q_ori": Q_o, "R_ori": R_o
            })
            
        elif mode == "fusion":
            # State-level Variance Fusion (Strategy B: Soft-Gating via Trace constraint)
            bsize = track_history.size(0)
            dtype = track_history.dtype
            Q_p, R_p, Q_s, R_s, Q_o, R_o = self._get_base_noise(bsize, dtype, class_ids)
            
            def fuse_covariance(Q_m: Tensor, Q_b: Tensor, strict_ratio: float = 2.0) -> Tensor:
                # trace shape: [B, 1, 1]
                tr_m = Q_m.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
                tr_b = Q_b.diagonal(dim1=-2, dim2=-1).sum(-1, keepdim=True).unsqueeze(-1)
                
                # If Mamba trace jumps heavily above baseline (meaning OOD/Out-of-control)
                # the weight drops to 0, smoothly falling back to physical priors.
                # If Mamba trace is tight and confident, weight stays 1.0 (trust Mamba).
                weight = torch.clamp(strict_ratio - (tr_m / (tr_b + 1e-6)), min=0.0, max=1.0)
                
                return weight * Q_m + (1.0 - weight) * Q_b
            
            # Fuse process noises (Q controls Kalman Gain scaling for prediction)
            mamba_out["Q_pos"] = fuse_covariance(mamba_out["Q_pos"], Q_p)
            mamba_out["Q_siz"] = fuse_covariance(mamba_out["Q_siz"], Q_s)
            
            # Fuse observation noises (R controls trustworthiness of incoming detections)
            mamba_out["R_pos"] = fuse_covariance(mamba_out["R_pos"], R_p)
            mamba_out["R_siz"] = fuse_covariance(mamba_out["R_siz"], R_s)

        # Execute KF predict step using the active (or fused) noise models
        pos_x, pos_P, siz_x, siz_P, ori_x, ori_P = self.kf.predict(
            delta_t,
            Q_pos=mamba_out["Q_pos"],
            Q_siz=mamba_out["Q_siz"],
            Q_ori=mamba_out["Q_ori"],
        )

        return mamba_out, pos_x, pos_P, siz_x, siz_P, ori_x, ori_P

    def update_with_mamba(
        self,
        z_pos: Tensor,
        z_siz: Tensor,
        z_ori: Tensor,
        mamba_out: Dict[str, Tensor],
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run KF update using Mamba-predicted R matrices.

        Args:
            z_pos     : [B, 3, 1]
            z_siz     : [B, 3, 1]
            z_ori     : [B, 1, 1]
            mamba_out : dict from predict_with_mamba

        Returns:
            pos_x, pos_P, siz_x, siz_P, ori_x, ori_P  (updated states)
        """
        return self.kf.update(
            z_pos, z_siz, z_ori,
            R_pos=mamba_out["R_pos"],
            R_siz=mamba_out["R_siz"],
            R_ori=mamba_out["R_ori"],
        )
