# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track
# Module A: Decoupled Adaptive Kalman Filters (Kinematics)
# Module B: Mamba Soft-Coupler & Multi-Head Noise Prediction
#
# Three independent filters prevent "dimension confusion" and error propagation:
#   1. PositionFilter   : CA model, state [x, y, z, vx, vy, vz, ax, ay]  dim=8
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

def cholesky_to_psd(L_raw: Tensor, eps: float = 1e-5) -> Tensor:
    """
    Reconstruct a strictly positive-definite matrix from a raw lower-triangular
    Cholesky factor by applying Softplus + eps on the diagonal.

    # PSD reconstruction formula:
    #   L_diag = Softplus(L_raw_diag) + eps   (guarantees L_diag > eps > 0)
    #   L      = tril(L_raw) with corrected diagonal
    #   M      = L @ L^T                       (strictly PD by construction)

    Args:
        L_raw: Raw lower-triangular factor. Shape: [B, D, D]
        eps:   Minimum diagonal value.

    Returns:
        PSD matrix. Shape: [B, D, D]
    """
    # [B, D, D] — extract lower triangular part
    L = torch.tril(L_raw)
    diag_idx = torch.arange(L.shape[-1], device=L.device)
    # Softplus + eps on diagonal — prevents zero/negative eigenvalues
    L = L.clone()
    L[:, diag_idx, diag_idx] = F.softplus(L_raw[:, diag_idx, diag_idx]) + eps
    # Q = L @ L^T  →  [B, D, D]
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
# Filter 1: Position Filter  (Constant Acceleration model)
# State: [x, y, z, vx, vy, vz, ax, ay]  dim=8
# Obs  : [x, y, z]                       dim=3
# ============================================================

class PositionFilter(nn.Module):
    """
    Batched EKF for 3-D position under a Constant Acceleration (CA) model.

    State vector (dim=8): [x, y, z, vx, vy, vz, ax, ay]
      - ax, ay are horizontal accelerations; az is assumed zero (flat-world prior).

    Observation vector (dim=3): [x, y, z]
    """

    STATE_DIM: int = 8
    OBS_DIM: int = 3

    def __init__(self, batch_size: int, device: torch.device = torch.device("cpu")) -> None:
        """
        Args:
            batch_size : Number of tracklets processed in parallel.
            device     : Torch device.

        Initialised buffers
        -------------------
        x : [B, 8, 1]   — state mean
        P : [B, 8, 8]   — state covariance
        H : [3, 8]       — observation matrix (constant)
        """
        super().__init__()
        self.B = batch_size
        self.device = device

        self.x: Tensor = torch.zeros(batch_size, self.STATE_DIM, 1, device=device)
        self.P: Tensor = torch.eye(self.STATE_DIM, device=device).unsqueeze(0).repeat(batch_size, 1, 1)

        # H selects [x, y, z] from the 8-dim state — constant across batch
        # H = [[1,0,0,0,0,0,0,0],
        #      [0,1,0,0,0,0,0,0],
        #      [0,0,1,0,0,0,0,0]]   shape: [3, 8]
        self.register_buffer("H", torch.zeros(self.OBS_DIM, self.STATE_DIM))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0

        # default Q/R when Module B is not yet active (explicit .to(device) avoids CPU/GPU mismatch)
        self.register_buffer("_default_Q", (0.01 * torch.eye(self.STATE_DIM)).to(device))
        self.register_buffer("_default_R", (0.1 * torch.eye(self.OBS_DIM)).to(device))

    def init_state(self, x0: Tensor, P0: Tensor) -> None:
        """
        Args:
            x0 : [B, 8, 1]
            P0 : [B, 8, 8]
        """
        self.x = x0.to(self.device)
        self.P = P0.to(self.device)
        self.B = x0.shape[0]

    def build_F(self, delta_t: float) -> Tensor:
        """
        Construct the CA state-transition matrix F for a given delta_t.

        # CA kinematics (dt = delta_t, dt2 = 0.5*dt^2):
        #   x_{k+1}  = x_k  + vx_k*dt + 0.5*ax_k*dt^2
        #   y_{k+1}  = y_k  + vy_k*dt + 0.5*ay_k*dt^2
        #   z_{k+1}  = z_k  + vz_k*dt
        #   vx_{k+1} = vx_k + ax_k*dt
        #   vy_{k+1} = vy_k + ay_k*dt
        #   vz_{k+1} = vz_k
        #   ax_{k+1} = ax_k
        #   ay_{k+1} = ay_k
        #
        # F = [[1, 0, 0, dt,  0,  0, dt2,   0],
        #      [0, 1, 0,  0, dt,  0,   0, dt2],
        #      [0, 0, 1,  0,  0, dt,   0,   0],
        #      [0, 0, 0,  1,  0,  0,  dt,   0],
        #      [0, 0, 0,  0,  1,  0,   0,  dt],
        #      [0, 0, 0,  0,  0,  1,   0,   0],
        #      [0, 0, 0,  0,  0,  0,   1,   0],
        #      [0, 0, 0,  0,  0,  0,   0,   1]]

        Args:
            delta_t : Elapsed time in seconds.

        Returns:
            F : [8, 8]
        """
        dt = delta_t
        dt2 = 0.5 * dt * dt
        F = torch.eye(self.STATE_DIM, device=self.device)
        # position <- velocity
        F[0, 3] = dt   # x  <- vx*dt
        F[1, 4] = dt   # y  <- vy*dt
        F[2, 5] = dt   # z  <- vz*dt
        # position <- acceleration
        F[0, 6] = dt2  # x  <- 0.5*ax*dt^2
        F[1, 7] = dt2  # y  <- 0.5*ay*dt^2
        # velocity <- acceleration
        F[3, 6] = dt   # vx <- ax*dt
        F[4, 7] = dt   # vy <- ay*dt
        return F

    def predict(
        self,
        delta_t: float,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Kalman predict step (batched, no for-loop).

        # x_pred = F @ x                      [B,8,8] @ [B,8,1] -> [B,8,1]
        # P_pred = F @ P @ F^T + Q             [B,8,8]

        Args:
            delta_t : Elapsed time in seconds since last update.
            Q       : [B, 8, 8] — process noise from Module B. If None, uses default.

        Returns:
            x_pred : [B, 8, 1]
            P_pred : [B, 8, 8]
        """
        F = self.build_F(delta_t)                           # [8, 8]
        F_batch = F.unsqueeze(0).expand(self.B, -1, -1)     # [B, 8, 8]

        if Q is None:
            Q = self._default_Q.unsqueeze(0).expand(self.B, -1, -1)  # [B, 8, 8]

        # x_pred = F @ x   →  [B, 8, 1]
        self.x = torch.bmm(F_batch, self.x)

        # P_pred = F @ P @ F^T + Q   →  [B, 8, 8]
        self.P = torch.bmm(torch.bmm(F_batch, self.P), F_batch.transpose(-1, -2)) + Q

        return self.x, self.P

    def update(
        self,
        z: Tensor,
        R: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Kalman update step (batched, no for-loop).

        # H       : [3, 8] (constant)
        # S       = H @ P @ H^T + R            [B,3,3]
        # K       = P @ H^T @ S^{-1}           [B,8,3]
        # innov   = z - H @ x                  [B,3,1]
        # x_upd   = x + K @ innov              [B,8,1]
        # P_upd   = (I - K @ H) @ P            [B,8,8]

        Args:
            z : [B, 3, 1] — observation [x, y, z]
            R : [B, 3, 3] — measurement noise from Module B. If None, uses default.

        Returns:
            x_upd : [B, 8, 1]
            P_upd : [B, 8, 8]
        """
        if R is None:
            R = self._default_R.unsqueeze(0).expand(self.B, -1, -1)  # [B, 3, 3]

        H = self.H.unsqueeze(0).expand(self.B, -1, -1)      # [B, 3, 8]
        Ht = H.transpose(-1, -2)                             # [B, 8, 3]

        # S = H @ P @ H^T + R   →  [B, 3, 3]
        S = torch.bmm(torch.bmm(H, self.P), Ht) + R

        # K = P @ H^T @ S^{-1}  →  [B, 8, 3]
        K = torch.bmm(torch.bmm(self.P, Ht), torch.linalg.inv(S))

        # innovation = z - H @ x   →  [B, 3, 1]
        innov = z - torch.bmm(H, self.x)

        # x_upd = x + K @ innovation   →  [B, 8, 1]
        self.x = self.x + torch.bmm(K, innov)

        # P_upd = (I - K @ H) @ P   →  [B, 8, 8]
        I = torch.eye(self.STATE_DIM, device=self.device).unsqueeze(0).expand(self.B, -1, -1)
        self.P = torch.bmm(I - torch.bmm(K, H), self.P)

        return self.x, self.P

    def get_state(self) -> Tuple[Tensor, Tensor]:
        """
        Returns:
            x : [B, 8, 1]
            P : [B, 8, 8]
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
        delta_t: float,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        Predict step under constant model (F = I_3, H = I_3).

        # x_pred = x          (size unchanged)
        # P_pred = I @ P @ I^T + Q = P + Q

        Args:
            delta_t : Elapsed time (unused for constant model; kept for API consistency).
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
        self.register_buffer("H", torch.tensor([[1.0, 0.0]]))

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

    def build_F(self, delta_t: float) -> Tensor:
        """
        CV state-transition Jacobian for orientation.

        # theta_{k+1} = theta_k + omega_k * dt
        # omega_{k+1} = omega_k
        #
        # F = [[1, dt],
        #      [0,  1]]

        Args:
            delta_t : Elapsed time in seconds.

        Returns:
            F : [2, 2]
        """
        F = torch.eye(self.STATE_DIM, device=self.device)
        F[0, 1] = delta_t
        return F

    def predict(
        self,
        delta_t: float,
        Q: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """
        EKF predict step with angle wrapping (batched, no for-loop).

        # x_pred = F(dt) @ x                  [B,2,2] @ [B,2,1] -> [B,2,1]
        # theta_pred = wrap_to_pi(theta_pred)
        # P_pred = F @ P @ F^T + Q            [B,2,2]

        Args:
            delta_t : Elapsed time in seconds.
            Q       : [B, 2, 2] — process noise from Module B.

        Returns:
            x_pred : [B, 2, 1]
            P_pred : [B, 2, 2]
        """
        F = self.build_F(delta_t)                            # [2, 2]
        F_batch = F.unsqueeze(0).expand(self.B, -1, -1)      # [B, 2, 2]

        if Q is None:
            Q = self._default_Q.unsqueeze(0).expand(self.B, -1, -1)

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
        pos_x  : [B, 8, 1]    pos_P  : [B, 8, 8]
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
            pos_x0 : [B, 8, 1]   pos_P0 : [B, 8, 8]
            siz_x0 : [B, 3, 1]   siz_P0 : [B, 3, 3]
            ori_x0 : [B, 2, 1]   ori_P0 : [B, 2, 2]
        """
        self.pos_filter.init_state(pos_x0, pos_P0)
        self.siz_filter.init_state(siz_x0, siz_P0)
        self.ori_filter.init_state(ori_x0, ori_P0)

    def predict(
        self,
        delta_t: float,
        Q_pos: Optional[Tensor] = None,
        Q_siz: Optional[Tensor] = None,
        Q_ori: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run predict step on all three filters independently.

        Args:
            delta_t : Elapsed time in seconds.
            Q_pos   : [B, 8, 8]  — from Module B (PSD-guaranteed)
            Q_siz   : [B, 3, 3]  — from Module B (PSD-guaranteed)
            Q_ori   : [B, 2, 2]  — from Module B (PSD-guaranteed)

        Returns:
            pos_x_pred, pos_P_pred : [B,8,1], [B,8,8]
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
            pos_x_upd, pos_P_upd : [B,8,1], [B,8,8]
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
    the Softplus + eps safety lock on the diagonal.

    Input  : Mamba temporal embedding   [B, d_model]
    Output : PSD covariance matrix      [B, D, D]
    """

    def __init__(self, d_model: int, mat_dim: int, hidden_dim: int = 64) -> None:
        """
        Args:
            d_model   : Mamba output feature dimension.
            mat_dim   : Dimension of the target covariance matrix (D).
            hidden_dim: Hidden layer width.

        The MLP predicts D*(D+1)/2 free parameters — the lower-triangular entries.
        """
        super().__init__()
        self.mat_dim = mat_dim
        # number of free parameters in a lower-triangular DxD matrix
        n_tril = mat_dim * (mat_dim + 1) // 2

        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_tril),
        )

        # precompute tril indices
        self.register_buffer(
            "_tril_rows",
            torch.tril_indices(mat_dim, mat_dim)[0],
        )
        self.register_buffer(
            "_tril_cols",
            torch.tril_indices(mat_dim, mat_dim)[1],
        )

    def forward(self, embedding: Tensor) -> Tensor:
        """
        Args:
            embedding : [B, d_model] — Mamba temporal embedding

        Returns:
            PSD matrix : [B, D, D] — strictly positive-definite, via Cholesky + Softplus + eps

        Internal steps:
            1. MLP predicts D*(D+1)/2 raw values                  [B, n_tril]
            2. Fill lower-triangular matrix L_raw                  [B, D, D]
            3. cholesky_to_psd(L_raw) applies Softplus+eps on diag
            4. Return L @ L^T                                     [B, D, D]
        """
        B = embedding.shape[0]
        raw = self.mlp(embedding)                                    # [B, n_tril]

        # scatter into lower-triangular matrix
        L_raw = torch.zeros(B, self.mat_dim, self.mat_dim,
                            device=embedding.device, dtype=embedding.dtype)
        L_raw[:, self._tril_rows, self._tril_cols] = raw

        # PSD safety lock: Softplus + 1e-5 on diagonal, then L @ L^T
        return cholesky_to_psd(L_raw, eps=1e-5)


class TemporalMamba(nn.Module):
    """
    Module B: Mamba Soft-Coupler for temporal tracklet modeling.

    Processes joint historical states of each tracklet over the past T frames
    to capture latent correlations between orientation, velocity, and position.

    Architecture:
        Input projection → Mamba SSM backbone → Output projection
        → 6 CholeskyHeads (Q_pos, Q_siz, Q_ori, R_pos, R_siz, R_ori)
        → 1 Temporal Embedding head (for semantic data association)

    Complexity: O(1) per tracklet (no cross-attention between tracklets).

    Input feature per frame (dim = 8 + 3 + 2 = 13):
        [x, y, z, vx, vy, vz, ax, ay,  l, w, h,  theta, omega]
         ├── position state (8) ──┤  ├ size(3) ┤  ├ orient(2) ┤
    """

    # Joint input = pos(8) + size(3) + orientation(2) = 13
    INPUT_DIM: int = 13

    def __init__(
        self,
        d_model: int = 64,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_mamba_layers: int = 2,
        embed_dim: int = 32,
    ) -> None:
        """
        Args:
            d_model       : Internal Mamba feature dimension.
            d_state       : SSM state dimension (N in Mamba paper).
            d_conv        : Local convolution width.
            expand         : Expansion factor for inner dimension.
            n_mamba_layers : Number of stacked Mamba layers.
            embed_dim      : Output temporal embedding dimension for association.
        """
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim

        # input projection: 13 → d_model
        self.input_proj = nn.Linear(self.INPUT_DIM, d_model)

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

        # ---- Multi-Head Noise Prediction (6 heads) ----
        # Process noise Q: one per filter
        self.head_Q_pos = CholeskyHead(d_model, mat_dim=8)   # Q for Position  [B,8,8]
        self.head_Q_siz = CholeskyHead(d_model, mat_dim=3)   # Q for Size      [B,3,3]
        self.head_Q_ori = CholeskyHead(d_model, mat_dim=2)   # Q for Orient.   [B,2,2]

        # Measurement noise R: one per filter
        self.head_R_pos = CholeskyHead(d_model, mat_dim=3)   # R for Position  [B,3,3]
        self.head_R_siz = CholeskyHead(d_model, mat_dim=3)   # R for Size      [B,3,3]
        self.head_R_ori = CholeskyHead(d_model, mat_dim=1)   # R for Orient.   [B,1,1]

        # ---- Temporal Embedding head (for semantic association in Module C) ----
        self.embed_head = nn.Sequential(
            nn.Linear(d_model, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(
        self,
        track_history: Tensor,
    ) -> Dict[str, Tensor]:
        """
        Process joint historical states and produce noise matrices + embedding.

        Args:
            track_history : [B, T, 13] — past T frames of joint state per tracklet
                            B = number of active tracklets
                            T = temporal window length
                            13 = [x,y,z,vx,vy,vz,ax,ay, l,w,h, theta,omega]

        Returns:
            dict with keys:
                "Q_pos"     : [B, 8, 8]  — position process noise     (PSD)
                "Q_siz"     : [B, 3, 3]  — size process noise          (PSD)
                "Q_ori"     : [B, 2, 2]  — orientation process noise   (PSD)
                "R_pos"     : [B, 3, 3]  — position measurement noise  (PSD)
                "R_siz"     : [B, 3, 3]  — size measurement noise      (PSD)
                "R_ori"     : [B, 1, 1]  — orientation measurement noise (PSD)
                "embedding" : [B, embed_dim] — temporal embedding for association
        """
        # input projection: [B, T, 13] → [B, T, d_model]
        h = self.input_proj(track_history)

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

        # ---- Multi-Head Noise Prediction ----
        # Each head: [B, d_model] → [B, D, D] (PSD via Cholesky + Softplus + eps)
        Q_pos = self.head_Q_pos(h_last)    # [B, 8, 8]
        Q_siz = self.head_Q_siz(h_last)    # [B, 3, 3]
        Q_ori = self.head_Q_ori(h_last)    # [B, 2, 2]
        R_pos = self.head_R_pos(h_last)    # [B, 3, 3]
        R_siz = self.head_R_siz(h_last)    # [B, 3, 3]
        R_ori = self.head_R_ori(h_last)    # [B, 1, 1]

        # ---- Temporal Embedding for semantic association ----
        embedding = self.embed_head(h_last)  # [B, embed_dim]

        return {
            "Q_pos": Q_pos, "Q_siz": Q_siz, "Q_ori": Q_ori,
            "R_pos": R_pos, "R_siz": R_siz, "R_ori": R_ori,
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
        n_mamba_layers: int = 2,
        embed_dim: int = 32,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        super().__init__()
        self.mamba = TemporalMamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            n_mamba_layers=n_mamba_layers,
            embed_dim=embed_dim,
        )
        self.kf = DecoupledAdaptiveKF(batch_size, device)
        self.device = device

    def predict_with_mamba(
        self,
        track_history: Tensor,
        delta_t: float,
    ) -> Tuple[Dict[str, Tensor], Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Run Mamba to get adaptive Q/R, then run KF predict.

        Args:
            track_history : [B, T, 13] — joint historical states
            delta_t       : elapsed seconds

        Returns:
            mamba_out : dict with Q_pos/Q_siz/Q_ori/R_pos/R_siz/R_ori/embedding
            pos_x, pos_P, siz_x, siz_P, ori_x, ori_P  (predicted states)
        """
        mamba_out = self.mamba(track_history)

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
