# Mamba-Decoupled-EKF Track: Project Context & AI Guidelines (V2 SOTA)

## 1. Project Background & Goals
**Objective:** Develop a Fast, Efficient, and Accurate 3D Multi-Object Tracking (3DMOT) framework named **Mamba-Decoupled-EKF Track**.
**Target Publication:** IEEE T-ITS / IEEE TVT.

**Core SOTA Innovations:**
1. **Zero Transformer/GNN:** Strictly linear complexity `O(N)` for extremely high FPS.
2. **Partial Adaptive Mamba (MoE Logic):** Mamba inputs 12D features (including `det_score` for observational awareness) to dynamically predict process/measurement noise (Q/R) **only for Position**.
3. **Physical Prior Locking:** Size estimation is stripped from neural prediction and purely handled by EMA and strict lifecycle locking (Rigid Body Assumption).
4. **Directional Bayesian Modeling:** Orientation uncertainty uses the Von Mises distribution (predicting concentration $\kappa$) to eliminate topological wrapping tears ($-\pi$ to $\pi$).
5. **Two-Stage Cascade Association:** ByteTrack paradigm with strict birth to handle heavy occlusions and low-confidence detections.

## 2. Core Architecture Specifications

### Module A: Decoupled Physical Base (EKF & Containers)
* **Position Filter (Dynamic):** `[x, y, z, vx, vy, vz]` (6D Constant Velocity model). Avoid CA (Constant Acceleration) to prevent inference catapult/shock.
* **Size Filter (Static Locked):** `[l, w, h]`. Handled in `trajectory.py` via early EMA (Life <= 10) and Mature Locking (Life > 10).
* **Orientation Filter (Bayesian):** `[theta, omega]`. Covariance dynamically guided by Von Mises $\kappa$.
* **Trajectory Container:** `trajectory.py` is a PURE data container. Trajectory scores are evaluated solely on high-quality real observations, aggressively penalizing coasting/blind predictions to prevent ghost tracks.

### Module B: Mamba Brain (Soft-Coupler)
* **Input Features:** 12D `[Δx, Δy, z, vx, vy, vz, l, w, h, theta, omega, det_score]`. `det_score` MUST be injected (0.0 for coasted frames) to grant Mamba observational awareness.
* **Output Heads:**
  - `head_Q_pos` [B, 6, 6], `head_R_pos` [B, 3, 3] → Cholesky heads for position CV model (`Softplus + min_diag`).
  - `head_kappa_ori` [B, 1] → Von Mises concentration for orientation.
  - `raw_q_ori` (scalar) → Static learnable process noise for orientation.
  - `raw_q_siz_diag`, `raw_r_siz_diag` [3] → Static learnable size noise (rigid-body prior).

### Module C: Data Association
* **Two-Stage Matching (ByteTrack):**
  - Stage 1: High-score dets (`>= 0.4`) matched with all active trajectories.
  - Stage 2: Low-score dets (`0.1 ~ 0.4`) matched with unmatched trajectories (Threshold relaxed).
* **Strict Birth:** New trajectories can ONLY be spawned from unmatched high-score dets (`>= 0.4`).

## 3. Strict Development Rules (CRITICAL)
1. **Vectorization:** Use batched PyTorch tensor operations (`torch.bmm`) for all KFs. No python loops over tracklets for EKF states.
2. **Gradient Survival (Softplus Safety):** For all predicted Cholesky diagonals, NEVER use `torch.clamp`. Always use `F.softplus(x) + min_diag`.
3. **Zero-Initialization:** The final linear layer of ALL Mamba prediction heads (`CholeskyHead`, `KappaHead`) MUST be zero-initialized (`uniform_(-1e-4, 1e-4)`, `bias=0.0`) to prevent dead gradients at Epoch 1.
4. **Von Mises Stability:** When computing `von_mises_loss`, absolutely require `torch.special.i0e` (exponentially scaled Bessel function) to prevent NaN overflow.
5. **No Detach in Physics:** Never accidentally `.detach()` tensors flowing from Mamba into the NLL Loss.