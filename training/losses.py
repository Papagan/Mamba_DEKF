# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — Training Losses
#
# Loss design:
#   Position & Size  → Kalman NLL (Gaussian negative log-likelihood)
#                       jointly optimises x_pred and P_pred through the
#                       innovation covariance S = H@P@H^T + R.
#   Orientation      → Kalman NLL (angle-wrapped) + trace(S) penalty
#                       (prevents Mamba from predicting infinite R to
#                       zero out the Mahalanobis term)
#   Embedding        → InfoNCE (supervised contrastive)
#
# Why NOT MSE:
#   - MSE ignores covariance → Mamba-predicted Q/R receive no gradient
#     through the filter uncertainty pathway.
#   - Angle MSE explodes near ±π due to wrap discontinuity.
# ------------------------------------------------------------------------

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


def _reduce_per_sample(
    per_sample: torch.Tensor,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()


# ======================================================================
# Von Mises Negative Log-Likelihood (Orientation)
# ======================================================================
# Replaces the Gaussian NLL for angular predictions. The Von Mises
# distribution is the natural exponential-family distribution on the
# circle, eliminating the -pi/pi topological tear that causes gradient
# collapse under Gaussian NLL.

def von_mises_loss_per_sample(
    pred_yaw: torch.Tensor,   # [B] or [B, 1]
    gt_yaw: torch.Tensor,     # [B] or [B, 1]
    kappa: torch.Tensor,      # [B, 1] — concentration parameter
) -> torch.Tensor:
    pred = pred_yaw.squeeze(-1)       # [B]
    gt = gt_yaw.squeeze(-1)           # [B]
    k = kappa.squeeze(-1)             # [B]
    k = torch.clamp(k, min=0.3, max=5.0)    # align with TemporalMamba forward clamp ceiling

    diff = pred - gt
    cos_term = -k * torch.cos(diff)                        # [B]
    log_i0e = torch.log(torch.special.i0e(k) + 1e-7)      # [B]
    k_term = k                                              # [B]

    nll_per_sample = cos_term + log_i0e + k_term            # [B]
    return nll_per_sample


def von_mises_loss(
    pred_yaw: torch.Tensor,   # [B] or [B, 1]
    gt_yaw: torch.Tensor,     # [B] or [B, 1]
    kappa: torch.Tensor,      # [B, 1] — concentration parameter
    sample_weights: torch.Tensor = None,  # [B]
) -> torch.Tensor:
    """
    Von Mises negative log-likelihood for angular predictions.

    # VM log-pdf ∝ κ·cos(Δθ) − log(I₀(κ))
    # Using torch.special.i0e (exponentially-scaled Bessel):
    #   I₀(κ) = i0e(κ) · exp(κ)
    #   log I₀(κ) = log(i0e(κ) + ε) + κ
    # Therefore:
    #   NLL = −κ·cos(pred − gt) + log(i0e(κ) + ε) + κ

    torch.special.i0e avoids NaN overflow for large κ (i0e(κ) ≈ 1/√(2πκ)
    for large κ, which is well-behaved in log-space).

    Args:
        pred_yaw : Predicted yaw angle, shape [B] or [B, 1].
        gt_yaw   : Ground-truth yaw angle, shape [B] or [B, 1].
        kappa    : Concentration parameter, shape [B, 1]. Must be > 0.

    Returns:
        Scalar loss (mean over batch).
    """
    return _reduce_per_sample(
        von_mises_loss_per_sample(pred_yaw, gt_yaw, kappa),
        sample_weights=sample_weights,
    )


# ======================================================================
# Kalman Negative Log-Likelihood (NLL)
# ======================================================================

def kalman_nll_per_sample(
    z_gt: torch.Tensor,        # [B, m]      — ground-truth observation
    x_pred: torch.Tensor,      # [B, n, 1]   — predicted state mean
    P_pred: torch.Tensor,      # [B, n, n]   — predicted state covariance
    H: torch.Tensor,           # [m, n]       — observation matrix (constant)
    R_pred: torch.Tensor,      # [B, m, m]   — measurement noise covariance (PSD)
    eps: float = 1e-5,
) -> torch.Tensor:
    B = z_gt.shape[0]
    m = H.shape[0]
    dev = z_gt.device

    # defensive: ensure H is on the same device as the observations
    H = H.to(device=dev)

    # ---- innovation y = z_gt - H @ x_pred  →  [B, m, 1] ----
    z = z_gt.unsqueeze(-1)                                    # [B, m, 1]
    H_batch = H.unsqueeze(0).expand(B, -1, -1)                # [B, m, n]
    predicted_obs = torch.bmm(H_batch, x_pred)                # [B, m, 1]
    y = z - predicted_obs                                     # [B, m, 1]

    # ---- innovation covariance S = H @ P @ H^T + R  →  [B, m, m] ----
    S = torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(-1, -2)) + R_pred

    # ---- regularise: S_reg = S + eps * I ----
    I = torch.eye(m, device=dev, dtype=S.dtype).unsqueeze(0).expand(B, -1, -1)
    S_reg = S + eps * I

    # ---- logdet(S_reg)  →  [B] ----
    logdet = torch.linalg.slogdet(S_reg)[1]                   # log|S_reg|

    # ---- y^T @ inv(S_reg) @ y  (Cholesky solve with fallback) ----
    try:
        L = torch.linalg.cholesky(S_reg)                      # [B, m, m]
        sol = torch.cholesky_solve(y, L)                      # [B, m, 1]
    except RuntimeError:
        # Fallback: increase regularisation and retry
        S_reg = S + (eps * 100) * I
        L = torch.linalg.cholesky(S_reg)
        sol = torch.cholesky_solve(y, L)
    quad_form = torch.bmm(y.transpose(-1, -2), sol).squeeze(-1).squeeze(-1)  # [B]

    # ---- logdet safety guard: fires when eigenvalue < e^{-2} ≈ 0.135.
    # 0.5 coefficient matches the NLL 0.5 prefactor, providing symmetric
    # gradient push-back against logdet → -∞ (size NLL collapse).
    logdet_per_dim = logdet / m
    logdet_guard = 1.0 * F.relu(-logdet_per_dim - 1.5)  # [B]

    # ---- per-sample NLL + guard, then mean over batch ----
    nll_per_sample = 0.5 * (logdet + quad_form) + logdet_guard
    return nll_per_sample


def kalman_nll_loss(
    z_gt: torch.Tensor,        # [B, m]      — ground-truth observation
    x_pred: torch.Tensor,      # [B, n, 1]   — predicted state mean
    P_pred: torch.Tensor,      # [B, n, n]   — predicted state covariance
    H: torch.Tensor,           # [m, n]       — observation matrix (constant)
    R_pred: torch.Tensor,      # [B, m, m]   — measurement noise covariance (PSD)
    eps: float = 1e-5,
    sample_weights: torch.Tensor = None,  # [B]
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood for a Kalman filter prediction.

    # y = z_gt - H @ x_pred                          (innovation / residual)
    # S = H @ P_pred @ H^T + R_pred                   (innovation covariance)
    # NLL = 0.5 * ( logdet(S + eps*I) + y^T @ inv(S + eps*I) @ y )

    Gradients flow through x_pred (state mean) and P_pred (via S into
    Mamba-predicted Q/R), enabling end-to-end optimisation of the full
    Kalman pipeline.

    Args:
        z_gt:    Ground-truth observation vector, shape [B, m].
        x_pred:  Predicted state mean, shape [B, n, 1].
        P_pred:  Predicted state covariance, shape [B, n, n] (must be PSD).
        H:       Observation matrix, shape [m, n] (constant, shared across batch).
        R_pred:  Measurement noise covariance from Mamba, shape [B, m, m] (PSD).
        eps:     Small diagonal regularisation for numerical stability.

    Returns:
        Scalar loss (mean over batch).
    """
    return _reduce_per_sample(
        kalman_nll_per_sample(z_gt, x_pred, P_pred, H, R_pred, eps=eps),
        sample_weights=sample_weights,
    )


def kalman_nis_per_sample(
    z_gt: torch.Tensor,        # [B, m]
    x_pred: torch.Tensor,      # [B, n, 1]
    P_pred: torch.Tensor,      # [B, n, n]
    H: torch.Tensor,           # [m, n]
    R_pred: torch.Tensor,      # [B, m, m]
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Compute per-sample NIS = y^T S^{-1} y (no reduction).
    """
    B = z_gt.shape[0]
    m = H.shape[0]
    dev = z_gt.device
    H = H.to(device=dev)

    z = z_gt.unsqueeze(-1)                                    # [B, m, 1]
    H_batch = H.unsqueeze(0).expand(B, -1, -1)                # [B, m, n]
    y = z - torch.bmm(H_batch, x_pred)                        # [B, m, 1]

    S = torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(-1, -2)) + R_pred
    I = torch.eye(m, device=dev, dtype=S.dtype).unsqueeze(0).expand(B, -1, -1)
    S_reg = S + eps * I

    try:
        L = torch.linalg.cholesky(S_reg)
        sol = torch.cholesky_solve(y, L)
    except RuntimeError:
        S_reg = S + (eps * 100) * I
        L = torch.linalg.cholesky(S_reg)
        sol = torch.cholesky_solve(y, L)

    nis = torch.bmm(y.transpose(-1, -2), sol).squeeze(-1).squeeze(-1)  # [B]
    return nis


def wrap_to_pi_torch(x: torch.Tensor) -> torch.Tensor:
    return x - 2.0 * math.pi * torch.round(x / (2.0 * math.pi))


def wrapped_orientation_nll_per_sample(
    pred_yaw: torch.Tensor,
    gt_yaw: torch.Tensor,
    r_ori: torch.Tensor,
    eps: float = 1e-5,
) -> torch.Tensor:
    pred = pred_yaw.squeeze(-1)
    gt = gt_yaw.squeeze(-1)
    diff = wrap_to_pi_torch(pred - gt)
    var = torch.clamp(r_ori.squeeze(-1).squeeze(-1), min=eps)
    per_sample = 0.5 * (torch.log(var) + diff.pow(2) / var)
    return per_sample


def wrapped_orientation_nll(
    pred_yaw: torch.Tensor,
    gt_yaw: torch.Tensor,
    r_ori: torch.Tensor,
    sample_weights: torch.Tensor = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Wrapped Gaussian NLL on the circle using the runtime R_ori variance.
    """
    return _reduce_per_sample(
        wrapped_orientation_nll_per_sample(pred_yaw, gt_yaw, r_ori, eps=eps),
        sample_weights=sample_weights,
    )


def circular_orientation_state_loss_per_sample(
    pred_yaw: torch.Tensor,
    gt_yaw: torch.Tensor,
) -> torch.Tensor:
    pred = pred_yaw.squeeze(-1)
    gt = gt_yaw.squeeze(-1)
    per_sample = 1.0 - torch.cos(wrap_to_pi_torch(pred - gt))
    return per_sample


def circular_orientation_state_loss(
    pred_yaw: torch.Tensor,
    gt_yaw: torch.Tensor,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    return _reduce_per_sample(
        circular_orientation_state_loss_per_sample(pred_yaw, gt_yaw),
        sample_weights=sample_weights,
    )


def orientation_saturation_penalty(
    kappa_ori_unc: torch.Tensor,
    *,
    max_effective_kappa: float,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    per_sample = F.relu(kappa_ori_unc.squeeze(-1) - float(max_effective_kappa)).pow(2)
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()


def log_ratio_anchor_loss(
    gamma: torch.Tensor,
    sample_weights: torch.Tensor = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalise deviation from the prior ratio gamma=1 in log-space.
    """
    per_sample = torch.abs(torch.log(torch.clamp(gamma.squeeze(-1), min=eps)))
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()


def log_ratio_bound_loss(
    gamma: torch.Tensor,
    min_ratio,
    max_ratio,
    sample_weights: torch.Tensor = None,
    lower_weight: float = 0.25,
    upper_weight: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Penalise log-space bound violations with stronger upper-bound pressure.
    """
    gamma_flat = gamma.squeeze(-1)
    min_ratio_t = torch.as_tensor(min_ratio, device=gamma_flat.device, dtype=gamma_flat.dtype)
    max_ratio_t = torch.as_tensor(max_ratio, device=gamma_flat.device, dtype=gamma_flat.dtype)
    log_gamma = torch.log(torch.clamp(gamma_flat, min=eps))
    log_min = torch.log(torch.clamp(min_ratio_t, min=eps))
    log_max = torch.log(torch.clamp(max_ratio_t, min=eps))
    lower_violation = F.relu(log_min - log_gamma)
    upper_violation = F.relu(log_gamma - log_max)
    per_sample = lower_weight * lower_violation.pow(2) + upper_weight * upper_violation.pow(2)
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()


# ======================================================================
# Smooth Angle Loss (retained for reference; NOT used in training)
# ======================================================================

def angle_loss(pred_yaw: torch.Tensor, gt_yaw: torch.Tensor) -> torch.Tensor:
    """
    Smooth, wrap-safe heading loss.

    # L_ori = 1 - cos(pred - gt)

    Cosine handles the ±π discontinuity naturally — there is no need for
    explicit angle wrapping.  The loss is bounded in [0, 2] and has zero
    gradient at the identity.

    Args:
        pred_yaw: Predicted yaw angle, shape [B].
        gt_yaw:   Ground-truth yaw angle, shape [B].

    Returns:
        Scalar loss (mean over batch).
    """
    diff = pred_yaw - gt_yaw
    return (1.0 - torch.cos(diff)).mean()


# ======================================================================
# State Prediction Loss (NLL for pos/siz, angle loss for ori)
# ======================================================================

class StatePredictionLoss(nn.Module):
    """
    Prediction loss for the three decoupled Kalman filters.

    Position & Size  →  kalman_nll_loss (Gaussian NLL)
    Orientation      →  von_mises_loss (Von Mises NLL, no topological tear)

    All three losses receive gradient through predicted covariance P_pred
    and measurement noise R_pred, giving Mamba's Q/R heads direct signal.

    Degeneracy prevention is handled at the architecture level (min_diag in
    CholeskyHead) with mild logdet guards in the loss as belt-and-suspenders.
    """

    def __init__(
        self,
        w_pos: float = 1.0,
        w_siz: float = 0.5,
        w_ori: float = 1.0,
        w_vel: float = 0.3,
        w_nis: float = 0.0,
    ) -> None:
        super().__init__()
        self.w_pos = w_pos
        self.w_siz = w_siz
        self.w_ori = w_ori
        self.w_vel = w_vel
        self.w_nis = w_nis

        # H_pos: [3, 6] — selects [x, y, z] from 6-dim state for NLL loss.
        # The KF update uses a 5D observation [x,y,z,vx,vy], but the NLL loss
        # only evaluates position prediction quality (velocity is auxiliary).
        self.register_buffer("H_pos", torch.zeros(3, 6))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0

        # H_vel: [2, 6] — selects [vx, vy] from 6-dim state for velocity NLL.
        self.register_buffer("H_vel", torch.zeros(2, 6))
        self.H_vel[0, 3] = 1.0
        self.H_vel[1, 4] = 1.0

        # H_siz: [3, 3] — identity (observation = state for size filter)
        self.register_buffer("H_siz", torch.eye(3))
        self.register_buffer("H_ori", torch.tensor([[1.0, 0.0]]))

    def forward(
        self,
        pos_x_pred: torch.Tensor,    # [B, 6, 1]
        pos_P_pred: torch.Tensor,    # [B, 6, 6]
        siz_x_pred: torch.Tensor,    # [B, 3, 1]
        siz_P_pred: torch.Tensor,    # [B, 3, 3]
        ori_x_pred: torch.Tensor,    # [B, 2, 1]
        ori_P_pred: torch.Tensor,    # [B, 2, 2]  (unused, kept for API compat)
        gt_next_pos: torch.Tensor,   # [B, 3]
        gt_next_siz: torch.Tensor,   # [B, 3]
        gt_next_ori: torch.Tensor,   # [B, 1]
        R_pos: torch.Tensor,         # [B, 5, 5]
        R_siz: torch.Tensor,         # [B, 3, 3]
        R_ori: torch.Tensor,         # [B, 1, 1]
        kappa_ori: torch.Tensor = None,  # [B, 1] — Von Mises concentration
        gt_next_vel: torch.Tensor = None,  # [B, 2] — GT velocity for vel NLL
        in_warmup: bool = False,
        ori_nll_alpha: float = None,       # smooth transition 0(angle) -> 1(VM-NLL)
        class_ids: torch.Tensor = None,     # [B]
        class_weights: torch.Tensor = None,  # [C]
        use_wrapped_orientation_nll: bool = False,
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor (differentiable)
            detail : dict with individual loss values for logging
        """
        sample_weights = None
        if class_ids is not None and class_weights is not None:
            cids = torch.clamp(class_ids, 0, class_weights.shape[0] - 1)
            sample_weights = class_weights[cids]
            sample_weights = sample_weights / (sample_weights.mean() + 1e-8)

        # Position: Kalman NLL (on xyz sub-block)
        loss_pos_per_sample = kalman_nll_per_sample(
            z_gt=gt_next_pos,
            x_pred=pos_x_pred,
            P_pred=pos_P_pred,
            H=self.H_pos,
            R_pred=R_pos[:, :3, :3],  # [B,3,3] position sub-block of [B,5,5] R_pos
        )
        loss_pos = _reduce_per_sample(loss_pos_per_sample, sample_weights=sample_weights)

        # Velocity: Kalman NLL (on vx,vy sub-block).
        # GT velocity is approximated from initial GT (constant-vel assumption
        # over short rollouts). NaN guard prevents one bad batch from crashing training.
        loss_vel_val = 0.0
        loss_vel_per_sample = torch.zeros_like(loss_pos_per_sample)
        if gt_next_vel is not None and self.w_vel > 0:
            loss_vel_candidate_per_sample = kalman_nll_per_sample(
                z_gt=gt_next_vel,
                x_pred=pos_x_pred,
                P_pred=pos_P_pred,
                H=self.H_vel,
                R_pred=R_pos[:, 3:5, 3:5],  # [B,2,2] velocity sub-block
            )
            loss_vel = _reduce_per_sample(loss_vel_candidate_per_sample, sample_weights=sample_weights)
            if not torch.isnan(loss_vel) and not torch.isinf(loss_vel):
                loss_vel_per_sample = loss_vel_candidate_per_sample
                loss_pos = loss_pos + self.w_vel * loss_vel
                loss_vel_val = loss_vel.item()

        # Size: Kalman NLL
        loss_siz_per_sample = kalman_nll_per_sample(
            z_gt=gt_next_siz,
            x_pred=siz_x_pred,
            P_pred=siz_P_pred,
            H=self.H_siz,
            R_pred=R_siz,
        )
        loss_siz = _reduce_per_sample(loss_siz_per_sample, sample_weights=sample_weights)

        # Orientation loss:
        # - legacy path: hard switch by in_warmup
        # - smooth path: ori_nll_alpha in [0,1], blends angle -> Von Mises NLL
        loss_ori_angle_per_sample = circular_orientation_state_loss_per_sample(
            pred_yaw=ori_x_pred[:, 0:1, 0],
            gt_yaw=gt_next_ori,
        )
        loss_ori_angle = _reduce_per_sample(
            loss_ori_angle_per_sample,
            sample_weights=sample_weights,
        )

        loss_ori_wrapped_per_sample = wrapped_orientation_nll_per_sample(
            pred_yaw=ori_x_pred[:, 0:1, 0],
            gt_yaw=gt_next_ori,
            r_ori=R_ori,
        )
        loss_ori_wrapped = _reduce_per_sample(
            loss_ori_wrapped_per_sample,
            sample_weights=sample_weights,
        )

        if use_wrapped_orientation_nll:
            loss_ori_vm = torch.tensor(0.0, device=pos_x_pred.device)
            loss_ori_vm_per_sample = torch.zeros_like(loss_ori_wrapped_per_sample)
            loss_ori = loss_ori_wrapped
            loss_ori_per_sample = loss_ori_wrapped_per_sample
        else:
            loss_ori_vm_per_sample = von_mises_loss_per_sample(
                pred_yaw=ori_x_pred[:, 0, 0],
                gt_yaw=gt_next_ori,
                kappa=kappa_ori,
            )
            loss_ori_vm = _reduce_per_sample(loss_ori_vm_per_sample, sample_weights=sample_weights)
            if ori_nll_alpha is None:
                loss_ori = loss_ori_angle if in_warmup else loss_ori_vm
                loss_ori_per_sample = loss_ori_angle_per_sample if in_warmup else loss_ori_vm_per_sample
            else:
                alpha = float(max(0.0, min(1.0, ori_nll_alpha)))
                loss_ori = (1.0 - alpha) * loss_ori_angle + alpha * loss_ori_vm
                loss_ori_per_sample = (
                    (1.0 - alpha) * loss_ori_angle_per_sample
                    + alpha * loss_ori_vm_per_sample
                )

        loss_nis = torch.tensor(0.0, device=pos_x_pred.device)
        loss_nis_per_sample = torch.zeros_like(loss_pos_per_sample)
        if self.w_nis > 0:
            nis_pos = kalman_nis_per_sample(
                z_gt=gt_next_pos, x_pred=pos_x_pred, P_pred=pos_P_pred,
                H=self.H_pos, R_pred=R_pos[:, :3, :3],
            )  # [B]
            nis_siz = kalman_nis_per_sample(
                z_gt=gt_next_siz, x_pred=siz_x_pred, P_pred=siz_P_pred,
                H=self.H_siz, R_pred=R_siz,
            )  # [B]

            nis_terms = [
                F.smooth_l1_loss(nis_pos / 3.0, torch.ones_like(nis_pos), reduction="none"),
                F.smooth_l1_loss(nis_siz / 3.0, torch.ones_like(nis_siz), reduction="none"),
            ]
            if gt_next_vel is not None:
                nis_vel = kalman_nis_per_sample(
                    z_gt=gt_next_vel, x_pred=pos_x_pred, P_pred=pos_P_pred,
                    H=self.H_vel, R_pred=R_pos[:, 3:5, 3:5],
                )  # [B]
                nis_terms.append(
                    F.smooth_l1_loss(nis_vel / 2.0, torch.ones_like(nis_vel), reduction="none")
                )

            # Orientation NIS (wrap innovation on circle).
            pred_yaw = ori_x_pred[:, 0, 0]
            gt_yaw = gt_next_ori.squeeze(-1)
            wrapped_diff = wrap_to_pi_torch(gt_yaw - pred_yaw)
            z_ori_adj = (pred_yaw + wrapped_diff).unsqueeze(-1)  # [B,1]
            nis_ori = kalman_nis_per_sample(
                z_gt=z_ori_adj, x_pred=ori_x_pred, P_pred=ori_P_pred,
                H=self.H_ori, R_pred=R_ori,
            )  # [B]
            nis_terms.append(
                F.smooth_l1_loss(nis_ori, torch.ones_like(nis_ori), reduction="none")
            )

            nis_stack = torch.stack(nis_terms, dim=0).mean(dim=0)  # [B]
            loss_nis_per_sample = nis_stack
            if sample_weights is None:
                loss_nis = nis_stack.mean()
            else:
                w = sample_weights / (sample_weights.sum() + 1e-8)
                loss_nis = (nis_stack * w).sum()

        loss_state_per_sample = (
            self.w_pos * (loss_pos_per_sample + self.w_vel * loss_vel_per_sample)
            + self.w_siz * loss_siz_per_sample
            + self.w_ori * loss_ori_per_sample
            + self.w_nis * loss_nis_per_sample
        )
        loss = (
            self.w_pos * loss_pos
            + self.w_siz * loss_siz
            + self.w_ori * loss_ori
            + self.w_nis * loss_nis
        )

        detail = {
            "loss_pos": loss_pos.item(),
            "loss_siz": loss_siz.item(),
            "loss_ori": loss_ori.item(),
            "loss_ori_angle": loss_ori_angle.item(),
            "loss_ori_vm": loss_ori_vm.item(),
            "loss_ori_wrapped": loss_ori_wrapped.item(),
            "loss_vel": loss_vel_val,
            "loss_nis": loss_nis.item(),
            "loss_ori_state_tensor": loss_ori_angle,
            "loss_ori_wrapped_tensor": loss_ori_wrapped,
            "loss_ori_vm_tensor": loss_ori_vm,
            "loss_ori_tensor": loss_ori,
        }
        if return_per_sample:
            detail.update({
                "loss_state_per_sample": loss_state_per_sample,
                "_loss_pos_per_sample": loss_pos_per_sample,
                "_loss_vel_per_sample": loss_vel_per_sample,
                "_loss_siz_per_sample": loss_siz_per_sample,
                "_loss_ori_state_per_sample": loss_ori_angle_per_sample,
                "_loss_ori_wrapped_per_sample": loss_ori_wrapped_per_sample,
                "_loss_ori_vm_per_sample": loss_ori_vm_per_sample,
                "_loss_ori_per_sample": loss_ori_per_sample,
                "_loss_nis_per_sample": loss_nis_per_sample,
            })
        return loss, detail


# ======================================================================
# InfoNCE Contrastive Loss (unchanged)
# ======================================================================

class InfoNCELoss(nn.Module):
    """
    Supervised contrastive loss (InfoNCE) on Mamba embeddings.

    Within a batch, samples from the same instance_token are positives,
    different instance_tokens are negatives.

    L = -log( sum(exp(sim(i,p)/τ)) / sum(exp(sim(i,j)/τ)) )
    averaged over all anchors that have at least one positive.
    """

    def __init__(self, temperature: float = 0.07, hard_negative_topk: int = 0) -> None:
        super().__init__()
        self.temperature = temperature
        self.hard_negative_topk = hard_negative_topk

    def forward(
        self,
        embeddings: torch.Tensor,       # [B, D]
        instance_tokens: list,           # list of B strings
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        B = embeddings.shape[0]
        device = embeddings.device

        if B < 2:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"loss_contrastive": 0.0, "n_valid_anchors": 0}

        emb_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D]

        sim_matrix = torch.mm(emb_norm, emb_norm.t()) / self.temperature  # [B, B]

        token_to_id = {}
        labels = []
        for tok in instance_tokens:
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
            labels.append(token_to_id[tok])
        labels = torch.tensor(labels, device=device)  # [B]

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, B]
        pos_mask.fill_diagonal_(False)

        has_positive = pos_mask.any(dim=1)  # [B]
        n_valid = has_positive.sum().item()

        if n_valid == 0:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"loss_contrastive": 0.0, "n_valid_anchors": 0}

        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_stable = sim_matrix - sim_max.detach()

        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        exp_pos = torch.exp(sim_stable) * pos_mask.float()             # [B, B]
        pos_sum = exp_pos.sum(dim=1)                                   # [B]

        neg_mask = self_mask & (~pos_mask)
        exp_neg = torch.exp(sim_stable) * neg_mask.float()             # [B, B]
        if self.hard_negative_topk > 0:
            neg_logits = sim_stable.masked_fill(~neg_mask, float("-inf"))
            k = min(self.hard_negative_topk, B - 1)
            topk_vals, _ = torch.topk(neg_logits, k=k, dim=1)
            topk_vals = topk_vals.masked_fill(torch.isinf(topk_vals), float("-inf"))
            hard_neg_sum = torch.exp(topk_vals).sum(dim=1)             # [B]
            denom = pos_sum + hard_neg_sum + 1e-8
        else:
            denom = pos_sum + exp_neg.sum(dim=1) + 1e-8

        loss_per_anchor = -torch.log(pos_sum / denom + 1e-8)  # [B]
        loss = loss_per_anchor[has_positive].mean()

        detail = {
            "loss_contrastive": loss.item(),
            "n_valid_anchors": n_valid,
        }
        return loss, detail


# ======================================================================
# Joint Loss
# ======================================================================

class JointLoss(nn.Module):
    """
    Combined training loss:
        L_total = L_state + λ * L_contrastive

    where:
        L_state       = w_pos * NLL_pos + w_siz * NLL_siz + w_ori * NLL_yaw
        L_contrastive = InfoNCE on Mamba embeddings
    """

    def __init__(
        self,
        w_pos: float = 1.0,
        w_siz: float = 0.5,
        w_ori: float = 1.0,
        w_vel: float = 0.3,
        w_nis: float = 0.0,
        lambda_contrast: float = 0.1,
        temperature: float = 0.07,
        physics_scale: float = 50.0,
        hard_negative_topk: int = 0,
        class_weights: list = None,
        residual_supervision: dict = None,
    ) -> None:
        super().__init__()
        self.state_loss = StatePredictionLoss(w_pos, w_siz, w_ori, w_vel, w_nis)
        self.contrastive_loss = InfoNCELoss(temperature, hard_negative_topk=hard_negative_topk)
        self.lambda_contrast = lambda_contrast
        self.physics_scale = physics_scale
        self.class_weights = class_weights
        self.residual_supervision = dict(residual_supervision or {})

    def forward(
        self,
        pos_x_pred: torch.Tensor,    # [B, 6, 1]
        pos_P_pred: torch.Tensor,    # [B, 6, 6]
        siz_x_pred: torch.Tensor,    # [B, 3, 1]
        siz_P_pred: torch.Tensor,    # [B, 3, 3]
        ori_x_pred: torch.Tensor,    # [B, 2, 1]
        ori_P_pred: torch.Tensor,    # [B, 2, 2]
        gt_next_pos: torch.Tensor,   # [B, 3]
        gt_next_siz: torch.Tensor,   # [B, 3]
        gt_next_ori: torch.Tensor,   # [B, 1]
        embeddings: torch.Tensor,    # [B, D]
        instance_tokens: list,        # list of B strings
        R_pos: torch.Tensor,          # [B, 5, 5]
        R_siz: torch.Tensor,          # [B, 3, 3]
        R_ori: torch.Tensor,          # [B, 1, 1]
        kappa_ori: torch.Tensor = None,  # [B, 1]
        gt_next_vel: torch.Tensor = None,  # [B, 2]
        in_warmup: bool = False,
        ori_nll_alpha: float = None,
        class_ids: torch.Tensor = None,  # [B]
        use_wrapped_orientation_nll: bool = False,
        return_per_sample: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor for backward()
            detail : dict with all sub-loss values
        """
        class_weights_tensor = None
        if self.class_weights is not None:
            class_weights_tensor = torch.tensor(
                self.class_weights, device=pos_x_pred.device, dtype=pos_x_pred.dtype,
            )

        loss_state, detail_state = self.state_loss(
            pos_x_pred, pos_P_pred,
            siz_x_pred, siz_P_pred,
            ori_x_pred, ori_P_pred,
            gt_next_pos, gt_next_siz, gt_next_ori,
            R_pos, R_siz, R_ori,
            kappa_ori=kappa_ori,
            gt_next_vel=gt_next_vel,
            in_warmup=in_warmup,
            ori_nll_alpha=ori_nll_alpha,
            class_ids=class_ids,
            class_weights=class_weights_tensor,
            use_wrapped_orientation_nll=use_wrapped_orientation_nll,
            return_per_sample=return_per_sample,
        )

        # Contrastive loss only on step 0 (embeddings not None).
        # Rollout steps k>0 share the same Mamba embedding — computing
        # contrastive again would double-count without adding signal.
        if embeddings is not None and instance_tokens is not None:
            loss_contrast, detail_contrast = self.contrastive_loss(
                embeddings, instance_tokens,
            )
        else:
            loss_contrast = torch.tensor(0.0, device=pos_x_pred.device, requires_grad=True)
            detail_contrast = {"loss_contrastive": 0.0, "n_valid_anchors": 0}

        loss_total = self.physics_scale * loss_state + self.lambda_contrast * loss_contrast

        detail = {
            **detail_state,
            **detail_contrast,
            "loss_state": loss_state.item(),
            "loss_total": loss_total.item(),
        }
        if return_per_sample:
            detail["loss_total_per_sample"] = (
                self.physics_scale * detail_state["loss_state_per_sample"]
            )
        return loss_total, detail
