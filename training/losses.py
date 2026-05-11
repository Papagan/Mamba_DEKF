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


# ======================================================================
# Von Mises Negative Log-Likelihood (Orientation)
# ======================================================================
# Replaces the Gaussian NLL for angular predictions. The Von Mises
# distribution is the natural exponential-family distribution on the
# circle, eliminating the -pi/pi topological tear that causes gradient
# collapse under Gaussian NLL.

def von_mises_loss(
    pred_yaw: torch.Tensor,   # [B] or [B, 1]
    gt_yaw: torch.Tensor,     # [B] or [B, 1]
    kappa: torch.Tensor,      # [B, 1] — concentration parameter
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
    pred = pred_yaw.squeeze(-1)       # [B]
    gt = gt_yaw.squeeze(-1)           # [B]
    k = kappa.squeeze(-1)             # [B]

    diff = pred - gt
    cos_term = -k * torch.cos(diff)                        # [B]
    log_i0e = torch.log(torch.special.i0e(k) + 1e-7)      # [B]
    k_term = k                                              # [B]

    nll_per_sample = cos_term + log_i0e + k_term            # [B]
    return nll_per_sample.mean()


# ======================================================================
# Kalman Negative Log-Likelihood (NLL)
# ======================================================================

def kalman_nll_loss(
    z_gt: torch.Tensor,        # [B, m]      — ground-truth observation
    x_pred: torch.Tensor,      # [B, n, 1]   — predicted state mean
    P_pred: torch.Tensor,      # [B, n, n]   — predicted state covariance
    H: torch.Tensor,           # [m, n]       — observation matrix (constant)
    R_pred: torch.Tensor,      # [B, m, m]   — measurement noise covariance (PSD)
    eps: float = 1e-5,
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

    # ---- y^T @ inv(S_reg) @ y  (solve avoids explicit inverse) ----
    sol = torch.linalg.solve(S_reg, y)                        # [B, m, 1]
    quad_form = torch.bmm(y.transpose(-1, -2), sol).squeeze(-1).squeeze(-1)  # [B]

    # ---- logdet safety guard (belt-and-suspenders) ----
    # min_diag in CholeskyHead already prevents R → 0. This penalty only
    # fires when logdet(S) per observation-dimension falls below -5, which
    # corresponds to mean eigenvalue < e^{-5} ≈ 0.007. In healthy operation
    # with min_diag = 0.1, eigenvalues are ≥ 0.01, so this is a no-op.
    logdet_per_dim = logdet / m
    logdet_guard = 0.01 * F.relu(-logdet_per_dim - 5.0)  # [B]

    # ---- per-sample NLL + guard, then mean over batch ----
    nll_per_sample = 0.5 * (logdet + quad_form) + logdet_guard
    return nll_per_sample.mean()


# ======================================================================
# Kalman NLL for Yaw (Orientation)
# ======================================================================

def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to [-pi, pi]."""
    return angles - 2.0 * math.pi * torch.round(angles / (2.0 * math.pi))


def kalman_nll_loss_yaw(
    z_gt: torch.Tensor,        # [B, 1]      — ground-truth yaw
    x_pred: torch.Tensor,      # [B, 2, 1]   — predicted state [theta, omega]
    P_pred: torch.Tensor,      # [B, 2, 2]   — predicted state covariance
    H: torch.Tensor,           # [1, 2]       — observation matrix (selects theta)
    R_pred: torch.Tensor,      # [B, 1, 1]   — measurement noise covariance (PSD)
    logdet_guard_weight: float = 0.1,
    eps: float = 1e-5,
) -> torch.Tensor:
    """
    Gaussian NLL specialised for orientation (yaw) angle.

    Key differences from generic kalman_nll_loss:

    1. Residual y is wrapped to [-pi, pi] so that angular differences
       are geometrically correct (e.g., π - (-π) = 0, not 2π).
       # y = wrap_to_pi(z_gt - H @ x_pred)

    2. A logdet guard prevents the "cheating" solution where Mamba
       predicts infinite R to drive the Mahalanobis term to zero.
       # penalty = λ * max(0, -logdet(S) - 2.0)

       Unlike the old trace penalty (λ * trace(S)), this does NOT
       push R toward zero — it only activates when the innovation
       covariance S is pathologically large, which is the anti-cheat
       case. The min_diag constraint in CholeskyHead already prevents
       R → 0, so a separate lower-bound penalty is unnecessary.

    Args:
        z_gt:               Ground-truth yaw, shape [B, 1].
        x_pred:             Predicted state [theta, omega], shape [B, 2, 1].
        P_pred:             Predicted state covariance, shape [B, 2, 2].
        H:                  Observation matrix [[1, 0]], shape [1, 2].
        R_pred:             Measurement noise from Mamba, shape [B, 1, 1].
        logdet_guard_weight: Weight λ for the anti-cheat logdet guard.
        eps:                Small diagonal regularisation.

    Returns:
        Scalar loss (mean over batch).
    """
    B = z_gt.shape[0]
    m = H.shape[0]          # = 1 for yaw
    dev = z_gt.device

    H = H.to(device=dev)

    # ---- innovation  y = wrap_to_pi(z_gt - H @ x_pred)  → [B, 1, 1] ----
    z = z_gt.unsqueeze(-1)                                    # [B, 1, 1]
    H_batch = H.unsqueeze(0).expand(B, -1, -1)                # [B, 1, 2]
    predicted_obs = torch.bmm(H_batch, x_pred)                # [B, 1, 1]
    y_raw = z - predicted_obs                                  # [B, 1, 1]
    y = wrap_to_pi(y_raw)                                      # [B, 1, 1]

    # ---- innovation covariance  S = H @ P @ H^T + R  → [B, 1, 1] ----
    S = torch.bmm(torch.bmm(H_batch, P_pred), H_batch.transpose(-1, -2)) + R_pred

    # ---- regularise: S_reg = S + eps * I ----
    I = torch.eye(m, device=dev, dtype=S.dtype).unsqueeze(0).expand(B, -1, -1)
    S_reg = S + eps * I

    # ---- logdet(S_reg) → [B] ----
    logdet = torch.linalg.slogdet(S_reg)[1]

    # ---- y^T @ inv(S_reg) @ y  (solve avoids explicit inverse) ----
    sol = torch.linalg.solve(S_reg, y)                        # [B, 1, 1]
    quad_form = torch.bmm(y.transpose(-1, -2), sol).squeeze(-1).squeeze(-1)  # [B]

    # ---- logdet guard: prevents Mamba from predicting infinite R ----
    # Only penalises pathologically LARGE S (logdet → +∞), never pushes R toward zero.
    # Gradient is zero when logdet(S) < 2.0 (the vast majority of healthy cases).
    logdet_penalty = logdet_guard_weight * F.relu(logdet - 2.0)   # [B]

    # per-sample NLL + anti-cheat guard, mean over batch
    nll_per_sample = 0.5 * (logdet + quad_form) + logdet_penalty
    return nll_per_sample.mean()


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
        w_ori: float = 50.0,
    ) -> None:
        super().__init__()
        self.w_pos = w_pos
        self.w_siz = w_siz
        self.w_ori = w_ori

        # H_pos: [3, 6] — selects [x, y, z] from 6-dim position state
        self.register_buffer("H_pos", torch.zeros(3, 6))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0

        # H_siz: [3, 3] — identity (observation = state for size filter)
        self.register_buffer("H_siz", torch.eye(3))

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
        R_pos: torch.Tensor,         # [B, 3, 3]
        R_siz: torch.Tensor,         # [B, 3, 3]
        R_ori: torch.Tensor,         # [B, 1, 1]  (unused, kept for API compat)
        kappa_ori: torch.Tensor = None,  # [B, 1] — Von Mises concentration
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor (differentiable)
            detail : dict with individual loss values for logging
        """
        # Position: Kalman NLL
        loss_pos = kalman_nll_loss(
            z_gt=gt_next_pos,
            x_pred=pos_x_pred,
            P_pred=pos_P_pred,
            H=self.H_pos,
            R_pred=R_pos,
        )

        # Size: Kalman NLL
        loss_siz = kalman_nll_loss(
            z_gt=gt_next_siz,
            x_pred=siz_x_pred,
            P_pred=siz_P_pred,
            H=self.H_siz,
            R_pred=R_siz,
        )

        # Orientation: Von Mises NLL (no topological tear at -pi/pi)
        # pred_yaw = ori_x_pred[:, 0, 0] extracts theta from [theta, omega]
        loss_ori = von_mises_loss(
            pred_yaw=ori_x_pred[:, 0, 0],
            gt_yaw=gt_next_ori,
            kappa=kappa_ori,
        )

        loss = self.w_pos * loss_pos + self.w_siz * loss_siz + self.w_ori * loss_ori

        detail = {
            "loss_pos": loss_pos.item(),
            "loss_siz": loss_siz.item(),
            "loss_ori": loss_ori.item(),
        }
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

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

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
        exp_sim = torch.exp(sim_stable) * self_mask.float()
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)

        exp_pos = torch.exp(sim_stable) * pos_mask.float()
        log_pos_sum = torch.log(exp_pos.sum(dim=1) + 1e-8)

        loss_per_anchor = log_denom - log_pos_sum  # [B]
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
        w_ori: float = 50.0,
        lambda_contrast: float = 0.1,
        temperature: float = 0.07,
        physics_scale: float = 50.0,
    ) -> None:
        super().__init__()
        self.state_loss = StatePredictionLoss(w_pos, w_siz, w_ori)
        self.contrastive_loss = InfoNCELoss(temperature)
        self.lambda_contrast = lambda_contrast
        self.physics_scale = physics_scale

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
        R_pos: torch.Tensor,          # [B, 3, 3]
        R_siz: torch.Tensor,          # [B, 3, 3]
        R_ori: torch.Tensor,          # [B, 1, 1]
        kappa_ori: torch.Tensor = None,  # [B, 1]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor for backward()
            detail : dict with all sub-loss values
        """
        loss_state, detail_state = self.state_loss(
            pos_x_pred, pos_P_pred,
            siz_x_pred, siz_P_pred,
            ori_x_pred, ori_P_pred,
            gt_next_pos, gt_next_siz, gt_next_ori,
            R_pos, R_siz, R_ori,
            kappa_ori=kappa_ori,
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
        return loss_total, detail
