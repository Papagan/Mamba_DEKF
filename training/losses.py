# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — Training Losses
#
# Loss design:
#   Position & Size  → Kalman NLL (Gaussian negative log-likelihood)
#                       jointly optimises x_pred and P_pred through the
#                       innovation covariance S = H@P@H^T + R.
#   Orientation      → 1 - cos(pred - gt)   (smooth, wrap-safe)
#   Embedding        → InfoNCE (supervised contrastive)
#
# Why NOT MSE:
#   - MSE ignores covariance → Mamba-predicted Q/R receive no gradient
#     through the filter uncertainty pathway.
#   - Angle MSE explodes near ±π due to wrap discontinuity.
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple


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

    # ---- per-sample NLL, then mean over batch ----
    nll_per_sample = 0.5 * (logdet + quad_form)
    return nll_per_sample.mean()


# ======================================================================
# Smooth Angle Loss
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

    Position & Size  →  kalman_nll_loss (X, P, H, R)
    Orientation      →  angle_loss (1 - cos(Δθ))

    The NLL jointly optimises state mean and covariance, giving Mamba's
    Q/R prediction heads a direct gradient through the filter uncertainty.
    """

    def __init__(
        self,
        w_pos: float = 1.0,
        w_siz: float = 0.5,
        w_ori: float = 0.5,
    ) -> None:
        super().__init__()
        self.w_pos = w_pos
        self.w_siz = w_siz
        self.w_ori = w_ori

        # H_pos: [3, 8] — selects [x, y, z] from 8-dim position state
        self.register_buffer("H_pos", torch.zeros(3, 8))
        self.H_pos[0, 0] = 1.0
        self.H_pos[1, 1] = 1.0
        self.H_pos[2, 2] = 1.0

        # H_siz: [3, 3] — identity (observation = state for size filter)
        self.register_buffer("H_siz", torch.eye(3))

    def forward(
        self,
        pos_x_pred: torch.Tensor,    # [B, 8, 1]
        pos_P_pred: torch.Tensor,    # [B, 8, 8]
        siz_x_pred: torch.Tensor,    # [B, 3, 1]
        siz_P_pred: torch.Tensor,    # [B, 3, 3]
        ori_x_pred: torch.Tensor,    # [B, 2, 1]
        gt_next_pos: torch.Tensor,   # [B, 3]
        gt_next_siz: torch.Tensor,   # [B, 3]
        gt_next_ori: torch.Tensor,   # [B, 1]
        R_pos: torch.Tensor,         # [B, 3, 3]
        R_siz: torch.Tensor,         # [B, 3, 3]
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

        # Orientation: smooth angle loss (1 - cos(Δθ))
        pred_yaw = ori_x_pred[:, 0, 0]        # [B]
        gt_yaw = gt_next_ori[:, 0]             # [B]
        loss_ori = angle_loss(pred_yaw, gt_yaw)

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
        L_state       = w_pos * NLL_pos + w_siz * NLL_siz + w_ori * AngleLoss
        L_contrastive = InfoNCE on Mamba embeddings
    """

    def __init__(
        self,
        w_pos: float = 1.0,
        w_siz: float = 0.5,
        w_ori: float = 0.5,
        lambda_contrast: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.state_loss = StatePredictionLoss(w_pos, w_siz, w_ori)
        self.contrastive_loss = InfoNCELoss(temperature)
        self.lambda_contrast = lambda_contrast

    def forward(
        self,
        pos_x_pred: torch.Tensor,    # [B, 8, 1]
        pos_P_pred: torch.Tensor,    # [B, 8, 8]
        siz_x_pred: torch.Tensor,    # [B, 3, 1]
        siz_P_pred: torch.Tensor,    # [B, 3, 3]
        ori_x_pred: torch.Tensor,    # [B, 2, 1]
        gt_next_pos: torch.Tensor,   # [B, 3]
        gt_next_siz: torch.Tensor,   # [B, 3]
        gt_next_ori: torch.Tensor,   # [B, 1]
        embeddings: torch.Tensor,    # [B, D]
        instance_tokens: list,        # list of B strings
        R_pos: torch.Tensor,          # [B, 3, 3]
        R_siz: torch.Tensor,          # [B, 3, 3]
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor for backward()
            detail : dict with all sub-loss values
        """
        loss_state, detail_state = self.state_loss(
            pos_x_pred, pos_P_pred,
            siz_x_pred, siz_P_pred,
            ori_x_pred,
            gt_next_pos, gt_next_siz, gt_next_ori,
            R_pos, R_siz,
        )

        loss_contrast, detail_contrast = self.contrastive_loss(
            embeddings, instance_tokens,
        )

        loss_total = loss_state + self.lambda_contrast * loss_contrast

        detail = {
            **detail_state,
            **detail_contrast,
            "loss_state": loss_state.item(),
            "loss_total": loss_total.item(),
        }
        return loss_total, detail
