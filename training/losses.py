# ------------------------------------------------------------------------
# Mamba-Decoupled-EKF Track — Training
# Loss Functions: State Prediction MSE + InfoNCE Contrastive + Joint Loss
#
# Gradient flow:
#   MSE → x_pred → F @ x + Q → TemporalMamba weights
#   InfoNCE → embedding → TemporalMamba weights
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


def wrap_to_pi(angle: torch.Tensor) -> torch.Tensor:
    """Wrap angle tensor to [-pi, pi] (differentiable-friendly)."""
    return angle - 2.0 * torch.pi * torch.round(angle / (2.0 * torch.pi))


class StatePredictionLoss(nn.Module):
    """
    MSE loss between KF-predicted state and GT next-frame state.

    Three independent terms for the three decoupled filters:
      L_pos = MSE(pred_xyz, gt_xyz)           — position [x, y, z]
      L_siz = MSE(pred_lwh, gt_lwh)           — size [l, w, h]
      L_ori = mean(wrap(pred_θ - gt_θ)^2)     — orientation (angle-wrapped)

    L_state = w_pos * L_pos + w_siz * L_siz + w_ori * L_ori
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

    def forward(
        self,
        pos_x_pred: torch.Tensor,   # [B, 8, 1] — predicted position state
        siz_x_pred: torch.Tensor,   # [B, 3, 1] — predicted size state
        ori_x_pred: torch.Tensor,   # [B, 2, 1] — predicted orientation state
        gt_next_pos: torch.Tensor,  # [B, 3]    — GT position [x, y, z]
        gt_next_siz: torch.Tensor,  # [B, 3]    — GT size [l, w, h]
        gt_next_ori: torch.Tensor,  # [B, 1]    — GT yaw
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor (differentiable)
            detail : dict with individual loss values for logging
        """
        # Position: compare predicted [x, y, z] (first 3 of 8-dim state)
        pred_pos = pos_x_pred[:, :3, 0]  # [B, 3]
        loss_pos = F.mse_loss(pred_pos, gt_next_pos)

        # Size: compare predicted [l, w, h]
        pred_siz = siz_x_pred[:, :, 0]   # [B, 3]
        loss_siz = F.mse_loss(pred_siz, gt_next_siz)

        # Orientation: angle-wrapped MSE
        pred_ori = ori_x_pred[:, 0, 0]   # [B]
        gt_ori = gt_next_ori[:, 0]        # [B]
        angle_diff = wrap_to_pi(pred_ori - gt_ori)
        loss_ori = (angle_diff ** 2).mean()

        loss = self.w_pos * loss_pos + self.w_siz * loss_siz + self.w_ori * loss_ori

        detail = {
            "loss_pos": loss_pos.item(),
            "loss_siz": loss_siz.item(),
            "loss_ori": loss_ori.item(),
        }
        return loss, detail


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
        """
        Returns:
            loss   : scalar tensor
            detail : dict with loss value and stats
        """
        B = embeddings.shape[0]
        device = embeddings.device

        if B < 2:
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"loss_contrastive": 0.0, "n_valid_anchors": 0}

        # L2-normalize embeddings
        emb_norm = F.normalize(embeddings, p=2, dim=1)  # [B, D]

        # Pairwise cosine similarity / temperature
        sim_matrix = torch.mm(emb_norm, emb_norm.t()) / self.temperature  # [B, B]

        # Build positive mask: same instance_token
        # Convert tokens to integer labels for fast comparison
        token_to_id = {}
        labels = []
        for tok in instance_tokens:
            if tok not in token_to_id:
                token_to_id[tok] = len(token_to_id)
            labels.append(token_to_id[tok])
        labels = torch.tensor(labels, device=device)  # [B]

        pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # [B, B]
        pos_mask.fill_diagonal_(False)  # exclude self

        # Check if any anchor has at least one positive
        has_positive = pos_mask.any(dim=1)  # [B]
        n_valid = has_positive.sum().item()

        if n_valid == 0:
            # No positive pairs in this batch — return zero loss
            zero = torch.tensor(0.0, device=device, requires_grad=True)
            return zero, {"loss_contrastive": 0.0, "n_valid_anchors": 0}

        # For numerical stability, subtract max per row
        sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
        sim_stable = sim_matrix - sim_max.detach()

        # Denominator: sum over all j != i
        self_mask = ~torch.eye(B, dtype=torch.bool, device=device)
        exp_sim = torch.exp(sim_stable) * self_mask.float()  # [B, B]
        log_denom = torch.log(exp_sim.sum(dim=1) + 1e-8)     # [B]

        # Numerator: mean log-sum-exp over positives for each anchor
        # For each anchor i with positives P(i):
        #   loss_i = -mean_{p in P(i)} [ sim(i,p) - log(sum_j exp(sim(i,j))) ]
        exp_pos = torch.exp(sim_stable) * pos_mask.float()    # [B, B]
        log_pos_sum = torch.log(exp_pos.sum(dim=1) + 1e-8)    # [B]

        loss_per_anchor = log_denom - log_pos_sum  # [B]

        # Average only over anchors that have positives
        loss = loss_per_anchor[has_positive].mean()

        detail = {
            "loss_contrastive": loss.item(),
            "n_valid_anchors": n_valid,
        }
        return loss, detail


class JointLoss(nn.Module):
    """
    Combined training loss:
        L_total = L_state + λ * L_contrastive

    where:
        L_state       = w_pos * MSE_pos + w_siz * MSE_siz + w_ori * MSE_ori
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
        pos_x_pred: torch.Tensor,   # [B, 8, 1]
        siz_x_pred: torch.Tensor,   # [B, 3, 1]
        ori_x_pred: torch.Tensor,   # [B, 2, 1]
        gt_next_pos: torch.Tensor,  # [B, 3]
        gt_next_siz: torch.Tensor,  # [B, 3]
        gt_next_ori: torch.Tensor,  # [B, 1]
        embeddings: torch.Tensor,   # [B, D]
        instance_tokens: list,       # list of B strings
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns:
            loss   : scalar tensor for backward()
            detail : dict with all sub-loss values
        """
        loss_state, detail_state = self.state_loss(
            pos_x_pred, siz_x_pred, ori_x_pred,
            gt_next_pos, gt_next_siz, gt_next_ori,
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
