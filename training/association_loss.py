from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor


def association_ranking_loss(
    track_embeddings: Tensor,
    det_embeddings: Tensor,
    class_ids: Tensor,
    match_mask: Tensor,
    *,
    margin: float = 0.2,
    hard_negative_topk: int = 4,
) -> Tuple[Tensor, Dict[str, float]]:
    """Rank each matched future detection above same-class batch negatives."""
    device = track_embeddings.device
    zero = torch.zeros((), device=device, dtype=track_embeddings.dtype)
    if (
        track_embeddings.numel() == 0
        or det_embeddings.numel() == 0
        or track_embeddings.shape != det_embeddings.shape
    ):
        return zero, {"loss_association": 0.0, "association_valid_anchors": 0}

    class_ids = class_ids.to(device=device, dtype=torch.long)
    match_mask = match_mask.to(device=device, dtype=torch.bool)
    trk = F.normalize(track_embeddings, p=2, dim=-1)
    det = F.normalize(det_embeddings, p=2, dim=-1)
    sim = trk @ det.transpose(0, 1)

    losses = []
    batch = int(sim.shape[0])
    for idx in range(batch):
        if not bool(match_mask[idx].item()):
            continue
        same_class = class_ids == class_ids[idx]
        negative_mask = same_class & match_mask
        negative_mask[idx] = False
        if not bool(negative_mask.any().item()):
            continue

        pos_sim = sim[idx, idx]
        neg_sims = sim[idx][negative_mask]
        if hard_negative_topk > 0 and neg_sims.numel() > hard_negative_topk:
            neg_sims = torch.topk(neg_sims, k=int(hard_negative_topk), largest=True).values
        losses.append(F.relu(float(margin) - pos_sim + neg_sims).mean())

    if not losses:
        return zero, {"loss_association": 0.0, "association_valid_anchors": 0}

    loss = torch.stack(losses).mean()
    return loss, {
        "loss_association": float(loss.detach().item()),
        "association_valid_anchors": len(losses),
    }
