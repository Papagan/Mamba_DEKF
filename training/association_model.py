from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor


class ClassConditionedAssociationHeadBank(nn.Module):
    """Route pair features to an independent binary association head per class."""

    def __init__(
        self,
        *,
        input_dim: int,
        num_classes: int = 7,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_classes = int(num_classes)
        self.heads = nn.ModuleList([
            self._make_head(self.input_dim, hidden_dims, dropout)
            for _ in range(self.num_classes)
        ])

    @staticmethod
    def _make_head(input_dim: int, hidden_dims: Sequence[int], dropout: float) -> nn.Sequential:
        layers = []
        prev = int(input_dim)
        for hidden in hidden_dims:
            hidden = int(hidden)
            layers.extend([
                nn.Linear(prev, hidden),
                nn.LayerNorm(hidden),
                nn.SiLU(),
            ])
            if float(dropout) > 0.0:
                layers.append(nn.Dropout(float(dropout)))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        return nn.Sequential(*layers)

    def forward(self, pair_features: Tensor, class_ids: Tensor) -> Tensor:
        if pair_features.ndim != 2:
            raise ValueError("pair_features must have shape [B, D]")
        if pair_features.shape[1] != self.input_dim:
            raise ValueError(f"pair feature dim mismatch: got {pair_features.shape[1]}, expected {self.input_dim}")

        class_ids = torch.clamp(class_ids.to(device=pair_features.device, dtype=torch.long), 0, self.num_classes - 1)
        logits = pair_features.new_zeros(pair_features.shape[0])
        for class_id in torch.unique(class_ids).tolist():
            mask = class_ids == int(class_id)
            logits[mask] = self.heads[int(class_id)](pair_features[mask]).squeeze(-1)
        return logits


class PairwiseAssociationModel(nn.Module):
    """Shared temporal encoder plus class-conditioned association heads."""

    def __init__(
        self,
        temporal_encoder: nn.Module,
        *,
        embed_dim: int,
        pair_feature_dim: int,
        num_classes: int = 7,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.1,
        encoder_mode: str = "mamba_multihead_closure",
    ) -> None:
        super().__init__()
        self.temporal_encoder = temporal_encoder
        self.embed_dim = int(embed_dim)
        self.pair_feature_dim = int(pair_feature_dim)
        self.encoder_mode = str(encoder_mode)
        head_input_dim = self.embed_dim * 4 + self.pair_feature_dim
        self.association_heads = ClassConditionedAssociationHeadBank(
            input_dim=head_input_dim,
            num_classes=num_classes,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )

    def encode(self, history: Tensor, class_ids: Tensor, state_buckets=None) -> Tensor:
        current_range = torch.linalg.norm(history[:, -1, 0:2], dim=-1)
        history_mask = history.abs().sum(dim=-1) > 0
        out = self.temporal_encoder(
            history,
            class_ids=class_ids,
            current_range=current_range,
            detection_driven_mask=history_mask[:, -1],
            history_mask=history_mask,
            history_match_mask=history_mask,
            state_buckets=state_buckets,
            mode=self.encoder_mode,
        )
        return out["embedding"]

    def forward(
        self,
        anchor_history: Tensor,
        candidate_history: Tensor,
        pair_features: Tensor,
        class_ids: Tensor,
        state_buckets=None,
    ) -> Tensor:
        track_embedding = self.encode(anchor_history, class_ids, state_buckets=state_buckets)
        det_embedding = self.encode(candidate_history, class_ids, state_buckets=["matched"] * int(class_ids.shape[0]))
        pair = torch.cat(
            [
                track_embedding,
                det_embedding,
                torch.abs(track_embedding - det_embedding),
                track_embedding * det_embedding,
                pair_features,
            ],
            dim=-1,
        )
        return self.association_heads(pair, class_ids)
