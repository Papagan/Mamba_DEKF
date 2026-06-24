from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
from torch import Tensor


def _build_diag_ratio(
    batch_size: int,
    *,
    device,
    dtype,
    groups: Iterable[Tuple[int, int, Tensor]],
) -> Tensor:
    pieces = []
    for start, end, ratio in groups:
        width = end - start
        pieces.append(ratio.view(batch_size, 1).expand(batch_size, width))
    return torch.cat(pieces, dim=1)


def scale_prior_covariance(base_cov: Tensor, diag_ratio: Tensor) -> Tensor:
    std_ratio = torch.sqrt(torch.clamp(diag_ratio, min=0.0))
    scaled = base_cov * std_ratio.unsqueeze(-1) * std_ratio.unsqueeze(-2)
    return 0.5 * (scaled + scaled.transpose(-1, -2))


def apply_factorized_ratio_to_q_pos(base_cov: Tensor, ratios: Dict[str, Tensor]) -> Tensor:
    diag_ratio = _build_diag_ratio(
        base_cov.shape[0],
        device=base_cov.device,
        dtype=base_cov.dtype,
        groups=[
            (0, 3, ratios["q_pos_xyz"]),
            (3, 6, ratios["q_pos_vxyz"]),
        ],
    )
    return scale_prior_covariance(base_cov, diag_ratio)


def apply_factorized_ratio_to_r_pos(base_cov: Tensor, ratios: Dict[str, Tensor]) -> Tensor:
    diag_ratio = _build_diag_ratio(
        base_cov.shape[0],
        device=base_cov.device,
        dtype=base_cov.dtype,
        groups=[
            (0, 3, ratios["r_pos_xyz"]),
            (3, 5, ratios["r_pos_vxy"]),
        ],
    )
    return scale_prior_covariance(base_cov, diag_ratio)


def apply_factorized_ratio_to_r_siz(base_cov: Tensor, ratios: Dict[str, Tensor]) -> Tensor:
    diag_ratio = _build_diag_ratio(
        base_cov.shape[0],
        device=base_cov.device,
        dtype=base_cov.dtype,
        groups=[
            (0, 2, ratios["r_siz_lw"]),
            (2, 3, ratios["r_siz_h"]),
        ],
    )
    return scale_prior_covariance(base_cov, diag_ratio)


def apply_factorized_ratio_to_r_ori(base_cov: Tensor, ratios: Dict[str, Tensor]) -> Tensor:
    return scale_prior_covariance(base_cov, ratios["r_ori"].view(base_cov.shape[0], 1))


class BoundedRatioHead(nn.Module):
    def __init__(self, d_model: int, out_dim: int = 1, max_ratio: float = 4.0) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)
        self.log_limit = nn.Parameter(torch.full((out_dim,), math.log(float(max_ratio))))
        nn.init.uniform_(self.proj.weight, -1e-4, 1e-4)
        nn.init.constant_(self.proj.bias, 0.0)

    def forward(self, h: Tensor) -> Tensor:
        raw = self.proj(h)
        limit = torch.clamp(self.log_limit, min=0.0).view(1, -1)
        return torch.exp(limit * torch.tanh(raw))


class PriorConditionedHeadBank(nn.Module):
    def __init__(self, d_model: int, num_classes: int) -> None:
        super().__init__()
        self.num_classes = int(num_classes)
        family_max_ratio = {
            "q_pos_xyz": 4.0,
            "q_pos_vxyz": 4.0,
            "r_pos_xyz": 4.0,
            "r_pos_vxy": 4.0,
            "r_siz_lw": 3.0,
            "r_siz_h": 3.0,
            "r_ori": 4.0,
        }
        self.ratio_limits = {
            name: (1.0 / max_ratio, max_ratio)
            for name, max_ratio in family_max_ratio.items()
        }
        self.family_heads = nn.ModuleDict(
            {
                name: nn.ModuleList(
                    [BoundedRatioHead(d_model=d_model, out_dim=1, max_ratio=max_ratio) for _ in range(self.num_classes)]
                )
                for name, max_ratio in family_max_ratio.items()
            }
        )

    def forward(self, h: Tensor, class_ids: Tensor) -> Dict[str, Tensor]:
        if class_ids is None:
            class_ids = torch.zeros(h.shape[0], dtype=torch.long, device=h.device)
        class_ids = torch.clamp(class_ids.to(device=h.device, dtype=torch.long), 0, self.num_classes - 1)

        outputs = {
            name: h.new_zeros((h.shape[0], 1))
            for name in self.family_heads.keys()
        }
        unique_classes = torch.unique(class_ids).tolist()
        for class_id in unique_classes:
            mask = class_ids == int(class_id)
            class_h = h[mask]
            for name, heads in self.family_heads.items():
                outputs[name][mask] = heads[int(class_id)](class_h)
        return outputs

