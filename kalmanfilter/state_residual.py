from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


def wrap_to_pi(values: Tensor) -> Tensor:
    return values - 2.0 * math.pi * torch.round(values / (2.0 * math.pi))


def _normalize_active_class_states(values: Optional[dict]) -> Dict[int, set]:
    out: Dict[int, set] = {}
    for key, states in (values or {}).items():
        class_id = int(key)
        out[class_id] = {str(state).strip().lower() for state in (states or [])}
    return out


def _bound_for(cfg: dict, class_id: int, state_bucket: str, name: str) -> float:
    default_bounds = cfg.get("DEFAULT_BOUNDS") or {}
    class_bounds = cfg.get("CLASS_BOUNDS") or {}
    class_cfg = class_bounds.get(class_id, class_bounds.get(str(class_id), {})) or {}
    state_cfg = class_cfg.get(str(state_bucket), {}) or {}
    return float(state_cfg.get(name, default_bounds.get(name, 0.0)))


def _active_mask(class_ids: Tensor, state_buckets: Sequence[str], cfg: dict) -> Tensor:
    active_cfg = _normalize_active_class_states(cfg.get("ACTIVE_CLASS_STATES") or {})
    if not active_cfg:
        return torch.zeros(class_ids.shape[0], device=class_ids.device, dtype=torch.bool)

    values = []
    for class_id, state in zip(class_ids.detach().cpu().tolist(), state_buckets):
        active_states = active_cfg.get(int(class_id), set())
        values.append(str(state).strip().lower() in active_states)
    return torch.tensor(values, device=class_ids.device, dtype=torch.bool)


def apply_bounded_state_residuals(
    pos_x: Tensor,
    siz_x: Tensor,
    ori_x: Tensor,
    residual: Optional[Tensor],
    *,
    class_ids: Optional[Tensor],
    state_buckets: Optional[Sequence[str]],
    cfg: Optional[dict],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    cfg = cfg or {}
    batch = pos_x.shape[0]
    device = pos_x.device
    inactive = torch.zeros(batch, device=device, dtype=torch.bool)
    if (
        not bool(cfg.get("ENABLED", False))
        or residual is None
        or class_ids is None
        or state_buckets is None
    ):
        return pos_x, siz_x, ori_x, inactive

    if len(state_buckets) != batch:
        raise ValueError("state_buckets length must match batch size")

    residual = residual.to(device=device, dtype=pos_x.dtype)
    if residual.shape[0] != batch or residual.shape[1] < 6:
        raise ValueError("state residual must have shape [B, >=6]")

    class_ids = class_ids.to(device=device, dtype=torch.long)
    mask = _active_mask(class_ids, state_buckets, cfg)
    if not bool(mask.any().item()):
        return pos_x, siz_x, ori_x, mask

    out_pos = pos_x.clone()
    out_siz = siz_x.clone()
    out_ori = ori_x.clone()

    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        class_id = int(class_ids[idx].item())
        state_bucket = str(state_buckets[idx]).strip().lower()
        pos_xy_bound = _bound_for(cfg, class_id, state_bucket, "POS_XY")
        pos_z_bound = _bound_for(cfg, class_id, state_bucket, "POS_Z")
        vel_xy_bound = _bound_for(cfg, class_id, state_bucket, "VEL_XY")
        yaw_bound = _bound_for(cfg, class_id, state_bucket, "YAW")

        out_pos[idx, 0:2, 0] = out_pos[idx, 0:2, 0] + torch.clamp(
            residual[idx, 0:2], -pos_xy_bound, pos_xy_bound
        )
        out_pos[idx, 2, 0] = out_pos[idx, 2, 0] + torch.clamp(
            residual[idx, 2], -pos_z_bound, pos_z_bound
        )
        out_pos[idx, 3:5, 0] = out_pos[idx, 3:5, 0] + torch.clamp(
            residual[idx, 3:5], -vel_xy_bound, vel_xy_bound
        )
        out_ori[idx, 0, 0] = wrap_to_pi(
            out_ori[idx, 0, 0]
            + torch.clamp(residual[idx, 5], -yaw_bound, yaw_bound)
        )

    return out_pos, out_siz, out_ori, mask
