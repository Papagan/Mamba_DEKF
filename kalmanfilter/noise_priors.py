from __future__ import annotations

import torch
from torch import Tensor
from typing import Dict, Optional


TRACKING_CAT_ID_MAP = {
    "car": 0,
    "pedestrian": 1,
    "bicycle": 2,
    "motorcycle": 3,
    "bus": 4,
    "trailer": 5,
    "truck": 6,
}


def category_to_tracking_name(category: str) -> str:
    if not isinstance(category, str):
        return "car"
    if category in TRACKING_CAT_ID_MAP:
        return category
    suffix = category.split(".")[-1]
    if suffix in TRACKING_CAT_ID_MAP:
        return suffix
    if "pedestrian" in category:
        return "pedestrian"
    return "car"


def categories_to_class_ids(categories, device: Optional[torch.device] = None) -> Tensor:
    return torch.tensor(
        [TRACKING_CAT_ID_MAP.get(category_to_tracking_name(c), 0) for c in categories],
        dtype=torch.long,
        device=device,
    )


def _as_std_xy_map(cfg: dict, key_xy: str, key_scalar: str) -> list:
    vals = cfg.get(key_xy, None)
    if isinstance(vals, list) and len(vals) > 0 and isinstance(vals[0], (list, tuple)):
        return [[float(v[0]), float(v[1])] for v in vals]
    sc = cfg.get(key_scalar, None)
    if isinstance(sc, list) and len(sc) > 0:
        return [[float(v), float(v)] for v in sc]
    raise KeyError(f"Missing {key_xy}/{key_scalar} in noise config.")


def _as_std_1d_map(cfg: dict, key: str, default: float = 1.0, n: int = 7) -> list:
    vals = cfg.get(key, None)
    if isinstance(vals, list) and len(vals) > 0:
        return [float(v) for v in vals]
    return [float(default)] * n


def _as_std_optional_1d_map(cfg: dict, key: str, n: int) -> Optional[list]:
    vals = cfg.get(key, None)
    if isinstance(vals, list) and len(vals) == n:
        return [float(v) for v in vals]
    return None


def _history_valid_mask(track_history: Tensor) -> Tensor:
    return track_history.abs().sum(dim=-1) > 1e-6


def _last_valid_features(track_history: Tensor) -> Tensor:
    valid = _history_valid_mask(track_history)
    B, T, D = track_history.shape
    rev_idx = torch.argmax(valid.flip(dims=[1]).float(), dim=1)
    last_idx = (T - 1 - rev_idx).clamp(min=0)
    batch_idx = torch.arange(B, device=track_history.device)
    feats = track_history[batch_idx, last_idx]
    no_valid = ~valid.any(dim=1)
    if no_valid.any():
        feats = feats.clone()
        feats[no_valid] = 0.0
    return feats


def compute_condition_scales(
    track_history: Tensor,
    cond_cfg: Optional[dict],
    current_range: Optional[Tensor] = None,
    detection_driven_mask: Optional[Tensor] = None,
    history_mask: Optional[Tensor] = None,
    history_match_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    B = track_history.shape[0]
    device = track_history.device
    dtype = track_history.dtype

    one = torch.ones(B, device=device, dtype=dtype)
    if not cond_cfg or not bool(cond_cfg.get("ENABLED", False)):
        return {"pos": one, "siz": one, "vel": one, "ori": one, "valid_ratio": one}

    if history_mask is None:
        valid = _history_valid_mask(track_history)
    else:
        valid = history_mask.to(device=device)
        if valid.dtype != torch.bool:
            valid = valid > 0

    if history_match_mask is None:
        matched = _history_valid_mask(track_history)
    else:
        matched = history_match_mask.to(device=device)
        if matched.dtype != torch.bool:
            matched = matched > 0

    valid_ratio = valid.float().mean(dim=1).to(dtype=dtype)
    matched_ratio = matched.float().sum(dim=1) / torch.clamp(valid.float().sum(dim=1), min=1.0)
    last_feat = _last_valid_features(track_history)

    score = last_feat[:, 11].clamp(0.0, 1.0)
    speed = torch.linalg.norm(last_feat[:, 3:5], dim=1)
    yaw_rate = last_feat[:, 10].abs()

    if current_range is None:
        range_norm = torch.zeros(B, device=device, dtype=dtype)
    else:
        range_ref = float(cond_cfg.get("RANGE_REF", 30.0))
        range_norm = torch.clamp(current_range.to(device=device, dtype=dtype) / max(range_ref, 1e-6), min=0.0, max=3.0)

    speed_ref = float(cond_cfg.get("SPEED_REF", 15.0))
    yaw_rate_ref = float(cond_cfg.get("YAW_RATE_REF", 0.5))
    speed_norm = torch.clamp(speed / max(speed_ref, 1e-6), min=0.0, max=3.0)
    yaw_rate_norm = torch.clamp(yaw_rate / max(yaw_rate_ref, 1e-6), min=0.0, max=3.0)
    miss_ratio = 1.0 - matched_ratio

    score_w = float(cond_cfg.get("SCORE_WEIGHT", 0.8))
    miss_w = float(cond_cfg.get("MISS_WEIGHT", 0.8))
    range_w = float(cond_cfg.get("RANGE_WEIGHT", 0.3))
    speed_w = float(cond_cfg.get("SPEED_WEIGHT", 0.2))
    yaw_rate_w = float(cond_cfg.get("YAW_RATE_WEIGHT", 0.2))
    size_score_w = float(cond_cfg.get("SIZE_SCORE_WEIGHT", score_w * 0.5))
    size_miss_w = float(cond_cfg.get("SIZE_MISS_WEIGHT", miss_w * 0.25))

    pos_scale = 1.0 + score_w * (1.0 - score) + miss_w * miss_ratio + range_w * range_norm
    vel_scale = 1.0 + score_w * (1.0 - score) + miss_w * miss_ratio + speed_w * speed_norm
    ori_scale = 1.0 + score_w * (1.0 - score) + miss_w * miss_ratio + yaw_rate_w * yaw_rate_norm
    siz_scale = 1.0 + size_score_w * (1.0 - score) + size_miss_w * miss_ratio

    min_scale = float(cond_cfg.get("MIN_SCALE", 0.75))
    max_scale = float(cond_cfg.get("MAX_SCALE", 2.5))
    pos_scale = pos_scale.clamp(min=min_scale, max=max_scale)
    vel_scale = vel_scale.clamp(min=min_scale, max=max_scale)
    ori_scale = ori_scale.clamp(min=min_scale, max=max_scale)
    siz_scale = siz_scale.clamp(min=min_scale, max=max_scale)

    if detection_driven_mask is not None:
        dd = detection_driven_mask.to(device=device)
        if dd.dtype != torch.bool:
            dd = dd > 0
        dd = dd.to(dtype=dtype)
        pos_scale = one + (pos_scale - one) * dd
        vel_scale = one + (vel_scale - one) * dd
        ori_scale = one + (ori_scale - one) * dd
        siz_scale = one + (siz_scale - one) * dd
        valid_ratio = one + (valid_ratio - one) * dd

    return {
        "pos": pos_scale,
        "siz": siz_scale,
        "vel": vel_scale,
        "ori": ori_scale,
        "valid_ratio": valid_ratio,
        "matched_ratio": matched_ratio,
    }


def gather_base_noise_stats(
    r_cfg: dict,
    class_ids: Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, Tensor]:
    pos_std_xy_map = torch.tensor(_as_std_xy_map(r_cfg, "POS_STD_XY", "POS_STD"), device=device, dtype=dtype)
    siz_std_lw_map = torch.tensor(_as_std_xy_map(r_cfg, "SIZ_STD_LW", "SIZ_STD"), device=device, dtype=dtype)
    vel_std_xy_map = torch.tensor(_as_std_xy_map(r_cfg, "VEL_STD_XY", "VEL_STD"), device=device, dtype=dtype)
    ori_std_map = torch.tensor(
        _as_std_1d_map(r_cfg, "ORI_STD", default=0.5, n=pos_std_xy_map.shape[0]),
        device=device,
        dtype=dtype,
    )
    pos_std_z_map = _as_std_optional_1d_map(r_cfg, "POS_STD_Z", pos_std_xy_map.shape[0])
    siz_std_h_map = _as_std_optional_1d_map(r_cfg, "SIZ_STD_H", siz_std_lw_map.shape[0])

    max_cat = int(min(
        pos_std_xy_map.shape[0] - 1,
        siz_std_lw_map.shape[0] - 1,
        vel_std_xy_map.shape[0] - 1,
        ori_std_map.shape[0] - 1,
    ))
    safe_ids = torch.clamp(class_ids, min=0, max=max_cat)

    pos_std_xy = pos_std_xy_map[safe_ids]
    siz_std_lw = siz_std_lw_map[safe_ids]
    vel_std_xy = vel_std_xy_map[safe_ids]
    ori_std = ori_std_map[safe_ids]

    if pos_std_z_map is None:
        pos_std_z = pos_std_xy.mean(dim=1, keepdim=True)
    else:
        pos_std_z = torch.tensor(pos_std_z_map, device=device, dtype=dtype)[safe_ids].unsqueeze(1)

    if siz_std_h_map is None:
        siz_std_h = siz_std_lw.mean(dim=1, keepdim=True)
    else:
        siz_std_h = torch.tensor(siz_std_h_map, device=device, dtype=dtype)[safe_ids].unsqueeze(1)

    return {
        "pos_std_xyz": torch.cat([pos_std_xy, pos_std_z], dim=1),
        "siz_std_lwh": torch.cat([siz_std_lw, siz_std_h], dim=1),
        "vel_std_xy": vel_std_xy,
        "ori_std": ori_std,
    }


def build_base_covariances(
    base_noise_cfg: dict,
    class_ids: Tensor,
    dtype: torch.dtype,
    device: torch.device,
    track_history: Optional[Tensor] = None,
    current_range: Optional[Tensor] = None,
    detection_driven_mask: Optional[Tensor] = None,
    history_mask: Optional[Tensor] = None,
    history_match_mask: Optional[Tensor] = None,
) -> Dict[str, Tensor]:
    q_cfg = base_noise_cfg.get("Q", {})
    r_cfg = base_noise_cfg.get("R", {})
    cond_cfg = base_noise_cfg.get("CONDITIONAL_NOISE", {})
    bsize = int(class_ids.shape[0])

    stats = gather_base_noise_stats(r_cfg, class_ids, dtype=dtype, device=device)
    if track_history is None:
        scales = {
            "pos": torch.ones(bsize, device=device, dtype=dtype),
            "siz": torch.ones(bsize, device=device, dtype=dtype),
            "vel": torch.ones(bsize, device=device, dtype=dtype),
            "ori": torch.ones(bsize, device=device, dtype=dtype),
            "valid_ratio": torch.ones(bsize, device=device, dtype=dtype),
        }
    else:
        scales = compute_condition_scales(
            track_history=track_history,
            cond_cfg=cond_cfg,
            current_range=current_range,
            detection_driven_mask=detection_driven_mask,
            history_mask=history_mask,
            history_match_mask=history_match_mask,
        )

    pos_std_xyz = stats["pos_std_xyz"] * scales["pos"].unsqueeze(1)
    siz_std_lwh = stats["siz_std_lwh"] * scales["siz"].unsqueeze(1)
    vel_std_xy = stats["vel_std_xy"] * scales["vel"].unsqueeze(1)
    ori_std = stats["ori_std"] * scales["ori"]

    pos_per_cat = q_cfg.get("POS_PER_CAT", None)
    if class_ids is not None and isinstance(pos_per_cat, dict) and len(pos_per_cat) > 0:
        max_key = max(int(k) for k in pos_per_cat.keys())
        c_ids = torch.clamp(class_ids, 0, max_key)
        q_map_list = [pos_per_cat.get(i, q_cfg.get("POS", [0.5] * 6)) for i in range(max_key + 1)]
        q_pos_diag = torch.tensor(q_map_list, device=device, dtype=dtype)[c_ids]
    else:
        q_pos_diag = torch.tensor(q_cfg.get("POS", [0.5] * 6), device=device, dtype=dtype).view(1, 6).expand(bsize, 6).clone()

    q_pos_diag = q_pos_diag.clone()
    q_pos_diag[:, 0:3] *= scales["pos"].unsqueeze(1).pow(2)
    q_pos_diag[:, 3:5] *= scales["vel"].unsqueeze(1).pow(2)
    q_pos_diag[:, 5:6] *= scales["pos"].unsqueeze(1).pow(2)

    q_siz_diag = torch.tensor(q_cfg.get("SIZ", [0.05, 0.05, 0.05]), device=device, dtype=dtype).view(1, 3).expand(bsize, 3).clone()
    q_siz_diag *= scales["siz"].unsqueeze(1).pow(2)

    q_ori_base = float(q_cfg.get("ORI", [0.1])[0])
    q_ori_diag = torch.stack([
        torch.full((bsize,), q_ori_base, device=device, dtype=dtype),
        torch.full((bsize,), 0.5, device=device, dtype=dtype),
    ], dim=1)
    q_ori_diag *= scales["ori"].unsqueeze(1).pow(2)

    r_pos_diag = torch.cat([pos_std_xyz.pow(2), vel_std_xy.pow(2)], dim=1)
    r_siz_diag = siz_std_lwh.pow(2)
    r_ori_diag = ori_std.pow(2).unsqueeze(1)

    return {
        "Q_pos_base": torch.diag_embed(q_pos_diag),
        "Q_siz_base": torch.diag_embed(q_siz_diag),
        "Q_ori_base": torch.diag_embed(q_ori_diag),
        "R_pos_base": torch.diag_embed(r_pos_diag),
        "R_siz_base": torch.diag_embed(r_siz_diag),
        "R_ori_base": torch.diag_embed(r_ori_diag),
        "pos_std_xyz": pos_std_xyz,
        "siz_std_lwh": siz_std_lwh,
        "vel_std_xy": vel_std_xy,
        "ori_std": ori_std,
        "scales": scales,
    }


def apply_residual_anchor(
    pred_cov: Tensor,
    base_cov: Tensor,
    min_ratio: float,
    max_ratio: float,
) -> Tensor:
    eps = 1e-8
    diag_pred = torch.clamp(pred_cov.diagonal(dim1=-2, dim2=-1), min=eps)
    diag_base = torch.clamp(base_cov.diagonal(dim1=-2, dim2=-1), min=eps)

    std_pred = torch.sqrt(diag_pred)
    std_base = torch.sqrt(diag_base)
    ratio = torch.clamp(std_pred / torch.clamp(std_base, min=eps), min=min_ratio, max=max_ratio)
    std_new = std_base * ratio

    outer_pred = torch.bmm(std_pred.unsqueeze(-1), std_pred.unsqueeze(1))
    corr = pred_cov / torch.clamp(outer_pred, min=eps)
    D = pred_cov.shape[-1]
    eye = torch.eye(D, device=pred_cov.device, dtype=pred_cov.dtype).unsqueeze(0)
    corr = corr * (1.0 - eye) + eye

    outer_new = torch.bmm(std_new.unsqueeze(-1), std_new.unsqueeze(1))
    cov_new = corr * outer_new
    cov_new = 0.5 * (cov_new + cov_new.transpose(-1, -2))
    cov_new = cov_new + 1e-6 * eye
    return cov_new
