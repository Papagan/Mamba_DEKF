from __future__ import annotations

import torch


def build_future_detection_history(
    *,
    obs_future_pos,
    obs_future_siz,
    obs_future_ori,
    obs_future_score=None,
    obs_future_match,
    class_ids,
    history_len: int,
    step_index: int = 0,
):
    batch = obs_future_pos.shape[0]
    device = obs_future_pos.device
    dtype = obs_future_pos.dtype
    det_history = torch.zeros(batch, history_len, 12, device=device, dtype=dtype)
    det_history_mask = torch.zeros(batch, history_len, device=device, dtype=torch.bool)
    det_history_match_mask = torch.zeros(batch, history_len, device=device, dtype=torch.bool)
    match_mask = obs_future_match[:, step_index].to(device=device, dtype=torch.bool)
    det_history[:, -1, 0:3] = obs_future_pos[:, step_index, 0:3]
    det_history[:, -1, 3:5] = obs_future_pos[:, step_index, 3:5]
    det_history[:, -1, 6:9] = obs_future_siz[:, step_index, :]
    det_history[:, -1, 9] = obs_future_ori[:, step_index, 0]
    if obs_future_score is not None:
        det_history[:, -1, 11] = obs_future_score[:, step_index].to(device=device, dtype=dtype)
    else:
        det_history[:, -1, 11] = match_mask.to(dtype=dtype)
    det_history_mask[:, -1] = match_mask
    det_history_match_mask[:, -1] = match_mask
    det_range = torch.linalg.norm(obs_future_pos[:, step_index, 0:2], dim=-1)
    return {
        "history": det_history,
        "history_mask": det_history_mask,
        "history_match_mask": det_history_match_mask,
        "match_mask": match_mask,
        "class_ids": class_ids,
        "current_range": det_range,
    }
