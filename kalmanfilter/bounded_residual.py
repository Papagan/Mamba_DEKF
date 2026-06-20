from __future__ import annotations

from typing import Optional

try:
    import torch
except ImportError:
    torch = None


STATE_MATCHED = "matched"
STATE_UNMATCHED = "unmatched"

PROFILE_STABLE_LARGE = "stable_large"
PROFILE_AGILE_WEAK = "agile_weak"
PROFILE_HEAVY_LONG = "heavy_long"
PROFILE_HUMAN = "human"

_CLASS_TO_PROFILE = {
    0: PROFILE_STABLE_LARGE,
    4: PROFILE_STABLE_LARGE,
    2: PROFILE_AGILE_WEAK,
    3: PROFILE_AGILE_WEAK,
    5: PROFILE_HEAVY_LONG,
    6: PROFILE_HEAVY_LONG,
    1: PROFILE_HUMAN,
}


def map_class_to_profile(class_id: int) -> Optional[str]:
    return _CLASS_TO_PROFILE.get(int(class_id))


def infer_state_bucket(unmatch_length: int) -> str:
    return STATE_MATCHED if int(unmatch_length) == 0 else STATE_UNMATCHED


def clamp_ratio_value(value: float, *, min_ratio: float, max_ratio: float) -> float:
    return max(min_ratio, min(max_ratio, float(value)))


def get_family_ratio_bounds(
    class_id: int,
    state_bucket: str,
    family_name: str,
    closure_cfg: Optional[dict],
):
    if not closure_cfg or not bool(closure_cfg.get("ENABLED", False)):
        return None

    profiles = closure_cfg.get("PROFILES") or {}
    profile_name = map_class_to_profile(int(class_id))
    if profile_name is None:
        return None

    profile_cfg = profiles.get(profile_name) or {}
    state_cfg = profile_cfg.get(str(state_bucket)) or {}
    bounds = state_cfg.get(family_name)
    if not isinstance(bounds, (list, tuple)) or len(bounds) != 2:
        return None
    return float(bounds[0]), float(bounds[1])


def _covariance_trace_batch(cov):
    return cov.diagonal(dim1=-2, dim2=-1).sum(-1)


def apply_bounded_residuals(
    *,
    raw_tensors: dict,
    prior_tensors: dict,
    class_ids,
    state_buckets,
    closure_cfg: Optional[dict],
):
    bounded = dict(raw_tensors)
    if not closure_cfg or not bool(closure_cfg.get("ENABLED", False)):
        return bounded
    if torch is None:
        raise RuntimeError("apply_bounded_residuals requires torch")
    if class_ids is None or state_buckets is None:
        return bounded

    class_id_list = [
        int(value.item()) if hasattr(value, "item") else int(value)
        for value in class_ids
    ]
    state_bucket_list = [str(bucket) for bucket in state_buckets]
    if len(class_id_list) != len(state_bucket_list):
        raise ValueError("class_ids and state_buckets must have the same batch length")

    family_by_tensor = {
        "Q_pos": "q_pos",
        "R_pos": "r_pos",
        "R_siz": "r_siz",
        "R_ori": "r_ori",
    }

    for tensor_name, family_name in family_by_tensor.items():
        raw_cov = raw_tensors.get(tensor_name)
        prior_cov = prior_tensors.get(tensor_name)
        if raw_cov is None or prior_cov is None:
            continue

        min_ratios = []
        max_ratios = []
        bound_mask = []
        has_bounded_sample = False
        for class_id, state_bucket in zip(class_id_list, state_bucket_list):
            bounds = get_family_ratio_bounds(class_id, state_bucket, family_name, closure_cfg)
            if bounds is None:
                min_ratios.append(0.0)
                max_ratios.append(float("inf"))
                bound_mask.append(False)
                continue
            has_bounded_sample = True
            min_ratios.append(bounds[0])
            max_ratios.append(bounds[1])
            bound_mask.append(True)

        if not has_bounded_sample:
            continue

        min_ratio_t = torch.tensor(min_ratios, device=raw_cov.device, dtype=raw_cov.dtype)
        max_ratio_t = torch.tensor(max_ratios, device=raw_cov.device, dtype=raw_cov.dtype)
        bound_mask_t = torch.tensor(bound_mask, device=raw_cov.device, dtype=raw_cov.dtype)
        raw_trace = _covariance_trace_batch(raw_cov)
        prior_trace = _covariance_trace_batch(prior_cov)
        eps = torch.tensor(1e-8, device=raw_cov.device, dtype=raw_cov.dtype)

        safe_prior_trace = torch.clamp(prior_trace, min=eps)
        raw_ratio = raw_trace / safe_prior_trace
        raw_valid = torch.isfinite(raw_ratio) & (torch.abs(raw_ratio) > eps)
        prior_valid = torch.isfinite(prior_trace) & (prior_trace > eps)
        ratio_for_clamp = torch.where(raw_valid, raw_ratio, max_ratio_t)
        bounded_ratio = torch.maximum(min_ratio_t, torch.minimum(max_ratio_t, ratio_for_clamp))
        has_bounds = bound_mask_t > eps
        use_scaled_raw = has_bounds & raw_valid & prior_valid

        scale = bounded_ratio / torch.where(raw_valid, raw_ratio, torch.ones_like(raw_ratio))
        scaled_raw = raw_cov * scale.view(-1, 1, 1)
        fallback = prior_cov * bounded_ratio.view(-1, 1, 1)
        bounded_candidate = torch.where(
            use_scaled_raw.view(-1, 1, 1),
            scaled_raw,
            fallback,
        )
        bounded[tensor_name] = torch.where(
            has_bounds.view(-1, 1, 1),
            bounded_candidate,
            raw_cov,
        )

    return bounded
