import numpy as np


def normalize_tracker_compat_mode(mode) -> str:
    if mode is None:
        return "default"
    return str(mode).strip().lower()


def initial_status_flag_for_mode(mode) -> int:
    mode = normalize_tracker_compat_mode(mode)
    return 1 if mode == "mctrack" else 0


def score_for_unmatched_fake_bbox(last_real_score: float, unmatch_length: int, mode) -> float:
    mode = normalize_tracker_compat_mode(mode)
    if mode == "mctrack":
        return 0.0
    return float(last_real_score) * (0.8 ** int(unmatch_length))


def select_output_tracking_score(
    current_score: float,
    real_scores,
    quality_scores,
    compat_mode,
) -> float:
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    if compat_mode == "mctrack":
        return float(current_score)
    if quality_scores:
        return float(sum(quality_scores) / len(quality_scores))
    if real_scores:
        return float(max(real_scores))
    return float(current_score)


def select_filtered_tracking_score(
    compat_mode,
    original_scores,
    transformed_scores,
    quality_scores,
    fallback_score: float,
) -> float:
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    if compat_mode == "mctrack":
        detected_scores = [float(score) for score in transformed_scores if score > -10000]
        if detected_scores:
            return float(sum(detected_scores) / (len(detected_scores) + 1e-5))
        return float(fallback_score)
    if quality_scores:
        return float(sum(quality_scores) / len(quality_scores))
    if original_scores:
        return float(fallback_score)
    return float(fallback_score)


def allow_single_stage_birth_under_mode(compat_mode, gate_allowed: bool) -> bool:
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    if compat_mode == "mctrack":
        return True
    return bool(gate_allowed)


def sync_bbox_fields_from_state(
    bbox,
    state,
    *,
    update_fusion: bool = True,
    update_predict: bool = False,
) -> None:
    state_list = [float(v) for v in state]
    bbox.global_xyz = state_list[:3]
    bbox.lwh = state_list[3:6]
    bbox.global_yaw = state_list[6]
    bbox.global_xyz_lwh_yaw = list(state_list)
    if update_fusion:
        bbox.global_xyz_lwh_yaw_fusion = np.array(state_list)
        bbox.lwh_fusion = list(state_list[3:6])
        bbox.global_yaw_fusion = float(state_list[6])
    if update_predict:
        bbox.global_xyz_lwh_yaw_predict = list(state_list)
