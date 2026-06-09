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
