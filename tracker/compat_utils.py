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


def _per_class_float(cfg_dict, category_num: int, default: float) -> float:
    if isinstance(cfg_dict, dict):
        return float(cfg_dict.get(category_num, default))
    return float(default)


def compute_track_quality_score(
    traj,
    *,
    raw_scores=None,
    current_score: float | None = None,
) -> float | None:
    """
    Heuristic trajectory-quality score in [0, 1].

    This upgrades tracking_score from simple detection-score aggregation to a
    track-level confidence estimate using:
      - real detection confidence
      - association quality
      - fake/coast ratio
      - low-score rescue ratio
      - trajectory maturity

    The score is intentionally post-hoc and does not require retraining.
    """
    cfg = getattr(traj, "cfg", {}) or {}
    score_cfg = cfg.get("TRACK_SCORE", {}) or {}
    if not bool(score_cfg.get("ENABLED", False)):
        return None
    if str(score_cfg.get("MODE", "quality_v1")).strip().lower() != "quality_v1":
        return None

    bboxes = list(getattr(traj, "bboxes", []) or [])
    if not bboxes:
        if current_score is None:
            return None
        return float(np.clip(current_score, 0.0, 1.0))

    if raw_scores is None:
        raw_scores = [
            float(getattr(b, "raw_det_score", getattr(b, "det_score", 0.0)))
            for b in bboxes
        ]

    raw_scores = [float(np.clip(s, 0.0, 1.0)) for s in raw_scores]
    current_bbox = bboxes[-1]
    category_num = int(getattr(traj, "category_num", 0))
    current_score = (
        float(np.clip(current_score, 0.0, 1.0))
        if current_score is not None
        else float(raw_scores[-1] if raw_scores else 0.0)
    )

    real_scores = []
    recent_real_scores = []
    fake_count = 0
    low_score_real = 0
    matched_costs = []
    good_match_count = 0
    output_score_thre = float(getattr(traj, "_output_score", 0.4))
    match_score_thre = max(float(getattr(traj, "_confirmed_match_score", 0.35)), 1e-3)

    for bbox, raw_score in zip(bboxes, raw_scores):
        if getattr(bbox, "is_fake", False):
            fake_count += 1
            continue
        real_scores.append(raw_score)
        recent_real_scores.append(raw_score)
        if getattr(bbox, "is_low_score_match", False) or raw_score < output_score_thre:
            low_score_real += 1

        matched_score = getattr(bbox, "matched_score", None)
        if matched_score is None:
            continue
        matched_score = float(matched_score)
        if not np.isfinite(matched_score) or matched_score <= 0.0:
            continue
        matched_costs.append(matched_score)
        if matched_score <= match_score_thre:
            good_match_count += 1

    if not real_scores:
        return current_score

    topk = max(1, int(score_cfg.get("REAL_SCORE_TOPK", 5)))
    recent_window = max(1, int(score_cfg.get("RECENT_WINDOW", 5)))
    mature_default = max(3, int(getattr(traj, "_confirmed_track_length", 1)) + 2)
    mature_len = max(
        1,
        int((score_cfg.get("MATURE_LEN", {}) or {}).get(category_num, mature_default)),
    )
    current_fake_scale = _per_class_float(
        score_cfg.get("CURRENT_FAKE_SCALE", {}),
        category_num,
        float(score_cfg.get("DEFAULT_CURRENT_FAKE_SCALE", 0.75)),
    )

    topk_mean = float(np.mean(sorted(real_scores, reverse=True)[:topk]))
    recent_mean = float(np.mean(recent_real_scores[-recent_window:]))
    det_conf = 0.6 * topk_mean + 0.4 * recent_mean

    real_count = len(real_scores)
    total_count = max(len(bboxes), 1)
    real_hit_ratio = real_count / total_count
    fake_ratio = fake_count / total_count
    low_score_ratio = low_score_real / max(real_count, 1)

    if matched_costs:
        mean_match_cost = float(np.mean(matched_costs))
        assoc_decay = float(np.exp(-mean_match_cost / match_score_thre))
        good_match_ratio = good_match_count / max(len(matched_costs), 1)
        assoc_conf = 0.5 * assoc_decay + 0.5 * good_match_ratio
    else:
        assoc_conf = 0.5

    cont_conf = float(np.clip(
        real_hit_ratio * (1.0 - 0.7 * fake_ratio) * (1.0 - 0.5 * low_score_ratio),
        0.0,
        1.0,
    ))
    maturity_conf = float(min(real_count / mature_len, 1.0))

    w_det = _per_class_float(score_cfg.get("W_DET", {}), category_num, 0.35)
    w_assoc = _per_class_float(score_cfg.get("W_ASSOC", {}), category_num, 0.25)
    w_cont = _per_class_float(score_cfg.get("W_CONT", {}), category_num, 0.25)
    w_mature = _per_class_float(score_cfg.get("W_MATURE", {}), category_num, 0.15)
    w_sum = max(w_det + w_assoc + w_cont + w_mature, 1e-6)

    score = (
        w_det * det_conf
        + w_assoc * assoc_conf
        + w_cont * cont_conf
        + w_mature * maturity_conf
    ) / w_sum

    if getattr(current_bbox, "is_fake", False):
        score *= current_fake_scale

    min_score = float(score_cfg.get("MIN_SCORE", 0.01))
    max_score = float(score_cfg.get("MAX_SCORE", 0.995))
    return float(np.clip(score, min_score, max_score))


def allow_single_stage_birth_under_mode(compat_mode, gate_allowed: bool) -> bool:
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    if compat_mode == "mctrack":
        return True
    return bool(gate_allowed)


def use_mctrack_single_stage_flow(compat_mode, use_bytetrack: bool) -> bool:
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    return compat_mode == "mctrack" and not bool(use_bytetrack)


def extract_bbox_history_fields(bbox, compat_mode):
    compat_mode = normalize_tracker_compat_mode(compat_mode)
    if compat_mode == "mctrack":
        xyz_fusion = getattr(bbox, "global_xyz_lwh_yaw_fusion", None)
        vel_fusion = getattr(bbox, "global_velocity_fusion", None)
        lwh_fusion = getattr(bbox, "lwh_fusion", None)
        yaw_fusion = getattr(bbox, "global_yaw_fusion", None)
        if xyz_fusion is not None:
            xyz = [float(v) for v in xyz_fusion[:3]]
        else:
            xyz = [float(v) for v in bbox.global_xyz]
        if vel_fusion is not None:
            vel = [float(v) for v in vel_fusion[:2]]
        else:
            vel = [float(v) for v in bbox.global_velocity[:2]]
        if lwh_fusion is not None:
            lwh = [float(v) for v in lwh_fusion[:3]]
        else:
            lwh = [float(v) for v in bbox.lwh]
        if yaw_fusion is not None:
            yaw = float(yaw_fusion)
        else:
            yaw = float(bbox.global_yaw)
        return xyz, vel, lwh, yaw

    return (
        [float(v) for v in bbox.global_xyz],
        [float(v) for v in bbox.global_velocity[:2]],
        [float(v) for v in bbox.lwh],
        float(bbox.global_yaw),
    )


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
