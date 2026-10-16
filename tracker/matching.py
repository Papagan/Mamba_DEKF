# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import copy
import os
from collections.abc import Mapping

import lap
import numpy as np

from tracker.cost_function import *
from tracker.compat_utils import resolve_bev_cost_state
from utils.utils import mask_tras_dets
from utils.debug_log import emit_debug_line


def _lookup_mapping_value(values, key, default=None):
    if not isinstance(values, dict):
        return default
    if key in values:
        return values[key]
    text_key = str(key)
    if text_key in values:
        return values[text_key]
    return default


def _normalize_state_set(values):
    return {str(value).strip().lower() for value in (values or [])}


def _association_state_bucket(traj):
    return "unmatched" if int(getattr(traj, "unmatch_length", 0) or 0) > 0 else "matched"


def _mamba_association_alpha(cfg, class_id, state_bucket):
    assoc_cfg = (cfg.get("MAMBA_ASSOCIATION_PRIOR", {}) or {})
    active_cfg = assoc_cfg.get("ACTIVE_CLASS_STATES", {}) or {}
    active_states = _normalize_state_set(_lookup_mapping_value(active_cfg, int(class_id), []))
    if str(state_bucket).strip().lower() not in active_states:
        return 0.0

    alpha_cfg = assoc_cfg.get("ALPHA", {}) or {}
    class_alpha = _lookup_mapping_value(alpha_cfg, int(class_id), {})
    if isinstance(class_alpha, dict):
        return float(_lookup_mapping_value(class_alpha, str(state_bucket).strip().lower(), 0.0))
    return float(class_alpha or 0.0)


def apply_mamba_association_prior_to_cost_matrix(
    cost_matrix,
    trajs,
    dets,
    cfg,
    *,
    trk_embeddings=None,
    det_embeddings=None,
    state_buckets=None,
):
    """Apply a bounded Mamba association penalty without replacing geometry.

    The correction is deliberately one-sided: low semantic/temporal similarity
    can make a candidate less attractive, but Mamba cannot create a match that
    the geometric cost did not already consider. This protects the frozen
    baseline from the historical failure mode where learned branches overrode
    the stable association path.
    """
    out = np.array(cost_matrix, copy=True)
    assoc_cfg = (cfg.get("MAMBA_ASSOCIATION_PRIOR", {}) or {})
    if (
        not bool(assoc_cfg.get("ENABLED", False))
        or trk_embeddings is None
        or det_embeddings is None
        or len(trajs) == 0
        or len(dets) == 0
    ):
        return out

    trk_embeddings = np.asarray(trk_embeddings, dtype=np.float32)
    det_embeddings = np.asarray(det_embeddings, dtype=np.float32)
    if trk_embeddings.shape[0] != len(trajs) or det_embeddings.shape[0] != len(dets):
        return out

    cos_sim_matrix = compute_cosine_similarity_matrix(trk_embeddings, det_embeddings)
    category_map = cfg.get("CATEGORY_MAP_TO_NUMBER", {}) or {}
    max_delta = float(assoc_cfg.get("MAX_DELTA", 0.05))
    det_score_weight = float(assoc_cfg.get("DET_SCORE_WEIGHT", 0.0))
    min_score = float(assoc_cfg.get("MIN_DET_SCORE", 0.0))

    for t, traj in enumerate(trajs):
        traj_category = traj.bboxes[-1].category
        class_id = category_map.get(traj_category, None)
        if class_id is None:
            continue
        state_bucket = (
            str(state_buckets[t]).strip().lower()
            if state_buckets is not None and t < len(state_buckets)
            else _association_state_bucket(traj)
        )
        alpha = _mamba_association_alpha(cfg, int(class_id), state_bucket)
        if alpha <= 0.0:
            continue

        for d, det in enumerate(dets):
            if det.category != traj_category or not np.isfinite(out[t, d]):
                continue
            semantic_penalty = max(0.0, 1.0 - float(cos_sim_matrix[t, d]))
            det_score = float(getattr(det, "det_score", 1.0))
            score_penalty = max(0.0, min_score - det_score) * det_score_weight
            delta = alpha * (semantic_penalty + score_penalty)
            out[t, d] = out[t, d] + min(max_delta, max(0.0, delta))

    return out


def apply_pairwise_association_head_to_cost_matrix(
    cost_matrix,
    trajs,
    dets,
    cfg,
    *,
    association_scores=None,
    state_buckets=None,
    audit_callback=None,
):
    """Apply a bounded learned correction inside the existing geometric gate.

    Legacy modes keep the one-sided low-score penalty. ``bounded_log_factor``
    interprets the classifier logit as a learned log-likelihood-ratio residual:
    strong pairs may be promoted and weak pairs penalized, but only among the
    near-best geometric candidates. ``PRESERVE_ROW_MIN`` prevents the learned
    branch from making any row more matchable than its original geometric best.
    """
    out = np.array(cost_matrix, copy=True)
    head_cfg = (cfg.get("MAMBA_ASSOCIATION_HEAD", {}) or {})
    if (
        not bool(head_cfg.get("ENABLED", False))
        or association_scores is None
        or len(trajs) == 0
        or len(dets) == 0
    ):
        return out

    scores = np.asarray(association_scores, dtype=np.float32)
    if scores.shape != out.shape:
        return out

    category_map = cfg.get("CATEGORY_MAP_TO_NUMBER", {}) or {}
    active_cfg = head_cfg.get("ACTIVE_CLASS_STATES", {}) or {}
    apply_mode = str(head_cfg.get("APPLY_MODE", "penalty")).strip().lower()
    cost_margin_eps = float(head_cfg.get("COST_MARGIN_EPS", 0.05))
    min_score = float(head_cfg.get("MIN_SCORE", 0.5))
    alpha = float(head_cfg.get("ALPHA", 0.05))
    max_delta = float(head_cfg.get("MAX_DELTA", 0.03))
    log_factor_bound = float(head_cfg.get("LOG_FACTOR_BOUND", 1.0))
    logit_temperature = float(head_cfg.get("LOGIT_TEMPERATURE", 1.0))
    preserve_row_min = bool(head_cfg.get("PRESERVE_ROW_MIN", True))
    if log_factor_bound <= 0.0 or not np.isfinite(log_factor_bound):
        raise ValueError("MAMBA_ASSOCIATION_HEAD.LOG_FACTOR_BOUND must be finite and positive")
    if logit_temperature <= 0.0 or not np.isfinite(logit_temperature):
        raise ValueError("MAMBA_ASSOCIATION_HEAD.LOGIT_TEMPERATURE must be finite and positive")

    for t, traj in enumerate(trajs):
        traj_category = traj.bboxes[-1].category
        class_id = category_map.get(traj_category, None)
        if class_id is None:
            continue
        state_bucket = (
            str(state_buckets[t]).strip().lower()
            if state_buckets is not None and t < len(state_buckets)
            else _association_state_bucket(traj)
        )
        active_states = _normalize_state_set(_lookup_mapping_value(active_cfg, int(class_id), []))
        if state_bucket not in active_states:
            continue

        row_cost = out[t]
        same_class_finite = [
            float(row_cost[d])
            for d, det in enumerate(dets)
            if det.category == traj_category and np.isfinite(row_cost[d])
        ]
        best_cost = min(same_class_finite) if same_class_finite else None
        sorted_same_class_costs = sorted(same_class_finite)

        for d, det in enumerate(dets):
            if det.category != traj_category or not np.isfinite(out[t, d]):
                continue
            score = float(scores[t, d])
            if not np.isfinite(score):
                continue
            cost_before = float(out[t, d])

            mode_active = True
            if apply_mode in {
                "margin_tiebreak",
                "margin",
                "tiebreak",
                "tie_break",
                "bounded_log_factor",
                "log_factor",
                "likelihood_residual",
            }:
                if best_cost is None or cost_before > best_cost + cost_margin_eps:
                    mode_active = False
            cost_rank = 1
            for rank, value in enumerate(sorted_same_class_costs, start=1):
                if abs(float(value) - cost_before) <= 1e-7:
                    cost_rank = rank
                    break

            delta = 0.0
            if mode_active:
                if apply_mode in {"bounded_log_factor", "log_factor", "likelihood_residual"}:
                    probability = float(np.clip(score, 1.0e-6, 1.0 - 1.0e-6))
                    raw_log_factor = float(np.log(probability) - np.log1p(-probability))
                    bounded_log_factor = log_factor_bound * float(
                        np.tanh(raw_log_factor / logit_temperature)
                    )
                    delta = float(np.clip(-alpha * bounded_log_factor, -max_delta, max_delta))
                    corrected_cost = cost_before + delta
                    if preserve_row_min and best_cost is not None:
                        corrected_cost = max(float(best_cost), corrected_cost)
                    out[t, d] = corrected_cost
                    delta = float(corrected_cost - cost_before)
                else:
                    delta = alpha * max(0.0, min_score - score)
                    delta = min(max_delta, max(0.0, delta))
                    out[t, d] = out[t, d] + delta
            if audit_callback is not None:
                audit_callback({
                    "class_id": int(class_id),
                    "class_name": str(traj_category),
                    "state_bucket": state_bucket,
                    "score": float(score),
                    "delta": float(delta),
                    "cost_before": cost_before,
                    "cost_after": float(out[t, d]),
                    "active": bool(mode_active),
                    "finite": True,
                    "sample": {
                        "track_index": int(t),
                        "det_index": int(d),
                        "cost_rank": int(cost_rank),
                        "best_cost": float(best_cost) if best_cost is not None else None,
                        "cost_margin_eps": float(cost_margin_eps),
                        "margin_gap": float(cost_before - best_cost) if best_cost is not None else None,
                        "correction_mode": apply_mode,
                    },
                })

    return out


def Greedy(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Greedy matching algorithm.
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         - N_cls: Number of classes.
                         - N_det: Number of detections.
                         - N_tra: Number of trajectories.
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be egual to cost matrix number."

    # solve cost matrix
    m_det, m_tra = [], []
    costs = []
    num_det, num_tra = cost_matrix.shape[1:]
    for cls_idx, cls_cost in enumerate(cost_matrix):
        for det_idx in range(num_det):
            tra_idx = cls_cost[det_idx].argmin()
            if cls_cost[det_idx][tra_idx] <= thresholds[cls_idx]:
                costs.append(cls_cost[det_idx, tra_idx])
                cost_matrix[cls_idx, :, tra_idx] = 1e18
                m_det.append(det_idx)
                m_tra.append(tra_idx)

    # unmatched tra and det
    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)


def Hungarian(cost_matrix, thresholds):
    """
    Refer: https://github.com/lixiaoyu2000/Poly-MOT/blob/main/utils/matching.py
    Info: This function implements the Hungarian algorithm using the Linear Assignment Problem solver (lapjv).
    Parameters:
        input:
            cost_matrix: np.array, either 2D or 3D cost matrix with shape [N_cls, N_det, N_tra] or [N_det, N_tra].
                         Invalid costs are represented by np.inf.
            thresholds: dict, class-specific matching thresholds to restrict false positive matches.
        output:
            m_det: list, indices of matched detections.
            m_tra: list, indices of matched trajectories.
            um_det: np.array, indices of unmatched detections.
            um_tra: np.array, indices of unmatched trajectories.
            costs: np.array, matching costs for the matched pairs.
    """
    assert cost_matrix.ndim == 2 or cost_matrix.ndim == 3, "cost matrix must be valid."
    if cost_matrix.ndim == 2:
        cost_matrix = cost_matrix[None, :, :]
    assert (
        len(thresholds) == cost_matrix.shape[0]
    ), "the number of thresholds should be equal to cost matrix number."

    m_det, m_tra = [], []
    costs = []
    num_det, num_tra = cost_matrix.shape[1:]

    for cls_idx, cls_cost in enumerate(cost_matrix):
        # ---- NaN guard: replace NaN with inf (unmatchable) ----
        if np.isnan(cls_cost).any():
            cls_cost = np.nan_to_num(cls_cost, nan=np.inf, posinf=np.inf, neginf=np.inf)

        # ---- All-inf guard: lapjv crashes on all-inf matrix ----
        if np.all(np.isinf(cls_cost)):
            continue  # no valid matches for this class

        try:
            _, x, y = lap.lapjv(cls_cost, extend_cost=True,
                                cost_limit=thresholds[cls_idx])
        except Exception:
            continue  # lapjv failure → no matches for this class

        for ix, mx in enumerate(x):
            if mx >= 0:
                m_det.append(ix)
                m_tra.append(mx)
                costs.append(cls_cost[ix, mx])

    if len(m_det) == 0:
        um_det, um_tra = np.arange(num_det), np.arange(num_tra)
    else:
        um_det = np.setdiff1d(np.arange(num_det), np.array(m_det))
        um_tra = np.setdiff1d(np.arange(num_tra), np.array(m_tra))

    return m_det, m_tra, um_det, um_tra, np.array(costs)


def solve_matching_cost_matrix(
    cost_matrix,
    trajs_category,
    dets_category,
    cfg,
    match_type,
):
    """Solve a track-by-detection cost matrix without modifying its inputs."""
    cost_matrix = np.asarray(cost_matrix)
    trajs_category = np.asarray(trajs_category)
    dets_category = np.asarray(dets_category)
    expected_shape = (len(trajs_category), len(dets_category))
    if cost_matrix.ndim != 2 or cost_matrix.shape != expected_shape:
        raise ValueError(
            f"cost_matrix must have shape {expected_shape}, got {cost_matrix.shape}"
        )
    if min(cost_matrix.shape) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    category_map = cfg["CATEGORY_MAP_TO_NUMBER"]
    vectorized_map = np.vectorize(category_map.get)
    dets_label = vectorized_map(dets_category)
    trajs_label = vectorized_map(trajs_category)

    cls_num = len(cfg["CATEGORY_LIST"])
    valid_mask = mask_tras_dets(cls_num, trajs_label, dets_label)
    trans_valid_mask = valid_mask.transpose(0, 2, 1)

    # Greedy mutates its working matrix, so the solver always owns this copy.
    trans_cost_matrix = np.array(cost_matrix.T, copy=True)
    trans_cost_matrix = trans_cost_matrix[None, :, :].repeat(cls_num, axis=0)
    trans_cost_matrix[np.where(~trans_valid_mask)] = np.inf

    matching_mode = cfg["MATCHING"][match_type]["MATCHING_MODE"]
    thresholds = cfg["THRESHOLD"][match_type]["COST_THRE"]
    if matching_mode == "Hungarian":
        m_det, m_tra, _, _, costs = Hungarian(trans_cost_matrix, thresholds)
    elif matching_mode == "Greedy":
        m_det, m_tra, _, _, costs = Greedy(trans_cost_matrix, thresholds)
    else:
        raise ValueError(f"Unknown matching mode: {matching_mode}")

    assert len(m_det) == len(m_tra)
    matched_indices = np.column_stack((m_tra, m_det))
    return matched_indices, costs


def _association_shadow_records(
    *,
    production_matches,
    production_costs,
    shadow_matches,
    shadow_costs,
    base_cost_matrix,
    corrected_cost_matrix,
    association_scores,
    trajs,
    trajs_category,
    cfg,
    state_buckets,
    metadata,
    class_thresholds=None,
):
    metadata = metadata if isinstance(metadata, dict) else {}
    track_count = len(trajs)
    track_ids = metadata.get("track_ids")
    if track_ids is None:
        track_ids = [getattr(traj, "track_id", None) for traj in trajs]

    from tracker.association_shadow_audit import compare_assignment_results

    scores = np.asarray(association_scores) if association_scores is not None else None
    if scores is None or scores.shape != base_cost_matrix.shape:
        scores = np.full(base_cost_matrix.shape, np.nan, dtype=float)

    class_names = metadata.get("class_names", list(trajs_category))
    class_ids = metadata.get(
        "class_ids",
        [cfg["CATEGORY_MAP_TO_NUMBER"].get(category, -1) for category in trajs_category],
    )
    shadow_states = metadata.get("state_buckets", state_buckets)
    if shadow_states is None or len(shadow_states) != track_count:
        shadow_states = [_association_state_bucket(traj) for traj in trajs]

    context = metadata.get("context")
    if context is None:
        context = {
            key: metadata.get(key)
            for key in ("scene_id", "sample_token")
            if key in metadata
        }

    return compare_assignment_results(
        production_matches=production_matches,
        production_costs=production_costs,
        shadow_matches=shadow_matches,
        shadow_costs=shadow_costs,
        base_cost_matrix=base_cost_matrix,
        corrected_cost_matrix=corrected_cost_matrix,
        association_scores=scores,
        track_ids=track_ids,
        class_ids=class_ids,
        class_names=class_names,
        state_buckets=shadow_states,
        track_gt_tokens=metadata.get("track_gt_tokens"),
        det_gt_tokens=metadata.get("det_gt_tokens"),
        context=context,
        class_thresholds=(
            cfg["THRESHOLD"]["BEV"]["COST_THRE"]
            if class_thresholds is None
            else class_thresholds
        ),
    )


def _association_shadow_error_record(exc, *, stage, metadata):
    metadata = metadata if isinstance(metadata, dict) else {}
    context = metadata.get("context")
    if not isinstance(context, dict):
        context = {
            key: metadata.get(key)
            for key in ("scene_id", "sample_token", "frame_id")
            if key in metadata
        }
    return {
        "stage": stage,
        "type": type(exc).__name__,
        "message": str(exc) or type(exc).__name__,
        "context": context,
    }


def _inference_pair_cache_error_record(exc, *, stage, payload):
    from tracker.inference_pair_cache import validate_error_record

    def native(value):
        return value.item() if isinstance(value, np.generic) else value

    def canonical_string(value, fallback):
        try:
            value = native(value)
        except Exception:
            return fallback
        if type(value) is str and value and value.strip() == value:
            return value
        return fallback

    error_type = canonical_string(type(exc).__name__, "Exception")
    try:
        message = str(exc).strip()
    except Exception:
        message = error_type
    if not message:
        message = error_type
    stage = canonical_string(stage, "export")

    try:
        if not isinstance(payload, Mapping):
            raise TypeError("payload context must be a mapping")
        record = {
            "scene_id": native(payload.get("scene_id")),
            "sample_token": native(payload.get("sample_token")),
            "frame_id": native(payload.get("frame_id")),
            "stage": stage,
            "error_type": error_type,
            "message": message,
        }
        return validate_error_record(record)
    except Exception:
        return validate_error_record(
            {
                "scene_id": "unknown",
                "sample_token": "unknown",
                "frame_id": 0,
                "stage": stage,
                "error_type": error_type,
                "message": message,
            }
        )


def _class_threshold_mapping(thresholds):
    if isinstance(thresholds, Mapping):
        return dict(thresholds)
    try:
        return {
            class_id: threshold
            for class_id, threshold in enumerate(thresholds)
        }
    except TypeError as exc:
        raise TypeError(
            "class matching thresholds must be a mapping or sequence"
        ) from exc


def _uncertainty_geometric_audit_cost_matrix(
    trajs,
    dets,
    cfg,
    trk_embeddings,
    det_embeddings,
    trk_pos_P,
    trk_ori_P,
    cal_flag,
):
    geometric_cfg = dict(cfg)
    threshold_cfg = dict(cfg.get("THRESHOLD", {}) or {})
    bev_cfg = dict(threshold_cfg.get("BEV", {}) or {})
    bev_cfg["COST_MODE"] = "geometric"
    threshold_cfg["BEV"] = bev_cfg
    geometric_cfg["THRESHOLD"] = threshold_cfg
    geometric_cost_matrix, _, _ = cost_calculate_uncertainty_aware(
        trajs,
        dets,
        geometric_cfg,
        trk_embeddings,
        det_embeddings,
        trk_pos_P,
        trk_ori_P,
        cal_flag=cal_flag,
    )
    return geometric_cost_matrix


def _export_inference_pair_cache_groups(
    *,
    payload,
    callback,
    error_callback,
    gt_provider,
    strict,
    base_cost_matrix,
    production_matches,
    production_costs,
    thresholds,
    cfg,
):
    """Run the post-assignment cache exporter against isolated copies.

    ``gt_provider`` is a zero-argument callable returning a mapping containing
    aligned ``track_gt_tokens`` and ``det_gt_tokens`` sequences.
    """
    if payload is None or callback is None:
        return

    stage = "payload"
    try:
        if not isinstance(payload, Mapping):
            raise TypeError("inference pair cache payload must be a mapping")
        export_payload = dict(payload)
        for name in ("track_gt_tokens", "det_gt_tokens"):
            if name in export_payload:
                export_payload[name] = copy.deepcopy(export_payload[name])
        if gt_provider is not None:
            stage = "gt_provider"
            gt_payload = gt_provider()
            if not isinstance(gt_payload, Mapping):
                raise TypeError("inference pair cache GT provider must return a mapping")
            for name in ("track_gt_tokens", "det_gt_tokens"):
                if name not in gt_payload:
                    raise ValueError(
                        f"inference pair cache GT provider is missing {name}"
                    )
                export_payload[name] = copy.deepcopy(gt_payload[name])

        stage = "candidate_groups"
        from tracker.inference_pair_cache import build_candidate_groups

        cache_cfg = cfg.get("INFERENCE_PAIR_CACHE", {}) or {}
        stage = "base_cost_matrix"
        export_base_cost_matrix = (
            base_cost_matrix()
            if callable(base_cost_matrix)
            else base_cost_matrix
        )
        stage = "thresholds"
        export_thresholds = _class_threshold_mapping(thresholds)
        export_matches = np.array(production_matches, copy=True)
        if export_matches.size == 0:
            export_matches = np.empty((0, 2), dtype=int)
        stage = "candidate_groups"
        groups = build_candidate_groups(
            payload=export_payload,
            base_cost_matrix=np.array(export_base_cost_matrix, copy=True),
            production_matches=export_matches,
            production_costs=np.array(production_costs, copy=True),
            thresholds=export_thresholds,
            include_embeddings=cache_cfg.get("INCLUDE_EMBEDDINGS", True),
            include_pair_vectors=cache_cfg.get("INCLUDE_PAIR_VECTORS", True),
            max_candidates_per_group=cache_cfg.get("MAX_CANDIDATES_PER_GROUP", 0),
            candidate_cost_margin=cache_cfg.get("CANDIDATE_COST_MARGIN"),
        )
        stage = "callback"
        callback(groups)
    except Exception as exc:
        if error_callback is not None:
            try:
                error_callback(
                    _inference_pair_cache_error_record(
                        exc, stage=stage, payload=payload
                    )
                )
            except Exception:
                pass
        if strict:
            raise


def match_trajs_and_dets(
    trajs,
    dets,
    cfg,
    transform_matrix=None,
    is_rv=False,
    trk_embeddings=None,
    det_embeddings=None,
    state_buckets=None,
    association_scores=None,
    association_audit_callback=None,
    association_shadow_callback=None,
    association_shadow_metadata=None,
    association_shadow_error_callback=None,
    association_shadow_strict=False,
    inference_pair_cache_payload=None,
    inference_pair_cache_callback=None,
    inference_pair_cache_error_callback=None,
    inference_pair_cache_gt_provider=None,
    inference_pair_cache_strict=False,
):
    """
    Info: This function matches trajectories with detections using a cost matrix and a specified matching algorithm (Hungarian or Greedy).
    Parameters:
        input:
            trajs: List of trajectory objects.
            dets: List of detection objects.
            cfg: Configuration dictionary, includes matching and category information.
            transform_matrix: (Optional) Matrix for transforming coordinates (if needed for the matching process).
            is_rv: bool, flag to indicate whether to use re-projected view (RV) or bird's eye view (BEV) for matching.
        output:
            matched_indices: np.array, array of matched trajectory and detection indices (shape [n, 2]).
            costs: np.array, corresponding costs for each match.
    """
    if len(trajs) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    if len(dets) == 0:
        production_matches = np.empty((0, 2), dtype=int)
        production_costs = np.empty((0,), dtype=float)
        if (
            inference_pair_cache_payload is not None
            and inference_pair_cache_callback is not None
        ):
            match_type = "RV" if is_rv else "BEV"
            _export_inference_pair_cache_groups(
                payload=inference_pair_cache_payload,
                callback=inference_pair_cache_callback,
                error_callback=inference_pair_cache_error_callback,
                gt_provider=inference_pair_cache_gt_provider,
                strict=inference_pair_cache_strict,
                base_cost_matrix=np.empty((len(trajs), 0), dtype=float),
                production_matches=production_matches,
                production_costs=production_costs,
                thresholds=cfg["THRESHOLD"][match_type]["COST_THRE"],
                cfg=cfg,
            )
        return production_matches, production_costs

    match_type = "RV" if is_rv else "BEV"
    cost_matrix, trajs_category, dets_category = cost_calculate_general(
        trajs, dets, cfg, transform_matrix, is_rv
    )
    export_enabled = (
        inference_pair_cache_payload is not None
        and inference_pair_cache_callback is not None
    )
    geometric_cost_matrix = (
        np.array(cost_matrix, copy=True) if export_enabled else None
    )
    if not is_rv:
        base_cost_matrix = apply_mamba_association_prior_to_cost_matrix(
            cost_matrix,
            trajs,
            dets,
            cfg,
            trk_embeddings=trk_embeddings,
            det_embeddings=det_embeddings,
            state_buckets=state_buckets,
        )
        base_cost_matrix = np.array(base_cost_matrix, copy=True)
    else:
        base_cost_matrix = np.array(cost_matrix, copy=True)

    audit_cfg = cfg.get("ASSOCIATION_HEAD_AUDIT", {}) or {}
    shadow_mode = bool(
        not is_rv
        and audit_cfg.get("SHADOW_MODE", False)
    )

    if shadow_mode:
        production_matches, production_costs = solve_matching_cost_matrix(
            base_cost_matrix, trajs_category, dets_category, cfg, match_type
        )
        _export_inference_pair_cache_groups(
            payload=inference_pair_cache_payload,
            callback=inference_pair_cache_callback,
            error_callback=inference_pair_cache_error_callback,
            gt_provider=inference_pair_cache_gt_provider,
            strict=inference_pair_cache_strict,
            base_cost_matrix=geometric_cost_matrix,
            production_matches=production_matches,
            production_costs=production_costs,
            thresholds=cfg["THRESHOLD"][match_type]["COST_THRE"],
            cfg=cfg,
        )
        shadow_stage = "shadow_correction"
        try:
            corrected_cost_matrix = apply_pairwise_association_head_to_cost_matrix(
                np.array(base_cost_matrix, copy=True),
                trajs,
                dets,
                cfg,
                association_scores=association_scores,
                state_buckets=state_buckets,
                audit_callback=association_audit_callback,
            )
            corrected_cost_matrix = np.array(corrected_cost_matrix, copy=True)
            shadow_stage = "shadow_solver"
            shadow_matches, shadow_costs = solve_matching_cost_matrix(
                corrected_cost_matrix, trajs_category, dets_category, cfg, match_type
            )
            if association_shadow_callback is not None:
                shadow_stage = "decision_records"
                records = _association_shadow_records(
                    production_matches=production_matches,
                    production_costs=production_costs,
                    shadow_matches=shadow_matches,
                    shadow_costs=shadow_costs,
                    base_cost_matrix=base_cost_matrix,
                    corrected_cost_matrix=corrected_cost_matrix,
                    association_scores=association_scores,
                    trajs=trajs,
                    trajs_category=trajs_category,
                    cfg=cfg,
                    state_buckets=state_buckets,
                    metadata=association_shadow_metadata,
                )
                if records is not None:
                    shadow_stage = "decision_callback"
                    association_shadow_callback(records)
        except Exception as exc:
            if association_shadow_error_callback is not None:
                try:
                    association_shadow_error_callback(
                        _association_shadow_error_record(
                            exc,
                            stage=shadow_stage,
                            metadata=association_shadow_metadata,
                        )
                    )
                except Exception:
                    if association_shadow_strict:
                        raise exc
            if association_shadow_strict:
                raise
        return production_matches, production_costs

    production_cost_matrix = base_cost_matrix
    if not is_rv:
        production_cost_matrix = apply_pairwise_association_head_to_cost_matrix(
            np.array(base_cost_matrix, copy=True),
            trajs,
            dets,
            cfg,
            association_scores=association_scores,
            state_buckets=state_buckets,
            audit_callback=association_audit_callback,
        )
    production_matches, production_costs = solve_matching_cost_matrix(
        production_cost_matrix, trajs_category, dets_category, cfg, match_type
    )
    _export_inference_pair_cache_groups(
        payload=inference_pair_cache_payload,
        callback=inference_pair_cache_callback,
        error_callback=inference_pair_cache_error_callback,
        gt_provider=inference_pair_cache_gt_provider,
        strict=inference_pair_cache_strict,
        base_cost_matrix=geometric_cost_matrix,
        production_matches=production_matches,
        production_costs=production_costs,
        thresholds=cfg["THRESHOLD"][match_type]["COST_THRE"],
        cfg=cfg,
    )
    return production_matches, production_costs


def cost_calculate_general(trajs, dets, cfg, transform_matrix, is_rv=False):
    cost_matrix = np.zeros((len(trajs), len(dets)))

    def choose_cost_func(is_rv, cost_mode):
        if is_rv:
            if cost_mode == "IOU_2D":
                cal_cost_func = cal_iou_inrv
            elif cost_mode == "GIOU_2D":
                cal_cost_func = cal_giou_inrv
            elif cost_mode == "DIOU_2D":
                cal_cost_func = cal_diou_inrv
            elif cost_mode == "SDIOU_2D":
                cal_cost_func = cal_sdiou_inrv
        else:
            if cost_mode == "RO_GDIOU_3D":
                cal_cost_func = cal_rotation_gdiou_inbev
        return cal_cost_func

    for t, trk in enumerate(trajs):
        for d, det in enumerate(dets):
            trk_category = cfg["CATEGORY_MAP_TO_NUMBER"][trk.bboxes[-1].category]
            match_type = "BEV"
            if is_rv:
                match_type = "RV"
            cost_mode = cfg["MATCHING"][match_type]["COST_MODE"][trk_category]
            cost_state = cfg["MATCHING"][match_type]["COST_STATE"][trk_category]
            if not is_rv:
                cost_state = resolve_bev_cost_state(cfg, trk_category, cost_state)
            cost_state_predict_ratio = cfg["THRESHOLD"]["COST_STATE_PREDICT_RATION"][
                trk_category
            ]
            cal_cost_func = choose_cost_func(is_rv, cost_mode)
            pred_cost = cal_cost_func(trk, det, cfg, cal_flag="Predict")
            no_pred_cost = cal_cost_func(trk, det, cfg, cal_flag="BackPredict")

            if cost_state == "Predict":
                cost_matrix[t, d] = pred_cost
            elif cost_state == "BackPredict":
                cost_matrix[t, d] = no_pred_cost
            elif cost_state == "Fusion":
                cost_matrix[t, d] = (
                    cost_state_predict_ratio * pred_cost
                    + (1 - cost_state_predict_ratio) * no_pred_cost
                )

    trajs_category = np.array([traj.bboxes[-1].category for traj in trajs])
    dets_category = np.array([det.category for det in dets])
    same_category_mask = (trajs_category[:, np.newaxis] == dets_category).astype(int)
    cost_matrix[same_category_mask == 0] = -np.inf

    return 1 - cost_matrix, trajs_category, dets_category


# ============================================================
# Module C: Uncertainty-Aware Cost Calculation
# ============================================================
# Upgrades the pure-geometric Ro_GDIoU cost matrix to incorporate:
#   1. Mamba temporal embedding cosine similarity (semantic affinity)
#   2. Position/Orientation covariance trace (uncertainty penalty)
#
# This function is called instead of cost_calculate_general when the
# Mamba module is active.

def cost_calculate_uncertainty_aware(
    trajs: list,
    dets: list,
    cfg: dict,
    trk_embeddings: np.ndarray,
    det_embeddings: np.ndarray,
    trk_pos_P: list,
    trk_ori_P: list,
    is_rv: bool = False,
    cal_flag = "Predict",
) -> tuple:
    """
    Compute uncertainty-aware cost matrix combining geometric affinity,
    semantic affinity (Mamba embeddings), and uncertainty penalty (P trace).

    # For each (trk, det) pair:
    #   cos_sim     = cosine_similarity(emb_trk, emb_det)
    #   penalty     = trace(P_pos[:3,:3]) + trace(P_ori)
    #   cost_i,j    = (1-w_s) * (1 - Ro_GDIoU)
    #                 + w_s   * (1 - cos_sim)
    #                 + w_u   * penalty
    #
    # The uncertainty penalty dynamically widens the matching gate when the
    # filter is uncertain (high covariance), and tightens it when confident.

    Args:
        trajs          : list of Trajectory objects              (N_trk)
        dets           : list of BBox detection objects           (N_det)
        cfg            : config dict
        trk_embeddings : [N_trk, embed_dim] — Mamba temporal embeddings
        det_embeddings : [N_det, embed_dim] — detection embeddings
                         (for new dets with no history, use zero vectors)
        trk_pos_P      : list of [6, 6] np arrays — position covariance
        trk_ori_P      : list of [2, 2] np arrays — orientation covariance
        is_rv          : flag for re-projected view matching
        cal_flag       : "Predict", "BackPredict", or list of them (len=N_trk).
                         "Fusion" is resolved to "Predict".

    Returns:
        cost_matrix       : [N_trk, N_det] — combined cost (lower = better)
        trajs_category    : [N_trk] — category labels
        dets_category     : [N_det] — category labels
    """
    N_trk = len(trajs)
    N_det = len(dets)
    cost_matrix = np.zeros((N_trk, N_det))

    # ---- precompute pairwise cosine similarity ----
    # [N_trk, N_det]
    if trk_embeddings is not None and det_embeddings is not None:
        cos_sim_matrix = compute_cosine_similarity_matrix(trk_embeddings, det_embeddings)
    else:
        cos_sim_matrix = np.zeros((N_trk, N_det))

    # ---- precompute per-trajectory uncertainty penalty ----
    # [N_trk]
    if trk_pos_P is not None and trk_ori_P is not None:
        uncertainty_vec = compute_uncertainty_penalty(trk_pos_P, trk_ori_P)
    else:
        uncertainty_vec = np.zeros(N_trk)

    # ---- resolve per-trajectory cal_flag ----
    if isinstance(cal_flag, list):
        cal_flags = cal_flag
    else:
        cal_flags = [cal_flag] * N_trk

    # ---- compute combined cost per pair ----
    for t, trk in enumerate(trajs):
        cf = cal_flags[t]
        if (cfg.get("LEARNED_MOTION_MATCHING", {}) or {}).get("ENABLED", False):
            class_id = cfg["CATEGORY_MAP_TO_NUMBER"][trk.bboxes[-1].category]
            cf = resolve_bev_cost_state(cfg, class_id, cf)
        for d, det in enumerate(dets):
            cost_matrix[t, d] = cal_uncertainty_aware_cost(
                trk, det, cfg,
                cal_flag=cf,
                cos_sim=float(cos_sim_matrix[t, d]),
                uncertainty=float(uncertainty_vec[t]),
            )

    # ---- mask cross-category pairs ----
    trajs_category = np.array([traj.bboxes[-1].category for traj in trajs])
    dets_category = np.array([det.category for det in dets])
    same_category_mask = (trajs_category[:, np.newaxis] == dets_category).astype(int)
    cost_matrix[same_category_mask == 0] = np.inf

    return cost_matrix, trajs_category, dets_category


def match_trajs_and_dets_uncertainty_aware(
    trajs: list,
    dets: list,
    cfg: dict,
    trk_embeddings: np.ndarray = None,
    det_embeddings: np.ndarray = None,
    trk_pos_P: list = None,
    trk_ori_P: list = None,
    cal_flag = "Predict",
    association_scores=None,
    association_audit_callback=None,
    state_buckets=None,
    association_shadow_callback=None,
    association_shadow_metadata=None,
    association_shadow_error_callback=None,
    association_shadow_strict=False,
    inference_pair_cache_payload=None,
    inference_pair_cache_callback=None,
    inference_pair_cache_error_callback=None,
    inference_pair_cache_gt_provider=None,
    inference_pair_cache_strict=False,
) -> tuple:
    """
    Uncertainty-aware matching pipeline (Module C entry point).

    Replaces match_trajs_and_dets when Mamba module is active.
    Falls back to original geometric matching when embeddings/covariances
    are not provided.

    Args:
        trajs          : list of Trajectory objects
        dets           : list of BBox detections
        cfg            : config dict
        trk_embeddings : [N_trk, embed_dim] or None
        det_embeddings : [N_det, embed_dim] or None
        trk_pos_P      : list of [6,6] arrays or None
        trk_ori_P      : list of [2,2] arrays or None
        cal_flag       : "Predict", "BackPredict", or list of them (len=N_trk)

    Returns:
        matched_indices : [M, 2] — (trajectory_idx, detection_idx) pairs
        costs           : [M]   — matching costs for each pair
    """
    if len(trajs) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    if len(dets) == 0:
        production_matches = np.empty((0, 2), dtype=int)
        production_costs = np.empty((0,), dtype=float)
        if (
            inference_pair_cache_payload is not None
            and inference_pair_cache_callback is not None
        ):
            _export_inference_pair_cache_groups(
                payload=inference_pair_cache_payload,
                callback=inference_pair_cache_callback,
                error_callback=inference_pair_cache_error_callback,
                gt_provider=inference_pair_cache_gt_provider,
                strict=inference_pair_cache_strict,
                base_cost_matrix=np.empty((len(trajs), 0), dtype=float),
                production_matches=production_matches,
                production_costs=production_costs,
                thresholds=cfg["THRESHOLD"]["BEV"]["COST_THRE"],
                cfg=cfg,
            )
        return production_matches, production_costs

    # Geometric mode is intended as a stable fallback during early training
    # or debugging. Route directly to the original geometric matcher so
    # uncertainty/embedding branches cannot suppress all matches.
    cost_mode = cfg.get("THRESHOLD", {}).get("BEV", {}).get("COST_MODE", "geometric")
    if cost_mode != "full":
        return match_trajs_and_dets(
            trajs,
            dets,
            cfg,
            association_scores=association_scores,
            association_audit_callback=association_audit_callback,
            state_buckets=state_buckets,
            association_shadow_callback=association_shadow_callback,
            association_shadow_metadata=association_shadow_metadata,
            association_shadow_error_callback=association_shadow_error_callback,
            association_shadow_strict=association_shadow_strict,
            inference_pair_cache_payload=inference_pair_cache_payload,
            inference_pair_cache_callback=inference_pair_cache_callback,
            inference_pair_cache_error_callback=inference_pair_cache_error_callback,
            inference_pair_cache_gt_provider=inference_pair_cache_gt_provider,
            inference_pair_cache_strict=inference_pair_cache_strict,
        )

    # fallback to original if no Mamba outputs available
    has_mamba = (trk_embeddings is not None and det_embeddings is not None)
    if not has_mamba:
        return match_trajs_and_dets(
            trajs,
            dets,
            cfg,
            association_scores=association_scores,
            association_audit_callback=association_audit_callback,
            state_buckets=state_buckets,
            association_shadow_callback=association_shadow_callback,
            association_shadow_metadata=association_shadow_metadata,
            association_shadow_error_callback=association_shadow_error_callback,
            association_shadow_strict=association_shadow_strict,
            inference_pair_cache_payload=inference_pair_cache_payload,
            inference_pair_cache_callback=inference_pair_cache_callback,
            inference_pair_cache_error_callback=inference_pair_cache_error_callback,
            inference_pair_cache_gt_provider=inference_pair_cache_gt_provider,
            inference_pair_cache_strict=inference_pair_cache_strict,
        )

    cost_matrix, trajs_category, dets_category = cost_calculate_uncertainty_aware(
        trajs, dets, cfg,
        trk_embeddings, det_embeddings,
        trk_pos_P, trk_ori_P,
        cal_flag=cal_flag,
    )
    shadow_mode = bool(
        (cfg.get("ASSOCIATION_HEAD_AUDIT", {}) or {}).get("SHADOW_MODE", False)
    )
    base_cost_matrix = np.array(cost_matrix, copy=True)
    export_enabled = (
        inference_pair_cache_payload is not None
        and inference_pair_cache_callback is not None
    )
    geometric_audit_cost_matrix = None
    if export_enabled:
        def build_geometric_audit_cost_matrix():
            return _uncertainty_geometric_audit_cost_matrix(
                trajs,
                dets,
                cfg,
                trk_embeddings,
                det_embeddings,
                trk_pos_P,
                trk_ori_P,
                cal_flag,
            )

        geometric_audit_cost_matrix = build_geometric_audit_cost_matrix

    if not shadow_mode:
        cost_matrix = apply_pairwise_association_head_to_cost_matrix(
            cost_matrix,
            trajs,
            dets,
            cfg,
            association_scores=association_scores,
            state_buckets=state_buckets,
            audit_callback=association_audit_callback,
        )

    category_map = cfg["CATEGORY_MAP_TO_NUMBER"]
    vectorized_map = np.vectorize(category_map.get)
    dets_label = vectorized_map(dets_category)
    trajs_label = vectorized_map(trajs_category)

    cls_num = len(cfg["CATEGORY_LIST"])
    valid_mask = mask_tras_dets(cls_num, trajs_label, dets_label)

    # cost_matrix is already [N_trk, N_det], transpose to [N_det, N_trk]
    trans_cost_matrix = cost_matrix.T
    trans_cost_matrix = trans_cost_matrix[None, :, :].repeat(cls_num, axis=0)
    trans_valid_mask = valid_mask.transpose(0, 2, 1)
    trans_cost_matrix[np.where(~trans_valid_mask)] = np.inf

    if min(cost_matrix.shape) > 0:
        matching_mode = cfg["MATCHING"]["BEV"]["MATCHING_MODE"]
        dbg_assoc = os.environ.get("DEBUG_ASSOC", "")
        if dbg_assoc:
            finite_mask = np.isfinite(trans_cost_matrix)
            finite_count = int(finite_mask.sum())
            total_count = int(trans_cost_matrix.size)
            all_inf_rows = int(np.sum(np.all(~finite_mask, axis=2)))
            finite_vals = trans_cost_matrix[finite_mask]
            if finite_vals.size > 0:
                finite_min = float(np.min(finite_vals))
                finite_max = float(np.max(finite_vals))
                finite_mean = float(np.mean(finite_vals))
            else:
                finite_min = float("inf")
                finite_max = float("inf")
                finite_mean = float("inf")
            emit_debug_line(
                f"[ASSOC] mode={matching_mode} trajs={len(trajs)} dets={len(dets)} "
                f"finite={finite_count}/{total_count} all_inf_rows={all_inf_rows} "
                f"cost_min={finite_min:.4f} cost_mean={finite_mean:.4f} cost_max={finite_max:.4f}"
            )

        # ---- Uncertainty-adaptive matching gate ----
        # Wider gate when KF is uncertain (high P trace), tighter when confident.
        base_thresholds = cfg["THRESHOLD"]["BEV"]["COST_THRE"]
        if trk_pos_P is not None and trk_ori_P is not None:
            uncertainty_vec = compute_uncertainty_penalty(trk_pos_P, trk_ori_P)
            # typical P trace range: 0.1 (confident) → 50 (lost). map to scale 1.0–2.5
            avg_unc = float(np.mean(uncertainty_vec))
            unc_scale = 1.0 + min(0.3 * avg_unc, 1.5)
        else:
            unc_scale = 1.0
        adaptive_thresholds = [v * unc_scale for v in (
            base_thresholds.values() if isinstance(base_thresholds, dict) else base_thresholds
        )]
        if dbg_assoc:
            cls_hits = []
            for cls_idx, thr in enumerate(adaptive_thresholds):
                cls_cost = trans_cost_matrix[cls_idx]
                cls_finite = np.isfinite(cls_cost)
                cls_total = int(cls_finite.sum())
                if cls_total == 0:
                    cls_hits.append(f"c{cls_idx}:0/0@{thr:.2f}")
                    continue
                cls_good = int(np.sum(cls_cost[cls_finite] <= thr))
                cls_hits.append(f"c{cls_idx}:{cls_good}/{cls_total}@{thr:.2f}")
            emit_debug_line(
                f"[ASSOC] unc_scale={unc_scale:.3f} thresholds={adaptive_thresholds}"
            )
            emit_debug_line(
                "[ASSOC] below-threshold " + " ".join(cls_hits)
            )

        # Safety fallback:
        # If full-mode costs are numerically exploded (or yield zero feasible
        # candidates), fall back to pure geometric association for this frame.
        # This prevents complete tracker collapse (all tracks stay unconfirmed).
        feasible_pairs = 0
        for cls_idx, thr in enumerate(adaptive_thresholds):
            cls_cost = trans_cost_matrix[cls_idx]
            cls_finite = np.isfinite(cls_cost)
            if np.any(cls_finite):
                feasible_pairs += int(np.sum(cls_cost[cls_finite] <= thr))

        finite_vals = trans_cost_matrix[np.isfinite(trans_cost_matrix)]
        exploded = (
            finite_vals.size > 0
            and float(np.min(finite_vals)) > (float(max(adaptive_thresholds)) * 50.0)
        )
        if feasible_pairs == 0 or exploded:
            if dbg_assoc:
                reason = "no-feasible-pairs" if feasible_pairs == 0 else "cost-exploded"
                emit_debug_line(
                    f"[ASSOC] fallback=geometric reason={reason} "
                    f"min_cost={float(np.min(finite_vals)) if finite_vals.size else float('inf'):.4f}"
                )
            return match_trajs_and_dets(
                trajs,
                dets,
                cfg,
                association_scores=association_scores,
                association_audit_callback=association_audit_callback,
                state_buckets=state_buckets,
                association_shadow_callback=association_shadow_callback,
                association_shadow_metadata=association_shadow_metadata,
                association_shadow_error_callback=association_shadow_error_callback,
                association_shadow_strict=association_shadow_strict,
                inference_pair_cache_payload=inference_pair_cache_payload,
                inference_pair_cache_callback=inference_pair_cache_callback,
                inference_pair_cache_error_callback=inference_pair_cache_error_callback,
                inference_pair_cache_gt_provider=inference_pair_cache_gt_provider,
                inference_pair_cache_strict=inference_pair_cache_strict,
            )

        def solve_adaptive(matrix):
            if matching_mode == "Hungarian":
                return Hungarian(matrix, adaptive_thresholds)
            if matching_mode == "Greedy":
                return Greedy(matrix, adaptive_thresholds)
            raise ValueError(f"Unknown matching mode: {matching_mode}")

        m_det, m_tra, _, _, costs = solve_adaptive(trans_cost_matrix)
        assert len(m_det) == len(m_tra)
        matched_indices = np.column_stack((m_tra, m_det))
        _export_inference_pair_cache_groups(
            payload=inference_pair_cache_payload,
            callback=inference_pair_cache_callback,
            error_callback=inference_pair_cache_error_callback,
            gt_provider=inference_pair_cache_gt_provider,
            strict=inference_pair_cache_strict,
            base_cost_matrix=geometric_audit_cost_matrix,
            production_matches=matched_indices,
            production_costs=costs,
            thresholds=base_thresholds,
            cfg=cfg,
        )

        if shadow_mode:
            shadow_stage = "shadow_correction"
            try:
                corrected_cost_matrix = apply_pairwise_association_head_to_cost_matrix(
                    np.array(base_cost_matrix, copy=True),
                    trajs,
                    dets,
                    cfg,
                    association_scores=association_scores,
                    state_buckets=state_buckets,
                    audit_callback=association_audit_callback,
                )
                corrected_cost_matrix = np.array(corrected_cost_matrix, copy=True)
                shadow_trans = corrected_cost_matrix.T
                shadow_trans = shadow_trans[None, :, :].repeat(cls_num, axis=0)
                shadow_trans[np.where(~trans_valid_mask)] = np.inf
                shadow_stage = "shadow_solver"
                shadow_det, shadow_tra, _, _, shadow_costs = solve_adaptive(
                    shadow_trans
                )
                shadow_matches = np.column_stack((shadow_tra, shadow_det))
                if association_shadow_callback is not None:
                    shadow_stage = "decision_records"
                    records = _association_shadow_records(
                        production_matches=matched_indices,
                        production_costs=costs,
                        shadow_matches=shadow_matches,
                        shadow_costs=shadow_costs,
                        base_cost_matrix=base_cost_matrix,
                        corrected_cost_matrix=corrected_cost_matrix,
                        association_scores=association_scores,
                        trajs=trajs,
                        trajs_category=trajs_category,
                        cfg=cfg,
                        state_buckets=state_buckets,
                        metadata=association_shadow_metadata,
                        class_thresholds=dict(enumerate(adaptive_thresholds)),
                    )
                    if records is not None:
                        shadow_stage = "decision_callback"
                        association_shadow_callback(records)
            except Exception as exc:
                if association_shadow_error_callback is not None:
                    try:
                        association_shadow_error_callback(
                            _association_shadow_error_record(
                                exc,
                                stage=shadow_stage,
                                metadata=association_shadow_metadata,
                            )
                        )
                    except Exception:
                        if association_shadow_strict:
                            raise exc
                if association_shadow_strict:
                    raise
    else:
        matched_indices = np.empty(shape=(0, 2))
        costs = np.empty(shape=(0,))

    return matched_indices, costs
