# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import numpy as np
import lap

from tracker.cost_function import *
from utils.utils import mask_tras_dets


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


def match_trajs_and_dets(trajs, dets, cfg, transform_matrix=None, is_rv=False):
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
    if len(trajs) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    cost_matrix, trajs_category, dets_category = cost_calculate_general(
        trajs, dets, cfg, transform_matrix, is_rv
    )
    match_type = "RV" if is_rv else "BEV"

    category_map = cfg["CATEGORY_MAP_TO_NUMBER"]
    vectorized_map = np.vectorize(category_map.get)
    dets_label = vectorized_map(dets_category)
    trajs_label = vectorized_map(trajs_category)

    cls_num = len(cfg["CATEGORY_LIST"])
    valid_mask = mask_tras_dets(cls_num, trajs_label, dets_label)
    trans_valid_mask = valid_mask.transpose(0, 2, 1)

    trans_cost_matrix = cost_matrix.T
    trans_cost_matrix = trans_cost_matrix[None, :, :].repeat(cls_num, axis=0)
    trans_cost_matrix[np.where(~trans_valid_mask)] = np.inf

    if min(cost_matrix.shape) > 0:
        if cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Hungarian":
            m_det, m_tra, um_det, um_tra, costs = Hungarian(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))
        elif cfg["MATCHING"][match_type]["MATCHING_MODE"] == "Greedy":
            m_det, m_tra, um_det, um_tra, costs = Greedy(
                trans_cost_matrix,
                cfg["THRESHOLD"][match_type]["COST_THRE"],
            )
            assert len(m_det) == len(m_tra)
            matched_indices = np.column_stack((m_tra, m_det))
    else:
        matched_indices = np.empty(shape=(0, 2))

    return matched_indices, costs


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
            trk_category = cfg["CATEGORY_MAP_TO_NUMBER"][trajs[0].bboxes[-1].category]
            match_type = "BEV"
            if is_rv:
                match_type = "RV"
            cost_mode = cfg["MATCHING"][match_type]["COST_MODE"][trk_category]
            cost_state = cfg["MATCHING"][match_type]["COST_STATE"][trk_category]
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
        cal_flags = [("Predict" if cf == "Fusion" else cf) for cf in cal_flag]
    else:
        cf = "Predict" if cal_flag == "Fusion" else cal_flag
        cal_flags = [cf] * N_trk

    # ---- compute combined cost per pair ----
    for t, trk in enumerate(trajs):
        cf = cal_flags[t]
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
    if len(trajs) == 0 or len(dets) == 0:
        return np.empty((0, 2), dtype=int), np.empty((0,), dtype=float)

    # fallback to original if no Mamba outputs available
    has_mamba = (trk_embeddings is not None and det_embeddings is not None)
    if not has_mamba:
        return match_trajs_and_dets(trajs, dets, cfg)

    cost_matrix, trajs_category, dets_category = cost_calculate_uncertainty_aware(
        trajs, dets, cfg,
        trk_embeddings, det_embeddings,
        trk_pos_P, trk_ori_P,
        cal_flag=cal_flag,
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
        if matching_mode == "Hungarian":
            m_det, m_tra, um_det, um_tra, costs = Hungarian(
                trans_cost_matrix,
                cfg["THRESHOLD"]["BEV"]["COST_THRE"],
            )
        elif matching_mode == "Greedy":
            m_det, m_tra, um_det, um_tra, costs = Greedy(
                trans_cost_matrix,
                cfg["THRESHOLD"]["BEV"]["COST_THRE"],
            )
        else:
            raise ValueError(f"Unknown matching mode: {matching_mode}")
        assert len(m_det) == len(m_tra)
        matched_indices = np.column_stack((m_tra, m_det))
    else:
        matched_indices = np.empty(shape=(0, 2))
        costs = np.empty(shape=(0,))

    return matched_indices, costs
