"""Shared detection/GT identity alignment for training and tracker replay.

GT identities are supervision sidecars only, never model input features.
"""

import numpy as np
from scipy.optimize import linear_sum_assignment


def center_distance_2d(a_xyz, b_xyz):
    return float(np.linalg.norm(np.asarray(a_xyz[:2], dtype=np.float32)
                                - np.asarray(b_xyz[:2], dtype=np.float32)))


def match_classwise(gt_items, det_items, dist_th):
    if not gt_items or not det_items:
        return [], list(range(len(gt_items))), list(range(len(det_items)))
    distances = np.asarray([
        [center_distance_2d(gt["global_xyz"], det["global_xyz"]) for det in det_items]
        for gt in gt_items
    ], dtype=np.float64)
    unmatched_cost = (min(len(gt_items), len(det_items)) + 1) * dist_th + 1.0
    cost = np.full((len(gt_items), len(det_items) + len(gt_items)), unmatched_cost)
    cost[:, :len(det_items)] = np.where(
        np.isfinite(distances) & (distances <= dist_th), distances, 2 * unmatched_cost
    )
    rows, columns = linear_sum_assignment(cost)
    matches = [(r, c, float(distances[r, c])) for r, c in zip(rows, columns)
               if c < len(det_items) and distances[r, c] <= dist_th]
    matched_gt = {r for r, _, _ in matches}
    matched_det = {c for _, c, _ in matches}
    return (matches, [i for i in range(len(gt_items)) if i not in matched_gt],
            [i for i in range(len(det_items)) if i not in matched_det])


def match_ambiguity_diagnostics(gt_items, det_items, gt_index, det_index, distance):
    other_det = [center_distance_2d(gt_items[gt_index]["global_xyz"], det["global_xyz"])
                 for index, det in enumerate(det_items) if index != det_index]
    other_gt = [center_distance_2d(gt["global_xyz"], det_items[det_index]["global_xyz"])
                for index, gt in enumerate(gt_items) if index != gt_index]
    row_gap = min(other_det) - distance if other_det else None
    col_gap = min(other_gt) - distance if other_gt else None
    gaps = [gap for gap in (row_gap, col_gap) if gap is not None]
    return {"match_row_margin": row_gap, "match_column_margin": col_gap,
            "match_identity_margin": min(gaps) if gaps else None}
