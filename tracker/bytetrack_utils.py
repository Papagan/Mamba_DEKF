from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union


ThresholdShape = Union[Dict[int, float], Sequence[float]]


def build_stage2_relaxed_thresholds(
    base_thresholds: ThresholdShape,
    relax_ratio: float = 1.0,
    per_class_overrides: Optional[Dict[int, float]] = None,
):
    """
    Build stage-2 rescue thresholds while preserving the original container shape.

    Dict inputs stay dicts so per-class threshold lookups remain stable.
    Optional overrides replace the relaxed value for specific classes.
    """
    per_class_overrides = per_class_overrides or {}

    if isinstance(base_thresholds, dict):
        relaxed = {
            cls_id: float(threshold) * float(relax_ratio)
            for cls_id, threshold in base_thresholds.items()
        }
        for cls_id, threshold in per_class_overrides.items():
            if cls_id in relaxed:
                relaxed[cls_id] = float(threshold)
        return relaxed

    relaxed = [float(threshold) * float(relax_ratio) for threshold in base_thresholds]
    for cls_id, threshold in per_class_overrides.items():
        if 0 <= int(cls_id) < len(relaxed):
            relaxed[int(cls_id)] = float(threshold)
    return relaxed


def build_stage2_rescue_groups(
    unmatched_traj_indices: Sequence[int],
    trajs: Sequence[object],
    rescue_det_indices: Sequence[int],
    dets: Sequence[object],
    category_map: dict,
) -> List[Tuple[int, List[int], List[int]]]:
    """
    Group unmatched trajectories and stage-2 rescue detections by class.

    This keeps stage-2 geometric rescue from depending on cross-class ordering
    when the geometric matcher consumes per-class threshold/state settings.
    """
    traj_groups: Dict[int, List[int]] = {}
    det_groups: Dict[int, List[int]] = {}

    for traj_idx in unmatched_traj_indices:
        cls_id = int(getattr(trajs[traj_idx], "category_num", -1))
        traj_groups.setdefault(cls_id, []).append(int(traj_idx))

    for det_idx in rescue_det_indices:
        det = dets[det_idx]
        cls_id = int(category_map.get(det.category, -1))
        det_groups.setdefault(cls_id, []).append(int(det_idx))

    shared_classes = sorted(set(traj_groups.keys()) & set(det_groups.keys()))
    return [
        (cls_id, traj_groups[cls_id], det_groups[cls_id])
        for cls_id in shared_classes
    ]


def classify_single_stage_birth(
    category: str,
    score: float,
    category_map: dict,
    birth_gate_cfg: Optional[dict] = None,
) -> bool:
    """
    Decide whether an unmatched detection may create a new track in
    single-stage mode.

    Behavior:
      - If a class has no configured gate, keep legacy single-stage behavior
        and always allow birth.
      - If a class has a configured gate, require score >= gate.
    """
    birth_gate_cfg = birth_gate_cfg or {}
    cat_num = category_map.get(category, None)
    if cat_num is None:
        return True

    birth_gate = birth_gate_cfg.get(cat_num, None)
    if birth_gate is None:
        return True
    return float(score) >= float(birth_gate)


def classify_bytetrack_score(
    score: float,
    birth_score: float,
    tentative_birth_score: Optional[float] = None,
    low_score_floor: float = 0.1,
) -> Optional[str]:
    """
    Classify a detection score for ByteTrack-style staging.

    Returns:
        "high"       : eligible for stage-1 matching and strict birth.
        "tentative"  : eligible for stage-2 matching and tentative birth.
        "low"        : eligible for stage-2 rescue only.
        None         : ignored entirely.
    """
    if score < low_score_floor:
        return None
    if score >= birth_score:
        return "high"
    if tentative_birth_score is None:
        return "low"
    if tentative_birth_score >= birth_score:
        return "low"
    if score >= tentative_birth_score:
        return "tentative"
    return "low"


def split_bytetrack_detections(
    dets: Sequence[object],
    category_map: dict,
    birth_cfg: dict,
    tentative_birth_cfg: Optional[dict] = None,
    low_score_floor: float = 0.1,
    tentative_birth_enabled: bool = True,
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split detections into high / tentative / low groups.

    The detection object only needs `.category` and `.det_score`.
    """
    high_indices: List[int] = []
    tentative_indices: List[int] = []
    low_indices: List[int] = []
    tentative_birth_cfg = tentative_birth_cfg or {}

    for i, det in enumerate(dets):
        cat_num = category_map.get(det.category, 0)
        birth_score = float(birth_cfg.get(cat_num, 0.4))
        tentative_score = tentative_birth_cfg.get(cat_num, birth_score)
        bucket = classify_bytetrack_score(
            float(det.det_score),
            birth_score=birth_score,
            tentative_birth_score=float(tentative_score),
            low_score_floor=low_score_floor,
        )
        if bucket == "high":
            high_indices.append(i)
        elif bucket == "tentative":
            if tentative_birth_enabled:
                tentative_indices.append(i)
            else:
                low_indices.append(i)
        elif bucket == "low":
            low_indices.append(i)

    return high_indices, tentative_indices, low_indices
