from typing import Iterable, List, Optional, Sequence, Tuple


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
            tentative_indices.append(i)
        elif bucket == "low":
            low_indices.append(i)

    return high_indices, tentative_indices, low_indices
