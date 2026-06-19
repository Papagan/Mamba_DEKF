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


def map_class_to_profile(class_id: int) -> str:
    return _CLASS_TO_PROFILE[int(class_id)]


def infer_state_bucket(unmatch_length: int) -> str:
    return STATE_MATCHED if int(unmatch_length) == 0 else STATE_UNMATCHED


def clamp_ratio_value(value: float, *, min_ratio: float, max_ratio: float) -> float:
    return max(min_ratio, min(max_ratio, float(value)))
