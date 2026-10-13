"""Shared MCTrack-compatible posterior corrections for position and size."""

from __future__ import annotations


def validate_exact_update_contract(train_cfg: dict, inference_cfg: dict) -> None:
    """Fail before training if online exact-update classes differ from rollout."""
    trained = ((train_cfg.get("BASE_NOISE", {}) or {}).get("MCTRACK_EXACT_UPDATE", {}) or {})
    compatible = str(inference_cfg.get("TRACKER_COMPAT_MODE", "default")).lower() == "mctrack"
    for state, enabled_key, class_key in (
        ("MATCHED_CLASSES", "MCTRACK_EXACT_MATCHED_UPDATE", "MCTRACK_EXACT_MATCHED_UPDATE_CLASSES"),
        ("UNMATCHED_CLASSES", "MCTRACK_EXACT_UNMATCH_UPDATE", "MCTRACK_EXACT_UNMATCH_UPDATE_CLASSES"),
    ):
        expected = (
            {int(value) for value in inference_cfg.get(class_key, [])}
            if compatible and inference_cfg.get(enabled_key, False) else set()
        )
        actual = {int(value) for value in trained.get(state, [])}
        if actual != expected:
            raise ValueError(
                f"Training BASE_NOISE.MCTRACK_EXACT_UPDATE.{state}={sorted(actual)} "
                f"does not match inference {class_key}={sorted(expected)}"
            )
    if (trained.get("MATCHED_CLASSES") or trained.get("UNMATCHED_CLASSES")) and not bool(
        ((train_cfg.get("BASE_NOISE", {}) or {}).get("DETECTION_UPDATE", {}) or {}).get("ENABLED", True)
    ):
        raise ValueError("Exact MCTrack rollout requires BASE_NOISE.DETECTION_UPDATE.ENABLED=true")


def restore_exact_matched_components(
    predicted_pos_x, predicted_pos_P, predicted_siz_x, predicted_siz_P,
    updated_pos_x, updated_pos_P, updated_siz_x, updated_siz_P,
    exact_mask,
):
    """Keep z/vz and height at their predicted values for selected tracks."""
    if not bool(exact_mask.any().item()):
        return updated_pos_x, updated_pos_P, updated_siz_x, updated_siz_P
    pos_x = updated_pos_x.clone()
    pos_P = updated_pos_P.clone()
    siz_x = updated_siz_x.clone()
    siz_P = updated_siz_P.clone()
    for index in (2, 5):
        pos_x[exact_mask, index, :] = predicted_pos_x[exact_mask, index, :]
        pos_P[exact_mask, index, :] = predicted_pos_P[exact_mask, index, :]
        pos_P[exact_mask, :, index] = predicted_pos_P[exact_mask, :, index]
    siz_x[exact_mask, 2, :] = predicted_siz_x[exact_mask, 2, :]
    siz_P[exact_mask, 2, :] = predicted_siz_P[exact_mask, 2, :]
    siz_P[exact_mask, :, 2] = predicted_siz_P[exact_mask, :, 2]
    return pos_x, pos_P, siz_x, siz_P
