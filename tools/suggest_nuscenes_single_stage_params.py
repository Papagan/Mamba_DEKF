#!/usr/bin/env python3
import argparse
import copy
import json
import os
from typing import Dict, List, Tuple

import yaml


WEAK_CLASSES = ["bicycle", "motorcycle", "trailer", "truck"]
STRONG_CLASSES = ["bus", "car", "pedestrian"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Suggest config/nuscenes_single_stage.yaml updates from calibration and comparison results."
    )
    parser.add_argument("--calibration", required=True, help="Path to track_score_calibration.json")
    parser.add_argument("--comparison", required=True, help="Path to comparison_summary.json")
    parser.add_argument("--config", required=True, help="Path to current nuscenes_single_stage.yaml")
    parser.add_argument("--output", required=True, help="Path to suggested YAML output")
    parser.add_argument("--report", required=True, help="Path to JSON report output")
    parser.add_argument("--history", default="", help="Optional optimization history JSON path")
    parser.add_argument(
        "--feedback-comparison",
        default="",
        help="Optional comparison_summary.json from the last deployed suggestion vs its baseline; used to update per-parameter step scales",
    )
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(path, payload):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def save_json(path, payload):
    out_dir = os.path.dirname(os.path.abspath(path))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def mean(xs: List[float]) -> float:
    if not xs:
        return 0.0
    return float(sum(xs) / len(xs))


def clamp(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def scale(delta: float, denom: float, hi: float = 1.5) -> float:
    if denom <= 1e-8:
        return 0.0
    return clamp(delta / denom, 0.0, hi)


def normalize_weights(weight_dicts: Dict[str, Dict[int, float]], class_id: int):
    vals = [max(0.0, float(weight_dicts[name][class_id])) for name in ["W_DET", "W_ASSOC", "W_CONT", "W_MATURE"]]
    total = sum(vals)
    if total <= 1e-8:
        vals = [0.25, 0.25, 0.25, 0.25]
        total = 1.0
    vals = [v / total for v in vals]
    for name, value in zip(["W_DET", "W_ASSOC", "W_CONT", "W_MATURE"], vals):
        weight_dicts[name][class_id] = round(value, 4)


def _param_key(path: str, class_name: str | None) -> str:
    return f"{path}|{class_name or '__global__'}"


def _sign(delta: float) -> int:
    if delta > 1e-12:
        return 1
    if delta < -1e-12:
        return -1
    return 0


def load_history(path: str) -> Dict:
    if not path or not os.path.exists(path):
        return {"entries": [], "param_state": {}}
    return load_json(path)


def _ensure_direction_state(param_state: Dict, key: str) -> Dict:
    state = param_state.setdefault(key, {})
    state.setdefault("1", 1.0)
    state.setdefault("-1", 1.0)
    return state


def get_directional_scale(history: Dict, path: str, class_name: str | None, delta: float) -> float:
    sign = _sign(delta)
    if sign == 0:
        return 1.0
    key = _param_key(path, class_name)
    state = _ensure_direction_state(history.setdefault("param_state", {}), key)
    return float(state[str(sign)])


def apply_scaled_delta(current: float, desired_delta: float, *, history: Dict, path: str, class_name: str | None, lo: float, hi: float) -> float:
    scale_factor = get_directional_scale(history, path, class_name, desired_delta)
    return clamp(current + desired_delta * scale_factor, lo, hi)


def update_history_with_feedback(history: Dict, feedback_comparison: Dict) -> Dict:
    entries = history.setdefault("entries", [])
    if not entries:
        return history
    last_entry = entries[-1]
    feedback = {
        "comparison": feedback_comparison,
        "aggregate_amota_delta": float(feedback_comparison.get("aggregate", {}).get("amota", {}).get("delta", 0.0)),
    }
    last_entry["feedback"] = feedback

    agg_delta = feedback["aggregate_amota_delta"]
    factor = 0.8 if agg_delta > 0 else 0.5 if agg_delta < 0 else 1.0
    param_state = history.setdefault("param_state", {})

    for change in last_entry.get("changes", []):
        old = change.get("old")
        new = change.get("new")
        if old is None or new is None:
            continue
        direction = _sign(float(new) - float(old))
        if direction == 0:
            continue
        key = _param_key(change["path"], change.get("class_name"))
        state = _ensure_direction_state(param_state, key)
        state[str(direction)] = clamp(float(state[str(direction)]) * factor, 0.2, 1.5)

    return history


def calibration_reliance(calibration: Dict) -> Dict[str, float]:
    feature_names = calibration.get("feature_names", [])
    weights = calibration.get("weights", [])
    abs_map = {name: abs(float(w)) for name, w in zip(feature_names, weights)}
    total = max(sum(abs_map.values()), 1e-8)
    score_rel = sum(abs_map.get(name, 0.0) for name in ["score_mean", "score_last", "score_std"]) / total
    quality_rel = sum(abs_map.get(name, 0.0) for name in ["tp_ratio", "purity", "dominant_recall"]) / total
    assoc_rel = abs_map.get("mean_match_dist", 0.0) / total
    cont_rel = sum(abs_map.get(name, 0.0) for name in ["gap_count", "num_frames", "duration_sec"]) / total
    return {
        "score_reliance": float(score_rel),
        "quality_reliance": float(quality_rel),
        "assoc_reliance": float(assoc_rel),
        "cont_reliance": float(cont_rel),
    }


def compute_diagnostics(comparison: Dict, calibration: Dict) -> Dict:
    agg_amota_delta = float(comparison["aggregate"]["amota"]["delta"])
    weak_mean_delta = mean([float(comparison["per_class"][cls]["amota"]["delta"]) for cls in WEAK_CLASSES if cls in comparison["per_class"]])
    strong_mean_delta = mean([float(comparison["per_class"][cls]["amota"]["delta"]) for cls in STRONG_CLASSES if cls in comparison["per_class"]])
    rel = calibration_reliance(calibration)
    agg_gain_scale = scale(agg_amota_delta, 0.10)
    weak_gain_scale = scale(weak_mean_delta, 0.20)
    strong_gain_scale = scale(strong_mean_delta, 0.08)
    weak_advantage_scale = scale(weak_mean_delta - strong_mean_delta, 0.15)

    if agg_amota_delta >= 0.08 or weak_mean_delta >= 0.15:
        strategy = "aggressive_weak_class_track_score"
    elif agg_amota_delta >= 0.01:
        if weak_mean_delta >= 0.01 and strong_mean_delta < 0.005:
            strategy = "weak_class_track_score"
        else:
            strategy = "global_track_score"
    elif abs(agg_amota_delta) < 0.005:
        strategy = "matching_lifecycle"
    elif weak_mean_delta > strong_mean_delta + 0.01:
        strategy = "weak_class_track_score"
    else:
        strategy = "hybrid_light"

    return {
        "strategy": strategy,
        "aggregate_amota_delta": agg_amota_delta,
        "weak_mean_delta": weak_mean_delta,
        "strong_mean_delta": strong_mean_delta,
        "agg_gain_scale": agg_gain_scale,
        "weak_gain_scale": weak_gain_scale,
        "strong_gain_scale": strong_gain_scale,
        "weak_advantage_scale": weak_advantage_scale,
        **rel,
    }


def _record(report, group: str, path: str, class_name: str | None, old, new, reason: str):
    if old == new:
        return
    report["changes"].append(
        {
            "group": group,
            "path": path,
            "class_name": class_name,
            "old": old,
            "new": new,
            "reason": reason,
        }
    )


def _adjust_track_score_for_class(cfg, report, class_name: str, class_id: int, *, diagnostics: Dict, history: Dict):
    track_score = cfg["TRACK_SCORE"]
    reason = "ranking gain indicates track-score sorting is the primary bottleneck"
    agg_gain = diagnostics.get("agg_gain_scale", 0.0)
    weak_gain = diagnostics.get("weak_gain_scale", 0.0)
    score_rel = diagnostics.get("score_reliance", 0.0)
    assoc_rel = diagnostics.get("assoc_reliance", 0.0)
    cont_rel = diagnostics.get("cont_reliance", 0.0)
    quality_rel = diagnostics.get("quality_reliance", 0.0)
    if class_name in ("bicycle", "motorcycle"):
        old = track_score["W_DET"][class_id]
        desired_delta = 0.04 * agg_gain + 0.04 * score_rel
        track_score["W_DET"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_DET", class_name=class_name, lo=0.0, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_DET", class_name, old, track_score["W_DET"][class_id], reason)
        old = track_score["W_ASSOC"][class_id]
        desired_delta = -(0.03 * weak_gain + 0.02 * assoc_rel)
        track_score["W_ASSOC"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_ASSOC", class_name=class_name, lo=0.01, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_ASSOC", class_name, old, track_score["W_ASSOC"][class_id], reason)
        old = track_score["W_CONT"][class_id]
        desired_delta = -(0.03 * weak_gain + 0.02 * cont_rel)
        track_score["W_CONT"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_CONT", class_name=class_name, lo=0.01, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_CONT", class_name, old, track_score["W_CONT"][class_id], reason)
        old = track_score["W_MATURE"][class_id]
        desired_delta = 0.02 * weak_gain + 0.02 * quality_rel
        track_score["W_MATURE"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_MATURE", class_name=class_name, lo=0.0, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_MATURE", class_name, old, track_score["W_MATURE"][class_id], reason)
        old = track_score["CURRENT_FAKE_SCALE"][class_id]
        desired_delta = 0.04 * weak_gain
        track_score["CURRENT_FAKE_SCALE"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.CURRENT_FAKE_SCALE", class_name=class_name, lo=0.55, hi=0.90)
        _record(report, "track_score", "TRACK_SCORE.CURRENT_FAKE_SCALE", class_name, old, track_score["CURRENT_FAKE_SCALE"][class_id], reason)
        old = track_score["MATURE_LEN"][class_id]
        desired_delta = -float(int(round(max(0.5, weak_gain))))
        track_score["MATURE_LEN"][class_id] = int(round(apply_scaled_delta(float(old), desired_delta, history=history, path="TRACK_SCORE.MATURE_LEN", class_name=class_name, lo=2.0, hi=12.0)))
        _record(report, "track_score", "TRACK_SCORE.MATURE_LEN", class_name, old, track_score["MATURE_LEN"][class_id], reason)
    else:
        old = track_score["W_DET"][class_id]
        desired_delta = -(0.03 * agg_gain + 0.02 * diagnostics.get("strong_gain_scale", 0.0))
        track_score["W_DET"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_DET", class_name=class_name, lo=0.01, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_DET", class_name, old, track_score["W_DET"][class_id], reason)
        old = track_score["W_ASSOC"][class_id]
        desired_delta = 0.02 * agg_gain + 0.02 * assoc_rel
        track_score["W_ASSOC"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_ASSOC", class_name=class_name, lo=0.01, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_ASSOC", class_name, old, track_score["W_ASSOC"][class_id], reason)
        old = track_score["W_CONT"][class_id]
        desired_delta = 0.02 * agg_gain + 0.02 * cont_rel
        track_score["W_CONT"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_CONT", class_name=class_name, lo=0.01, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_CONT", class_name, old, track_score["W_CONT"][class_id], reason)
        old = track_score["W_MATURE"][class_id]
        desired_delta = 0.01 * agg_gain + 0.01 * quality_rel
        track_score["W_MATURE"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.W_MATURE", class_name=class_name, lo=0.0, hi=1.0)
        _record(report, "track_score", "TRACK_SCORE.W_MATURE", class_name, old, track_score["W_MATURE"][class_id], reason)
        old = track_score["CURRENT_FAKE_SCALE"][class_id]
        desired_delta = -(0.03 * agg_gain)
        track_score["CURRENT_FAKE_SCALE"][class_id] = apply_scaled_delta(old, desired_delta, history=history, path="TRACK_SCORE.CURRENT_FAKE_SCALE", class_name=class_name, lo=0.55, hi=0.90)
        _record(report, "track_score", "TRACK_SCORE.CURRENT_FAKE_SCALE", class_name, old, track_score["CURRENT_FAKE_SCALE"][class_id], reason)
    normalize_weights(track_score, class_id)


def _apply_matching_lifecycle_rules(cfg, comparison, report, history: Dict):
    cmap = cfg["CATEGORY_MAP_TO_NUMBER"]
    input_online = cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]
    input_offline = cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
    confirmed = cfg["THRESHOLD"]["TRAJECTORY_THRE"]["CONFIRMED_DET_SCORE"]
    output_score = cfg["THRESHOLD"]["TRAJECTORY_THRE"]["OUTPUT_SCORE"]
    birth_score = cfg["THRESHOLD"]["TRAJECTORY_THRE"]["SINGLE_STAGE_BIRTH_SCORE"]
    cost_thre = cfg["THRESHOLD"]["BEV"]["COST_THRE"]

    for class_name, class_id in cmap.items():
        cls_metrics = comparison["per_class"].get(class_name)
        if cls_metrics is None:
            continue
        recall = float(cls_metrics["recall"]["orig"])
        mota = float(cls_metrics["mota"]["orig"])

        if recall < 0.65:
            reason = "low recall and weak calibration gain indicate under-matching / over-gating"
            old = input_online[class_id]
            input_online[class_id] = apply_scaled_delta(old, -0.02, history=history, path="THRESHOLD.INPUT_SCORE", class_name=class_name, lo=0.0, hi=0.5)
            input_offline[class_id] = input_online[class_id]
            _record(report, "matching_lifecycle", "THRESHOLD.INPUT_SCORE", class_name, old, input_online[class_id], reason)
            old = confirmed[class_id]
            confirmed[class_id] = apply_scaled_delta(old, -0.02, history=history, path="THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name=class_name, lo=0.2, hi=0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name, old, confirmed[class_id], reason)
            old = output_score[class_id]
            output_score[class_id] = apply_scaled_delta(old, -0.03, history=history, path="THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name=class_name, lo=0.2, hi=0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name, old, output_score[class_id], reason)
            old = cost_thre[class_id]
            cost_thre[class_id] = apply_scaled_delta(old, 0.05, history=history, path="THRESHOLD.BEV.COST_THRE", class_name=class_name, lo=0.8, hi=2.5)
            _record(report, "matching_lifecycle", "THRESHOLD.BEV.COST_THRE", class_name, old, cost_thre[class_id], reason)
        elif recall >= 0.75 and mota < 0.55:
            reason = "high recall but weak mota indicates too many noisy births / false continuations"
            old = input_online[class_id]
            input_online[class_id] = apply_scaled_delta(old, 0.02, history=history, path="THRESHOLD.INPUT_SCORE", class_name=class_name, lo=0.0, hi=0.5)
            input_offline[class_id] = input_online[class_id]
            _record(report, "matching_lifecycle", "THRESHOLD.INPUT_SCORE", class_name, old, input_online[class_id], reason)
            old = output_score[class_id]
            output_score[class_id] = apply_scaled_delta(old, 0.02, history=history, path="THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name=class_name, lo=0.2, hi=0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name, old, output_score[class_id], reason)
            old = confirmed[class_id]
            confirmed[class_id] = apply_scaled_delta(old, 0.01, history=history, path="THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name=class_name, lo=0.2, hi=0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name, old, confirmed[class_id], reason)
            old = cost_thre[class_id]
            cost_thre[class_id] = apply_scaled_delta(old, -0.05, history=history, path="THRESHOLD.BEV.COST_THRE", class_name=class_name, lo=0.8, hi=2.5)
            _record(report, "matching_lifecycle", "THRESHOLD.BEV.COST_THRE", class_name, old, cost_thre[class_id], reason)
            old = birth_score.get(class_id, None)
            desired_delta = max(input_online[class_id] + 0.08, birth_score.get(class_id, 0.0)) - float(birth_score.get(class_id, 0.0))
            new_birth = apply_scaled_delta(float(birth_score.get(class_id, 0.0)), desired_delta, history=history, path="THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name=class_name, lo=0.1, hi=0.85)
            birth_score[class_id] = new_birth
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name, old, new_birth, reason)


def _apply_aggressive_weak_thresholds(cfg, report, history: Dict):
    input_online = cfg["THRESHOLD"]["INPUT_SCORE"]["ONLINE"]
    input_offline = cfg["THRESHOLD"]["INPUT_SCORE"]["OFFLINE"]
    traj = cfg["THRESHOLD"]["TRAJECTORY_THRE"]
    confirmed = traj["CONFIRMED_DET_SCORE"]
    output_score = traj["OUTPUT_SCORE"]
    max_unmatch = traj["MAX_UNMATCH_LENGTH"]
    confirmed_len = traj["CONFIRMED_TRACK_LENGTH"]
    birth_score = traj["SINGLE_STAGE_BIRTH_SCORE"]
    cmap = cfg["CATEGORY_MAP_TO_NUMBER"]
    reason = "large calibrated gain indicates weak-class recall is suppressed by conservative score/lifecycle gates"
    diag = report["diagnostics"]
    weak_gain = diag.get("weak_gain_scale", 0.0)
    weak_adv = diag.get("weak_advantage_scale", 0.0)
    score_rel = diag.get("score_reliance", 0.0)
    quality_rel = diag.get("quality_reliance", 0.0)
    agg_gain = diag.get("agg_gain_scale", 0.0)

    for class_name, class_id in cmap.items():
        if class_name in ("bicycle", "motorcycle"):
            input_drop = 0.025 * weak_gain + 0.02 * score_rel
            confirm_drop = 0.042 * weak_gain + 0.026 * quality_rel
            output_drop = 0.05 * weak_gain + 0.025 * score_rel
            extra_unmatch = int(round(max(1.0, weak_adv)))
            extra_confirm_relax = int(round(max(1.0, weak_adv)))
        elif class_name in ("trailer", "truck"):
            input_drop = 0.02 * weak_gain + 0.015 * score_rel
            confirm_drop = 0.035 * weak_gain + 0.02 * quality_rel
            output_drop = 0.04 * weak_gain + 0.02 * score_rel
            extra_unmatch = int(round(max(1.0, 0.8 * weak_adv)))
            extra_confirm_relax = 0
        elif class_name == "car":
            input_drop = 0.0
            confirm_drop = 0.015 * agg_gain
            output_drop = 0.02 * agg_gain
            extra_unmatch = 0
            extra_confirm_relax = 0
        elif class_name == "pedestrian":
            input_drop = 0.01 * agg_gain
            confirm_drop = 0.015 * agg_gain
            output_drop = 0.02 * agg_gain
            extra_unmatch = 0
            extra_confirm_relax = 0
        else:  # bus
            input_drop = 0.01 * agg_gain
            confirm_drop = 0.015 * agg_gain
            output_drop = 0.02 * agg_gain
            extra_unmatch = 0
            extra_confirm_relax = 0

        old = input_online[class_id]
        input_online[class_id] = apply_scaled_delta(old, -input_drop, history=history, path="THRESHOLD.INPUT_SCORE", class_name=class_name, lo=0.0, hi=0.5)
        input_offline[class_id] = input_online[class_id]
        _record(report, "aggressive_thresholds", "THRESHOLD.INPUT_SCORE", class_name, old, input_online[class_id], reason)

        old = confirmed[class_id]
        confirmed[class_id] = apply_scaled_delta(old, -confirm_drop, history=history, path="THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name=class_name, lo=0.2, hi=0.8)
        _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name, old, confirmed[class_id], reason)

        old = output_score[class_id]
        output_score[class_id] = apply_scaled_delta(old, -output_drop, history=history, path="THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name=class_name, lo=0.2, hi=0.8)
        _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name, old, output_score[class_id], reason)

        if extra_unmatch > 0 and class_name in ("bicycle", "motorcycle", "trailer", "truck"):
            old = max_unmatch[class_id]
            max_unmatch[class_id] = int(round(apply_scaled_delta(float(old), float(extra_unmatch), history=history, path="THRESHOLD.TRAJECTORY_THRE.MAX_UNMATCH_LENGTH", class_name=class_name, lo=0.0, hi=6.0)))
            _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.MAX_UNMATCH_LENGTH", class_name, old, max_unmatch[class_id], reason)

        if extra_confirm_relax > 0 and class_name in ("bicycle", "motorcycle"):
            old = confirmed_len[class_id]
            confirmed_len[class_id] = int(round(apply_scaled_delta(float(old), -float(min(extra_confirm_relax, 2)), history=history, path="THRESHOLD.TRAJECTORY_THRE.CONFIRMED_TRACK_LENGTH", class_name=class_name, lo=1.0, hi=8.0)))
            _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_TRACK_LENGTH", class_name, old, confirmed_len[class_id], reason)

        if class_name == "car":
            old = birth_score.get(class_id, None)
            base = float(birth_score.get(class_id, 0.0))
            desired_delta = max(input_online[class_id] + 0.10, base) - base
            birth_score[class_id] = apply_scaled_delta(base, desired_delta, history=history, path="THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name=class_name, lo=0.1, hi=0.85)
            _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name, old, birth_score[class_id], reason)
        elif class_name == "bus":
            old = birth_score.get(class_id, None)
            base = float(birth_score.get(class_id, 0.0))
            desired_delta = max(input_online[class_id] + 0.08, base) - base
            birth_score[class_id] = apply_scaled_delta(base, desired_delta, history=history, path="THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name=class_name, lo=0.1, hi=0.85)
            _record(report, "aggressive_thresholds", "THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name, old, birth_score[class_id], reason)


def apply_suggestions(cfg: Dict, comparison: Dict, calibration: Dict, diagnostics: Dict, history: Dict | None = None) -> Tuple[Dict, Dict]:
    new_cfg = copy.deepcopy(cfg)
    history = history or {"entries": [], "param_state": {}}
    report = {
        "strategy": diagnostics["strategy"],
        "diagnostics": diagnostics,
        "changes": [],
    }
    cmap = new_cfg["CATEGORY_MAP_TO_NUMBER"]

    if diagnostics["strategy"] == "aggressive_weak_class_track_score":
        for class_name in WEAK_CLASSES:
            class_id = cmap[class_name]
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                diagnostics=diagnostics,
                history=history,
            )
        for class_name in ("bicycle", "motorcycle", "trailer", "truck"):
            class_id = cmap[class_name]
            track_score = new_cfg["TRACK_SCORE"]
            reason = "very large calibrated gain indicates the weak-class track-score model must be shifted aggressively toward det confidence and maturity"
            weak_gain = diagnostics.get("weak_gain_scale", 0.0)
            score_rel = diagnostics.get("score_reliance", 0.0)
            assoc_rel = diagnostics.get("assoc_reliance", 0.0)
            cont_rel = diagnostics.get("cont_reliance", 0.0)
            quality_rel = diagnostics.get("quality_reliance", 0.0)
            old = track_score["W_DET"][class_id]
            det_bump = (0.03 if class_name in ("trailer", "truck") else 0.04) * weak_gain + 0.02 * score_rel
            track_score["W_DET"][class_id] = apply_scaled_delta(old, det_bump, history=history, path="TRACK_SCORE.W_DET", class_name=class_name, lo=0.0, hi=1.0)
            _record(report, "track_score", "TRACK_SCORE.W_DET", class_name, old, track_score["W_DET"][class_id], reason)
            old = track_score["W_ASSOC"][class_id]
            assoc_drop = ((0.015 if class_name in ("trailer", "truck") else 0.02) * weak_gain + 0.01 * assoc_rel)
            track_score["W_ASSOC"][class_id] = apply_scaled_delta(old, -assoc_drop, history=history, path="TRACK_SCORE.W_ASSOC", class_name=class_name, lo=0.01, hi=1.0)
            _record(report, "track_score", "TRACK_SCORE.W_ASSOC", class_name, old, track_score["W_ASSOC"][class_id], reason)
            old = track_score["W_CONT"][class_id]
            cont_drop = ((0.015 if class_name in ("trailer", "truck") else 0.02) * weak_gain + 0.01 * cont_rel)
            track_score["W_CONT"][class_id] = apply_scaled_delta(old, -cont_drop, history=history, path="TRACK_SCORE.W_CONT", class_name=class_name, lo=0.01, hi=1.0)
            _record(report, "track_score", "TRACK_SCORE.W_CONT", class_name, old, track_score["W_CONT"][class_id], reason)
            old = track_score["W_MATURE"][class_id]
            mature_bump = (0.015 if class_name in ("trailer", "truck") else 0.02) * weak_gain + 0.01 * quality_rel
            track_score["W_MATURE"][class_id] = apply_scaled_delta(old, mature_bump, history=history, path="TRACK_SCORE.W_MATURE", class_name=class_name, lo=0.0, hi=1.0)
            _record(report, "track_score", "TRACK_SCORE.W_MATURE", class_name, old, track_score["W_MATURE"][class_id], reason)
            old = track_score["CURRENT_FAKE_SCALE"][class_id]
            fake_bump = (0.05 if class_name in ("bicycle", "motorcycle") else 0.06) * weak_gain
            track_score["CURRENT_FAKE_SCALE"][class_id] = apply_scaled_delta(old, fake_bump, history=history, path="TRACK_SCORE.CURRENT_FAKE_SCALE", class_name=class_name, lo=0.55, hi=0.90)
            _record(report, "track_score", "TRACK_SCORE.CURRENT_FAKE_SCALE", class_name, old, track_score["CURRENT_FAKE_SCALE"][class_id], reason)
            old = track_score["MATURE_LEN"][class_id]
            mature_delta = -float(int(round(max(0.8, diagnostics.get("weak_advantage_scale", 0.0)))))
            track_score["MATURE_LEN"][class_id] = int(round(apply_scaled_delta(float(old), mature_delta, history=history, path="TRACK_SCORE.MATURE_LEN", class_name=class_name, lo=2.0, hi=12.0)))
            _record(report, "track_score", "TRACK_SCORE.MATURE_LEN", class_name, old, track_score["MATURE_LEN"][class_id], reason)
            normalize_weights(track_score, class_id)
        _apply_aggressive_weak_thresholds(new_cfg, report, history)
    elif diagnostics["strategy"] == "global_track_score":
        for class_name, class_id in cmap.items():
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                diagnostics=diagnostics,
                history=history,
            )
    elif diagnostics["strategy"] == "weak_class_track_score":
        for class_name in WEAK_CLASSES:
            class_id = cmap[class_name]
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                diagnostics=diagnostics,
                history=history,
            )
    elif diagnostics["strategy"] == "matching_lifecycle":
        _apply_matching_lifecycle_rules(new_cfg, comparison, report, history)
    else:  # hybrid_light
        for class_name in WEAK_CLASSES:
            class_id = cmap[class_name]
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                diagnostics=diagnostics,
                history=history,
            )
        _apply_matching_lifecycle_rules(new_cfg, comparison, report, history)

    return new_cfg, report


def main():
    args = parse_args()
    calibration = load_json(args.calibration)
    comparison = load_json(args.comparison)
    cfg = load_yaml(args.config)
    history = load_history(args.history)
    if args.feedback_comparison:
        feedback_payload = load_json(args.feedback_comparison)
        history = update_history_with_feedback(history, feedback_payload)

    diagnostics = compute_diagnostics(comparison, calibration)
    new_cfg, report = apply_suggestions(cfg, comparison, calibration, diagnostics, history=history)
    report["meta"] = {
        "calibration": os.path.abspath(args.calibration),
        "comparison": os.path.abspath(args.comparison),
        "config": os.path.abspath(args.config),
        "output": os.path.abspath(args.output),
        "history": os.path.abspath(args.history) if args.history else "",
        "feedback_comparison": os.path.abspath(args.feedback_comparison) if args.feedback_comparison else "",
    }

    if args.history:
        history["entries"].append(
            {
                "diagnostics": diagnostics,
                "changes": report["changes"],
                "meta": report["meta"],
            }
        )

    save_yaml(args.output, new_cfg)
    save_json(args.report, report)
    if args.history:
        save_json(args.history, history)
    print(f"[suggest-config] strategy={diagnostics['strategy']} changes={len(report['changes'])}")
    print(f"[suggest-config] wrote suggested config to {args.output}")
    print(f"[suggest-config] wrote report to {args.report}")
    if args.history:
        print(f"[suggest-config] updated history at {args.history}")


if __name__ == "__main__":
    main()
