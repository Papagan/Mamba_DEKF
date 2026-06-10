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


def normalize_weights(weight_dicts: Dict[str, Dict[int, float]], class_id: int):
    vals = [max(0.0, float(weight_dicts[name][class_id])) for name in ["W_DET", "W_ASSOC", "W_CONT", "W_MATURE"]]
    total = sum(vals)
    if total <= 1e-8:
        vals = [0.25, 0.25, 0.25, 0.25]
        total = 1.0
    vals = [v / total for v in vals]
    for name, value in zip(["W_DET", "W_ASSOC", "W_CONT", "W_MATURE"], vals):
        weight_dicts[name][class_id] = round(value, 4)


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

    if agg_amota_delta >= 0.01:
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


def _adjust_track_score_for_class(cfg, report, class_name: str, class_id: int, *, small_weak: bool, diagnostics: Dict):
    track_score = cfg["TRACK_SCORE"]
    reason = "ranking gain indicates track-score sorting is the primary bottleneck"
    if small_weak:
        old = track_score["W_DET"][class_id]
        track_score["W_DET"][class_id] = old + 0.06
        _record(report, "track_score", "TRACK_SCORE.W_DET", class_name, old, track_score["W_DET"][class_id], reason)
        old = track_score["W_ASSOC"][class_id]
        track_score["W_ASSOC"][class_id] = max(0.01, old - 0.04)
        _record(report, "track_score", "TRACK_SCORE.W_ASSOC", class_name, old, track_score["W_ASSOC"][class_id], reason)
        old = track_score["W_CONT"][class_id]
        track_score["W_CONT"][class_id] = max(0.01, old - 0.04)
        _record(report, "track_score", "TRACK_SCORE.W_CONT", class_name, old, track_score["W_CONT"][class_id], reason)
        old = track_score["W_MATURE"][class_id]
        track_score["W_MATURE"][class_id] = old + 0.02
        _record(report, "track_score", "TRACK_SCORE.W_MATURE", class_name, old, track_score["W_MATURE"][class_id], reason)
        old = track_score["CURRENT_FAKE_SCALE"][class_id]
        track_score["CURRENT_FAKE_SCALE"][class_id] = clamp(old + 0.04, 0.55, 0.90)
        _record(report, "track_score", "TRACK_SCORE.CURRENT_FAKE_SCALE", class_name, old, track_score["CURRENT_FAKE_SCALE"][class_id], reason)
        old = track_score["MATURE_LEN"][class_id]
        track_score["MATURE_LEN"][class_id] = max(2, old - 1)
        _record(report, "track_score", "TRACK_SCORE.MATURE_LEN", class_name, old, track_score["MATURE_LEN"][class_id], reason)
    else:
        old = track_score["W_DET"][class_id]
        track_score["W_DET"][class_id] = max(0.01, old - 0.04)
        _record(report, "track_score", "TRACK_SCORE.W_DET", class_name, old, track_score["W_DET"][class_id], reason)
        old = track_score["W_ASSOC"][class_id]
        track_score["W_ASSOC"][class_id] = old + 0.03
        _record(report, "track_score", "TRACK_SCORE.W_ASSOC", class_name, old, track_score["W_ASSOC"][class_id], reason)
        old = track_score["W_CONT"][class_id]
        track_score["W_CONT"][class_id] = old + 0.03
        _record(report, "track_score", "TRACK_SCORE.W_CONT", class_name, old, track_score["W_CONT"][class_id], reason)
        old = track_score["W_MATURE"][class_id]
        track_score["W_MATURE"][class_id] = old + 0.01
        _record(report, "track_score", "TRACK_SCORE.W_MATURE", class_name, old, track_score["W_MATURE"][class_id], reason)
        old = track_score["CURRENT_FAKE_SCALE"][class_id]
        track_score["CURRENT_FAKE_SCALE"][class_id] = clamp(old - 0.04, 0.55, 0.90)
        _record(report, "track_score", "TRACK_SCORE.CURRENT_FAKE_SCALE", class_name, old, track_score["CURRENT_FAKE_SCALE"][class_id], reason)
    normalize_weights(track_score, class_id)


def _apply_matching_lifecycle_rules(cfg, comparison, report):
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
            input_online[class_id] = clamp(old - 0.02, 0.0, 0.5)
            input_offline[class_id] = input_online[class_id]
            _record(report, "matching_lifecycle", "THRESHOLD.INPUT_SCORE", class_name, old, input_online[class_id], reason)
            old = confirmed[class_id]
            confirmed[class_id] = clamp(old - 0.02, 0.2, 0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name, old, confirmed[class_id], reason)
            old = output_score[class_id]
            output_score[class_id] = clamp(old - 0.03, 0.2, 0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name, old, output_score[class_id], reason)
            old = cost_thre[class_id]
            cost_thre[class_id] = clamp(old + 0.05, 0.8, 2.5)
            _record(report, "matching_lifecycle", "THRESHOLD.BEV.COST_THRE", class_name, old, cost_thre[class_id], reason)
        elif recall >= 0.75 and mota < 0.55:
            reason = "high recall but weak mota indicates too many noisy births / false continuations"
            old = input_online[class_id]
            input_online[class_id] = clamp(old + 0.02, 0.0, 0.5)
            input_offline[class_id] = input_online[class_id]
            _record(report, "matching_lifecycle", "THRESHOLD.INPUT_SCORE", class_name, old, input_online[class_id], reason)
            old = output_score[class_id]
            output_score[class_id] = clamp(old + 0.02, 0.2, 0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.OUTPUT_SCORE", class_name, old, output_score[class_id], reason)
            old = confirmed[class_id]
            confirmed[class_id] = clamp(old + 0.01, 0.2, 0.8)
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.CONFIRMED_DET_SCORE", class_name, old, confirmed[class_id], reason)
            old = cost_thre[class_id]
            cost_thre[class_id] = clamp(old - 0.05, 0.8, 2.5)
            _record(report, "matching_lifecycle", "THRESHOLD.BEV.COST_THRE", class_name, old, cost_thre[class_id], reason)
            old = birth_score.get(class_id, None)
            new_birth = clamp(max(input_online[class_id] + 0.08, birth_score.get(class_id, 0.0)), 0.1, 0.85)
            birth_score[class_id] = new_birth
            _record(report, "matching_lifecycle", "THRESHOLD.TRAJECTORY_THRE.SINGLE_STAGE_BIRTH_SCORE", class_name, old, new_birth, reason)


def apply_suggestions(cfg: Dict, comparison: Dict, calibration: Dict, diagnostics: Dict) -> Tuple[Dict, Dict]:
    new_cfg = copy.deepcopy(cfg)
    report = {
        "strategy": diagnostics["strategy"],
        "diagnostics": diagnostics,
        "changes": [],
    }
    cmap = new_cfg["CATEGORY_MAP_TO_NUMBER"]

    if diagnostics["strategy"] == "global_track_score":
        for class_name, class_id in cmap.items():
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                small_weak=class_name in ("bicycle", "motorcycle"),
                diagnostics=diagnostics,
            )
    elif diagnostics["strategy"] == "weak_class_track_score":
        for class_name in WEAK_CLASSES:
            class_id = cmap[class_name]
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                small_weak=class_name in ("bicycle", "motorcycle"),
                diagnostics=diagnostics,
            )
    elif diagnostics["strategy"] == "matching_lifecycle":
        _apply_matching_lifecycle_rules(new_cfg, comparison, report)
    else:  # hybrid_light
        for class_name in WEAK_CLASSES:
            class_id = cmap[class_name]
            _adjust_track_score_for_class(
                new_cfg,
                report,
                class_name,
                class_id,
                small_weak=class_name in ("bicycle", "motorcycle"),
                diagnostics=diagnostics,
            )
        _apply_matching_lifecycle_rules(new_cfg, comparison, report)

    return new_cfg, report


def main():
    args = parse_args()
    calibration = load_json(args.calibration)
    comparison = load_json(args.comparison)
    cfg = load_yaml(args.config)

    diagnostics = compute_diagnostics(comparison, calibration)
    new_cfg, report = apply_suggestions(cfg, comparison, calibration, diagnostics)
    report["meta"] = {
        "calibration": os.path.abspath(args.calibration),
        "comparison": os.path.abspath(args.comparison),
        "config": os.path.abspath(args.config),
        "output": os.path.abspath(args.output),
    }

    save_yaml(args.output, new_cfg)
    save_json(args.report, report)
    print(f"[suggest-config] strategy={diagnostics['strategy']} changes={len(report['changes'])}")
    print(f"[suggest-config] wrote suggested config to {args.output}")
    print(f"[suggest-config] wrote report to {args.report}")


if __name__ == "__main__":
    main()
