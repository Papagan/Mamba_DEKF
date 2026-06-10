#!/usr/bin/env python3
import argparse
import json
import math
import os

import numpy as np


DEFAULT_FEATURES = [
    "score_mean",
    "score_last",
    "score_std",
    "num_frames",
    "duration_sec",
    "tp_ratio",
    "purity",
    "dominant_recall",
    "straightness",
    "mean_match_dist",
    "gap_count",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit a lightweight post-hoc track-score calibrator from exported track-quality features."
    )
    parser.add_argument("--features", required=True, help="JSON exported by export_track_quality_features.py")
    parser.add_argument("--output", required=True, help="Output calibration JSON path")
    parser.add_argument(
        "--feature-list",
        default=",".join(DEFAULT_FEATURES),
        help="Comma-separated feature names to use",
    )
    parser.add_argument("--lr", type=float, default=0.05, help="Learning rate")
    parser.add_argument("--iters", type=int, default=3000, help="Gradient descent iterations")
    parser.add_argument("--l2", type=float, default=1e-3, help="L2 regularization")
    parser.add_argument(
        "--blend-alpha",
        type=float,
        default=0.75,
        help="Blend alpha between calibrated score and original score_mean when rewriting results",
    )
    parser.add_argument("--results", default="", help="Optional source results.json to rewrite")
    parser.add_argument("--nusc-dataroot", default="", help="Required with --results to map sample->scene")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--output-results", default="", help="Optional rewritten results JSON path")
    return parser.parse_args()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_feature_payload(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def prepare_matrix(records, feature_names):
    X = []
    y = []
    weights = []
    orig_scores = []
    track_keys = []
    for rec in records:
        row = []
        for name in feature_names:
            value = float(rec.get(name, 0.0))
            if name == "mean_match_dist":
                value = min(value, 5.0)
            row.append(value)
        X.append(row)
        y.append(float(rec.get("is_good_track", 0)))
        weights.append(0.5 + float(rec.get("quality_target", 0.0)))
        orig_scores.append(float(rec.get("score_mean", 0.0)))
        track_keys.append(rec["track_key"])
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    orig_scores = np.asarray(orig_scores, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    Z = (X - mean) / std
    return Z, y, weights, orig_scores, track_keys, mean, std


def fit_logistic_regression(Z, y, weights, lr, iters, l2):
    n, d = Z.shape
    w = np.zeros(d, dtype=np.float64)
    b = 0.0
    for _ in range(iters):
        logits = Z @ w + b
        probs = sigmoid(logits)
        err = (probs - y) * weights
        grad_w = (Z.T @ err) / max(n, 1) + l2 * w
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def build_calibration_payload(feature_names, mean, std, w, b, blend_alpha, records, probs):
    ranked = sorted(
        zip(records, probs),
        key=lambda x: x[1],
        reverse=True,
    )
    preview = [
        {
            "track_key": rec["track_key"],
            "class_name": rec["class_name"],
            "quality_target": rec["quality_target"],
            "score_mean": rec["score_mean"],
            "calibrated_score": float(prob),
        }
        for rec, prob in ranked[:20]
    ]
    return {
        "feature_names": feature_names,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "weights": w.tolist(),
        "bias": float(b),
        "blend_alpha": float(blend_alpha),
        "preview_top20": preview,
    }


def apply_calibration_to_results(calib, results_path, nusc_dataroot, version, output_path):
    from nuscenes.nuscenes import NuScenes

    with open(results_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    results = payload.get("results", {})
    sample_tokens = sorted(results.keys())

    nusc = NuScenes(version=version, dataroot=nusc_dataroot, verbose=False)
    sample_to_scene = {}
    for sample_token in sample_tokens:
        sample = nusc.get("sample", sample_token)
        scene = nusc.get("scene", sample["scene_token"])
        sample_to_scene[sample_token] = scene["name"]

    track_score_map = {}
    for item in calib.get("track_scores", []):
        track_score_map[item["track_key"]] = float(item["calibrated_score"])

    for sample_token, boxes in results.items():
        scene_name = sample_to_scene[sample_token]
        for box in boxes:
            track_key = f"{scene_name}|{box['tracking_name']}|{box['tracking_id']}"
            if track_key not in track_score_map:
                continue
            box["tracking_score"] = float(track_score_map[track_key])

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)


def main():
    args = parse_args()
    payload = load_feature_payload(args.features)
    records = payload.get("records", [])
    feature_names = [name.strip() for name in args.feature_list.split(",") if name.strip()]
    Z, y, weights, orig_scores, track_keys, mean, std = prepare_matrix(records, feature_names)
    w, b = fit_logistic_regression(Z, y, weights, args.lr, args.iters, args.l2)
    probs = sigmoid(Z @ w + b)
    calibrated_scores = args.blend_alpha * probs + (1.0 - args.blend_alpha) * orig_scores
    calibrated_scores = np.clip(calibrated_scores, 1e-3, 0.999)

    calib = build_calibration_payload(
        feature_names,
        mean,
        std,
        w,
        b,
        args.blend_alpha,
        records,
        calibrated_scores,
    )
    calib["track_scores"] = [
        {
            "track_key": track_key,
            "calibrated_score": float(score),
        }
        for track_key, score in zip(track_keys, calibrated_scores)
    ]

    out_dir = os.path.dirname(os.path.abspath(args.output))
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    print(f"[track-score-calibrate] tracks={len(track_keys)} output={args.output}")

    if args.results or args.output_results:
        if not (args.results and args.nusc_dataroot and args.output_results):
            raise ValueError("--results, --nusc-dataroot, and --output-results must be set together")
        apply_calibration_to_results(
            calib,
            results_path=args.results,
            nusc_dataroot=args.nusc_dataroot,
            version=args.version,
            output_path=args.output_results,
        )
        print(f"[track-score-calibrate] wrote rescored results to {args.output_results}")


if __name__ == "__main__":
    main()
