#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, List


DEFAULT_CLASSES = [
    "bicycle",
    "bus",
    "car",
    "motorcycle",
    "pedestrian",
    "trailer",
    "truck",
]
DEFAULT_METRICS = ["amota", "recall", "mota"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare original vs calibrated nuScenes tracking results."
    )
    parser.add_argument("--orig-results", required=True, help="Path to original results.json")
    parser.add_argument("--cal-results", required=True, help="Path to calibrated results.json")
    parser.add_argument("--nusc-dataroot", required=True, help="nuScenes dataroot")
    parser.add_argument("--output-dir", required=True, help="Directory to store eval outputs and comparison JSON")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--eval-set", default="val", help="Evaluation split")
    parser.add_argument(
        "--metrics",
        default=",".join(DEFAULT_METRICS),
        help="Comma-separated aggregate/per-class metrics to compare",
    )
    parser.add_argument(
        "--class-names",
        default=",".join(DEFAULT_CLASSES),
        help="Comma-separated class names to compare",
    )
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse existing metrics_summary.json if present",
    )
    return parser.parse_args()


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def eval_nuscenes_results(result_path: str, output_dir: str, *, nusc_dataroot: str, version: str, eval_set: str):
    from nuscenes.eval.tracking.evaluate import TrackingEval
    from nuscenes.eval.common.config import config_factory as track_configs

    ensure_dir(output_dir)
    cfg = track_configs("tracking_nips_2019")
    evaluator = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=True,
        nusc_version=version,
        nusc_dataroot=nusc_dataroot,
    )
    metrics_summary = evaluator.main()
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if metrics_summary is None:
        with open(summary_path, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
    return metrics_summary


def load_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_comparison_payload(orig: Dict, cal: Dict, class_names: List[str], metrics: List[str] | None = None):
    metrics = metrics or list(DEFAULT_METRICS)
    aggregate = {}
    for metric in metrics:
        o = float(orig.get(metric, 0.0))
        c = float(cal.get(metric, 0.0))
        aggregate[metric] = {
            "orig": o,
            "cal": c,
            "delta": c - o,
        }

    per_class = {}
    for cls_name in class_names:
        cls_payload = {}
        for metric in metrics:
            o = float(orig.get("label_metrics", {}).get(metric, {}).get(cls_name, 0.0))
            c = float(cal.get("label_metrics", {}).get(metric, {}).get(cls_name, 0.0))
            cls_payload[metric] = {
                "orig": o,
                "cal": c,
                "delta": c - o,
            }
        per_class[cls_name] = cls_payload

    return {
        "aggregate": aggregate,
        "per_class": per_class,
    }


def print_comparison(payload: Dict, class_names: List[str], metrics: List[str]):
    print("Aggregated results:")
    for metric in metrics:
        item = payload["aggregate"][metric]
        print(f"  {metric.upper():8s} {item['orig']:.3f} -> {item['cal']:.3f}  delta={item['delta']:+.3f}")

    print("\nPer-class deltas:")
    header = "class".ljust(12) + "".join(f"{metric.upper():>24s}" for metric in metrics)
    print(header)
    for cls_name in class_names:
        row = cls_name.ljust(12)
        for metric in metrics:
            item = payload["per_class"][cls_name][metric]
            row += f"{item['orig']:.3f}->{item['cal']:.3f} ({item['delta']:+.3f})".rjust(24)
        print(row)


def maybe_eval(result_path: str, output_dir: str, *, reuse_existing: bool, nusc_dataroot: str, version: str, eval_set: str):
    summary_path = os.path.join(output_dir, "metrics_summary.json")
    if reuse_existing and os.path.exists(summary_path):
        return load_json(summary_path)
    return eval_nuscenes_results(
        result_path,
        output_dir,
        nusc_dataroot=nusc_dataroot,
        version=version,
        eval_set=eval_set,
    )


def main():
    args = parse_args()
    metrics = [m.strip().lower() for m in args.metrics.split(",") if m.strip()]
    class_names = [c.strip() for c in args.class_names.split(",") if c.strip()]

    ensure_dir(args.output_dir)
    orig_eval_dir = os.path.join(args.output_dir, "orig_eval")
    cal_eval_dir = os.path.join(args.output_dir, "cal_eval")

    orig_summary = maybe_eval(
        args.orig_results,
        orig_eval_dir,
        reuse_existing=args.reuse_existing,
        nusc_dataroot=args.nusc_dataroot,
        version=args.version,
        eval_set=args.eval_set,
    )
    cal_summary = maybe_eval(
        args.cal_results,
        cal_eval_dir,
        reuse_existing=args.reuse_existing,
        nusc_dataroot=args.nusc_dataroot,
        version=args.version,
        eval_set=args.eval_set,
    )

    payload = build_comparison_payload(orig_summary, cal_summary, class_names=class_names, metrics=metrics)
    payload["meta"] = {
        "orig_results": os.path.abspath(args.orig_results),
        "cal_results": os.path.abspath(args.cal_results),
        "orig_eval_dir": os.path.abspath(orig_eval_dir),
        "cal_eval_dir": os.path.abspath(cal_eval_dir),
        "metrics": metrics,
        "class_names": class_names,
    }

    out_path = os.path.join(args.output_dir, "comparison_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print_comparison(payload, class_names, metrics)
    print(f"\nSaved comparison to {out_path}")


if __name__ == "__main__":
    main()
