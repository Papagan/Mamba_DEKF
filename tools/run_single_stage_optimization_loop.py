#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from typing import Dict, Iterable, List

import yaml

from tools.compare_nuscenes_results import (
    DEFAULT_CLASSES,
    DEFAULT_METRICS,
    build_comparison_payload,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run eval -> export -> calibrate -> compare -> suggest in a loop for nuScenes single-stage tuning."
    )
    parser.add_argument("--config", required=True, help="Starting config path, e.g. config/nuscenes_single_stage.yaml")
    parser.add_argument("--iterations", type=int, default=1, help="Number of optimization iterations")
    parser.add_argument("--processes", "-p", type=int, default=12, help="Process count forwarded to main.py -p")
    parser.add_argument("--dataset", default="nuscenes", help="Dataset name passed to main.py")
    parser.add_argument("--nusc-dataroot", required=True, help="nuScenes dataroot")
    parser.add_argument("--work-dir", default="debug/optimization_loop", help="Workspace for intermediate artifacts")
    parser.add_argument("--docs-dir", default="docs/optimization_loop", help="Directory for markdown iteration reports")
    parser.add_argument("--history", default="debug/nuscenes_single_stage_history.json", help="Suggestion history JSON")
    parser.add_argument("--final-config", default="config/nuscenes_single_stage_suggested.yaml", help="Final suggested config output path")
    parser.add_argument("--keep-last-results", type=int, default=2, help="Keep only the most recent N heavy result directories")
    parser.add_argument("--python-bin", default=sys.executable, help="Python executable used to invoke scripts")
    parser.add_argument("--feature-list", default="", help="Optional comma-separated feature list for calibrate_track_score.py")
    parser.add_argument("--blend-alpha", type=float, default=0.75, help="Blend alpha for calibrate_track_score.py")
    parser.add_argument("--dist-th", type=float, default=2.0, help="Distance threshold for export_track_quality_features.py")
    return parser.parse_args()


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(path: str, payload: Dict):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def load_json(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def derive_result_root(cfg: Dict) -> str:
    return os.path.join(os.path.dirname(cfg["SAVE_PATH"]), cfg["DATASET"])


def list_run_dirs(result_root: str) -> List[str]:
    if not os.path.isdir(result_root):
        return []
    return [
        os.path.join(result_root, name)
        for name in os.listdir(result_root)
        if os.path.isdir(os.path.join(result_root, name))
    ]


def pick_new_run_dir(before: Iterable[str], after: Iterable[str], mtime_lookup=None) -> str:
    before_set = set(before)
    candidates = [path for path in after if path not in before_set]
    if not candidates:
        raise RuntimeError("No new result directory detected after evaluation")
    mtime_lookup = mtime_lookup or os.path.getmtime
    return max(candidates, key=mtime_lookup)


def prune_old_run_dirs(run_dirs: List[str], keep_last: int) -> List[str]:
    if keep_last < 0:
        return run_dirs
    if len(run_dirs) <= keep_last:
        return run_dirs
    kept = list(run_dirs)
    while len(kept) > keep_last:
        victim = kept.pop(0)
        if os.path.isdir(victim):
            shutil.rmtree(victim)
    return kept


def run_command(cmd: List[str], cwd: str):
    print("[loop] run:", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def build_feedback_comparison(prev_summary: Dict, curr_summary: Dict, class_names: List[str], metrics: List[str]) -> Dict:
    payload = build_comparison_payload(prev_summary, curr_summary, class_names=class_names, metrics=metrics)
    payload["meta"] = {
        "kind": "real_eval_feedback",
        "class_names": class_names,
        "metrics": metrics,
    }
    return payload


def write_iteration_report(path: str, payload: Dict):
    lines = [
        f"# Optimization Iteration {payload['iteration']}",
        "",
        f"- Timestamp: `{payload['timestamp']}`",
        f"- Input config: `{payload['input_config']}`",
        f"- Run dir: `{payload['run_dir']}`",
        f"- Suggested config: `{payload['suggested_config']}`",
        f"- Comparison summary: `{payload['comparison_summary']}`",
    ]
    if payload.get("feedback_comparison"):
        lines.append(f"- Feedback comparison: `{payload['feedback_comparison']}`")
    lines.extend(
        [
            "",
            "## Aggregate",
            "",
            f"- Original AMOTA: `{payload['aggregate']['orig_amota']:.3f}`",
            f"- Calibrated AMOTA: `{payload['aggregate']['cal_amota']:.3f}`",
            f"- Delta: `{payload['aggregate']['delta_amota']:+.3f}`",
            "",
            "## Suggestion",
            "",
            f"- Strategy: `{payload['suggestion']['strategy']}`",
            f"- Aggregate gain scale: `{payload['suggestion']['agg_gain_scale']:.3f}`",
            f"- Weak gain scale: `{payload['suggestion']['weak_gain_scale']:.3f}`",
            f"- Weak advantage scale: `{payload['suggestion']['weak_advantage_scale']:.3f}`",
            f"- Suggested changes: `{payload['suggestion']['change_count']}`",
            "",
        ]
    )
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    args = parse_args()
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(args.docs_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    loop_root = os.path.join(args.work_dir, f"loop_{timestamp}")
    os.makedirs(loop_root, exist_ok=True)

    current_config = os.path.abspath(args.config)
    prev_orig_metrics = None
    prev_feedback_path = ""
    tracked_run_dirs: List[str] = []

    for iteration in range(1, args.iterations + 1):
        iter_name = f"iter_{iteration:02d}"
        iter_dir = os.path.join(loop_root, iter_name)
        os.makedirs(iter_dir, exist_ok=True)

        cfg = load_yaml(current_config)
        result_root = derive_result_root(cfg)
        before_run_dirs = list_run_dirs(result_root)

        run_command(
            [
                args.python_bin,
                "main.py",
                "--dataset",
                args.dataset,
                "--eval",
                "--config",
                current_config,
                "-p",
                str(args.processes),
            ],
            cwd=os.getcwd(),
        )

        after_run_dirs = list_run_dirs(result_root)
        run_dir = pick_new_run_dir(before_run_dirs, after_run_dirs)
        tracked_run_dirs.append(run_dir)
        tracked_run_dirs = prune_old_run_dirs(tracked_run_dirs, args.keep_last_results)

        results_path = os.path.join(run_dir, "results.json")
        features_path = os.path.join(iter_dir, "track_quality_features.json")
        calibration_path = os.path.join(iter_dir, "track_score_calibration.json")
        calibrated_results_path = os.path.join(run_dir, "results_calibrated.json")
        compare_dir = os.path.join(iter_dir, "compare_calibrated")
        comparison_path = os.path.join(compare_dir, "comparison_summary.json")
        suggestion_report_path = os.path.join(iter_dir, "nuscenes_single_stage_suggestion.json")
        suggested_config_path = os.path.join(iter_dir, "nuscenes_single_stage_suggested.yaml")

        export_cmd = [
            args.python_bin,
            "tools/export_track_quality_features.py",
            "--results",
            results_path,
            "--nusc-dataroot",
            args.nusc_dataroot,
            "--output",
            features_path,
            "--dist-th",
            str(args.dist_th),
        ]
        run_command(export_cmd, cwd=os.getcwd())

        calibrate_cmd = [
            args.python_bin,
            "tools/calibrate_track_score.py",
            "--features",
            features_path,
            "--output",
            calibration_path,
            "--results",
            results_path,
            "--nusc-dataroot",
            args.nusc_dataroot,
            "--output-results",
            calibrated_results_path,
            "--blend-alpha",
            str(args.blend_alpha),
        ]
        if args.feature_list:
            calibrate_cmd.extend(["--feature-list", args.feature_list])
        run_command(calibrate_cmd, cwd=os.getcwd())

        compare_cmd = [
            args.python_bin,
            "tools/compare_nuscenes_results.py",
            "--orig-results",
            results_path,
            "--cal-results",
            calibrated_results_path,
            "--nusc-dataroot",
            args.nusc_dataroot,
            "--output-dir",
            compare_dir,
        ]
        run_command(compare_cmd, cwd=os.getcwd())

        orig_metrics_path = os.path.join(compare_dir, "orig_eval", "metrics_summary.json")
        cal_metrics_path = os.path.join(compare_dir, "cal_eval", "metrics_summary.json")
        orig_metrics = load_json(orig_metrics_path)
        cal_metrics = load_json(cal_metrics_path)
        comparison_summary = load_json(comparison_path)

        feedback_path = ""
        if prev_orig_metrics is not None:
            feedback_payload = build_feedback_comparison(prev_orig_metrics, orig_metrics, DEFAULT_CLASSES, DEFAULT_METRICS)
            feedback_path = os.path.join(iter_dir, "feedback_comparison_summary.json")
            save_json(feedback_path, feedback_payload)

        suggest_cmd = [
            args.python_bin,
            "tools/suggest_nuscenes_single_stage_params.py",
            "--calibration",
            calibration_path,
            "--comparison",
            comparison_path,
            "--config",
            current_config,
            "--output",
            suggested_config_path,
            "--report",
            suggestion_report_path,
            "--history",
            args.history,
        ]
        if feedback_path:
            suggest_cmd.extend(["--feedback-comparison", feedback_path])
        run_command(suggest_cmd, cwd=os.getcwd())

        suggestion_report = load_json(suggestion_report_path)
        doc_report_path = os.path.join(args.docs_dir, f"{timestamp}_{iter_name}.md")
        write_iteration_report(
            doc_report_path,
            {
                "iteration": iteration,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "input_config": current_config,
                "run_dir": run_dir,
                "suggested_config": suggested_config_path,
                "comparison_summary": comparison_path,
                "feedback_comparison": feedback_path,
                "aggregate": {
                    "orig_amota": float(orig_metrics.get("amota", 0.0)),
                    "cal_amota": float(cal_metrics.get("amota", 0.0)),
                    "delta_amota": float(comparison_summary["aggregate"]["amota"]["delta"]),
                },
                "suggestion": {
                    "strategy": suggestion_report["strategy"],
                    "agg_gain_scale": float(suggestion_report["diagnostics"].get("agg_gain_scale", 0.0)),
                    "weak_gain_scale": float(suggestion_report["diagnostics"].get("weak_gain_scale", 0.0)),
                    "weak_advantage_scale": float(suggestion_report["diagnostics"].get("weak_advantage_scale", 0.0)),
                    "change_count": len(suggestion_report.get("changes", [])),
                },
            },
        )

        prev_orig_metrics = orig_metrics
        prev_feedback_path = feedback_path
        current_config = suggested_config_path

    if current_config:
        os.makedirs(os.path.dirname(os.path.abspath(args.final_config)), exist_ok=True)
        shutil.copyfile(current_config, args.final_config)
        print(f"[loop] copied final suggested config to {args.final_config}")

    state_path = os.path.join(loop_root, "loop_summary.json")
    save_json(
        state_path,
        {
            "timestamp": timestamp,
            "iterations": args.iterations,
            "final_config": os.path.abspath(args.final_config),
            "history": os.path.abspath(args.history),
            "loop_root": os.path.abspath(loop_root),
            "docs_dir": os.path.abspath(args.docs_dir),
            "kept_run_dirs": tracked_run_dirs,
            "last_feedback_comparison": prev_feedback_path,
        },
    )
    print(f"[loop] wrote summary to {state_path}")


if __name__ == "__main__":
    main()
