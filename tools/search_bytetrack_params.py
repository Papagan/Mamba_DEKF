#!/usr/bin/env python3
"""
Auto-search ByteTrack parameters by iterating:
  params -> run main.py -> run per-class eval -> rank objective.
"""

from __future__ import annotations

import argparse
import ast
import copy
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Iterable, List, Tuple

import yaml


@dataclass
class SearchParam:
    path: str
    values: List


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def load_search_space(path: str) -> Tuple[List[SearchParam], str | None]:
    with open(path, "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            obj = json.load(f)
        else:
            obj = yaml.safe_load(f)

    params_raw = obj.get("parameters", [])
    if not params_raw:
        raise ValueError(f"No `parameters` found in search-space file: {path}")

    params: List[SearchParam] = []
    for p in params_raw:
        pth = p.get("path", "").strip()
        vals = p.get("values", [])
        if not pth or not isinstance(vals, list) or len(vals) == 0:
            raise ValueError(f"Invalid search parameter entry: {p}")
        params.append(SearchParam(path=pth, values=vals))
    default_objective = obj.get("objective", None)
    return params, default_objective


def _key_candidates(seg: str) -> List:
    out = [seg]
    if seg.isdigit():
        out.append(int(seg))
    return out


def set_by_path(d: Dict, path: str, value) -> None:
    segs = path.split(".")
    cur = d
    for seg in segs[:-1]:
        next_key = None
        for cand in _key_candidates(seg):
            if isinstance(cur, dict) and cand in cur:
                next_key = cand
                break
        if next_key is None:
            # create with string key by default
            cur[seg] = {}
            next_key = seg
        if not isinstance(cur[next_key], dict):
            raise ValueError(f"Path conflict at `{seg}` for path `{path}`")
        cur = cur[next_key]

    last = segs[-1]
    for cand in _key_candidates(last):
        if cand in cur:
            cur[cand] = value
            return
    cur[last] = value


def gen_trials_grid(params: List[SearchParam]) -> Iterable[Dict[str, object]]:
    names = [p.path for p in params]
    value_lists = [p.values for p in params]
    for combo in itertools.product(*value_lists):
        yield {k: v for k, v in zip(names, combo)}


def gen_trials_random(params: List[SearchParam], n: int, seed: int) -> List[Dict[str, object]]:
    rng = random.Random(seed)
    trials = []
    seen = set()
    max_unique = 1
    for p in params:
        max_unique *= max(1, len(p.values))
    target = min(n, max_unique)

    while len(trials) < target:
        sample = {p.path: rng.choice(p.values) for p in params}
        key = tuple((k, json.dumps(sample[k], sort_keys=True)) for k in sorted(sample.keys()))
        if key in seen:
            continue
        seen.add(key)
        trials.append(sample)
    return trials


def safe_eval_expression(expr: str) -> float:
    node = ast.parse(expr, mode="eval")
    allowed = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
        ast.FloorDiv,
    )
    for n in ast.walk(node):
        if not isinstance(n, allowed):
            raise ValueError(f"Unsupported node in objective expression: {type(n).__name__}")
    return float(eval(compile(node, "<objective>", "eval"), {"__builtins__": {}}, {}))


def evaluate_objective(eval_obj: Dict, objective: str) -> float:
    # Token format: overall.idf1 or car.mota etc.
    token_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

    def lookup(scope: str, metric: str) -> float:
        if scope == "overall":
            v = eval_obj.get("overall", {}).get(metric, float("nan"))
        else:
            v = eval_obj.get("per_class", {}).get(scope, {}).get(metric, float("nan"))
        try:
            fv = float(v)
        except Exception:
            return float("nan")
        return fv

    replaced = objective
    seen_tokens = set(token_pattern.findall(objective))
    for scope, metric in seen_tokens:
        val = lookup(scope, metric)
        if not (val == val):  # nan
            val = -1e9
        replaced = replaced.replace(f"{scope}.{metric}", f"({val})")
    return safe_eval_expression(replaced)


def run_and_capture(cmd: List[str], cwd: str, log_path: str, quiet: bool) -> Tuple[int, str]:
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as logf:
        proc = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: List[str] = []
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            logf.write(line)
            if not quiet:
                print(line, end="")
        ret = proc.wait()
    return ret, "".join(lines)


def parse_result_json_path(stdout_text: str) -> str | None:
    m = re.findall(r"\[SAVE\].*?path=(\S*results\.json)", stdout_text)
    if m:
        return m[-1]
    return None


def find_latest_results_json(root_dir: str, ts_start: float) -> str | None:
    best_path = None
    best_mtime = -1.0
    if not os.path.isdir(root_dir):
        return None
    for dirpath, _, filenames in os.walk(root_dir):
        if "results.json" not in filenames:
            continue
        p = os.path.join(dirpath, "results.json")
        mt = os.path.getmtime(p)
        if mt >= ts_start and mt > best_mtime:
            best_mtime = mt
            best_path = p
    return best_path


def build_trials(
    params: List[SearchParam],
    mode: str,
    max_trials: int,
    seed: int,
) -> List[Dict[str, object]]:
    if mode == "random":
        return gen_trials_random(params=params, n=max_trials, seed=seed)

    # grid mode
    all_trials = list(gen_trials_grid(params))
    if len(all_trials) <= max_trials:
        return all_trials
    rng = random.Random(seed)
    idxs = list(range(len(all_trials)))
    rng.shuffle(idxs)
    return [all_trials[i] for i in idxs[:max_trials]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-search ByteTrack parameters for nuScenes.")
    parser.add_argument("--base-config", default="config/nuscenes.yaml", help="Base YAML config for tracking.")
    parser.add_argument("--search-space", default="tools/bytetrack_search_space.nuscenes.json", help="JSON/YAML search space file.")
    parser.add_argument("--dataset", default="nuscenes", help="Dataset argument for main.py")
    parser.add_argument("--process", type=int, default=1, help="main.py --process")
    parser.add_argument("--mode", choices=["grid", "random"], default="random", help="Parameter sampler mode.")
    parser.add_argument("--max-trials", type=int, default=20, help="Max number of trials.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--objective", default="", help="Objective expression, e.g. '0.7*overall.idf1 + 0.3*overall.mota'")
    parser.add_argument("--eval-script", default="tools/eval_bytetrack_by_class_iter.py", help="Path to iterative evaluator script.")
    parser.add_argument("--dataroot", required=True, help="nuScenes dataroot for eval script.")
    parser.add_argument("--version", default="v1.0-trainval", help="nuScenes version")
    parser.add_argument("--eval-set", default="val", help="nuScenes split")
    parser.add_argument("--classes", default="bicycle,bus,car,motorcycle,pedestrian,trailer,truck", help="Classes for eval script.")
    parser.add_argument("--dist-th", type=float, default=2.0, help="Center distance threshold for evaluator.")
    parser.add_argument("--score-thr", type=float, default=0.0, help="Score threshold for evaluator.")
    parser.add_argument("--workdir", default=".", help="Repo root containing main.py")
    parser.add_argument("--run-dir", default="", help="Optional output directory for this search run.")
    parser.add_argument("--quiet-subprocess", action="store_true", help="Suppress trial subprocess stdout.")
    parser.add_argument("--dry-run", action="store_true", help="Only sample and print trials, do not run tracking.")
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    base_cfg_path = os.path.abspath(os.path.join(workdir, args.base_config))
    search_space_path = os.path.abspath(os.path.join(workdir, args.search_space))
    eval_script_path = os.path.abspath(os.path.join(workdir, args.eval_script))

    if not os.path.exists(base_cfg_path):
        raise FileNotFoundError(base_cfg_path)
    if not os.path.exists(search_space_path):
        raise FileNotFoundError(search_space_path)
    if not os.path.exists(eval_script_path):
        raise FileNotFoundError(eval_script_path)

    params, objective_from_space = load_search_space(search_space_path)
    objective = args.objective.strip() or (objective_from_space or "0.7*overall.idf1 + 0.3*overall.mota")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = os.path.abspath(os.path.join(workdir, args.run_dir))
    else:
        run_dir = os.path.abspath(os.path.join(workdir, "tools", "search_runs", f"bytetrack_{stamp}"))
    os.makedirs(run_dir, exist_ok=True)

    base_cfg = load_yaml(base_cfg_path)
    trials = build_trials(params=params, mode=args.mode, max_trials=args.max_trials, seed=args.seed)
    if not trials:
        raise RuntimeError("No trial generated.")

    manifest = {
        "base_config": base_cfg_path,
        "search_space": search_space_path,
        "mode": args.mode,
        "max_trials": args.max_trials,
        "seed": args.seed,
        "objective": objective,
        "n_trials": len(trials),
        "created_at": stamp,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SEARCH] run_dir={run_dir}")
    print(f"[SEARCH] objective={objective}")
    print(f"[SEARCH] trials={len(trials)}")

    if args.dry_run:
        for i, t in enumerate(trials):
            print(f"[DRY] trial={i:03d} params={t}")
        return

    results = []
    save_root_hint = os.path.join(os.path.dirname(base_cfg.get("SAVE_PATH", ".")), base_cfg.get("DATASET", "nuscenes"))

    for idx, trial_params in enumerate(trials):
        trial_tag = f"trial_{idx:03d}"
        trial_dir = os.path.join(run_dir, trial_tag)
        os.makedirs(trial_dir, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        for pth, val in trial_params.items():
            set_by_path(cfg, pth, val)

        cfg_path = os.path.join(trial_dir, "config.yaml")
        dump_yaml(cfg_path, cfg)

        print(f"[TRIAL {idx+1}/{len(trials)}] {trial_tag} params={trial_params}")

        main_cmd = [
            sys.executable,
            "main.py",
            "--dataset",
            args.dataset,
            "--config",
            cfg_path,
            "--process",
            str(args.process),
        ]

        ts_start = time.time()
        main_log = os.path.join(trial_dir, "main.log")
        main_rc, main_out = run_and_capture(main_cmd, cwd=workdir, log_path=main_log, quiet=args.quiet_subprocess)
        if main_rc != 0:
            results.append(
                {
                    "trial": trial_tag,
                    "params": trial_params,
                    "status": "main_failed",
                    "main_rc": main_rc,
                    "objective": float("-inf"),
                }
            )
            print(f"[TRIAL {trial_tag}] main.py failed rc={main_rc}")
            continue

        result_json_path = parse_result_json_path(main_out)
        if result_json_path is None:
            result_json_path = find_latest_results_json(save_root_hint, ts_start)
        if result_json_path is None or not os.path.exists(result_json_path):
            results.append(
                {
                    "trial": trial_tag,
                    "params": trial_params,
                    "status": "result_json_not_found",
                    "objective": float("-inf"),
                }
            )
            print(f"[TRIAL {trial_tag}] results.json not found.")
            continue

        eval_json = os.path.join(trial_dir, "eval_iter.json")
        eval_cmd = [
            sys.executable,
            eval_script_path,
            "-r",
            result_json_path,
            "--dataroot",
            args.dataroot,
            "--version",
            args.version,
            "--eval-set",
            args.eval_set,
            "--classes",
            args.classes,
            "--dist-th",
            str(args.dist_th),
            "--score-thr",
            str(args.score_thr),
            "-o",
            eval_json,
        ]
        eval_log = os.path.join(trial_dir, "eval.log")
        eval_rc, _ = run_and_capture(eval_cmd, cwd=workdir, log_path=eval_log, quiet=args.quiet_subprocess)
        if eval_rc != 0 or not os.path.exists(eval_json):
            results.append(
                {
                    "trial": trial_tag,
                    "params": trial_params,
                    "status": "eval_failed",
                    "eval_rc": eval_rc,
                    "result_json": result_json_path,
                    "objective": float("-inf"),
                }
            )
            print(f"[TRIAL {trial_tag}] eval failed rc={eval_rc}")
            continue

        with open(eval_json, "r", encoding="utf-8") as f:
            eval_obj = json.load(f)

        try:
            objective_value = evaluate_objective(eval_obj, objective)
        except Exception as exc:
            objective_value = float("-inf")
            print(f"[TRIAL {trial_tag}] objective parse/eval failed: {exc}")

        trial_result = {
            "trial": trial_tag,
            "params": trial_params,
            "status": "ok",
            "objective": objective_value,
            "result_json": result_json_path,
            "eval_json": eval_json,
            "overall": eval_obj.get("overall", {}),
        }
        results.append(trial_result)

        print(
            f"[TRIAL {trial_tag}] objective={objective_value:.6f} "
            f"overall.idf1={trial_result['overall'].get('idf1', float('nan'))} "
            f"overall.mota={trial_result['overall'].get('mota', float('nan'))}",
            flush=True,
        )

        with open(os.path.join(run_dir, "results_partial.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

    # Rank trials
    ranked = sorted(results, key=lambda x: x.get("objective", float("-inf")), reverse=True)
    output = {
        "manifest": manifest,
        "best": ranked[0] if ranked else None,
        "ranked": ranked,
    }
    out_path = os.path.join(run_dir, "results_ranked.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[DONE] ranked results saved: {out_path}")
    if ranked:
        best = ranked[0]
        print(f"[BEST] trial={best['trial']} objective={best.get('objective')} params={best.get('params')}")


if __name__ == "__main__":
    main()

