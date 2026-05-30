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
import math
import os
import random
import re
import statistics
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


@dataclass
class ConstraintSpec:
    scope: str
    metric: str
    min_value: float | None = None
    max_value: float | None = None
    mode: str = "hard"  # hard | soft
    weight: float = 1.0


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def dump_yaml(path: str, data: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=False)


def load_search_space(path: str) -> Tuple[List[SearchParam], str | None, List[ConstraintSpec]]:
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

    constraints_raw = obj.get("constraints", [])
    constraints: List[ConstraintSpec] = []
    for c in constraints_raw:
        scope = str(c.get("scope", "")).strip()
        metric = str(c.get("metric", "")).strip()
        if not scope or not metric:
            raise ValueError(f"Invalid constraint (missing scope/metric): {c}")

        min_raw = c.get("min", c.get("min_value", None))
        max_raw = c.get("max", c.get("max_value", None))
        if min_raw is None and max_raw is None:
            raise ValueError(f"Invalid constraint (need min or max): {c}")
        if min_raw is not None and max_raw is not None:
            raise ValueError(f"Invalid constraint (min and max cannot coexist): {c}")

        min_value = float(min_raw) if min_raw is not None else None
        max_value = float(max_raw) if max_raw is not None else None
        mode = str(c.get("mode", c.get("type", "hard"))).strip().lower()
        if mode not in {"hard", "soft"}:
            raise ValueError(f"Invalid constraint mode={mode}: {c}")
        weight = float(c.get("weight", 1.0))
        if weight < 0:
            raise ValueError(f"Constraint weight must be >= 0: {c}")

        constraints.append(
            ConstraintSpec(
                scope=scope,
                metric=metric,
                min_value=min_value,
                max_value=max_value,
                mode=mode,
                weight=weight,
            )
        )

    default_objective = obj.get("objective", None)
    return params, default_objective, constraints


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


def metric_lookup(eval_obj: Dict, scope: str, metric: str) -> float:
    if scope == "overall":
        v = eval_obj.get("overall", {}).get(metric, float("nan"))
    else:
        v = eval_obj.get("per_class", {}).get(scope, {}).get(metric, float("nan"))
    try:
        fv = float(v)
    except Exception:
        return float("nan")
    return fv


def evaluate_objective(eval_obj: Dict, objective: str) -> float:
    # Token format: overall.idf1 or car.mota etc.
    token_pattern = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)\.([A-Za-z_][A-Za-z0-9_]*)\b")

    replaced = objective
    seen_tokens = set(token_pattern.findall(objective))
    for scope, metric in seen_tokens:
        val = metric_lookup(eval_obj, scope, metric)
        if not (val == val):  # nan
            val = -1e9
        replaced = replaced.replace(f"{scope}.{metric}", f"({val})")
    return safe_eval_expression(replaced)


def _calc_normalized_violation(value: float, c: ConstraintSpec) -> Tuple[bool, float]:
    if not (value == value):  # nan
        return True, 1e6
    if c.min_value is not None:
        if value >= c.min_value:
            return False, 0.0
        denom = max(abs(c.min_value), 1e-6)
        return True, (c.min_value - value) / denom
    # c.max_value is not None
    if value <= c.max_value:
        return False, 0.0
    denom = max(abs(c.max_value), 1e-6)
    return True, (value - c.max_value) / denom


def evaluate_constraints(eval_obj: Dict, constraints: List[ConstraintSpec]) -> Dict:
    details = []
    hard_violations = []
    soft_penalty = 0.0

    for c in constraints:
        value = metric_lookup(eval_obj, c.scope, c.metric)
        violated, gap = _calc_normalized_violation(value, c)
        penalty = c.weight * gap if (violated and c.mode == "soft") else 0.0
        if violated and c.mode == "hard":
            hard_violations.append(f"{c.scope}.{c.metric}")
        soft_penalty += penalty
        details.append(
            {
                "scope": c.scope,
                "metric": c.metric,
                "mode": c.mode,
                "value": value,
                "min": c.min_value,
                "max": c.max_value,
                "violated": violated,
                "normalized_gap": gap,
                "weight": c.weight,
                "penalty": penalty,
            }
        )

    return {
        "passed_hard": len(hard_violations) == 0,
        "hard_violations": hard_violations,
        "soft_penalty": soft_penalty,
        "details": details,
    }


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


def _trial_key(trial: Dict[str, object]) -> Tuple[Tuple[str, str], ...]:
    return tuple((k, json.dumps(trial[k], sort_keys=True)) for k in sorted(trial.keys()))


def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool)


def _finite(v: float) -> bool:
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def build_scene_folds(
    eval_set: str,
    n_folds: int,
    seed: int,
    run_dir: str,
    scene_names_file: str = "",
) -> List[Dict]:
    if n_folds <= 1:
        return [{"name": "all", "scene_list_path": None, "scene_count": -1}]

    if scene_names_file:
        with open(scene_names_file, "r", encoding="utf-8") as f:
            scene_names = [x.strip() for x in f.readlines() if x.strip() and not x.strip().startswith("#")]
    else:
        try:
            from nuscenes.utils.splits import create_splits_scenes
        except Exception as exc:
            raise RuntimeError(
                "n-fold mode requires nuscenes-devkit, or pass --scene-names-file."
            ) from exc

        splits = create_splits_scenes()
        if eval_set not in splits:
            raise ValueError(f"Unsupported eval_set={eval_set}. Available: {list(splits.keys())}")
        scene_names = list(splits[eval_set])

    if len(scene_names) < n_folds:
        raise ValueError(f"n_folds={n_folds} is larger than number of scenes={len(scene_names)}")

    rng = random.Random(seed)
    rng.shuffle(scene_names)
    buckets: List[List[str]] = [[] for _ in range(n_folds)]
    for idx, s in enumerate(scene_names):
        buckets[idx % n_folds].append(s)

    fold_dir = os.path.join(run_dir, "fold_splits")
    os.makedirs(fold_dir, exist_ok=True)
    out: List[Dict] = []
    for i, scenes in enumerate(buckets):
        scenes_sorted = sorted(scenes)
        scene_file = os.path.join(fold_dir, f"fold_{i:02d}.txt")
        with open(scene_file, "w", encoding="utf-8") as f:
            for s in scenes_sorted:
                f.write(s + "\n")
        out.append(
            {
                "name": f"fold_{i:02d}",
                "scene_list_path": scene_file,
                "scene_count": len(scenes_sorted),
            }
        )
    return out


def aggregate_overall_metrics(eval_objs: List[Dict]) -> Dict:
    if not eval_objs:
        return {}
    key_to_vals: Dict[str, List[float]] = {}
    for eo in eval_objs:
        for k, v in eo.get("overall", {}).items():
            if _is_num(v) and _finite(float(v)):
                key_to_vals.setdefault(k, []).append(float(v))
    out = {}
    for k, vals in key_to_vals.items():
        if vals:
            out[k] = sum(vals) / float(len(vals))
    return out


def build_refine_trials(
    params: List[SearchParam],
    anchor_trials: List[Dict[str, object]],
    n_trials: int,
    seed: int,
    jitter: float,
) -> List[Dict[str, object]]:
    if n_trials <= 0 or not anchor_trials:
        return []

    rng = random.Random(seed)
    out: List[Dict[str, object]] = []
    seen = set()

    for t in anchor_trials:
        k = _trial_key(t)
        if k in seen:
            continue
        seen.add(k)
        out.append(copy.deepcopy(t))
        if len(out) >= n_trials:
            return out

    attempts = 0
    max_attempts = max(2000, n_trials * 200)
    while len(out) < n_trials and attempts < max_attempts:
        attempts += 1
        anchor = rng.choice(anchor_trials)
        sample: Dict[str, object] = {}
        for p in params:
            base_val = anchor.get(p.path, rng.choice(p.values))
            vals = p.values
            numeric_vals = [float(v) for v in vals if _is_num(v)]
            all_numeric = len(numeric_vals) == len(vals) and len(vals) > 0
            all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in vals)

            if all_numeric and _is_num(base_val):
                v = float(base_val)
                lo = min(numeric_vals)
                hi = max(numeric_vals)
                span = max(hi - lo, max(abs(v), 1.0) * 0.1)
                perturb = rng.uniform(-jitter, jitter) * span
                nv = max(lo, min(hi, v + perturb))
                if all_int:
                    sample[p.path] = int(round(nv))
                else:
                    sample[p.path] = round(float(nv), 6)
            else:
                anchor_vals = [a[p.path] for a in anchor_trials if p.path in a]
                choices = list(dict.fromkeys(anchor_vals + vals))
                if anchor_vals and rng.random() < 0.8:
                    sample[p.path] = rng.choice(anchor_vals)
                else:
                    sample[p.path] = rng.choice(choices)

        k = _trial_key(sample)
        if k in seen:
            continue
        seen.add(k)
        out.append(sample)

    return out


def run_eval_once(
    *,
    workdir: str,
    eval_script_path: str,
    result_json_path: str,
    dataroot: str,
    version: str,
    eval_set: str,
    classes: str,
    dist_th: float,
    score_thr: float,
    scene_list_path: str | None,
    eval_json_path: str,
    eval_log_path: str,
    quiet: bool,
) -> Tuple[int, Dict | None]:
    eval_cmd = [
        sys.executable,
        eval_script_path,
        "-r",
        result_json_path,
        "--dataroot",
        dataroot,
        "--version",
        version,
        "--eval-set",
        eval_set,
        "--classes",
        classes,
        "--dist-th",
        str(dist_th),
        "--score-thr",
        str(score_thr),
    ]
    if scene_list_path:
        eval_cmd.extend(["--scene-list", scene_list_path])
    eval_cmd.extend(["-o", eval_json_path])

    eval_rc, _ = run_and_capture(eval_cmd, cwd=workdir, log_path=eval_log_path, quiet=quiet)
    if eval_rc != 0 or not os.path.exists(eval_json_path):
        return eval_rc, None
    with open(eval_json_path, "r", encoding="utf-8") as f:
        return eval_rc, json.load(f)


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
    parser.add_argument("--n-folds", type=int, default=1, help="Scene folds for stability scoring (1 = full eval-set).")
    parser.add_argument("--fold-seed", type=int, default=42, help="Random seed for fold split.")
    parser.add_argument("--scene-names-file", default="", help="Optional scene list file for fold split (one scene per line).")
    parser.add_argument("--stability-weight", type=float, default=0.0, help="Objective penalty: weight * std(fold_objective).")
    parser.add_argument("--two-stage", action="store_true", help="Enable coarse->refine two-stage search.")
    parser.add_argument("--coarse-trials", type=int, default=30, help="Number of coarse-stage trials when --two-stage.")
    parser.add_argument("--refine-trials", type=int, default=60, help="Number of refine-stage trials when --two-stage.")
    parser.add_argument("--topk-refine", type=int, default=10, help="Use top-k coarse trials as anchors for refine stage.")
    parser.add_argument("--refine-jitter", type=float, default=0.1, help="Relative jitter for numeric params in refine stage.")
    parser.add_argument(
        "--hard-constraint-policy",
        choices=["drop", "keep"],
        default="drop",
        help="drop=hard-constraint violation => objective=-inf; keep=still rank by penalized score.",
    )
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

    if args.n_folds < 1:
        raise ValueError("--n-folds must be >= 1")
    if args.coarse_trials < 1 or args.refine_trials < 1:
        raise ValueError("--coarse-trials and --refine-trials must be >= 1")
    if args.topk_refine < 1:
        raise ValueError("--topk-refine must be >= 1")
    if args.refine_jitter < 0:
        raise ValueError("--refine-jitter must be >= 0")
    if args.stability_weight < 0:
        raise ValueError("--stability-weight must be >= 0")
    if args.scene_names_file:
        scene_names_file = os.path.abspath(os.path.join(workdir, args.scene_names_file))
        if not os.path.exists(scene_names_file):
            raise FileNotFoundError(scene_names_file)
    else:
        scene_names_file = ""

    params, objective_from_space, constraints = load_search_space(search_space_path)
    objective = args.objective.strip() or (objective_from_space or "0.7*overall.idf1 + 0.3*overall.mota")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.run_dir:
        run_dir = os.path.abspath(os.path.join(workdir, args.run_dir))
    else:
        run_dir = os.path.abspath(os.path.join(workdir, "tools", "search_runs", f"bytetrack_{stamp}"))
    os.makedirs(run_dir, exist_ok=True)

    base_cfg = load_yaml(base_cfg_path)
    fold_specs = build_scene_folds(
        eval_set=args.eval_set,
        n_folds=args.n_folds,
        seed=args.fold_seed,
        run_dir=run_dir,
        scene_names_file=scene_names_file,
    )

    if args.two_stage:
        coarse_trials = build_trials(
            params=params,
            mode=args.mode,
            max_trials=args.coarse_trials,
            seed=args.seed,
        )
        if not coarse_trials:
            raise RuntimeError("No coarse trial generated.")
        n_total_trials = args.coarse_trials + args.refine_trials
    else:
        coarse_trials = build_trials(
            params=params,
            mode=args.mode,
            max_trials=args.max_trials,
            seed=args.seed,
        )
        if not coarse_trials:
            raise RuntimeError("No trial generated.")
        n_total_trials = len(coarse_trials)

    manifest = {
        "base_config": base_cfg_path,
        "search_space": search_space_path,
        "mode": args.mode,
        "max_trials": args.max_trials,
        "two_stage": args.two_stage,
        "coarse_trials": args.coarse_trials,
        "refine_trials": args.refine_trials,
        "topk_refine": args.topk_refine,
        "refine_jitter": args.refine_jitter,
        "n_folds": args.n_folds,
        "fold_seed": args.fold_seed,
        "scene_names_file": scene_names_file,
        "stability_weight": args.stability_weight,
        "fold_specs": fold_specs,
        "seed": args.seed,
        "objective": objective,
        "hard_constraint_policy": args.hard_constraint_policy,
        "constraints": [c.__dict__ for c in constraints],
        "n_trials": n_total_trials,
        "created_at": stamp,
    }
    with open(os.path.join(run_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[SEARCH] run_dir={run_dir}")
    print(f"[SEARCH] objective={objective}")
    print(f"[SEARCH] constraints={len(constraints)}")
    print(f"[SEARCH] n_folds={args.n_folds} stability_weight={args.stability_weight}")
    print(f"[SEARCH] two_stage={args.two_stage} planned_trials={n_total_trials}")
    if args.two_stage:
        print(f"[SEARCH] coarse_trials={len(coarse_trials)} refine_trials={args.refine_trials}")
    else:
        print(f"[SEARCH] trials={len(coarse_trials)}")

    if args.dry_run:
        for i, t in enumerate(coarse_trials):
            stage = "coarse" if args.two_stage else "single"
            print(f"[DRY] {stage} trial={i:03d} params={t}")
        if args.two_stage:
            print(
                "[DRY] refine trials are generated from top-k coarse results at runtime; "
                "run without --dry-run to materialize refine stage."
            )
        if constraints:
            print("[DRY] constraints:")
            for c in constraints:
                if c.min_value is not None:
                    rule = f">= {c.min_value}"
                else:
                    rule = f"<= {c.max_value}"
                print(f"[DRY]   {c.mode} {c.scope}.{c.metric} {rule} weight={c.weight}")
        if args.n_folds > 1:
            for fd in fold_specs:
                print(f"[DRY] fold {fd['name']} scenes={fd['scene_count']} file={fd['scene_list_path']}")
        return

    results = []
    save_root_hint = os.path.join(os.path.dirname(base_cfg.get("SAVE_PATH", ".")), base_cfg.get("DATASET", "nuscenes"))
    global_idx = 0

    def run_trial(stage_name: str, stage_i: int, stage_n: int, trial_params: Dict[str, object]) -> Dict:
        nonlocal global_idx
        trial_tag = f"trial_{global_idx:03d}"
        trial_dir = os.path.join(run_dir, trial_tag)
        os.makedirs(trial_dir, exist_ok=True)

        cfg = copy.deepcopy(base_cfg)
        for pth, val in trial_params.items():
            set_by_path(cfg, pth, val)
        cfg_path = os.path.join(trial_dir, "config.yaml")
        dump_yaml(cfg_path, cfg)

        print(
            f"[TRIAL {global_idx+1}/{n_total_trials}] stage={stage_name} "
            f"({stage_i+1}/{stage_n}) {trial_tag} params={trial_params}"
        )

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
            return {
                "trial": trial_tag,
                "stage": stage_name,
                "params": trial_params,
                "status": "main_failed",
                "main_rc": main_rc,
                "objective": float("-inf"),
            }

        result_json_path = parse_result_json_path(main_out)
        if result_json_path is None:
            result_json_path = find_latest_results_json(save_root_hint, ts_start)
        if result_json_path is None or not os.path.exists(result_json_path):
            return {
                "trial": trial_tag,
                "stage": stage_name,
                "params": trial_params,
                "status": "result_json_not_found",
                "objective": float("-inf"),
            }

        fold_results = []
        eval_objs = []
        for fold_idx, fd in enumerate(fold_specs):
            eval_json = os.path.join(
                trial_dir,
                "eval_iter.json" if len(fold_specs) == 1 else f"eval_iter_{fd['name']}.json",
            )
            eval_log = os.path.join(
                trial_dir,
                "eval.log" if len(fold_specs) == 1 else f"eval_{fd['name']}.log",
            )
            eval_rc, eval_obj = run_eval_once(
                workdir=workdir,
                eval_script_path=eval_script_path,
                result_json_path=result_json_path,
                dataroot=args.dataroot,
                version=args.version,
                eval_set=args.eval_set,
                classes=args.classes,
                dist_th=args.dist_th,
                score_thr=args.score_thr,
                scene_list_path=fd.get("scene_list_path"),
                eval_json_path=eval_json,
                eval_log_path=eval_log,
                quiet=args.quiet_subprocess,
            )
            if eval_rc != 0 or eval_obj is None:
                return {
                    "trial": trial_tag,
                    "stage": stage_name,
                    "params": trial_params,
                    "status": "eval_failed",
                    "eval_rc": eval_rc,
                    "result_json": result_json_path,
                    "objective": float("-inf"),
                }

            try:
                objective_raw = evaluate_objective(eval_obj, objective)
            except Exception as exc:
                objective_raw = float("-inf")
                print(f"[TRIAL {trial_tag}][{fd['name']}] objective parse/eval failed: {exc}")

            c_eval = evaluate_constraints(eval_obj, constraints) if constraints else {
                "passed_hard": True,
                "hard_violations": [],
                "soft_penalty": 0.0,
                "details": [],
            }
            objective_penalized = objective_raw - float(c_eval["soft_penalty"])

            fold_results.append(
                {
                    "fold": fd["name"],
                    "scene_count": fd["scene_count"],
                    "objective_raw": objective_raw,
                    "soft_penalty": float(c_eval["soft_penalty"]),
                    "objective_penalized": objective_penalized,
                    "constraint_eval": c_eval,
                    "eval_json": eval_json,
                    "overall": eval_obj.get("overall", {}),
                }
            )
            eval_objs.append(eval_obj)

        raw_vals = [float(fr["objective_raw"]) for fr in fold_results if _finite(float(fr["objective_raw"]))]
        pen_vals = [float(fr["objective_penalized"]) for fr in fold_results if _finite(float(fr["objective_penalized"]))]
        hard_ok = all(bool(fr["constraint_eval"].get("passed_hard", False)) for fr in fold_results)
        hard_violations = []
        for fr in fold_results:
            hv = fr["constraint_eval"].get("hard_violations", [])
            hard_violations.extend([f"{fr['fold']}:{x}" for x in hv])

        objective_raw_mean = sum(raw_vals) / float(len(raw_vals)) if raw_vals else float("-inf")
        objective_pen_mean = sum(pen_vals) / float(len(pen_vals)) if pen_vals else float("-inf")
        objective_pen_std = statistics.pstdev(pen_vals) if len(pen_vals) > 1 else 0.0
        stability_penalty = args.stability_weight * objective_pen_std
        objective_stable = objective_pen_mean - stability_penalty

        if (not hard_ok) and args.hard_constraint_policy == "drop":
            objective_value = float("-inf")
            status = "constraint_failed"
        else:
            objective_value = objective_stable
            status = "ok"

        overall_agg = aggregate_overall_metrics(eval_objs)

        return {
            "trial": trial_tag,
            "stage": stage_name,
            "params": trial_params,
            "status": status,
            "objective": objective_value,
            "objective_raw_mean": objective_raw_mean,
            "objective_penalized_mean": objective_pen_mean,
            "objective_penalized_std": objective_pen_std,
            "stability_penalty": stability_penalty,
            "hard_ok": hard_ok,
            "hard_violations": hard_violations,
            "result_json": result_json_path,
            "fold_results": fold_results,
            "overall": overall_agg,
        }

    def run_stage(stage_name: str, stage_trials: List[Dict[str, object]]) -> List[Dict]:
        nonlocal global_idx
        stage_results: List[Dict] = []
        for i, tp in enumerate(stage_trials):
            tr = run_trial(stage_name=stage_name, stage_i=i, stage_n=len(stage_trials), trial_params=tp)
            stage_results.append(tr)
            results.append(tr)
            global_idx += 1

            print(
                f"[TRIAL {tr.get('trial')}] objective={tr.get('objective')} "
                f"mean={tr.get('objective_penalized_mean')} std={tr.get('objective_penalized_std')} "
                f"hard_ok={tr.get('hard_ok')} "
                f"overall.idf1={tr.get('overall', {}).get('idf1', float('nan'))} "
                f"overall.mota={tr.get('overall', {}).get('mota', float('nan'))}",
                flush=True,
            )

            with open(os.path.join(run_dir, "results_partial.json"), "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
        return stage_results

    coarse_results = run_stage("coarse" if args.two_stage else "single", coarse_trials)

    if args.two_stage:
        finite_coarse = [r for r in coarse_results if _finite(float(r.get("objective", float("-inf"))))]
        finite_coarse = sorted(finite_coarse, key=lambda x: x.get("objective", float("-inf")), reverse=True)
        anchor_trials = [r["params"] for r in finite_coarse[: args.topk_refine]]
        if not anchor_trials:
            anchor_trials = coarse_trials[: min(len(coarse_trials), args.topk_refine)]
            print("[REFINE] no valid coarse winners, fallback to first coarse trials as anchors.")

        refine_trials = build_refine_trials(
            params=params,
            anchor_trials=anchor_trials,
            n_trials=args.refine_trials,
            seed=args.seed + 100003,
            jitter=args.refine_jitter,
        )
        if not refine_trials:
            print("[REFINE] no refine trials generated; skip refine stage.")
        else:
            print(f"[REFINE] anchors={len(anchor_trials)} refine_trials={len(refine_trials)}")
            run_stage("refine", refine_trials)

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
