#!/usr/bin/env python3
"""Remote Stage C: export actual tracker vectors, train, then use one canonical head path.

The initial offline head is used only for collecting replay. The production
configuration rejects it until this step has produced tracker-replay weights.
"""

import argparse
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml


ROOT = Path(__file__).resolve().parents[1]


def build_replay_plan(train_config, inference_config, cache_dir, processes, bootstrap_path,
                      validation_cache=None):
    if int(processes) < 1:
        raise ValueError("processes must be positive")
    train_path, infer_path = Path(train_config).resolve(), Path(inference_config).resolve()
    train = yaml.safe_load(train_path.read_text(encoding="utf-8"))
    runtime = yaml.safe_load(infer_path.read_text(encoding="utf-8"))
    bootstrap = copy.deepcopy(runtime)
    bootstrap["MAMBA_ASSOCIATION_HEAD"].pop("REQUIRE_TRAINING_SOURCE", None)
    bootstrap.setdefault("INFERENCE_PAIR_CACHE", {})["REQUIRE_GT_POLICY_VERSION"] = 2
    cache_policy = bootstrap["INFERENCE_PAIR_CACHE"]
    cache_policy.setdefault("MAX_GROUPS_PER_CLASS_STATE", 256)
    cache_policy.setdefault("MAX_CANDIDATES_PER_GROUP", 16)
    cache_policy.setdefault("CANDIDATE_COST_MARGIN", 0.10)
    cache_policy.setdefault("SAMPLING_SEED", 0)
    if cache_policy["MAX_GROUPS_PER_CLASS_STATE"] <= 0 or cache_policy["MAX_CANDIDATES_PER_GROUP"] < 4:
        raise ValueError("Final replay needs positive group caps and at least four candidate slots")
    replay_path = str(Path(cache_dir).resolve() / "tracker_replay.pkl")
    export = [sys.executable, str(ROOT / "main.py"), "--dataset", "nuscenes", "--eval",
              "--config", str(bootstrap_path), "-p", str(processes),
              "--inference-pair-cache-output", replay_path,
              "--inference-pair-cache-summary", str(Path(cache_dir).resolve() / "tracker_replay_summary.json"),
              "--inference-pair-cache-strict", "--inference-pair-cache-require-gt"]
    finalize = [sys.executable, str(ROOT / "tools/train_pairwise_association.py"),
                "--train-config", str(train_path), "--train-pkl", replay_path,
                "--val-pkl", str(Path(validation_cache).resolve()) if validation_cache else replay_path,
                "--backbone-checkpoint", runtime["MAMBA"]["CHECKPOINT_PATH"],
                "--output", runtime["MAMBA_ASSOCIATION_HEAD"]["CHECKPOINT_PATH"],
                "--metrics-output", str(Path(cache_dir).resolve() / "replay_assoc_metrics.json"),
                "--freeze-backbone", "--precompute-embeddings", "--train-mode", "per_class"]
    if train.get("DATA", {}).get("STATE_OUTPUT_MODE") != runtime.get("STATE_OUTPUT_MODE"):
        raise ValueError("Train/runtime STATE_OUTPUT_MODE must match before collecting replay")
    return bootstrap, [export, finalize]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-config", default="config/train_nuscenes.yaml")
    parser.add_argument("--inference-config", default="config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml")
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--processes", type=int, default=4)
    parser.add_argument("--val-replay", help="Optional held-out replay cache; default retains the current smoke-run reuse policy")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running tracker/training")
    args = parser.parse_args(argv)
    with tempfile.TemporaryDirectory(prefix="association-replay-") as temporary:
        bootstrap_path = Path(temporary) / "bootstrap.yaml"
        bootstrap, commands = build_replay_plan(
            args.train_config, args.inference_config, args.cache_dir, args.processes, bootstrap_path,
            validation_cache=args.val_replay,
        )
        if args.dry_run:
            print(json.dumps(commands, indent=2, ensure_ascii=False))
            return
        bootstrap_path.write_text(yaml.safe_dump(bootstrap, sort_keys=False), encoding="utf-8")
        Path(args.cache_dir).mkdir(parents=True, exist_ok=True)
        for command in commands:
            subprocess.run(command, cwd=ROOT, check=True)
        print("Replay head saved at the production config's CHECKPOINT_PATH; rerun production evaluation.")


if __name__ == "__main__":
    main()
