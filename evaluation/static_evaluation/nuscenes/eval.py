import os
import json
from nuscenes.eval.tracking.evaluate import TrackingEval
from nuscenes.eval.common.config import config_factory as track_configs
    
def eval_nusc(args):
    result_path = os.path.join(args["SAVE_PATH"], "results.json")
    eval_path = os.path.join(args["SAVE_PATH"], "eval_result/")
    nusc_path = args["DATASET_ROOT"]
    cfg = track_configs("tracking_nips_2019")
    nusc_eval = TrackingEval(
        config=cfg,
        result_path=result_path,
        eval_set="val",
        output_dir=eval_path,
        verbose=True,
        nusc_version="v1.0-trainval",
        nusc_dataroot=nusc_path,
    )
    print("result in " + result_path)
    metrics_summary = nusc_eval.main()
    summary_path = os.path.join(eval_path, "metrics_summary.json")
    print(f"[EVAL] summary path: {summary_path}")

    if metrics_summary is not None:
        print("[EVAL] metrics_summary returned by TrackingEval.main():")
        print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))
    elif os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            metrics_summary = json.load(f)
        print("[EVAL] metrics_summary loaded from disk:")
        print(json.dumps(metrics_summary, indent=2, ensure_ascii=False))
    else:
        print("[EVAL] WARNING: metrics_summary is empty and metrics_summary.json was not found.")
