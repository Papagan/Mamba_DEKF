import os
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
    if metrics_summary is None and not os.path.exists(summary_path):
        print("[EVAL] WARNING: metrics_summary is empty and metrics_summary.json was not found.")
