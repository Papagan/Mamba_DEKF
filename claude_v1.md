# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

MCTrack is a research framework for 3D multi-object tracking (MOT) in autonomous driving. It takes 3D object detections and links them across frames into trajectories. It supports KITTI, nuScenes, and Waymo datasets through a unified intermediate format called **BaseVersion**.

Paper: https://arxiv.org/abs/2409.16149

## Setup

```bash
conda create -n MCTrack python=3.8
conda activate MCTrack
pip install -r requirements.txt
```

Waymo requires additional setup — follow the [official waymo-open-dataset tutorial](https://github.com/waymo-research/waymo-open-dataset/blob/master/tutorial/tutorial.ipynb).

## Commands

### Preprocess raw detections into BaseVersion format
```bash
python preprocess/convert2baseversion.py --dataset kitti   # or nuscenes / waymo
```
Edit the `kitti_cfg` / `nuscenes_cfg` / `waymo_cfg` dicts at the top of that file to set paths, detector, and split before running.

### Run tracking (online mode)
```bash
python main.py --dataset kitti      # kitti / nuscenes / waymo
python main.py --dataset kitti -e   # with evaluation
python main.py --dataset kitti -e -p 8  # multi-process (8 workers)
python main.py --dataset kitti --debug  # 2-scene subset for quick iteration
```

### Run tracking in offline/global mode
```bash
python main.py --dataset kitti -m   # -m flag switches to *_offline.yaml config
```

### Motion metric evaluation (nuScenes only)
```bash
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py
python evaluation/eval_motion.py
```

Results are written to `results/<dataset>/<timestamp>/`.

## Architecture

### Data flow
```
Raw detections  →  BaseVersion JSON  →  BaseVersionTrackingDataset  →  Base3DTracker  →  Results
(preprocess/)       data/base_version/    (dataset/)                    (tracker/)        (utils/)
```

### BaseVersion format
All datasets are normalized into a single JSON schema at `data/base_version/<dataset>/<detector>/<split>.json`. Each scene contains frames; each frame contains bboxes with fields: `global_xyz`, `global_yaw`, `lwh`, `global_velocity`, `global_acceleration`, `detection_score`, `category`, `bbox_image` (2D image coords), and `transform_matrix` (global↔ego↔lidar↔camera transforms). This abstraction is the key design decision that lets one tracker work across all three datasets.

### Tracker internals (`tracker/`)

**`Base3DTracker`** (`base_tracker.py`) — stateful tracker per scene. Each call to `track_single_frame()` runs:
1. Predict all active trajectory states via Kalman filter
2. BEV matching: build cost matrix (GDIoU by default) between predicted tracks and detections, solve with Hungarian or Greedy
3. Optional RV matching: secondary image-space pass for BEV-unmatched pairs (enabled via `IS_RV_MATCHING` in config)
4. Update matched trajectories, call `unmatch_update()` on unmatched ones
5. Spawn new `Trajectory` for each unmatched detection
6. Cull dead trajectories (`status_flag == 4`) into `all_dead_trajs`

In `GLOBAL` mode, `post_processing()` is called after all frames to interpolate gaps and filter low-score tracks.

**`Trajectory`** (`trajectory.py`) — one tracked object. Maintains three independent Kalman filters:
- `cv_filter_pose` — position + velocity (x, y, vx, vy)
- `cv_filter_size` — l, w + their rates
- `cv_filter_yaw` — yaw + yaw rate (EKF, handles angle wrapping)

`status_flag` values: `1` = confirmed/active, `2` = obscured, `4` = dead.

**`matching.py`** — builds per-category cost matrices and dispatches to `Hungarian` (via `lap`) or `Greedy`. Handles both BEV and RV matching modes.

**`cost_function.py`** — implements `RO_GDIOU_3D` (rotation-aware GDIoU in bird's-eye view) and several 2D image-space IoU variants (`IOU_2D`, `GIOU_2D`, `DIOU_2D`, `SDIOU_2D`) for RV matching.

### Configuration (`config/`)

All tracker behavior is controlled by YAML. Key fields:
- `TRACKING_MODE`: `ONLINE` or `GLOBAL`
- `IS_RV_MATCHING`: enable secondary image-space matching pass
- `MATCHING.BEV.COST_STATE`: per-category state used for cost (`Predict`, `BackPredict`, `Fusion`)
- `MATCHING.BEV.COST_MODE`: cost function per category (e.g. `RO_GDIOU_3D`)
- `THRESHOLD.TRAJECTORY_THRE`: lifecycle thresholds — `MAX_UNMATCH_LENGTH`, `CONFIRMED_TRACK_LENGTH`, `PREDICT_BBOX_LENGTH`, etc.
- `KALMAN_FILTER_POSE/SIZE/YAW`: motion model (`CV`, `CA`, `CTRA`) and noise matrices (P, Q, R) per category

Category indices (0–6 for nuScenes: car, pedestrian, bicycle, motorcycle, bus, trailer, truck) are used as keys throughout the config and code.

### Kalman filters (`kalmanfilter/`)

`extend_kalman.py` provides `EKF_CV`, `EKF_CA`, `EKF_CTRA` — extended Kalman filter variants for different motion models. The yaw filter uses an EKF because yaw is non-linear (angle wrapping). Pose and size use standard linear KF.

### Dataset-specific export (`utils/`)

After tracking, results are converted back to dataset-native formats:
- KITTI: per-scene `.txt` files via `kitti_utils.py`
- nuScenes: `result.json` + `result_for_motion.json` via `nusc_utils.py`
- Waymo: binary `.bin` via `waymo_utils/`

### Evaluation (`evaluation/`)

- `static_evaluation/kitti/`, `static_evaluation/nuscenes/`, `static_evaluation/waymo/` — standard MOT metrics (HOTA, MOTA, AMOTA)
- `eval_motion.py` — custom motion quality metrics (velocity/acceleration accuracy), nuScenes only, configured via `config/nuscenes_motion_eval.yaml`

