# Mamba-DEKF

`Mamba-DEKF` is a single-stage 3D MOT project built around a decoupled Kalman filtering stack and a Mamba-based noise predictor. The active development path is now:

1. detection-driven nuScenes cache construction
2. fusion-history augmentation aligned with the tracker runtime
3. Mamba training
4. single-stage nuScenes evaluation

Deprecated experimental branches such as ByteTrack rescue matching, post-hoc track-score calibration, and automated parameter-search loops have been removed from the mainline codebase.

## Current baseline

The frozen nuScenes baseline is:

- config: [config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml](/home/alvin/demo/Mamba-DEKF/config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml)
- mode: `pure_dekf` + `mctrack` compat single-stage
- status: protected reference config for future Mamba improvements
- reference AMOTA: `0.739`
- reference trailer AMOTA for the protected target: `0.533`

Important:

- local history also contains a nearby run with `trailer=0.534`, but that run corresponds to `AMOTA=0.737`, not `0.739`
- when validating regression-to-baseline, use `AMOTA=0.739` with the `trailer=0.533` profile as the primary target

`main.py` now defaults to this config when you run `--dataset nuscenes` without passing `--config`.

## Active files

Core runtime:

- [main.py](/home/alvin/demo/Mamba-DEKF/main.py)
- [tracker/base_tracker.py](/home/alvin/demo/Mamba-DEKF/tracker/base_tracker.py)
- [tracker/trajectory.py](/home/alvin/demo/Mamba-DEKF/tracker/trajectory.py)
- [kalmanfilter/mamba_adaptive_kf.py](/home/alvin/demo/Mamba-DEKF/kalmanfilter/mamba_adaptive_kf.py)

Training pipeline:

- [config/train_nuscenes.yaml](/home/alvin/demo/Mamba-DEKF/config/train_nuscenes.yaml)
- [training/train.py](/home/alvin/demo/Mamba-DEKF/training/train.py)
- [training/det_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/det_tracklet_dataset.py)
- [training/gt_tracklet_dataset.py](/home/alvin/demo/Mamba-DEKF/training/gt_tracklet_dataset.py)

Data/build tools:

- [tools/build_centerpoint_mini_train_dataset.py](/home/alvin/demo/Mamba-DEKF/tools/build_centerpoint_mini_train_dataset.py)
- [tools/augment_tracklet_cache_with_fusion.py](/home/alvin/demo/Mamba-DEKF/tools/augment_tracklet_cache_with_fusion.py)
- [tools/audit_det_tracklet_cache.py](/home/alvin/demo/Mamba-DEKF/tools/audit_det_tracklet_cache.py)
- [tools/summarize_noise_audit.py](/home/alvin/demo/Mamba-DEKF/tools/summarize_noise_audit.py)
- [tools/verify_pipeline.py](/home/alvin/demo/Mamba-DEKF/tools/verify_pipeline.py)

## Environment

```bash
conda create -n mamba-dekf python=3.10 -y
conda activate mamba-dekf

pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
pip install mamba-ssm>=1.2.0
pip install -r requirements.txt
```

Optional smoke test:

```bash
python tools/verify_pipeline.py
python tools/verify_pipeline.py --ckpt checkpoints/mamba_dekf/best.pt
```

## Required data layout

The active training/evaluation flow assumes:

- nuScenes metadata root: `/root/autodl-tmp/data/nuscenes/datasets/`
- BaseVersion CenterPoint detections: `/root/autodl-tmp/data/base_version/nuscenes/centerpoint/`

At minimum, the following should exist:

```text
/root/autodl-tmp/data/nuscenes/datasets/
├── maps/
├── samples/
├── sweeps/
├── v1.0-trainval/
└── v1.0-test/         # only needed when you evaluate test split

/root/autodl-tmp/data/base_version/nuscenes/centerpoint/
└── val.json
```

## End-to-end pipeline

### 1. Build the detection-driven training cache

This step aligns CenterPoint BaseVersion detections with nuScenes GT and writes a detection-driven tracklet cache.

Script:

- [tools/build_centerpoint_mini_train_dataset.py](/home/alvin/demo/Mamba-DEKF/tools/build_centerpoint_mini_train_dataset.py)

Recommended command:

```bash
python tools/build_centerpoint_mini_train_dataset.py \
  --det-json /root/autodl-tmp/data/base_version/nuscenes/centerpoint/val.json \
  --nusc-dataroot /root/autodl-tmp/data/nuscenes/datasets/ \
  --nusc-version v1.0-trainval \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --max-scenes 10 \
  --dist-th 2.0 \
  --min-frames 2 \
  --min-matched-frames 2 \
  --train-config config/train_nuscenes.yaml
```

What it produces:

- tracklet cache: `/root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl`
- summary json: `/root/autodl-tmp/data/training_cache/nuscenes/mini/train_det_summary.json`

Important behavior:

- each frame stores both GT features and detector observation features
- misses can be preserved unless `--no-misses` is used
- the script estimates effective sample counts using `HISTORY_LEN`, `ROLLOUT_STEPS`, and adaptive window settings from `config/train_nuscenes.yaml`

### 2. Add fusion-history fields

This step augments the aligned cache with MCTrack-compatible local fusion history so training inputs better match runtime semantics.

Script:

- [tools/augment_tracklet_cache_with_fusion.py](/home/alvin/demo/Mamba-DEKF/tools/augment_tracklet_cache_with_fusion.py)

Recommended command:

```bash
python tools/augment_tracklet_cache_with_fusion.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --train-config config/train_nuscenes.yaml
```

What it adds per frame:

- `fusion_valid`
- `fusion_is_fake`
- `fusion_global_xyz`
- `fusion_lwh`
- `fusion_yaw`
- `fusion_velocity`
- `fusion_feature_12`

It also writes:

- `/root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion_summary.json`

### 3. Audit the cache before training

This step is optional but recommended. It tells you whether the cache is large enough, balanced enough, and whether matched-frame ratios are acceptable.

Script:

- [tools/audit_det_tracklet_cache.py](/home/alvin/demo/Mamba-DEKF/tools/audit_det_tracklet_cache.py)

Recommended command:

```bash
python tools/audit_det_tracklet_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --train-config config/train_nuscenes.yaml \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion_audit.json
```

### 4. Point the training config at the generated cache

Edit [config/train_nuscenes.yaml](/home/alvin/demo/Mamba-DEKF/config/train_nuscenes.yaml) and make sure these paths are correct:

```yaml
DATA:
  TRAIN_SOURCE: det
  VAL_SOURCE: det
  TRAIN_TRACKLET_PKL: /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl
  VAL_TRACKLET_PKL: /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl
  HISTORY_SOURCE: fusion
  INIT_STATE_SOURCE: fusion
```

Notes:

- `TRAIN_SOURCE=det` and `VAL_SOURCE=det` mean training uses detection-driven caches instead of GT-only tracklets.
- `HISTORY_SOURCE=fusion` and `INIT_STATE_SOURCE=fusion` keep training aligned with the current single-stage runtime.
- `TRAINING.INFERENCE_CONFIG` already points at the frozen baseline config.

### 5. Train Mamba

Script:

- [training/train.py](/home/alvin/demo/Mamba-DEKF/training/train.py)

Standard training:

```bash
python training/train.py --config config/train_nuscenes.yaml
```

Resume training:

```bash
python training/train.py \
  --config config/train_nuscenes.yaml \
  --resume checkpoints/mamba_dekf/checkpoint_epoch70.pt
```

Optional GT-only extraction mode:

```bash
python training/train.py \
  --config config/train_nuscenes.yaml \
  --extract-only
```

Training outputs:

- periodic checkpoints in `checkpoints/mamba_dekf/`
- best checkpoint in `checkpoints/mamba_dekf/best.pt`
- TensorBoard logs under the configured save directory

### 6. Evaluate on nuScenes

The simplest command now uses the frozen baseline by default:

```bash
python main.py --dataset nuscenes --eval -p 12
```

Explicit command:

```bash
python main.py \
  --dataset nuscenes \
  --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml \
  -p 12
```

If you want to evaluate a newly trained Mamba checkpoint:

1. update `MAMBA.CHECKPOINT_PATH` in [config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml](/home/alvin/demo/Mamba-DEKF/config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml)
2. run the same evaluation command again

Results are written to:

- `SAVE_PATH/<dataset>/<timestamp>/results.json`
- official nuScenes metric files under the same result directory when `--eval` is enabled

## Suggested working commands

Build cache:

```bash
python tools/build_centerpoint_mini_train_dataset.py \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl
```

Augment cache:

```bash
python tools/augment_tracklet_cache_with_fusion.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_det.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl
```

Audit cache:

```bash
python tools/audit_det_tracklet_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --train-config config/train_nuscenes.yaml
```

Train:

```bash
python training/train.py --config config/train_nuscenes.yaml
```

Evaluate:

```bash
python main.py --dataset nuscenes --eval -p 12
```

## Notes for future work

The repo has been intentionally narrowed to support the next phase: improving AMOTA through better Mamba training/inference alignment, not through additional heuristic post-processing branches.

That means:

- single-stage association is the only retained matching path
- dirty suppressor remains because it is part of the protected `0.739` baseline
- ByteTrack rescue logic, track-score calibration, and automated heuristic search are no longer part of the maintained path
