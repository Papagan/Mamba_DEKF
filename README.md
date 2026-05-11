<div align="center"><img src="./docs/MC_logo.png" width="55%"></div>

## <p align="center">Mamba-Decoupled-EKF Track</p>
### <p align="center">Fast, Efficient, and Accurate 3D Multi-Object Tracking via Mamba-driven Adaptive Decoupled Kalman Filtering</p>

<p align="center">
  Based on <a href="https://arxiv.org/abs/2409.16149">MCTrack</a> &nbsp;|&nbsp; Target: IEEE TITS / IEEE TVT
</p>

---

## 1. Overview

`Mamba-DEKF` refactors the original MCTrack pipeline around three decoupled Kalman filters whose process and measurement noise are dynamically predicted by a Mamba-based temporal backbone. The framework keeps `O(N)` per-tracklet cost (no Transformer / no GNN) and adds a Mamba-driven semantic embedding plus an uncertainty-aware association cost.

**Module map**

| Module | File | Responsibility |
|---|---|---|
| A — Decoupled KFs | `kalmanfilter/mamba_adaptive_kf.py` | `PositionFilter` (CA, dim 8), `SizeFilter` (Const, dim 3), `OrientationFilter` (CV, dim 2). All batched via `torch.bmm`. |
| B — Mamba Soft-Coupler | `kalmanfilter/mamba_adaptive_kf.py` | `TemporalMamba` over `[B, T, 14]` history → six `CholeskyHead`s emit PSD-locked Q/R + a `[B, embed_dim]` embedding. Features: 13-dim kinematics + `det_score` for input conditioning. |
| C — Uncertainty-aware association | `tracker/cost_function.py`, `tracker/matching.py` | Combines Ro-GDIoU (geometry), cosine sim of Mamba embeddings (semantic), and `trace(P_pos[:3,:3]) + trace(P_ori)` (uncertainty). |
| D — Tracker lifecycle | `tracker/base_tracker.py`, `tracker/trajectory.py` | Birth / Active / Coasted / Death management. `Trajectory` is a stateless data container; all KF state lives in `Base3DTracker.kf_states`. |

```
Detection (per frame)
        |
        v
+---------------------+      track_history [B, T, 14]
|  Module D           | ─────────────────────────────→ +-------------------+
|  Tracker Manager    |                                |  Module B         |
+---------------------+                                |  TemporalMamba    |
        |                                              +-------------------+
        |                                                  |          |
        |  observations                  Q_pos/Q_siz/Q_ori |          | embedding [B,32]
        |  z_pos, z_siz, z_ori           R_pos/R_siz/R_ori |          |
        v                                                  v          |
+-------------------------------------------------------------+       |
|  Module A: Decoupled Adaptive KFs (batched, torch.bmm)       |←─────+
|    PositionFilter    [x,y,z,vx,vy,vz,ax,ay]   (CA, 8)        |
|    SizeFilter        [l,w,h]                  (Const, 3)     |
|    OrientationFilter [theta,omega]            (CV, 2)        |
|    PSD safety lock: Softplus + min_diag + 1e-5 on Cholesky diagonals |
+-------------------------------------------------------------+
        |
        | predicted state + covariance P
        v
+-------------------------------------------------------------+
|  Module C: Association  (two-stage ByteTrack paradigm)        |
|    Stage 1: high-score dets → strict Ro-GDIoU matching         |
|    Stage 2: low-score dets → relaxed matching (rescue only)   |
|    Pure geometric cost during early training;                   |
|    semantic + uncertainty terms re-enabled after convergence.  |
+-------------------------------------------------------------+
```

---

## 2. Project Layout

```
Mamba-DEKF/
├── config/
│   ├── kitti.yaml / nuscenes.yaml / waymo.yaml      # Inference configs (now include MAMBA: block)
│   ├── kitti_offline.yaml / nuscenes_offline.yaml
│   ├── nuscenes_motion_eval.yaml                     # Motion-quality eval config
│   └── train_nuscenes.yaml                           # Training config
├── kalmanfilter/
│   ├── mamba_adaptive_kf.py                          # Module A + B
│   ├── extend_kalman.py / base_kalman.py             # Legacy (kept for reference)
├── tracker/
│   ├── base_tracker.py                               # Module D + KF state batcher
│   ├── trajectory.py                                 # Stateless trajectory container
│   ├── matching.py                                   # Hungarian / Greedy + Module C entry
│   ├── cost_function.py                              # Ro-GDIoU + cosine + uncertainty
│   ├── bbox.py / frame.py
├── training/
│   ├── train.py                                      # Training entry point
│   ├── gt_tracklet_dataset.py                        # GT tracklet extraction + Dataset
│   └── losses.py                                     # Kalman NLL + InfoNCE joint loss
├── dataset/
│   └── baseversion_dataset.py                        # BaseVersion JSON → Frame/BBox loader
├── preprocess/
│   ├── convert2baseversion.py                        # Entry: raw detections → BaseVersion
│   ├── convert_{kitti,nuscenes,waymo}.py
│   └── motion_dataset/convert_nuscenes_result_to_pkl.py
├── evaluation/
│   ├── eval_motion.py                                # Custom motion-quality eval (nuScenes)
│   ├── motion_evaluation/                            # Pose / velocity metric helpers
│   └── static_evaluation/
│       ├── kitti/evaluation_HOTA/                    # HOTA / MOTA (KITTI)
│       ├── nuscenes/eval.py                          # AMOTA (nuScenes)
│       └── waymo/eval.py                             # MOTA L1/L2 (Waymo)
├── utils/                                            # KITTI / nuScenes / Waymo IO + math helpers
├── tools/
│   └── verify_pipeline.py                            # End-to-end smoke test (no data needed)
├── main.py                                           # Tracking + evaluation entry
├── requirements.txt
└── README.md  (this file)
```

---

## 3. Environment Setup

```bash
conda create -n mamba-dekf python=3.10 -y
conda activate mamba-dekf

# 1. PyTorch — match your CUDA toolkit
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121
# (CUDA 11.8 users: replace cu121 with cu118)

# 2. Mamba-SSM (Triton kernels). Linux x86_64 + CUDA only.
pip install mamba-ssm>=1.2.0
# If mamba-ssm cannot be installed (macOS / Windows / no CUDA),
# the code automatically falls back to a GRU backbone — no edits required,
# but trained Mamba weights cannot be reused under the GRU fallback.

# 3. Tracker + evaluation deps
pip install -r requirements.txt
```

**Smoke test** (no dataset, no GPU required):

```bash
python tools/verify_pipeline.py
# With a trained checkpoint:
python tools/verify_pipeline.py --ckpt checkpoints/mamba_dekf/best.pt
```

A successful run prints PSD checks for every Q/R, runs `predict_with_mamba` + `update_with_mamba`, and (if a checkpoint is provided) loads it with `strict=False`.

---

## 4. Data Preparation

The full pipeline expects raw detections to first be converted into the unified **BaseVersion** JSON format (one file per `<dataset>/<detector>/<split>.json`).

### 4.1 Directory layout

Create a `data/` folder at the project root and arrange it as below.

<details><summary><b>KITTI</b></summary>

```
data/
└── kitti/
    ├── datasets/
    │   ├── testing/
    │   │   ├── calib/
    │   │   └── pose/
    │   └── training/
    │       ├── calib/
    │       ├── label_02/
    │       └── pose/
    └── detectors/
        ├── casa/
        └── point_rcnn/
```

</details>

<details><summary><b>nuScenes</b></summary>

```
data/
└── nuscenes/
    ├── datasets/
    │   ├── maps/
    │   ├── samples/
    │   ├── sweeps/
    │   ├── v1.0-test/
    │   └── v1.0-trainval/
    └── detectors/
        ├── centerpoint/
        └── largekernel/
```

For training only, the **Metadata** archive of `v1.0-trainval` (or `v1.0-mini` for a small smoke-test) is sufficient — point clouds and camera images are not needed. Set `DATA.NUSC_VERSION` and `DATA.NUSC_DATAROOT` in `config/train_nuscenes.yaml`.

</details>

<details><summary><b>Waymo</b></summary>

```
data/
└── waymo/
    ├── datasets/
    │   ├── testing/
    │   │   ├── ego_info/
    │   │   └── ts_info/
    │   └── validation/
    │       ├── ego_info/
    │       └── ts_info/
    └── detectors/
        └── ctrl/
            ├── testing/
            └── validation/
```

</details>

Pre-converted BaseVersion JSONs are also available from the original MCTrack release ([Google Drive](https://drive.google.com/drive/folders/15QDnPR9t3FO18fVzCyqUu4h-7COl9Utd?usp=sharing) · [Baidu Pan](https://pan.baidu.com/s/1Fk6EPeIBxThFjBJuMMKQCw?pwd=6666)). Drop them at `data/base_version/<dataset>/<detector>/<split>.json`.

### 4.2 Convert raw detections → BaseVersion

Edit the `kitti_cfg` / `nuscenes_cfg` / `waymo_cfg` dicts at the top of `preprocess/convert2baseversion.py` (especially `raw_data_path`), then run:

```bash
python preprocess/convert2baseversion.py --dataset kitti      # or nuscenes / waymo
```

Output goes to `data/base_version/<dataset>/<detector>/<split>.json` and is what `main.py` reads via `cfg.DETECTIONS_ROOT`.

---

## 5. Training

`TemporalMamba` is the only learnable module. It is trained on **GT tracklets** of nuScenes; the three decoupled KFs consume the predicted `Q/R` directly and are not trained.

### 5.1 Run training

```bash
# Step 1 (run once): extract GT tracklets to a pickle cache
python training/train.py --config config/train_nuscenes.yaml --extract-only

# Step 2: train
python training/train.py --config config/train_nuscenes.yaml

# Resume from a checkpoint:
python training/train.py --config config/train_nuscenes.yaml \
    --resume checkpoints/mamba_dekf/checkpoint_epoch10.pt
```

Training writes periodic checkpoints (`checkpoint_epoch{N}.pt`) and the best validation checkpoint (`best.pt`) under `TRAINING.SAVE_DIR`.

### 5.2 Loss design

| Component | Loss | Purpose |
|---|---|---|
| Position | Kalman NLL: `0.5·(logdet(S_pos) + yᵀS⁻¹y)` | Jointly optimises `x_pred` and `P_pred` through the innovation covariance S |
| Size | Kalman NLL (same form) | Same for the size filter |
| Orientation | Angle-wrapped NLL + logdet guard: `0.5·(logdet(S_ori) + ỹᵀS⁻¹ỹ) + 0.1·ReLU(logdet(S_ori) − 2.0)` where `ỹ = wrap_to_pi(z - Hx)` | prevents ±π discontinuity; logdet guard penalises only pathologically LARGE S (never pushes R→0) |
| Embedding | InfoNCE (supervised contrastive) | Pulls same-instance embeddings together |

**Loss formula** (`losses.py:JointLoss`):

```
L_total = PHYSICS_SCALE · (w_pos·NLL_pos + w_siz·NLL_siz + w_ori·NLL_yaw) + λ · InfoNCE
```

Default weights (in `config/train_nuscenes.yaml`):

| Key | Value | Note |
|---|---|---|
| `W_POS` | 1.0 | |
| `W_SIZ` | 0.5 | |
| `W_ORI` | 50.0 | Large multiplier — yaw NLL is small in magnitude |
| `LAMBDA_CONTRAST` | 0.1 | |
| `PHYSICS_SCALE` | 50.0 | Global multiplier on all physics losses vs contrastive |

### 5.3 Training stability safeguards

- **Minimum diagonal constraint**: Cholesky diagonals use `F.softplus(x) + min_diag + 1e-5`. R heads get `min_diag=0.1` (min eigenvalue ≈ 0.01), Q heads get `min_diag=0.03` (min eigenvalue ≈ 0.001). This prevents the model from collapsing to near-zero eigenvalues that produce artificially negative NLL.
- **Logdet guard (anti-cheat)**: Orientation loss uses `0.1·ReLU(logdet(S_ori) − 2.0)` instead of a trace penalty. Unlike `λ·trace(S)` whose gradient `λ·I` pushes all R entries toward zero, the logdet guard only activates when the innovation covariance is pathologically large — it never encourages R→0.
- **Zero-init CholeskyHeads**: The last linear layer in every `CholeskyHead` is initialised with `Uniform(-1e-4, 1e-4)` and bias `0`. Output ≈ 0 → `Softplus(0) ≈ 0.69` — a healthy initial variance before epoch 1.
- **NaN guard**: Batches with NaN/Inf loss or gradients are skipped (not `nan_to_num`-ed).
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` on Mamba parameters.
- **Separated LR groups**: Mamba backbone `5e-5`, all other params `1e-3`.
- **Q/R diagonal std monitor**: Every epoch logs `std_R_ori`, `std_Q_ori` (mean over batch of per-dimension diagonal std). If these flat-line at 0, the Cholesky heads have degenerated to constant output.

### 5.4 How to verify training is healthy

After epoch 4–5, watch the log lines:

```
Epoch 5/100 (12.3s) | train_loss=X pos=X siz=X ori=X contrast=X | stdRori=0.12 stdQori=0.08 | NaN=0
```

- `stdRori` and `stdQori` should be **non-zero** and fluctuating across epochs — this proves the Mamba Q/R heads are producing diverse, data-dependent outputs.
- If either std stays at `0.0000` for multiple epochs, the Cholesky heads have collapsed to constant output (gradient death). Stop training, check the `cholesky_to_psd` implementation, and restart.

---

## 6. Unsealing the Full Cost Function (after initial convergence)

During early training, the association cost is **pure geometric** (`cost = 1.0 - Ro_GDIoU`). Mamba's semantic embeddings and uncertainty estimates are unreliable before the backbone converges and would cause massive ID switches and false positives.

The cost mode is controlled by `COST_MODE` in the inference config (`config/{kitti,nuscenes,waymo}.yaml`):

```yaml
THRESHOLD:
  BEV:
    COST_MODE: "geometric"   # "geometric" (safe) | "full" (converged)
```

### Option A: Auto-unseal (recommended)

Set `AUTO_UNSEAL_EPOCH` in `config/train_nuscenes.yaml`:

```yaml
TRAINING:
  AUTO_UNSEAL_EPOCH: 40           # toggle COST_MODE → "full" at this epoch
  INFERENCE_CONFIG: config/nuscenes.yaml   # which YAML to update
```

At epoch 40, the training script automatically edits the inference config to set `COST_MODE: "full"`. No manual intervention needed.

### Option B: Manual toggle

When `loss_pos` and `loss_siz` have stabilised and the Q/R std monitor shows healthy variance:

```bash
# Edit the inference config by hand:
sed -i 's/COST_MODE: "geometric"/COST_MODE: "full"/' config/nuscenes.yaml
```

Then re-run evaluation with `python main.py --dataset nuscenes -e -p 8`.

The full cost enables the Mamba-driven semantic affinity and Kalman uncertainty penalty in the matching gate, which tightens associations for well-tracked objects while widening the gate during ambiguous occlusion phases.

---

## 7. Loading the Trained Weights at Inference

The inference configs (`config/{kitti,nuscenes,waymo}.yaml`) now include a `MAMBA:` block:

```yaml
MAMBA:
  CHECKPOINT_PATH: "checkpoints/mamba_dekf/best.pt"   # ← put your trained weights here
  D_MODEL: 64
  D_STATE: 16
  D_CONV: 4
  EXPAND: 2
  N_MAMBA_LAYERS: 3
  EMBED_DIM: 32
  HISTORY_LEN: 10
  MAX_BATCH_SIZE: 256
```

**Required action: drop the trained checkpoint here**

```
Mamba-DEKF/
└── checkpoints/
    └── mamba_dekf/
        └── best.pt        ← move/copy your trained file here
```

Or override the path:

```bash
# either edit config YAML in place, or:
python main.py --dataset nuscenes -e -p 8   # config CHECKPOINT_PATH is read at startup
```

`Base3DTracker.__init__` will print one of three lines on startup so you can verify what happened:

```
[Base3DTracker] Loaded Mamba weights from checkpoints/mamba_dekf/best.pt
[Base3DTracker] WARNING: CHECKPOINT_PATH=... not found. Running with RANDOM Mamba weights — results will be poor.
[Base3DTracker] WARNING: cfg['MAMBA']['CHECKPOINT_PATH'] not set. Running with RANDOM Mamba weights — set CHECKPOINT_PATH to load trained weights.
```

The architecture in `MAMBA:` must match what was trained — keep `D_MODEL/D_STATE/D_CONV/EXPAND/N_MAMBA_LAYERS/EMBED_DIM/HISTORY_LEN` identical to `config/train_nuscenes.yaml`. Loading uses `strict=False`, so any minor checkpoint key drift is reported but does not crash.

---

## 8. Tracking + Evaluation

```bash
# Online tracking + evaluation
python main.py --dataset kitti     -e -p 8
python main.py --dataset nuscenes  -e -p 8
python main.py --dataset waymo     -e -p 8

# Offline / global mode (uses *_offline.yaml)
python main.py --dataset kitti     -m -e -p 8

# Quick 2-scene debug pass
python main.py --dataset nuscenes  --debug -e -p 1
```

Flags:
- `-e / --eval` runs the dataset-native evaluator after tracking.
- `-p N / --process N` parallelises across scenes.
- `-m / --mode` switches to the offline `*_offline.yaml` config (KITTI / nuScenes).
- `--debug` runs only the first 2 scenes.

Outputs: `results/<dataset>/<YYYYMMDD_HHMMSS>/...`.

### 8.1 Per-dataset evaluators

| Dataset | Metric | Implementation |
|---|---|---|
| KITTI | HOTA / MOTA / IDSW | `evaluation/static_evaluation/kitti/evaluation_HOTA/scripts/run_kitti.py` (`eval_kitti(cfg)`) |
| nuScenes | AMOTA / MOTA / IDS | `evaluation/static_evaluation/nuscenes/eval.py` (`eval_nusc(cfg)`) — uses `nuscenes-devkit` |
| Waymo | MOTA L1/L2, MOTP | `evaluation/static_evaluation/waymo/eval.py` (`eval_waymo(cfg, save_path)`) — requires `waymo-open-dataset` |

The evaluators are wired into `main.py` and called automatically when `--eval` is set. KITTI labels live under `evaluation/static_evaluation/kitti/gt/`. For Waymo, set the WOD repo path inside `evaluation/static_evaluation/waymo/eval.py` before running.

### 8.2 Motion-quality evaluation (nuScenes)

A custom evaluator measures per-track velocity / acceleration accuracy:

```bash
# Convert tracking results to the motion-eval pickle
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py

# Score
python evaluation/eval_motion.py
```

Settings live in `config/nuscenes_motion_eval.yaml`.

---

## 9. Submission to Public Benchmarks

<details><summary><b>KITTI</b></summary>

In `config/kitti.yaml` set `SPLIT: test` and `DETECTOR: virconv`, run tracking, then zip the per-scene `.txt` outputs and submit to the [KITTI tracking challenge](https://www.cvlibs.net/datasets/kitti/user_submit.php).

</details>

<details><summary><b>nuScenes</b></summary>

In `config/nuscenes.yaml` set `SPLIT: test` and `DETECTOR: largekernel`, run tracking, then submit `result.json` to the [nuScenes tracking challenge](https://eval.ai/web/challenges/challenge-page/476/overview).

</details>

<details><summary><b>Waymo</b></summary>

```bash
vim waymo-od/src/waymo_open_dataset/metrics/tools/submission.txtpb
mkdir test_result
waymo-od/src/bazel-bin/waymo_open_dataset/metrics/tools/create_submission \
    --input_filenames='results/waymo/testing/bin/pred.bin' \
    --output_filename='test_result/model' \
    --submission_filename='waymo-od/src/waymo_open_dataset/metrics/tools/submission.txtpb'
tar cvf test_result/my_model.tar test_result/
gzip test_result/my_model.tar
```

Submit to the [Waymo tracking challenge](https://waymo.com/open/challenges/2020/3d-tracking/).

</details>

---

## 10. End-to-End Recipe (assuming training is already done)

```bash
# 0. (one-time) place data/ and trained checkpoint
ls data/base_version/nuscenes/centerpoint/val.json    # detections in BaseVersion form
ls checkpoints/mamba_dekf/best.pt                     # trained TemporalMamba

# 1. sanity check the pipeline + checkpoint
python tools/verify_pipeline.py --ckpt checkpoints/mamba_dekf/best.pt

# 2. IF training is fresh (<50 epochs): ensure cost fallback is active
#    Open tracker/cost_function.py → cal_uncertainty_aware_cost →
#    verify: cost = geometric_cost  (pure geometry, no Mamba terms)

# 3. IF training converged (50+ epochs): unseal the full cost
#    See Section 6 — re-enable semantic + uncertainty terms

# 4. tracking on nuScenes val with evaluation
python main.py --dataset nuscenes -e -p 8

# 5. (optional) motion-quality evaluation
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py
python evaluation/eval_motion.py
```

Console output during step 4 will show:

```
[Base3DTracker] Loaded Mamba weights from checkpoints/mamba_dekf/best.pt
Loading data from data/base_version/nuscenes/centerpoint/val.json...
Processing scene-0103: 100%|████████████████| 41/41
...
```

Final results are written under `results/nuscenes/<timestamp>/result.json`.

---

## 11. Configuration Cheatsheet

| Key | Purpose |
|---|---|
| `MAMBA.CHECKPOINT_PATH` | Trained TemporalMamba weights. Empty / missing → random init (warning printed). |
| `MAMBA.HISTORY_LEN` | Length T of the joint-state history window fed to Mamba. **Must equal training value.** |
| `MAMBA.D_MODEL / D_STATE / D_CONV / EXPAND / N_MAMBA_LAYERS / EMBED_DIM` | Backbone architecture. Must match training. |
| `THRESHOLD.BEV.W_SEMANTIC` | Per-category weight on `1 - cos_sim(emb_trk, emb_det)`. **Disabled during early training (fallback mode).** |
| `THRESHOLD.BEV.W_UNCERTAINTY` | Per-category weight on `trace(P_pos[:3,:3]) + trace(P_ori)`. **Disabled during early training.** |
| `THRESHOLD.BEV.COST_THRE` | Per-category cost gate. Stage-2 matching uses 2× this threshold for low-score dets. |
| `TRACKING_MODE` | `ONLINE` (per-frame output) or `GLOBAL` (offline interpolation in `Trajectory.filtering`). |
| `FRAME_RATE` | Fallback `delta_t = 1/FRAME_RATE` when timestamps are missing. |
| `LOSS.PHYSICS_SCALE` | Global multiplier on state losses vs contrastive (default 50.0). Tune if contrast dominates. |
| `LOSS.W_ORI` | Orientation NLL weight — large (50.0) because yaw NLL is inherently small. |

---

## 12. Acknowledgement

- Detection: [CTRL](https://github.com/tusen-ai/SST), [VirConv](https://github.com/hailanyi/VirConv), [CenterPoint](https://github.com/tianweiy/CenterPoint), [LargeKernel3D](https://github.com/dvlab-research/LargeKernel3D)
- Tracking: [MCTrack](https://github.com/megvii-research/MCTrack), [PC3T](https://github.com/hailanyi/3D-Multi-Object-Tracker), [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT), [ImmortalTracker](https://github.com/esdolo/ImmortalTracker)
- State Space Models: [Mamba](https://github.com/state-spaces/mamba)

## 13. Citation

```bibtex
@article{wang2024mctrack,
  title={MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving},
  author={Wang, Xiyang and Qi, Shouzheng and Zhao, Jieyou and Zhou, Hangning and Zhang, Siyu and Wang, Guoan and Tu, Kai and Guo, Songlin and Zhao, Jianbo and Li, Jian and others},
  journal={arXiv preprint arXiv:2409.16149},
  year={2024}
}
```
