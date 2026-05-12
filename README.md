<div align="center"><img src="./docs/MC_logo.png" width="55%"></div>

## <p align="center">Mamba-Decoupled-EKF Track</p>
### <p align="center">Fast, Efficient, and Accurate 3D Multi-Object Tracking via Mamba-driven Adaptive Decoupled Kalman Filtering</p>

<p align="center">
  Based on <a href="https://arxiv.org/abs/2409.16149">MCTrack</a> &nbsp;|&nbsp; Target: IEEE TITS / IEEE TVT
</p>

---

## 1. Overview

`Mamba-DEKF` replaces MCTrack's Transformer/GNN backbone with a Mamba SSM that adaptively predicts noise covariances for three physically-decoupled Kalman filters. Only the **position** filter receives Mamba-predicted Q/R; **size** is locked by rigid-body EMA, and **orientation** uses Von Mises directional statistics. The framework keeps `O(N)` per-tracklet cost and adds a temporal embedding for semantic association.

**Module map**

| Module | File | Responsibility |
|---|---|---|
| A — Decoupled KFs | `kalmanfilter/mamba_adaptive_kf.py` | `PositionFilter` (CV, dim 6), `SizeFilter` (Const, dim 3, EMA + locked at frame 10), `OrientationFilter` (CV, dim 2, Von Mises κ). All batched via `torch.bmm`. |
| B — Mamba Soft-Coupler | `kalmanfilter/mamba_adaptive_kf.py` | `TemporalMamba` over `[B, T, 12]` history → 2 `CholeskyHead`s (position Q/R) + kappa head (orientation concentration) + static size/ori params + `[B, 32]` embedding. |
| C — Uncertainty-aware association | `tracker/cost_function.py`, `tracker/matching.py` | Combines Ro-GDIoU (geometry), cosine sim of Mamba embeddings (semantic), and `trace(P_pos[:3,:3]) + trace(P_ori)` (uncertainty). |
| D — Tracker lifecycle | `tracker/base_tracker.py`, `tracker/trajectory.py` | Birth / Active / Coasted / Death. `Trajectory` is a stateless container; KF state in `Base3DTracker.kf_states`. |

```
Detection (per frame)
        |
        v
+---------------------+      track_history [B, T, 12]
|  Module D           | ─────────────────────────────→ +-------------------+
|  Tracker Manager    |                                |  Module B         |
+---------------------+                                |  TemporalMamba    |
        |                                              +-------------------+
        |                                                  |          |
        |  observations                  Q_pos / Q_siz     |          | embedding [B,32]
        |  z_pos, z_siz, z_ori           R_pos / R_siz     |          |
        v                                      Q_ori / R_ori / kappa  |
+-------------------------------------------------------------+       |
|  Module A: Decoupled Adaptive KFs (batched, torch.bmm)       |←─────+
|    PositionFilter    [x,y,z,vx,vy,vz]          (CV, 6)        |
|      Q/R: Cholesky heads (softplus + min_diag, PSD-locked)    |
|    SizeFilter        [l,w,h]                  (Const, 3)      |
|      Q/R: static nn.Parameter (rigid-body prior)               |
|    OrientationFilter [theta,omega]            (CV, 2)         |
|      R: 1/κ (derived from Von Mises concentration)             |
|      Q: static nn.Parameter                                    |
+-------------------------------------------------------------+
        |
        | predicted state + covariance P
        v
+-------------------------------------------------------------+
|  Module C: Association  (two-stage ByteTrack paradigm)        |
|    Stage 1: high-score dets (≥0.4) → strict matching           |
|    Stage 2: low-score dets (0.1–0.4) → relaxed rescue-only    |
|    Pure geometric cost during early training;                   |
|    semantic + uncertainty terms re-enabled after convergence.  |
+-------------------------------------------------------------+
```

---

## 2. Project Layout

```
Mamba-DEKF/
├── config/
│   ├── kitti.yaml / nuscenes.yaml / waymo.yaml      # Inference configs (MAMBA block)
│   ├── kitti_offline.yaml / nuscenes_offline.yaml
│   ├── nuscenes_motion_eval.yaml                     # Motion-quality eval config
│   └── train_nuscenes.yaml                           # Training config
├── kalmanfilter/
│   ├── mamba_adaptive_kf.py                          # Module A + B
│   ├── extend_kalman.py / base_kalman.py             # Legacy (kept for reference)
├── tracker/
│   ├── base_tracker.py                               # Module D + KF state batcher
│   ├── trajectory.py                                 # Stateless container + size EMA/locking
│   ├── matching.py                                   # Hungarian / Greedy + Module C entry
│   ├── cost_function.py                              # Ro-GDIoU + cosine + uncertainty
│   ├── bbox.py / frame.py
├── training/
│   ├── train.py                                      # Training entry + multi-step rollout
│   ├── gt_tracklet_dataset.py                        # GT tracklet extraction + Dataset
│   └── losses.py                                     # Kalman NLL + Von Mises + InfoNCE
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

# 2. Mamba-SSM (Triton kernels). Linux x86_64 + CUDA only.
pip install mamba-ssm>=1.2.0
# Falls back to GRU backbone if mamba-ssm unavailable (macOS / Windows / no CUDA),
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

For training, the **Metadata** archive of `v1.0-trainval` (or `v1.0-mini` for a smoke-test) is sufficient — point clouds and images are not needed. Set `DATA.NUSC_VERSION` and `DATA.NUSC_DATAROOT` in `config/train_nuscenes.yaml`.
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

Pre-converted BaseVersion JSONs are available from the original MCTrack release ([Google Drive](https://drive.google.com/drive/folders/15QDnPR9t3FO18fVzCyqUu4h-7COl9Utd?usp=sharing) · [Baidu Pan](https://pan.baidu.com/s/1Fk6EPeIBxThFjBJuMMKQCw?pwd=6666)). Drop them at `data/base_version/<dataset>/<detector>/<split>.json`.

### 4.2 Convert raw detections → BaseVersion

Edit the `kitti_cfg` / `nuscenes_cfg` / `waymo_cfg` dicts at the top of `preprocess/convert2baseversion.py`, then run:

```bash
python preprocess/convert2baseversion.py --dataset kitti      # or nuscenes / waymo
```

Output goes to `data/base_version/<dataset>/<detector>/<split>.json`.

---

## 5. Training

Only `TemporalMamba` is trainable. It learns to predict **position** noise (Q/R via Cholesky heads) and **orientation concentration** (κ via kappa head). Size and orientation process noise are static `nn.Parameter`s trained alongside Mamba.

The training objective is **multi-step KF rollout**: the KF runs auto-regressively for K steps from a GT-initialised state, and the NLL at each step is averaged. This prevents the model from collapsing to trivial constant-minimum output that single-step prediction allows.

### 5.1 Run training

```bash
# Step 1 (run once): extract GT tracklets to a pickle cache
python training/train.py --config config/train_nuscenes.yaml --extract-only

# Step 2: train
python training/train.py --config config/train_nuscenes.yaml

# Resume from a checkpoint:
python training/train.py --config config/train_nuscenes.yaml \
    --resume checkpoints/mamba_dekf/checkpoint_epoch70.pt
```

Training writes periodic checkpoints (`checkpoint_epoch{N}.pt`) and the best validation checkpoint (`best.pt`) under `TRAINING.SAVE_DIR`.

### 5.2 Loss design

| Component | Loss | Purpose |
|---|---|---|
| Position | Kalman NLL: `0.5·(logdet(S_pos) + yᵀS⁻¹y)` | Jointly optimises `x_pred` and `P_pred` through the innovation covariance S |
| Size | Kalman NLL with static Q/R params | Learns per-category rigid-body size uncertainty; no Mamba adaptation needed |
| Orientation | Von Mises NLL: `−κ·cos(Δθ) + log(i0e(κ)+ε) + κ` | Handles circular topology naturally (−π≡π); κ predicted by Mamba |
| Embedding | InfoNCE (supervised contrastive) | Pulls same-instance embeddings together for semantic association |

**Loss formula** (`losses.py:JointLoss`):

```
L_total = PHYSICS_SCALE · (w_pos·NLL_pos + w_siz·NLL_siz + w_ori·VM_ori) + λ · InfoNCE
```

Default weights (in `config/train_nuscenes.yaml`):

| Key | Value | Note |
|---|---|---|
| `W_POS` | 1.0 | |
| `W_SIZ` | 0.5 | |
| `W_ORI` | 5.0 | Von Mises loss is ~0.1–1.0; Gaussian NLL was ~−1.0 |
| `LAMBDA_CONTRAST` | 0.1 | |
| `PHYSICS_SCALE` | 50.0 | Global multiplier on physics losses vs contrastive |
| `ROLLOUT_STEPS` | 5 | KF auto-regressive steps per training sample |

### 5.3 Training stability safeguards

- **Minimum diagonal constraint**: Position Cholesky diagonals use `F.softplus(x) + min_diag + 1e-5` (`min_diag=0.1`, min eigenvalue ≈ 0.01). Configured via `MODEL.MIN_DIAG_Q` / `MODEL.MIN_DIAG_R`.
- **Von Mises numerical safety**: Uses `torch.special.i0e` (exponentially-scaled Bessel) to prevent NaN overflow at large κ.
- **Zero-init heads**: The last linear layer in every `CholeskyHead` and `head_kappa_ori` is initialised with `Uniform(-1e-4, 1e-4)` and bias `0`. Prevents dead gradients at epoch 1.
- **NaN guard**: Batches with NaN/Inf loss or gradients are skipped (not `nan_to_num`-ed).
- **Gradient clipping**: `clip_grad_norm_(max_norm=1.0)` on Mamba parameters.
- **Separated LR groups**: Mamba backbone `5e-5`, all other params `1e-3`.
- **Kappa monitor**: Every epoch logs `std_kappa` and `mean_kappa`. `std_kappa` should be non-zero (proves kappa is data-dependent); `mean_kappa` should increase then stabilise.

### 5.4 How to verify training is healthy

After epoch 4–5, watch the log lines:

```
Epoch 5/100 (12.3s) | train_loss=X pos=X siz=X ori=X contrast=X | kappa_m=3.45 kappa_s=0.12 | NaN=0
```

- `kappa_m` (mean κ): should rise from ~0.7 toward 3–8 over epochs, then stabilise. Continuous unbounded increase → overfitting.
- `kappa_s` (κ std): should be **non-zero** — this proves Mamba is producing input-dependent κ rather than a constant.
- If `kappa_s ≈ 0.0000` and loss is stagnant, the kappa head has collapsed. Restart with lower `W_ORI` or higher `PHYSICS_SCALE`.

---

## 6. Unsealing the Full Cost Function (after initial convergence)

During early training, the association cost is **pure geometric** (`cost = 1.0 - Ro_GDIoU`). Mamba's semantic embeddings and uncertainty estimates are unreliable before the backbone converges.

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
  INFERENCE_CONFIG: config/nuscenes.yaml
```

At epoch 40, the training script automatically edits the inference config. No manual intervention needed.

### Option B: Manual toggle

When `kappa_m` has stabilised and `kappa_s` shows healthy variance:

```bash
sed -i 's/COST_MODE: "geometric"/COST_MODE: "full"/' config/nuscenes.yaml
```

Then re-run evaluation with `python main.py --dataset nuscenes -e -p 8`.

---

## 7. Loading the Trained Weights at Inference

The inference configs include a `MAMBA:` block:

```yaml
MAMBA:
  CHECKPOINT_PATH: "checkpoints/mamba_dekf/best.pt"
  D_MODEL: 64
  D_STATE: 16
  D_CONV: 4
  EXPAND: 2
  N_MAMBA_LAYERS: 3
  EMBED_DIM: 32
  HISTORY_LEN: 10
  MAX_BATCH_SIZE: 256
  MIN_DIAG_Q: 0.1           # position Q Cholesky floor
  MIN_DIAG_R: 0.1           # position R Cholesky floor
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
python main.py --dataset nuscenes -e -p 8
```

`Base3DTracker.__init__` prints one of three lines on startup:

```
[Base3DTracker] Loaded Mamba weights from checkpoints/mamba_dekf/best.pt
[Base3DTracker] WARNING: CHECKPOINT_PATH=... not found. Running with RANDOM Mamba weights.
[Base3DTracker] WARNING: cfg['MAMBA']['CHECKPOINT_PATH'] not set.
```

The architecture in `MAMBA:` must match what was trained. Loading uses `strict=False`, so minor checkpoint key drift is reported but does not crash.

---

## 8. Tracking + Evaluation

```bash
# Online tracking + evaluation
python main.py --dataset kitti     -e -p 2
python main.py --dataset nuscenes  -e -p 2
python main.py --dataset waymo     -e -p 2

# Offline / global mode (uses *_offline.yaml, enables Trajectory.filtering)
python main.py --dataset kitti     -m -e -p 2

# Quick 2-scene debug pass
python main.py --dataset nuscenes  --debug -e -p 1
```

Flags:
- `-e / --eval` runs the dataset-native evaluator after tracking.
- `-p N / --process N` parallelises across scenes (use 2 for typical 16GB GPU, 1 for debugging).
- `-m / --mode` switches to the offline `*_offline.yaml` config.
- `--debug` runs only the first 2 scenes.

Outputs: `results/<dataset>/<YYYYMMDD_HHMMSS>/...`.

### 8.1 Per-dataset evaluators

| Dataset | Metric | Implementation |
|---|---|---|
| KITTI | HOTA / MOTA / IDSW | `evaluation/static_evaluation/kitti/evaluation_HOTA/scripts/run_kitti.py` |
| nuScenes | AMOTA / MOTA / IDS | `evaluation/static_evaluation/nuscenes/eval.py` — uses `nuscenes-devkit` |
| Waymo | MOTA L1/L2, MOTP | `evaluation/static_evaluation/waymo/eval.py` — requires `waymo-open-dataset` |

### 8.2 Motion-quality evaluation (nuScenes)

```bash
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py
python evaluation/eval_motion.py
```

---

## 9. Submission to Public Benchmarks

<details><summary><b>KITTI</b></summary>

In `config/kitti.yaml` set `SPLIT: test` and `DETECTOR: virconv`, run tracking, then submit the per-scene `.txt` outputs to the [KITTI tracking challenge](https://www.cvlibs.net/datasets/kitti/user_submit.php).
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

## 10. End-to-End Recipe

```bash
# 0. (one-time) place data/ and trained checkpoint
ls data/base_version/nuscenes/centerpoint/val.json
ls checkpoints/mamba_dekf/best.pt

# 1. Sanity check the pipeline + checkpoint
python tools/verify_pipeline.py --ckpt checkpoints/mamba_dekf/best.pt

# 2. IF training is fresh (<50 epochs): ensure cost fallback is active
#    Config THRESHOLD.BEV.COST_MODE should be "geometric"

# 3. IF training converged (50+ epochs): unseal the full cost
#    See Section 6 — re-enable semantic + uncertainty terms

# 4. Tracking on nuScenes val with evaluation
python main.py --dataset nuscenes -e -p 2

# 5. (optional) motion-quality evaluation
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py
python evaluation/eval_motion.py
```

Final results are written under `results/nuscenes/<timestamp>/result.json`.

---

## 11. Configuration Cheatsheet

| Key | Purpose |
|---|---|
| `MAMBA.CHECKPOINT_PATH` | Trained TemporalMamba weights. Empty/missing → random init (warning printed). |
| `MAMBA.HISTORY_LEN` | Length T of the joint-state history window. **Must equal training value.** |
| `MAMBA.D_MODEL / D_STATE / D_CONV / EXPAND / N_MAMBA_LAYERS / EMBED_DIM` | Backbone architecture. Must match training. |
| `MAMBA.MIN_DIAG_Q` | Floor on position Q Cholesky diagonal (default 0.1). |
| `MAMBA.MIN_DIAG_R` | Floor on position R Cholesky diagonal (default 0.1). |
| `LOSS.W_ORI` | Von Mises loss weight (default 5.0). Tune if orientation loss dominates. |
| `LOSS.PHYSICS_SCALE` | Global multiplier on state losses vs contrastive (default 50.0). |
| `LOSS.ROLLOUT_STEPS` | Number of KF auto-regressive predict steps per sample (default 5). |
| `THRESHOLD.BEV.COST_MODE` | `"geometric"` (early training) or `"full"` (converged). |
| `THRESHOLD.BEV.COST_THRE` | Per-category cost gate. Stage-2 matching uses 2× this. |
| `THRESHOLD.BEV.W_SEMANTIC` / `W_UNCERTAINTY` | Per-category weights for full cost mode. |
| `TRACKING_MODE` | `ONLINE` (per-frame) or `GLOBAL` (offline + `Trajectory.filtering`). |
| `FRAME_RATE` | Fallback `delta_t = 1/FRAME_RATE` when timestamps are missing. |

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