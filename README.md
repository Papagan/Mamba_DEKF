
<div  align=center><img src="./docs/MC_logo.png" width="55%"></div>

## <p align=center>Mamba-Decoupled-EKF Track</p>
### <p align=center>Fast, Efficient, and Accurate 3D Multi-Object Tracking via Mamba-driven Adaptive Decoupled Kalman Filtering</p>

<p align="center">
  Based on <a href="https://arxiv.org/abs/2409.16149">MCTrack</a> &nbsp;|&nbsp; Target: IEEE TITS / IEEE TVT
</p>

---

## Architecture

```
Detection (per frame)
        |
        v
+-------------------+     track_history [B, T, 13]
|  Module D:        | ──────────────────────────────────>+-------------------+
|  Tracker Manager  |                                    |  Module B:        |
|  (Birth/Death/    |                                    |  TemporalMamba    |
|   Active/Coasted) |                                    |  (SSM backbone)   |
+-------------------+                                    +-------------------+
        |                                                   |           |
        |  observations                      Q_pos/Q_siz/Q_ori    embedding
        |  z_pos, z_siz, z_ori               R_pos/R_siz/R_ori    [B, 32]
        v                                                   |           |
+-------------------------------------------------------+   |           |
|  Module A: Decoupled Adaptive KFs                      |<--+           |
|                                                        |               |
|  PositionFilter    [x,y,z,vx,vy,vz,ax,ay] (CA, dim=8) |               |
|  SizeFilter        [l,w,h]                 (I,  dim=3) |               |
|  OrientationFilter [theta,omega]           (CV, dim=2) |               |
|                                                        |               |
|  * All batched via torch.bmm — NO for-loops            |               |
|  * PSD Safety Lock: Softplus + 1e-5 on Cholesky diag   |               |
+-------------------------------------------------------+               |
        |                                                                |
        | predicted state + covariance P                                 |
        v                                                                v
+------------------------------------------------------------------------+
|  Module C: Uncertainty-Aware Association                               |
|                                                                        |
|  cost = (1-w_s) * (1 - Ro_GDIoU)                                      |
|       + w_s     * (1 - cosine_sim(emb_trk, emb_det))                  |
|       + w_u     * [trace(P_pos[:3,:3]) + trace(P_ori)]                 |
+------------------------------------------------------------------------+
        |
        v
  Hungarian / Greedy matching
```

**Core design principles:**
1. **Zero Transformer/GNN** — strictly O(1) per-tracklet inference via Mamba SSM
2. **Dimensional isolation** — three independent KFs prevent cross-dimension error propagation
3. **Physics-infused adaptivity** — Mamba predicts per-step Q/R noise matrices through Cholesky decomposition with guaranteed PSD

---

## Quick Start

### 1. Environment Setup

```bash
# Create conda environment
conda create -n mamba-dekf python=3.10 -y
conda activate mamba-dekf

# Install PyTorch (adjust CUDA version to match your GPU)
# For CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# For CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

# Install mamba-ssm (requires torch >= 2.0 + CUDA)
pip install mamba-ssm>=1.2.0

# Install remaining dependencies
pip install -r requirements.txt
```

**If you cannot install `mamba-ssm`** (e.g., no GPU, macOS, Windows):
The code automatically falls back to a GRU backbone — no code changes needed.

### 2. Verify Installation

```bash
python -c "
from kalmanfilter.mamba_adaptive_kf import MambaDecoupledEKF, TemporalMamba
import torch
B, T = 4, 10
pipe = MambaDecoupledEKF(batch_size=B, d_model=64)
pipe.kf.init_states(
    torch.randn(B,8,1), torch.eye(8).unsqueeze(0).repeat(B,1,1),
    torch.randn(B,3,1), torch.eye(3).unsqueeze(0).repeat(B,1,1),
    torch.randn(B,2,1), torch.eye(2).unsqueeze(0).repeat(B,1,1),
)
mout, *_ = pipe.predict_with_mamba(torch.randn(B,T,13), delta_t=0.1)
print('Q_pos PSD:', (torch.linalg.eigvalsh(mout['Q_pos']) > 0).all().item())
print('Embedding shape:', mout['embedding'].shape)
print('All OK!')
"
```

Expected output:
```
Q_pos PSD: True
Embedding shape: torch.Size([4, 32])
All OK!
```

---

## Project Structure

```
MCTrack/
├── kalmanfilter/
│   ├── mamba_adaptive_kf.py    # Module A + B (Decoupled KFs + TemporalMamba + PSD Lock)
│   ├── extend_kalman.py        # Original unified EKFs (legacy, kept for reference)
│   └── base_kalman.py          # Base KF class (legacy)
├── tracker/
│   ├── matching.py             # Module C (uncertainty-aware association)
│   ├── cost_function.py        # Cost functions (Ro_GDIoU + semantic + uncertainty)
│   ├── base_tracker.py         # Module D (tracker lifecycle management)
│   ├── trajectory.py           # Trajectory state management
│   ├── bbox.py                 # Bounding box representation
│   └── frame.py                # Per-frame data container
├── config/                     # YAML configs per dataset
├── data/                       # Dataset root (see Data Preparation below)
├── evaluation/                 # Evaluation scripts (KITTI, nuScenes, Waymo)
├── preprocess/                 # Dataset conversion to BaseVersion format
├── utils/                      # Shared utilities
├── main.py                     # Entry point
├── CLAUDE.md                   # AI development guidelines
├── requirements.txt
└── README.md
```

### Key files for the new architecture

| File | Module | Description |
|------|--------|-------------|
| `kalmanfilter/mamba_adaptive_kf.py` | A + B | `PositionFilter` (CA), `SizeFilter` (Const), `OrientationFilter` (CV), `TemporalMamba`, `CholeskyHead`, `MambaDecoupledEKF` |
| `tracker/cost_function.py` | C | `cal_uncertainty_aware_cost`, `compute_cosine_similarity_matrix`, `compute_uncertainty_penalty` |
| `tracker/matching.py` | C | `match_trajs_and_dets_uncertainty_aware` |

---

## Data Preparation

### BaseVersion Data Generation
First, download the original datasets from [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking.php), [nuScenes](https://www.nuscenes.org/tracking/?externalData=all&mapData=all&modalities=Any), and [Waymo](https://waymo.com/open/download/), as well as their corresponding detection results, and organize them in the following directory structure.

Pre-converted data is available from [Google Drive](https://drive.google.com/drive/folders/15QDnPR9t3FO18fVzCyqUu4h-7COl9Utd?usp=sharing) and [Baidu Cloud](https://pan.baidu.com/s/1Fk6EPeIBxThFjBJuMMKQCw?pwd=6666).

<details>
<summary>Directory structure (click to expand)</summary>

For KITTI:
```
data/
└── kitti/
    ├── datasets/
    |    ├── testing/
    |    |    ├── calib/
    |    |    └── pose/
    |    └── training/
    |         ├── calib/
    |         ├── label_02/
    |         └── pose/
    └── detectors/
         ├── casa/
         └── point_rcnn/
```

For nuScenes:
```
data/
└── nuScenes/
    ├── datasets/
    |    ├── maps/
    |    ├── samples/
    |    ├── sweeps/
    |    ├── v1.0-test/
    |    └── v1.0-trainval/
    └── detectors/
         ├── centerpoint/
         └── largekernel/
```

For Waymo:
```
data/
└── Waymo/
    ├── datasets/
    |    ├── testing/
    |    |    ├── ego_info/
    |    |    └── ts_info/
    |    └── validation/
    |         ├── ego_info/
    |         └── ts_info/
    └── detectors/
         └── ctrl/
              ├── testing/
              └── validation/
```
</details>

Convert to BaseVersion format:
```bash
python preprocess/convert2baseversion.py --dataset kitti    # or nuscenes / waymo
```

---

## Training (Step 5)

> **Status:** Training pipeline is under development. The loss function will combine:
> - **State MSE Loss**: prediction accuracy for position, size, and orientation
> - **Contrastive Loss**: embedding quality for semantic association

```bash
# (Coming soon)
python train.py --dataset nuscenes --config config/nuscenes.yaml
```

---

## Evaluation

### Run tracking + evaluation
```bash
# Single process
python main.py --dataset kitti -e -p 1

# Multi-process (faster)
python main.py --dataset kitti -e -p 8

# nuScenes
python main.py --dataset nuscenes -e -p 8

# Waymo (set WOD path in evaluation/static_evaluation/waymo/eval.py first)
python main.py --dataset waymo -e -p 8
```

Results are saved in the `results/` folder. Modify evaluation parameters in `config/<dataset>.yaml`.

### Submit to benchmarks

<details>
<summary>KITTI submission</summary>

Set `SPLIT: test` and `DETECTOR: virconv` in `config/kitti.yaml`, run tracking, then zip the output `.txt` files and submit to [KITTI tracking challenge](https://www.cvlibs.net/datasets/kitti/user_submit.php).
</details>

<details>
<summary>nuScenes submission</summary>

Set `SPLIT: test` and `DETECTOR: largekernel` in `config/nuscenes.yaml`, run tracking, zip the `result.json` and submit to [nuScenes tracking challenge](https://eval.ai/web/challenges/challenge-page/476/overview).
</details>

<details>
<summary>Waymo submission</summary>

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
Submit to [Waymo tracking challenge](https://waymo.com/open/challenges/2020/3d-tracking/).
</details>

### Motion metric evaluation (nuScenes only)

```bash
python preprocess/motion_dataset/convert_nuscenes_result_to_pkl.py
python evaluation/eval_motion.py
```

---

## Results (MCTrack Baseline)

### [KITTI](https://www.cvlibs.net/datasets/kitti/eval_tracking_detail.php?result=236cb88ca975231d7a3ed33556025e177d0eab20)

| Method | Detector | Set | HOTA | MOTA | IDSW |
|--------|----------|-----|------|------|------|
| MCTrack | VirConv | test | 81.07 | 89.81 | 46 |
| MCTrack | VirConv | train | 82.65 | 85.19 | 22 |

### [nuScenes](https://www.nuscenes.org/tracking?externalData=all&mapData=all&modalities=Any)

| Method | Detector | Set | AMOTA | MOTA | IDS |
|--------|----------|-----|-------|------|-----|
| MCTrack | LargeKernel3D | test | 0.763 | 0.634 | 242 |
| MCTrack | CenterPoint | val | 0.740 | 0.640 | 275 |

### [Waymo](https://waymo.com/open/challenges/tracking-3d/results/90b4c398-afcf/1725037468534000/)

| Method | Detector | Set | MOTA/L1 | MOTP/L1 | MOTA/L2 | MOTP/L2 |
|--------|----------|-----|---------|---------|---------|---------|
| MCTrack | CTRL | test | 0.7504 | 0.2276 | 0.7344 | 0.2278 |
| MCTrack | CTRL | val | 0.7384 | 0.2288 | 0.7155 | 0.2293 |

---

## Acknowledgement

- Detection: [CTRL](https://github.com/tusen-ai/SST), [VirConv](https://github.com/hailanyi/VirConv), [CenterPoint](https://github.com/tianweiy/CenterPoint)
- Tracking: [PC3T](https://github.com/hailanyi/3D-Multi-Object-Tracker), [Poly-MOT](https://github.com/lixiaoyu2000/Poly-MOT), [ImmortalTracker](https://github.com/esdolo/ImmortalTracker)
- State Space Models: [Mamba](https://github.com/state-spaces/mamba)

## Citation
```bibtex
@article{wang2024mctrack,
  title={MCTrack: A Unified 3D Multi-Object Tracking Framework for Autonomous Driving},
  author={Wang, Xiyang and Qi, Shouzheng and Zhao, Jieyou and Zhou, Hangning and Zhang, Siyu and Wang, Guoan and Tu, Kai and Guo, Songlin and Zhao, Jianbo and Li, Jian and others},
  journal={arXiv preprint arXiv:2409.16149},
  year={2024}
}
```
