# Mamba-Decoupled-EKF — Core Innovations

## Architecture: Three Independent Batched Kalman Filters

```
PositionFilter:  CV model,  state [x,y,z,vx,vy,vz] (6D),  obs [x,y,z,vx,vy] (5D)
SizeFilter:      Const model, state [l,w,h]           (3D),  obs [l,w,h]     (3D)
OrientationFilter: CV model, state [θ,ω]              (2D),  obs [θ]         (1D)
```

- All filters: batched `torch.bmm` — **no for-loops over tracklets**
- Position OBS_DIM=5: velocity is observed directly (not inferred from position lag)
- Size F=I, H=I (rigid-body prior — physical objects don't change size)
- Orientation uses `wrap_to_pi` on innovation AND post-update θ

## Module B: TemporalMamba predicts adaptive Q/R only for Position

- History `[B,T,12]` → Mamba SSM → `h_last [B,d_model]`
  - Features: `[Δx,Δy,z, vx,vy,0, l,w,h, θ,ω, det_score]` (12D)
  - Δx,Δy are relative to latest frame (avoids gradient saturation)
  - `det_score=0.0` for coasted/fake bboxes → signals low quality to Mamba
- `head_Q_pos`: CholeskyHead(d_model, mat_dim=6, min_diag=0.1) → PSD [B,6,6]
- `head_R_pos`: CholeskyHead(d_model, mat_dim=5, min_diag=0.1) → PSD [B,5,5]
- Size noise: `nn.Embedding(num_classes, 3)` with log-space init from `noise.log`
- Orientation: `head_kappa_ori` (MLP→1); R_ori = 1/κ; Q_ori static `nn.Parameter`
- Two filter modes: `mamba` (Mamba Q/R), `pure_dekf` (baseline Q/R), `fusion` (trace-gated blend)

## PSD Safety Locks

- CholeskyHead: `softplus(L_raw_diag) + min_diag + eps` → eigenvalues ≥ min_diag²
- Size embeddings: `softplus(raw) + min_diag_siz` (0.05 floor, was `1e-4` — collapsed to -∞ NLL)
- kappa: `softplus(head) + min_kappa` (0.1 floor, was `1e-3` — collapsed to 0.001)
- **NEVER use torch.clamp on Cholesky diagonals. Always F.softplus + min_diag.**

## Training: Multi-step KF Rollout

- Mamba runs ONCE → Q/R/κ used for all K rollout steps
- KF init from noisy GT state (simulates tracking uncertainty)
- K-step predict-update with noisy teacher forcing
- NLL loss vs CLEAN GT at each step → gradients flow through x_pred, P_pred, Q, R
- Von Mises NLL for orientation: `i0e` for numerical stability (no ±π tear)
- InfoNCE on Mamba embeddings (instance-level contrastive)

### Confidence Warmup (epochs 0 to WARMUP_UNCERTAINTY_EPOCHS)

- **Detach Q/R/κ** → no gradient through uncertainty heads → Mamba backbone learns features first
- **Orientation uses angle_loss** `(1-cos(Δθ))` instead of Von Mises → bounded, no κ shortcut
- After warmup: release all uncertainty heads, full Von Mises NLL

### κ Overconfidence Penalty

- `kappa_reg = 1e-3 * ReLU(κ - 20.0).mean()` → only fires when κ>20 (R_ori<0.05)
- With min_kappa=0.1, normal κ ∈ [0.1, ~5] → penalty is a no-op

## Training Config Key

| Key | Default | Purpose |
|-----|---------|---------|
| WARMUP_UNCERTAINTY_EPOCHS | 3 | Epochs to detach uncertainty heads |
| MIN_DIAG_SIZ | 0.05 | Floor on size noise diagonal |
| MIN_KAPPA | 0.1 | Floor on Von Mises concentration |
| MIN_DIAG_Q / MIN_DIAG_R | 0.1 | Floor on position Cholesky diagonal |
| ROLLOUT_STEPS | 3 | KF auto-regressive steps per sample |
| PHYSICS_SCALE | 20.0 | Global multiplier on state vs contrastive loss |
| WARMUP_EPOCHS | 3 | LR scheduler warmup (separate from uncertainty warmup) |

## Per-Category Config

- MOTION_MODE (from MCTrack): CV for all nuScenes categories
- BIRTH_SCORE: higher for noisy detectors (bicycle=0.55, motorcycle=0.50, car=0.40)
- Velocity observation noise: 2× position std (CenterPoint velocity is often zero)
- VEL_STD: per-category velocity measurement noise std

## Key Differences from MCTrack

1. **Velocity observation**: MCTrack observed [x,y,vx,vy]; we observe [x,y,z,vx,vy] (5D)
2. **No per-category motion model selection**: single CV for all, but adaptive Q/R from Mamba
3. **Batched KF**: torch.bmm over all tracklets vs MCTrack's per-track numpy KF
4. **Adaptive noise**: Mamba predicts Q/R vs MCTrack's fixed per-category Q/R
5. **SizeFilter is const model** (F=I) vs MCTrack's KF_SIZE with size velocity
6. **Dual CholeskyHead**: predicts full PSD covariance (not just diagonal)
