# Mamba Prior-Conditioned Multi-Head Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current loosely coupled Mamba uncertainty path with a prior-conditioned, 7-class, multi-head, residual-token closure design that preserves the frozen `AMOTA 0.739` baseline and only ships if the new branch exceeds `AMOTA 0.740`.

**Architecture:** Keep the current tracker shell and frozen baseline config intact. Add explicit matched-residual history caching, build adaptive residual-token histories, replace the existing absolute-noise head semantics with per-class `pos / vel / ori / size` ratio heads over `pure_dekf` priors, and align training/inference through the same lifecycle buckets and loss stack. The implementation is staged so the old mainline remains reproducible while the new branch is isolated behind new configs and tests.

**Tech Stack:** Python, PyTorch, YAML, unittest, existing `tracker`, `kalmanfilter`, and `training` modules.

---

## File Structure

- Create: `kalmanfilter/prior_conditioned_heads.py`
  - Own class-specific ratio heads and bounded-ratio utilities for the new branch.
- Create: `tests/test_prior_conditioned_heads.py`
  - Verify per-class head routing and bounded ratio reconstruction.
- Modify: `tracker/trajectory.py`
  - Store matched residual history and lifecycle-aligned residual tokens.
- Modify: `tracker/base_tracker.py`
  - Build residual-token histories, adaptive windows, lifecycle buckets, and pass them into the Mamba branch.
- Modify: `kalmanfilter/mamba_adaptive_kf.py`
  - Replace direct covariance-head semantics with shared trunk + per-class multi-head ratio outputs.
- Modify: `training/losses.py`
  - Introduce wrapped-Gaussian orientation likelihood and ratio anchor / bound helpers.
- Modify: `training/train.py`
  - Align training input construction, lifecycle semantics, and loss stack with inference.
- Modify: `config/train_nuscenes.yaml`
  - Add category window defaults, per-class ratio bounds, and first-pass loss weights.
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`
  - Dedicated inference config for the new branch.
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml`
  - Audit-focused config for numeric validation without touching the frozen baseline.
- Modify: `README.md`
  - Add a short branch-specific section for the new training/eval path after implementation passes tests.
- Create: `tests/test_residual_history_tokens.py`
  - Verify residual history caching and adaptive token construction.
- Create: `tests/test_mamba_multihead_closure_config.py`
  - Verify new config schema, class windows, and lifecycle knobs.
- Modify: `tests/test_noise_audit_infer.py`
  - Verify the new branch still emits audit-compatible summaries.
- Modify: `tests/test_noise_audit_train.py`
  - Verify train-side lifecycle mapping and ratio bookkeeping under the new branch.

## Task 1: Add matched residual history storage to trajectories

**Files:**
- Modify: `tracker/trajectory.py`
- Test: `tests/test_residual_history_tokens.py`

- [ ] **Step 1: Write the failing trajectory residual-history tests**

```python
import unittest

from tracker.trajectory import Trajectory


class ResidualHistoryTest(unittest.TestCase):
    def test_record_matched_residual_appends_real_residual_entry(self):
        traj = Trajectory(track_id=7, init_bbox=_make_bbox(), cfg=_make_cfg())
        traj.record_matched_residual(
            pos_residual=[0.2, -0.1, 0.0, 0.3, -0.2],
            siz_residual=[0.1, 0.0, -0.1],
            ori_residual=0.05,
            det_score=0.61,
            timestamp=1000,
        )
        self.assertEqual(len(traj.residual_history), 1)
        self.assertTrue(traj.residual_history[-1]["is_matched"])
        self.assertAlmostEqual(traj.residual_history[-1]["ori_residual"], 0.05)

    def test_record_coast_residual_appends_masked_placeholder(self):
        traj = Trajectory(track_id=8, init_bbox=_make_bbox(), cfg=_make_cfg())
        traj.record_coast_residual(timestamp=2000)
        self.assertEqual(len(traj.residual_history), 1)
        self.assertFalse(traj.residual_history[-1]["is_matched"])
        self.assertEqual(traj.residual_history[-1]["det_score"], 0.0)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_residual_history_tokens.ResidualHistoryTest -v`

Expected: missing `residual_history` / `record_matched_residual` failures.

- [ ] **Step 3: Add residual-history storage and lifecycle helpers to `Trajectory`**

```python
# tracker/trajectory.py
class Trajectory:
    def __init__(...):
        ...
        self.residual_history = []

    def record_matched_residual(
        self,
        *,
        pos_residual,
        siz_residual,
        ori_residual,
        det_score,
        timestamp,
    ):
        self.residual_history.append(
            {
                "is_matched": True,
                "pos_residual": list(pos_residual),
                "siz_residual": list(siz_residual),
                "ori_residual": float(ori_residual),
                "det_score": float(det_score),
                "timestamp": int(timestamp),
            }
        )

    def record_coast_residual(self, *, timestamp):
        self.residual_history.append(
            {
                "is_matched": False,
                "pos_residual": [0.0] * 5,
                "siz_residual": [0.0] * 3,
                "ori_residual": 0.0,
                "det_score": 0.0,
                "timestamp": int(timestamp),
            }
        )
```

- [ ] **Step 4: Wire residual recording into matched and coast updates**

```python
# tracker/trajectory.py inside matched update path
self.record_matched_residual(
    pos_residual=matched_residual["pos"],
    siz_residual=matched_residual["siz"],
    ori_residual=matched_residual["ori"],
    det_score=bbox.det_score,
    timestamp=bbox.timestamp,
)

# tracker/trajectory.py inside coast / fake-bbox path
self.record_coast_residual(timestamp=fake_bbox.timestamp)
```

- [ ] **Step 5: Run the trajectory residual-history tests**

Run: `python -m unittest tests.test_residual_history_tokens.ResidualHistoryTest -v`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add tracker/trajectory.py tests/test_residual_history_tokens.py
git commit -m "feat: add residual history storage to trajectories"
```

## Task 2: Build adaptive residual-token histories in the tracker

**Files:**
- Modify: `tracker/base_tracker.py`
- Modify: `tests/test_residual_history_tokens.py`
- Modify: `tests/test_noise_audit_infer.py`

- [ ] **Step 1: Write failing tracker token-construction tests**

```python
def test_extract_residual_token_history_uses_recent_residuals_and_masks(self):
    tracker = _build_tracker_stub(history_len=8)
    traj = _make_traj_with_residual_history(
        class_id=2,
        matched_flags=[True, True, False, True],
    )
    tokens, valid_mask, match_mask = tracker._extract_residual_token_history([traj])
    self.assertEqual(tokens.shape, (1, 8, tracker.mamba_input_dim))
    self.assertEqual(valid_mask.sum().item(), 4)
    self.assertEqual(match_mask.sum().item(), 3)

def test_effective_history_is_shortened_for_unmatched_bicycle(self):
    tracker = _build_tracker_stub(history_len=8)
    traj = _make_traj_with_residual_history(class_id=2, unmatch_length=2, n_steps=8)
    effective = tracker._resolve_effective_history_len(traj, valid_history_len=8)
    self.assertLessEqual(effective, 4)
```

- [ ] **Step 2: Run the tracker token tests to verify they fail**

Run: `python -m unittest tests.test_residual_history_tokens -v`

Expected: missing residual-token extraction methods.

- [ ] **Step 3: Add category window lookup and adaptive shrink helpers**

```python
# tracker/base_tracker.py
def _class_window_cfg(self, class_name: str) -> dict:
    return (self.runtime_window_cfg.get(class_name) or self.default_window_cfg).copy()

def _resolve_effective_history_len(self, traj, valid_history_len: int) -> int:
    class_name = traj.bboxes[-1].category
    cfg = self._class_window_cfg(class_name)
    length = min(valid_history_len, int(cfg["MAX_HISTORY_LEN"]))
    if getattr(traj, "unmatch_length", 0) > 0:
        length = max(int(cfg["MIN_HISTORY_LEN"]), length - 2)
    if getattr(traj, "unmatch_length", 0) > 1:
        length = min(length, 3)
    return max(1, length)
```

- [ ] **Step 4: Add `_extract_residual_token_history()` and call it from `predict_before_associate()`**

```python
# tracker/base_tracker.py
def _extract_residual_token_history(self, trajs):
    B, T = len(trajs), self.history_len
    tokens = torch.zeros(B, T, self.mamba_input_dim, device=self.device)
    valid_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
    match_mask = torch.zeros(B, T, dtype=torch.bool, device=self.device)
    for i, traj in enumerate(trajs):
        residuals = traj.residual_history
        effective = self._resolve_effective_history_len(traj, len(residuals))
        recent = residuals[-effective:]
        for j, item in enumerate(recent, start=T - len(recent)):
            tokens[i, j, :] = self._encode_residual_token(traj, item, age_index=j)
            valid_mask[i, j] = True
            match_mask[i, j] = bool(item["is_matched"])
    return tokens, valid_mask, match_mask

# use these tensors in place of the legacy state-history path for the new branch
```

- [ ] **Step 5: Keep the legacy path intact behind a branch check**

```python
if self.filter_mode == "mamba_multihead_closure":
    history, history_mask, history_match_mask = self._extract_residual_token_history(trajs)
else:
    history, history_mask, history_match_mask = self._extract_track_history(trajs)
```

- [ ] **Step 6: Run tracker / audit inference tests**

Run:

```bash
python -m unittest \
  tests.test_residual_history_tokens \
  tests.test_noise_audit_infer \
  -v
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add tracker/base_tracker.py tests/test_residual_history_tokens.py tests/test_noise_audit_infer.py
git commit -m "feat: add adaptive residual token histories for mamba closure"
```

## Task 3: Replace absolute covariance heads with per-class multi-head ratio heads

**Files:**
- Create: `kalmanfilter/prior_conditioned_heads.py`
- Modify: `kalmanfilter/mamba_adaptive_kf.py`
- Test: `tests/test_prior_conditioned_heads.py`

- [ ] **Step 1: Write failing per-class head tests**

```python
import unittest
import torch

from kalmanfilter.prior_conditioned_heads import PriorConditionedHeadBank


class PriorConditionedHeadBankTest(unittest.TestCase):
    def test_selects_independent_heads_per_class(self):
        bank = PriorConditionedHeadBank(d_model=32, num_classes=7)
        h = torch.randn(2, 32)
        class_ids = torch.tensor([4, 6], dtype=torch.long)
        out = bank(h, class_ids)
        self.assertIn("r_pos_xyz", out)
        self.assertEqual(out["r_pos_xyz"].shape, (2, 1))

    def test_bounded_ratio_stays_within_limits(self):
        bank = PriorConditionedHeadBank(d_model=32, num_classes=7)
        h = torch.randn(1, 32) * 20.0
        class_ids = torch.tensor([2], dtype=torch.long)
        out = bank(h, class_ids)
        ratio = out["r_ori"]
        self.assertGreaterEqual(float(ratio.min()), 0.0)
        self.assertLessEqual(float(ratio.max()), 4.0)
```

- [ ] **Step 2: Run the head tests to verify they fail**

Run: `python -m unittest tests.test_prior_conditioned_heads -v`

Expected: missing module / symbol failures.

- [ ] **Step 3: Implement per-class head bank**

```python
# kalmanfilter/prior_conditioned_heads.py
class BoundedRatioHead(nn.Module):
    def __init__(self, d_model, out_dim, alpha_init):
        super().__init__()
        self.proj = nn.Linear(d_model, out_dim)
        self.alpha = nn.Parameter(torch.full((out_dim,), float(alpha_init)))

    def forward(self, h):
        raw = self.proj(h)
        return torch.exp(self.alpha * torch.tanh(raw))


class PriorConditionedHeadBank(nn.Module):
    def __init__(self, d_model, num_classes):
        super().__init__()
        self.pos_heads = nn.ModuleList([BoundedRatioHead(d_model, 2, 0.5) for _ in range(num_classes)])
        self.vel_heads = nn.ModuleList([BoundedRatioHead(d_model, 2, 0.5) for _ in range(num_classes)])
        self.ori_heads = nn.ModuleList([BoundedRatioHead(d_model, 1, 0.4) for _ in range(num_classes)])
        self.size_heads = nn.ModuleList([BoundedRatioHead(d_model, 2, 0.2) for _ in range(num_classes)])

    def forward(self, h, class_ids):
        ...
```

- [ ] **Step 4: Integrate the head bank into `TemporalMamba` and reconstruct final `Q/R` from priors**

```python
# kalmanfilter/mamba_adaptive_kf.py
self.head_bank = PriorConditionedHeadBank(d_model=d_model, num_classes=num_classes)

head_out = self.head_bank(h_last, class_ids)
base_cov = build_base_covariances(...)
Q_pos = apply_factorized_ratio_to_q_pos(base_cov["Q_pos_base"], head_out)
R_pos = apply_factorized_ratio_to_r_pos(base_cov["R_pos_base"], head_out)
R_siz = apply_factorized_ratio_to_r_siz(base_cov["R_siz_base"], head_out)
R_ori = apply_factorized_ratio_to_r_ori(base_cov["R_ori_base"], head_out)
```

- [ ] **Step 5: Disable the legacy direct covariance heads only for the new branch**

```python
if self.mode_name == "mamba_multihead_closure":
    # use head_bank + prior reconstruction
else:
    # keep existing Q_pos/R_pos/kappa path intact
```

- [ ] **Step 6: Run the head-bank and branch regression tests**

Run:

```bash
python -m unittest \
  tests.test_prior_conditioned_heads \
  tests.test_bounded_residual \
  -v
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add kalmanfilter/prior_conditioned_heads.py kalmanfilter/mamba_adaptive_kf.py tests/test_prior_conditioned_heads.py
git commit -m "feat: add prior-conditioned multi-head ratio outputs"
```

## Task 4: Align the loss stack with prior-ratio runtime semantics

**Files:**
- Modify: `training/losses.py`
- Modify: `training/train.py`
- Modify: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Write failing wrapped-orientation and ratio-loss tests**

```python
def test_wrapped_gaussian_orientation_loss_handles_pi_boundary(self):
    pred = torch.tensor([[3.13]])
    gt = torch.tensor([[-3.13]])
    var = torch.tensor([[[0.1]]])
    loss = wrapped_orientation_nll(pred, gt, var)
    self.assertTrue(torch.isfinite(loss))

def test_ratio_anchor_is_zero_when_gamma_is_one(self):
    gamma = torch.ones(4, 1)
    loss = log_ratio_anchor_loss(gamma)
    self.assertAlmostEqual(float(loss), 0.0, places=6)
```

- [ ] **Step 2: Run the training-loss tests to verify they fail**

Run: `python -m unittest tests.test_noise_audit_train -v`

Expected: missing helper failures.

- [ ] **Step 3: Add new loss helpers in `training/losses.py`**

```python
def wrapped_orientation_nll(pred_yaw, gt_yaw, r_ori, sample_weights=None):
    diff = wrap_to_pi_torch(pred_yaw - gt_yaw)
    var = torch.clamp(r_ori.squeeze(-1).squeeze(-1), min=1e-5)
    per = 0.5 * (torch.log(var) + diff.pow(2) / var)
    if sample_weights is None:
        return per.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per * w).sum()


def log_ratio_anchor_loss(gamma, sample_weights=None):
    per = torch.abs(torch.log(torch.clamp(gamma, min=1e-8)))
    if sample_weights is None:
        return per.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per.squeeze(-1) * w).sum()
```

- [ ] **Step 4: Replace the legacy `kappa`-centric orientation main loss in `training_step()`**

```python
# training/train.py
loss_ori = wrapped_orientation_nll(
    pred_yaw=ori_x_pred[:, 0:1, 0],
    gt_yaw=gt_future_ori[:, k, :],
    r_ori=R_ori[active_k],
    sample_weights=sample_weights,
)

# remove kappa_reg from the new branch path
if branch_name == "mamba_multihead_closure":
    kappa_reg = torch.zeros((), device=device)
```

- [ ] **Step 5: Add explicit ratio anchor + bound losses using the new head outputs**

```python
ratio_anchor = (
    log_ratio_anchor_loss(mamba_out["ratios"]["q_pos_xyz"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["q_pos_vxyz"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["r_pos_xyz"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["r_pos_vxy"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["r_siz_lw"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["r_siz_h"]) +
    log_ratio_anchor_loss(mamba_out["ratios"]["r_ori"])
)
real_loss = total_loss_tensor / norm + lambda_anchor * ratio_anchor + lambda_bound * ratio_bound + aux_loss
```

- [ ] **Step 6: Keep robust auxiliary terms small and branch-local**

```python
if branch_name == "mamba_multihead_closure":
    aux_pos = F.huber_loss(pos_x_pred[:, 0:3, 0], gt_future_pos[:, k, :], reduction="mean")
    aux_siz = F.huber_loss(siz_x_pred[:, :, 0], gt_future_siz[:, k, :], reduction="mean")
    aux_ori = (1.0 - torch.cos(wrap_to_pi_torch(ori_x_pred[:, 0, 0] - gt_future_ori[:, k, 0]))).mean()
    aux_loss = 0.05 * aux_pos + 0.02 * aux_siz + 0.02 * aux_ori
```

- [ ] **Step 7: Run the training-focused tests**

Run:

```bash
python -m unittest \
  tests.test_noise_audit_train \
  tests.test_prior_conditioned_heads \
  -v
```

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add training/losses.py training/train.py tests/test_noise_audit_train.py
git commit -m "feat: align mamba closure losses with prior-ratio runtime"
```

## Task 5: Update training and inference configs for the isolated branch

**Files:**
- Modify: `config/train_nuscenes.yaml`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml`
- Test: `tests/test_mamba_multihead_closure_config.py`

- [ ] **Step 1: Write failing config-schema tests**

```python
import unittest
import yaml


class MambaMultiheadClosureConfigTest(unittest.TestCase):
    def test_train_config_contains_class_windows_for_all_7_classes(self):
        cfg = yaml.safe_load(open("config/train_nuscenes.yaml"))
        class_window = cfg["DATA"]["CLASS_WINDOW"]
        for key in ["car", "pedestrian", "bicycle", "motorcycle", "bus", "trailer", "truck"]:
            self.assertIn(key, class_window)

    def test_branch_config_keeps_frozen_baseline_path_separate(self):
        cfg = yaml.safe_load(open("config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml"))
        self.assertEqual(cfg["FILTER_MODE"], "mamba_multihead_closure")
```

- [ ] **Step 2: Run config tests to verify they fail**

Run: `python -m unittest tests.test_mamba_multihead_closure_config -v`

Expected: missing file or missing branch-key failures.

- [ ] **Step 3: Update `config/train_nuscenes.yaml` with explicit 7-class windows and branch knobs**

```yaml
DATA:
  CLASS_WINDOW:
    car:        {MIN_HISTORY_LEN: 4, MAX_HISTORY_LEN: 8, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 3}
    pedestrian: {MIN_HISTORY_LEN: 4, MAX_HISTORY_LEN: 7, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 3}
    bicycle:    {MIN_HISTORY_LEN: 3, MAX_HISTORY_LEN: 6, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 2}
    motorcycle: {MIN_HISTORY_LEN: 3, MAX_HISTORY_LEN: 6, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 2}
    bus:        {MIN_HISTORY_LEN: 4, MAX_HISTORY_LEN: 8, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 3}
    trailer:    {MIN_HISTORY_LEN: 5, MAX_HISTORY_LEN: 8, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 3}
    truck:      {MIN_HISTORY_LEN: 4, MAX_HISTORY_LEN: 8, MIN_ROLLOUT_STEPS: 1, MAX_ROLLOUT_STEPS: 3}
MODEL:
  BRANCH_NAME: mamba_multihead_closure
LOSS:
  RATIO_ANCHOR_WEIGHT: 0.05
  RATIO_BOUND_WEIGHT: 0.10
  AUX_STATE_WEIGHT: 0.05
```

- [ ] **Step 4: Create isolated inference and audit configs**

```yaml
# config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml
FILTER_MODE: "mamba_multihead_closure"
TRACKER_COMPAT_MODE: "mctrack"
MAMBA_CLOSURE:
  ENABLED: true
  MODE: "prior_conditioned_multihead"
  FORCE_COAST_PRIOR_ONLY: true
```

```yaml
# config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml
FILTER_MODE: "mamba_multihead_closure"
AUDIT:
  NOISE_AUDIT:
    ENABLED: true
```

- [ ] **Step 5: Run config tests**

Run: `python -m unittest tests.test_mamba_multihead_closure_config -v`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add config/train_nuscenes.yaml config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml tests/test_mamba_multihead_closure_config.py
git commit -m "feat: add isolated configs for mamba multihead closure"
```

## Task 6: End-to-end verification, docs, and branch usage instructions

**Files:**
- Modify: `README.md`
- Modify: `tests/test_noise_audit_infer.py`
- Modify: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Add failing regression tests for branch-mode selection**

```python
def test_predict_with_mamba_uses_multihead_branch_when_configured(self):
    model = _build_mamba_stub(branch_name="mamba_multihead_closure")
    out = model.predict_with_mamba(...)
    self.assertIn("ratios", out[0])
```

- [ ] **Step 2: Run full focused test suite and verify failures are only from missing branch wiring**

Run:

```bash
python -m unittest \
  tests.test_residual_history_tokens \
  tests.test_prior_conditioned_heads \
  tests.test_mamba_multihead_closure_config \
  tests.test_noise_audit_infer \
  tests.test_noise_audit_train \
  -v
```

Expected: failures until branch integration is complete.

- [ ] **Step 3: Update README with the new branch workflow**

````markdown
## Experimental Mamba Multi-Head Closure Branch

Train:

```bash
python training/train.py --config config/train_nuscenes.yaml
```

Audit:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_audit.yaml -p 12
```

Full eval:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml -p 12
```
````

- [ ] **Step 4: Run the focused suite again and ensure it passes**

Run:

```bash
python -m unittest \
  tests.test_residual_history_tokens \
  tests.test_prior_conditioned_heads \
  tests.test_mamba_multihead_closure_config \
  tests.test_noise_audit_infer \
  tests.test_noise_audit_train \
  -v
```

Expected: `OK`

- [ ] **Step 5: Run syntax verification on touched modules**

Run:

```bash
python -m py_compile \
  tracker/trajectory.py \
  tracker/base_tracker.py \
  kalmanfilter/prior_conditioned_heads.py \
  kalmanfilter/mamba_adaptive_kf.py \
  training/losses.py \
  training/train.py
```

Expected: no output, exit code `0`

- [ ] **Step 6: Commit**

```bash
git add README.md tests/test_noise_audit_infer.py tests/test_noise_audit_train.py
git commit -m "docs: add mamba multihead closure branch workflow"
```

- [ ] **Step 7: Prepare experiment commands for remote evaluation**

Run:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml -p 12
```

Expected: reproduce the frozen `AMOTA 0.739` baseline before new-branch comparison.

Run:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml -p 12
```

Expected: compare against the frozen baseline and only proceed if `AMOTA > 0.740`.
