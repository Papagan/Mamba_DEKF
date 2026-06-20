# Bounded Residual Mamba Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace free-form `mamba` noise inference with a prior-anchored, class-profile + matched/unmatched bounded residual closure path, and only keep the branch if it reaches aggregated `AMOTA >= 0.740` without modifying the frozen `0.737` baseline.

**Architecture:** Introduce a focused bounded-residual helper that maps `(class profile, tracker state, noise family)` to allowed ratio envelopes, consume it in `predict_with_mamba()` so `mamba` produces bounded residual corrections around `pure_dekf` priors, and add aligned training-side ratio regularization so inference/runtime semantics match the trained objective. Keep all work behind new config blocks and dedicated experiment configs so the frozen baseline remains untouched.

**Tech Stack:** Python, PyTorch, YAML configs, unittest, existing `mamba_adaptive_kf` / tracker pipeline, existing noise audit infrastructure.

---

## File Structure

- Create: `kalmanfilter/bounded_residual.py`
  - Own the bounded-ratio config schema, class-profile mapping, matched/unmatched state mapping helpers, and ratio application utilities.
- Modify: `kalmanfilter/mamba_adaptive_kf.py`
  - Integrate bounded residual closure into `predict_with_mamba()`.
- Modify: `tracker/base_tracker.py`
  - Pass per-track matched/unmatched state buckets into `predict_with_mamba()`.
- Modify: `training/train.py`
  - Add ratio-anchor loss and state-aware/class-profile-aware regularization using the same helper semantics as inference.
- Modify: `config/train_nuscenes.yaml`
  - Add training-side bounded residual settings.
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml`
  - Dedicated branch config for full evaluation.
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba_closure.yaml`
  - Dedicated audit config for numeric closure comparison.
- Test: `tests/test_bounded_residual.py`
  - Unit-test the helper module.
- Modify: `tests/test_noise_audit_infer.py`
  - Verify state-bucket plumbing and audit output remains valid.
- Modify: `tests/test_noise_audit_train.py`
  - Verify training-side ratio bookkeeping and state projection.
- Create: `tests/test_mamba_closure_config.py`
  - Validate config schema and key branch knobs.
- Modify: `README.md`
  - Document branch configs, success gate, and execution commands.

## Task 1: Build the bounded residual helper module

**Files:**
- Create: `kalmanfilter/bounded_residual.py`
- Test: `tests/test_bounded_residual.py`

- [ ] **Step 1: Write the failing helper tests**

```python
import unittest

from kalmanfilter.bounded_residual import (
    STATE_MATCHED,
    STATE_UNMATCHED,
    PROFILE_STABLE_LARGE,
    infer_state_bucket,
    map_class_to_profile,
    clamp_ratio_value,
)


class BoundedResidualHelperTest(unittest.TestCase):
    def test_map_class_to_profile(self):
        self.assertEqual(map_class_to_profile(0), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(4), PROFILE_STABLE_LARGE)
        self.assertEqual(map_class_to_profile(2), "agile_weak")
        self.assertEqual(map_class_to_profile(3), "agile_weak")
        self.assertEqual(map_class_to_profile(5), "heavy_long")
        self.assertEqual(map_class_to_profile(6), "heavy_long")
        self.assertEqual(map_class_to_profile(1), "human")

    def test_infer_state_bucket(self):
        self.assertEqual(infer_state_bucket(0), STATE_MATCHED)
        self.assertEqual(infer_state_bucket(2), STATE_UNMATCHED)

    def test_clamp_ratio_value(self):
        self.assertEqual(clamp_ratio_value(0.5, min_ratio=0.8, max_ratio=1.6), 0.8)
        self.assertEqual(clamp_ratio_value(1.2, min_ratio=0.8, max_ratio=1.6), 1.2)
        self.assertEqual(clamp_ratio_value(2.4, min_ratio=0.8, max_ratio=1.6), 1.6)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_bounded_residual -v`

Expected: `ModuleNotFoundError` or missing symbol failures for `kalmanfilter.bounded_residual`.

- [ ] **Step 3: Write the minimal helper implementation**

```python
# kalmanfilter/bounded_residual.py
STATE_MATCHED = "matched"
STATE_UNMATCHED = "unmatched"

PROFILE_STABLE_LARGE = "stable_large"
PROFILE_AGILE_WEAK = "agile_weak"
PROFILE_HEAVY_LONG = "heavy_long"
PROFILE_HUMAN = "human"

_CLASS_TO_PROFILE = {
    0: PROFILE_STABLE_LARGE,  # car
    4: PROFILE_STABLE_LARGE,  # bus
    2: PROFILE_AGILE_WEAK,    # bicycle
    3: PROFILE_AGILE_WEAK,    # motorcycle
    5: PROFILE_HEAVY_LONG,    # trailer
    6: PROFILE_HEAVY_LONG,    # truck
    1: PROFILE_HUMAN,         # pedestrian
}


def map_class_to_profile(class_id: int) -> str:
    return _CLASS_TO_PROFILE[int(class_id)]


def infer_state_bucket(unmatch_length: int) -> str:
    return STATE_MATCHED if int(unmatch_length) == 0 else STATE_UNMATCHED


def clamp_ratio_value(value: float, *, min_ratio: float, max_ratio: float) -> float:
    return max(min_ratio, min(max_ratio, float(value)))
```

- [ ] **Step 4: Run the helper test to verify it passes**

Run: `python -m unittest tests.test_bounded_residual -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add kalmanfilter/bounded_residual.py tests/test_bounded_residual.py
git commit -m "test: add bounded residual helper scaffolding"
```

## Task 2: Wire matched/unmatched state into inference and bound `mamba` noise ratios

**Files:**
- Modify: `tracker/base_tracker.py:928-993`
- Modify: `kalmanfilter/mamba_adaptive_kf.py:1298-1419`
- Modify: `tests/test_noise_audit_infer.py`
- Test: `tests/test_bounded_residual.py`

- [ ] **Step 1: Write a failing inference-side state plumbing test**

```python
def test_predict_before_associate_passes_state_buckets_into_predict_with_mamba(self):
    tracker = _build_tracker_stub()
    tracker.all_trajs = {
        1: types.SimpleNamespace(unmatch_length=0, category_num=0),
        2: types.SimpleNamespace(unmatch_length=2, category_num=5),
    }
    captured = {}

    def _fake_predict_with_mamba(*args, **kwargs):
        captured["state_buckets"] = kwargs["state_buckets"]
        raise RuntimeError("stop after capture")

    tracker.mamba_ekf.predict_with_mamba = _fake_predict_with_mamba

    with self.assertRaisesRegex(RuntimeError, "stop after capture"):
        tracker.predict_before_associate([tracker.all_trajs[1], tracker.all_trajs[2]], delta_t=0.5)

    self.assertEqual(captured["state_buckets"], ["matched", "unmatched"])
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_noise_audit_infer.NoiseAuditInferTest.test_predict_before_associate_passes_state_buckets_into_predict_with_mamba -v`

Expected: failure because `state_buckets` is not passed.

- [ ] **Step 3: Add state bucket plumbing in the tracker**

```python
# tracker/base_tracker.py inside predict_before_associate(...)
state_buckets = [
    infer_state_bucket(getattr(traj, "unmatch_length", 0))
    for traj in trajs
]

mamba_out, px, pP, sx, sP, ox, oP = self.mamba_ekf.predict_with_mamba(
    track_history,
    delta_t=delta_t,
    class_ids=class_ids,
    mode=self.filter_mode,
    current_range=current_range,
    detection_driven_mask=detection_driven_mask,
    history_mask=history_mask,
    history_match_mask=history_match_mask,
    state_buckets=state_buckets,
)
```

- [ ] **Step 4: Add bounded-ratio application in `predict_with_mamba()`**

```python
# kalmanfilter/mamba_adaptive_kf.py inside predict_with_mamba(...)
raw_values = {
    "q_pos": _covariance_trace_batch(mamba_out["Q_pos"]),
    "r_pos": _covariance_trace_batch(mamba_out["R_pos"]),
    "r_siz": _covariance_trace_batch(mamba_out["R_siz"]),
    "r_ori": _covariance_trace_batch(mamba_out["R_ori"]),
}
prior_values = {
    "q_pos": _covariance_trace_batch(Q_p),
    "r_pos": _covariance_trace_batch(R_p),
    "r_siz": _covariance_trace_batch(R_s),
    "r_ori": _covariance_trace_batch(R_o),
}

bounded = apply_bounded_residuals(
    raw_tensors=mamba_out,
    prior_tensors={
        "Q_pos": Q_p,
        "R_pos": R_p,
        "R_siz": R_s,
        "R_ori": R_o,
    },
    class_ids=class_ids,
    state_buckets=state_buckets,
    closure_cfg=self.base_noise_cfg.get("MAMBA_CLOSURE", {}),
)

mamba_out["Q_pos"] = bounded["Q_pos"]
mamba_out["R_pos"] = bounded["R_pos"]
mamba_out["R_siz"] = bounded["R_siz"]
mamba_out["R_ori"] = bounded["R_ori"]
```

- [ ] **Step 5: Extend audit output so raw vs bounded ratios are visible**

```python
mamba_out["noise_audit_values"] = {
    "q_pos": _covariance_trace_batch(mamba_out["Q_pos"]),
    "r_pos": _covariance_trace_batch(mamba_out["R_pos"]),
    "r_siz": _covariance_trace_batch(mamba_out["R_siz"]),
    "r_ori": _covariance_trace_batch(mamba_out["R_ori"]),
}
mamba_out["noise_audit_priors"] = {
    "q_pos": _covariance_trace_batch(Q_p),
    "r_pos": _covariance_trace_batch(R_p),
    "r_siz": _covariance_trace_batch(R_s),
    "r_ori": _covariance_trace_batch(R_o),
}
```

- [ ] **Step 6: Run the inference-focused tests**

Run:

```bash
python -m unittest \
  tests.test_bounded_residual \
  tests.test_noise_audit_infer \
  -v
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add tracker/base_tracker.py kalmanfilter/mamba_adaptive_kf.py tests/test_noise_audit_infer.py tests/test_bounded_residual.py
git commit -m "feat: add bounded residual mamba inference closure"
```

## Task 3: Add training-side ratio-anchor regularization using the same closure semantics

**Files:**
- Modify: `training/train.py`
- Modify: `config/train_nuscenes.yaml`
- Modify: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Write a failing training-side bounded-ratio test**

```python
def test_training_step_reports_ratio_regularization_keys(self):
    detail = _run_minimal_training_step_with_closure()
    self.assertIn("loss_ratio_q_pos", detail)
    self.assertIn("loss_ratio_r_pos", detail)
    self.assertIn("loss_ratio_r_siz", detail)
    self.assertIn("loss_ratio_r_ori", detail)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m unittest tests.test_noise_audit_train.NoiseAuditTrainTest.test_training_step_reports_ratio_regularization_keys -v`

Expected: missing detail-key failure.

- [ ] **Step 3: Add a closure config block to the training config**

```yaml
# config/train_nuscenes.yaml
MAMBA_CLOSURE:
  ENABLED: true
  PROFILES:
    stable_large:
      matched:   {q_pos: [0.8, 1.8], r_pos: [0.8, 2.0], r_siz: [0.8, 2.0], r_ori: [0.6, 3.0]}
      unmatched: {q_pos: [0.8, 1.5], r_pos: [0.8, 1.6], r_siz: [0.8, 1.6], r_ori: [0.6, 2.5]}
    agile_weak:
      matched:   {q_pos: [0.8, 1.6], r_pos: [0.8, 1.8], r_siz: [0.8, 1.8], r_ori: [0.6, 3.0]}
      unmatched: {q_pos: [0.8, 1.3], r_pos: [0.8, 1.4], r_siz: [0.8, 1.4], r_ori: [0.6, 2.2]}
```

- [ ] **Step 4: Add ratio-anchor losses in `training_step()`**

```python
ratio_losses = compute_ratio_anchor_losses(
    predicted={
        "Q_pos": mamba_out["Q_pos"],
        "R_pos": mamba_out["R_pos"],
        "R_siz": mamba_out["R_siz"],
        "R_ori": mamba_out["R_ori"],
    },
    priors={
        "Q_pos": noise_bundle["Q_pos_base"],
        "R_pos": noise_bundle["R_pos_base"],
        "R_siz": noise_bundle["R_siz_base"],
        "R_ori": noise_bundle["R_ori_base"],
    },
    class_ids=class_ids,
    state_buckets=audit_state_per_sample,
    closure_cfg=base_noise_cfg.get("MAMBA_CLOSURE", {}),
)

loss = loss + (
    ratio_losses["q_pos"]
    + ratio_losses["r_pos"]
    + ratio_losses["r_siz"]
    + ratio_losses["r_ori"]
)
detail["loss_ratio_q_pos"] = ratio_losses["q_pos"].item()
detail["loss_ratio_r_pos"] = ratio_losses["r_pos"].item()
detail["loss_ratio_r_siz"] = ratio_losses["r_siz"].item()
detail["loss_ratio_r_ori"] = ratio_losses["r_ori"].item()
```

- [ ] **Step 5: Run the training-audit tests**

Run:

```bash
python -m unittest tests.test_noise_audit_train -v
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add training/train.py config/train_nuscenes.yaml tests/test_noise_audit_train.py
git commit -m "feat: add bounded residual ratio regularization"
```

## Task 4: Create branch configs and document the success gate

**Files:**
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba_closure.yaml`
- Create: `tests/test_mamba_closure_config.py`
- Modify: `README.md`

- [ ] **Step 1: Write the failing config-shape test**

```python
import unittest
import yaml


class MambaClosureConfigTest(unittest.TestCase):
    def test_closure_config_has_success_branch_knobs(self):
        with open("config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml", "r", encoding="utf-8") as handle:
            cfg = yaml.safe_load(handle)

        self.assertEqual(cfg["FILTER_MODE"], "mamba")
        self.assertTrue(cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]["ENABLED"])
        self.assertIn("PROFILES", cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"])
```

- [ ] **Step 2: Run the config test to verify it fails**

Run: `python -m unittest tests.test_mamba_closure_config -v`

Expected: file-not-found failure.

- [ ] **Step 3: Add the closure experiment configs**

```yaml
# config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml
FILTER_MODE: 'mamba'
AUDIT:
  NOISE_AUDIT:
    ENABLED: false
DEKF_BASE_NOISE:
  MAMBA_CLOSURE:
    ENABLED: true
    PROFILES:
      stable_large:
        matched:   {q_pos: [0.8, 1.8], r_pos: [0.8, 2.0], r_siz: [0.8, 2.0], r_ori: [0.6, 3.0]}
        unmatched: {q_pos: [0.8, 1.5], r_pos: [0.8, 1.6], r_siz: [0.8, 1.6], r_ori: [0.6, 2.5]}
      agile_weak:
        matched:   {q_pos: [0.8, 1.6], r_pos: [0.8, 1.8], r_siz: [0.8, 1.8], r_ori: [0.6, 3.0]}
        unmatched: {q_pos: [0.8, 1.3], r_pos: [0.8, 1.4], r_siz: [0.8, 1.4], r_ori: [0.6, 2.2]}
      heavy_long:
        matched:   {q_pos: [0.8, 1.8], r_pos: [0.8, 2.0], r_siz: [0.8, 1.8], r_ori: [0.6, 3.0]}
        unmatched: {q_pos: [0.8, 1.4], r_pos: [0.8, 1.5], r_siz: [0.8, 1.5], r_ori: [0.6, 2.5]}
      human:
        matched:   {q_pos: [0.8, 1.8], r_pos: [0.8, 2.0], r_siz: [0.8, 2.0], r_ori: [0.6, 3.0]}
        unmatched: {q_pos: [0.8, 1.5], r_pos: [0.8, 1.7], r_siz: [0.8, 1.7], r_ori: [0.6, 2.5]}
```

- [ ] **Step 4: Document the branch and success gate**

```markdown
## Bounded Residual Mamba Closure Branch

Use:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml -p 12
```

Success gate:

- frozen baseline stays `AMOTA 0.737`
- the branch only counts as successful if aggregated `AMOTA >= 0.740`
```

- [ ] **Step 5: Run config and smoke validation**

Run:

```bash
python -m unittest tests.test_mamba_closure_config -v
python -m py_compile kalmanfilter/bounded_residual.py kalmanfilter/mamba_adaptive_kf.py tracker/base_tracker.py training/train.py tests/test_bounded_residual.py tests/test_mamba_closure_config.py
```

Expected: both commands succeed.

- [ ] **Step 6: Commit**

```bash
git add config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba_closure.yaml tests/test_mamba_closure_config.py README.md
git commit -m "docs: add mamba closure branch configs and success gate"
```

## Task 5: Run the closure branch in stages and decide whether to keep it

**Files:**
- Use: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba_closure.yaml`
- Use: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml`
- Use: `debug/noise_audit/...`
- Document results in: `docs/project_context_log.md`

- [ ] **Step 1: Run the audit-only closure comparison**

Run:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba_closure.yaml -p 12
python tools/summarize_noise_audit.py --inputs debug/noise_audit/mamba_closure/infer_noise_audit.json --format text
```

Expected:
- audit file is generated
- `unmatched r_pos` medians materially shrink versus the earlier raw-`mamba` audit

- [ ] **Step 2: Run the full closure evaluation**

Run:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_closure.yaml -p 12
```

Expected:
- evaluation completes
- aggregate `AMOTA` is available for comparison against `0.737`

- [ ] **Step 3: Apply the success gate**

Record the result in `docs/project_context_log.md` using this template:

```markdown
### 2026-06-19 bounded residual mamba closure

- Frozen baseline: `0.737`
- Closure branch result: `<value>`
- Audit verdict: `<brief summary>`
- Decision:
  - keep branch only if `AMOTA >= 0.740`
  - otherwise retain frozen baseline and do not switch mainline
```

- [ ] **Step 4: Commit the evaluation note if code/doc changes were made**

```bash
git add docs/project_context_log.md
git commit -m "docs: record bounded residual mamba closure evaluation"
```

## Self-Review

- Spec coverage:
  - bounded residual inference closure: covered by Tasks 1-2
  - training/inference semantic alignment: covered by Task 3
  - `R_ori` inclusion: covered in Tasks 2-3
  - `0.737` baseline protection and `0.740` success gate: covered by Tasks 4-5
  - defer `TRACK_SCORE` until after closure result: enforced in Task 5 decision rule
- Placeholder scan:
  - no `TBD` / `TODO`
  - every task includes concrete files, commands, and code snippets
- Type consistency:
  - plan consistently uses `state_buckets`, `MAMBA_CLOSURE`, and family names `q_pos / r_pos / r_siz / r_ori`

