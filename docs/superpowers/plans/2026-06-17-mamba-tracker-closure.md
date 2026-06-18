# Mamba Tracker Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the current `AMOTA 0.737` baseline, then improve the project by first closing `mamba` inference/runtime contract gaps, then fixing clearly unreasonable ByteTrack behaviors, and only afterwards redesigning `TRACK_SCORE` as a denoising mechanism instead of a ranking override.

**Architecture:** Keep the current strong baseline isolated in exact-hybrid configs and treat every subsequent change as an opt-in branch path with explicit config gates and A/B evaluation. First tighten `mamba` runtime semantics and checkpoint/runtime contract checks without retraining, then repair ByteTrack’s two-stage implementation so it behaves like a conservative rescue path instead of a noise amplifier, and only after the main inference chain is stable introduce a lightweight dirty-track suppressor to replace the current heuristic track-quality reranker.

**Tech Stack:** Python, PyTorch, YAML configs, existing tracker/kalmanfilter/training modules, unittest, py_compile.

---

### Task 1: Freeze the `0.737` baseline and document the immutable reference

**Files:**
- Modify: `docs/project_context_log.md`
- Modify: `README.md`
- Test: none

- [ ] **Step 1: Add a baseline record entry**

Append an entry to `docs/project_context_log.md` that records:

```md
## 2026-06-17: Frozen Exact-Hybrid Baseline

- Baseline config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Baseline aggregated AMOTA: `0.737`
- Baseline purpose: immutable A/B reference for all route-A work
- Constraint: any new change must be evaluated against this config and must not replace it in-place
```

- [ ] **Step 2: Add README note for the frozen baseline**

Add a short section in `README.md` near the exact-baseline configs:

```md
### Frozen Route-A Baseline

Current protected baseline:

- Config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Aggregated AMOTA: `0.737`

All route-A experiments should branch from this baseline and be evaluated against it rather than modifying it in place.
```

- [ ] **Step 3: Verify docs changed as intended**

Run: `rg -n "0.737|Frozen Route-A Baseline|Frozen Exact-Hybrid Baseline" docs/project_context_log.md README.md`

Expected: both files show the new baseline markers.


### Task 2: Strengthen `mamba` runtime contract checks without changing current baseline behavior

**Files:**
- Modify: `training/train.py`
- Modify: `tracker/base_tracker.py`
- Test: `tests/test_mctrack_compat_utils.py` or new focused runtime-contract test

- [ ] **Step 1: Write the failing test for stronger contract validation**

Add a focused unit test in `tests/test_mctrack_compat_utils.py` or a new `tests/test_runtime_contract_checks.py`:

```python
def test_runtime_contract_warns_on_history_and_filter_mode_mismatch():
    runtime_contract = {
        "tracker_compat_mode": "mctrack",
        "expected_bev_cost_mode": "geometric",
        "history_source": "fusion",
        "init_state_source": "fusion",
        "filter_mode": "mamba",
    }
    cfg = {
        "TRACKER_COMPAT_MODE": "mctrack",
        "FILTER_MODE": "pure_dekf",
        "THRESHOLD": {"BEV": {"COST_MODE": "geometric"}},
        "MAMBA": {"CHECKPOINT_PATH": "dummy.pt"},
    }
    # Assert that the contract checker emits warnings for filter/history mismatch.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_runtime_contract_checks -v`

Expected: FAIL because filter/history contract checks do not yet exist.

- [ ] **Step 3: Save stricter runtime contract fields during training**

Update `training/train.py` runtime contract payload to include:

```python
runtime_contract = {
    "tracker_compat_mode": train_tracker_compat_mode,
    "history_source": history_source,
    "init_state_source": init_state_source,
    "filter_mode": str(cfg.get("INFERENCE", {}).get("FILTER_MODE_HINT", train_tracker_compat_mode)).strip().lower(),
    "expected_bev_cost_mode": str(data_cfg.get("EXPECTED_BEV_COST_MODE", "geometric")).strip().lower(),
}
```

For the existing training config flow, use the actual training-side mode source instead of inventing a new config hierarchy if a simpler direct field already exists.

- [ ] **Step 4: Enforce stronger warning checks in tracker init**

Extend the existing checkpoint warning block in `tracker/base_tracker.py` so it also compares:

```python
expected_history_source = str(runtime_contract.get("history_source", "unknown")).strip().lower()
expected_init_state_source = str(runtime_contract.get("init_state_source", "unknown")).strip().lower()
expected_filter_mode = str(runtime_contract.get("filter_mode", self.filter_mode)).strip().lower()
```

and warns when current runtime mode is inconsistent with the checkpoint contract.

- [ ] **Step 5: Run the new test**

Run: `python -m unittest tests.test_runtime_contract_checks -v`

Expected: PASS.

- [ ] **Step 6: Run regression tests**

Run: `python -m unittest tests.test_mctrack_compat_utils -v`

Expected: PASS.


### Task 3: Remove dead/no-op `mamba` conditioning paths or make them explicit no-ops

**Files:**
- Modify: `tracker/base_tracker.py`
- Modify: `training/train.py`
- Modify: `training/det_tracklet_dataset.py` if needed for metadata propagation
- Test: new focused test or reuse existing tests with lightweight assertions

- [ ] **Step 1: Write a failing test that documents current constant-mask behavior**

Add a test that asserts the project deliberately treats `detection_driven_mask` as all-ones in the current detection-driven path unless a future mode is explicitly added.

```python
def test_detection_driven_mask_is_explicitly_constant_in_current_detection_driven_path():
    # Assert helper or metadata builder marks the current path as fully detection-driven.
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest tests.test_runtime_contract_checks -v`

Expected: FAIL if no helper/explicit annotation exists.

- [ ] **Step 3: Introduce a helper that builds runtime conditioning metadata explicitly**

Refactor the inline `current_range` / `detection_driven_mask` creation in `tracker/base_tracker.py` into a helper such as:

```python
def _build_mamba_runtime_context(self, trajs):
    class_ids = ...
    current_range = ...
    detection_driven_mask = torch.ones(...)
    return class_ids, current_range, detection_driven_mask
```

Do the same idea on the training side if necessary so the current “all detection-driven” semantics are explicit instead of accidental.

- [ ] **Step 4: Add code comments that this branch is intentionally constant today**

Document that `detection_driven_mask` is intentionally constant in the current detection-driven training/inference path, and that future trajectory-state modes should change it explicitly rather than relying on implicit behavior.

- [ ] **Step 5: Run tests**

Run:
- `python -m unittest tests.test_runtime_contract_checks -v`
- `python -m py_compile tracker/base_tracker.py training/train.py training/det_tracklet_dataset.py`

Expected: PASS.


### Task 4: Make ByteTrack stage-2 thresholds class-aware and structurally safe

**Files:**
- Modify: `tracker/base_tracker.py`
- Modify: `tracker/bytetrack_utils.py` if helper support is needed
- Test: `tests/test_bytetrack_utils.py`, plus a new focused stage-2 config handling test

- [ ] **Step 1: Write failing tests for stage-2 relaxed threshold handling**

Add tests covering two facts:

```python
def test_stage2_relaxed_thresholds_preserve_per_class_dict_shape():
    orig = {0: 1.3, 1: 1.8, 2: 1.7}
    relaxed = build_relaxed_cost_thresholds(orig, relax_ratio=1.5)
    assert isinstance(relaxed, dict)
    assert relaxed[2] == 1.7 * 1.5

def test_stage2_relaxed_thresholds_can_use_per_class_overrides():
    orig = {2: 1.7, 5: 1.4}
    relaxed = build_relaxed_cost_thresholds(orig, relax_ratio=1.5, overrides={5: 1.2})
    assert relaxed[2] == 1.7 * 1.5
    assert relaxed[5] == 1.2
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_bytetrack_utils -v`

Expected: FAIL because no helper exists.

- [ ] **Step 3: Introduce a dedicated relaxed-threshold helper**

Add a helper in `tracker/bytetrack_utils.py`:

```python
def build_relaxed_cost_thresholds(cost_thre_cfg, relax_ratio=1.5, overrides=None):
    relaxed = {int(k): float(v) * float(relax_ratio) for k, v in dict(cost_thre_cfg).items()}
    for k, v in (overrides or {}).items():
        relaxed[int(k)] = float(v)
    return relaxed
```

and use it in `tracker/base_tracker.py` instead of converting the dict to a list.

- [ ] **Step 4: Replace the current `* 2.0` blanket relax logic**

In the ByteTrack stage-2 branch, replace:

```python
cfg_relaxed["THRESHOLD"]["BEV"]["COST_THRE"] = [
    v * 2.0 for v in orig_thre.values()
]
```

with:

```python
cfg_relaxed["THRESHOLD"]["BEV"]["COST_THRE"] = build_relaxed_cost_thresholds(
    orig_thre,
    relax_ratio=self.cfg.get("MATCHING", {}).get("BYTETRACK_RELAX_RATIO", 1.5),
    overrides=self.cfg.get("MATCHING", {}).get("BYTETRACK_RELAXED_COST_THRE", {}),
)
```

- [ ] **Step 5: Run tests**

Run:
- `python -m unittest tests.test_bytetrack_utils -v`
- `python -m py_compile tracker/base_tracker.py tracker/bytetrack_utils.py`

Expected: PASS.


### Task 5: Restrict ByteTrack stage-2 to conservative rescue behavior

**Files:**
- Modify: `tracker/base_tracker.py`
- Modify: `config/nuscenes_bytetrack.yaml`
- Test: new focused unit test for tentative/low birth gating behavior

- [ ] **Step 1: Write failing test for stage-2 birth policy**

Add a test that asserts low/tentative detections in stage-2 are not allowed to create new tracks unless explicitly enabled.

```python
def test_bytetrack_stage2_birth_disabled_by_default():
    cfg = {
        "MATCHING": {"ALLOW_TENTATIVE_STAGE2_BIRTH": False},
        ...
    }
    # Assert unmatched tentative/low detections do not create births.
```

- [ ] **Step 2: Run test to verify failure**

Run: `python -m unittest tests.test_bytetrack_utils -v`

Expected: FAIL because current path still births tentative detections.

- [ ] **Step 3: Gate tentative births behind an explicit config**

In `tracker/base_tracker.py`, only allow the current unmatched tentative-detection birth block when:

```python
allow_tentative_birth = bool(
    self.cfg.get("MATCHING", {}).get("ALLOW_TENTATIVE_STAGE2_BIRTH", False)
)
```

Otherwise stage-2 remains rescue-only.

- [ ] **Step 4: Update ByteTrack preset config explicitly**

In `config/nuscenes_bytetrack.yaml`, add explicit fields:

```yaml
MATCHING:
  USE_BYTETRACK: true
  BYTETRACK_RELAX_RATIO: 1.5
  BYTETRACK_RELAXED_COST_THRE: {}
  ALLOW_TENTATIVE_STAGE2_BIRTH: false
```

- [ ] **Step 5: Run tests**

Run:
- `python -m unittest tests.test_bytetrack_utils -v`
- `python -m py_compile tracker/base_tracker.py`

Expected: PASS.


### Task 6: Redefine `TRACK_SCORE` scope from “ranking override” to “dirty-track suppressor”

**Files:**
- Modify: `tracker/compat_utils.py`
- Modify: `tracker/base_tracker.py`
- Modify: `tracker/trajectory.py`
- Test: new focused tests around suppression-only behavior

- [ ] **Step 1: Write failing tests for suppression-only semantics**

Add tests such as:

```python
def test_track_score_dirty_suppressor_preserves_clean_current_score():
    # Clean trajectory should keep score close to current detector score.

def test_track_score_dirty_suppressor_penalizes_fake_heavy_track():
    # Track with many fake / low-score rescue frames should be reduced.
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python -m unittest tests.test_track_score_tools -v`

Expected: FAIL because current implementation is a full quality reranker.

- [ ] **Step 3: Implement a conservative suppressor mode**

Add a new mode in `compute_track_quality_score()` such as `dirty_suppress_v1`:

```python
base = current_score
penalty = 0.0
penalty += fake_ratio_weight * fake_ratio
penalty += low_score_weight * low_score_ratio
penalty += bad_assoc_weight * bad_assoc_ratio
return clip(base * (1.0 - penalty), min_score, max_score)
```

Do not average in maturity/det-conf as a replacement ranking model in this mode.

- [ ] **Step 4: Wire online/offline use to the new semantics only behind opt-in config**

Ensure current `TRACK_SCORE=false` baseline is untouched. Only configs that explicitly choose the new mode should use it.

- [ ] **Step 5: Run tests**

Run:
- `python -m unittest tests.test_track_score_tools -v`
- `python -m py_compile tracker/compat_utils.py tracker/base_tracker.py tracker/trajectory.py`

Expected: PASS.


### Task 7: Verification matrix against the frozen baseline

**Files:**
- Modify: `docs/project_context_log.md`
- Test: real eval commands only

- [ ] **Step 1: Record the comparison matrix template**

Append a section to `docs/project_context_log.md`:

```md
## Route-A Verification Matrix

- Frozen baseline: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml` → `AMOTA 0.737`
- Candidate 1: stronger mamba runtime contract
- Candidate 2: ByteTrack stage-2 threshold fix
- Candidate 3: ByteTrack rescue-only birth policy
- Candidate 4: dirty-track suppressor
```

- [ ] **Step 2: Define evaluation commands**

Run baseline reference when needed:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml -p 12
```

Run ByteTrack candidate:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_bytetrack.yaml -p 12
```

Run current single-stage fusion reference:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage.yaml -p 12
```

- [ ] **Step 3: Verification rule**

Only accept a change into the working line if:
- it does not lower the frozen baseline config result, or
- it improves an opt-in branch (for example ByteTrack) without regressing unrelated reference configs.

