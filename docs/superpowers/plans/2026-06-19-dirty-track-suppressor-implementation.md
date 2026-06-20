# Dirty Track Suppressor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a conservative dirty-track suppressor that only down-weights or rejects clearly dirty trajectories at output time, while preserving the frozen `AMOTA 0.737` baseline when disabled.

**Architecture:** Keep the existing tracking, matching, birth, KF update, and lifecycle logic unchanged. Add a small output-stage feature collector plus suppressor evaluator that compute online-visible dirty-track signals and convert them into a soft multiplicative penalty and rare hard-reject decision. Wire the new behavior behind a dedicated `DIRTY_TRACK_SUPPRESSOR` config block so the frozen baseline remains untouched unless an explicit suppressor config is used.

**Tech Stack:** Python, existing tracker pipeline in `tracker/base_tracker.py`, helper logic in `tracker/compat_utils.py`, YAML configs, `unittest`

---

## File map

- Modify: `tracker/compat_utils.py`
  - Add profile mapping for suppressor, dirty feature extraction helpers, suppressor evaluator, and safe defaults.
- Modify: `tracker/base_tracker.py`
  - Integrate suppressor into final output path only.
- Modify: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
  - Keep baseline config unchanged or explicitly disabled.
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor.yaml`
  - Dedicated suppressor-on branch config for evaluation.
- Modify: `tests/test_mctrack_compat_utils.py`
  - Add unit tests for suppressor feature extraction and evaluator behavior.
- Modify: `tests/test_runtime_contract_checks.py` only if config/runtime checks need to acknowledge the new config block; otherwise do not touch.
- Optionally modify: `README.md`
  - Add a short evaluation entry for the suppressor branch after implementation stabilizes.

## Task 1: Add suppressor helper tests and minimal helper surface

**Files:**
- Modify: `tests/test_mctrack_compat_utils.py`
- Modify: `tracker/compat_utils.py`

- [ ] **Step 1: Write failing tests for profile mapping and clean/dirty evaluator behavior**

Add tests that expect:

```python
def test_dirty_track_profile_mapping():
    self.assertEqual(map_class_to_dirty_profile(0), "stable_large")
    self.assertEqual(map_class_to_dirty_profile(2), "agile_weak")
    self.assertEqual(map_class_to_dirty_profile(5), "heavy_long")
    self.assertEqual(map_class_to_dirty_profile(1), "human")
    self.assertIsNone(map_class_to_dirty_profile(99))


def test_dirty_suppressor_returns_identity_for_clean_track():
    suppress = dirty_track_suppressor(
        features={
            "recent_fake_len": 0,
            "fake_ratio": 0.0,
            "recent_low_score_match_count": 0,
            "low_score_ratio": 0.0,
            "recent_match_cost_mean": 0.3,
            "current_det_score": 0.8,
            "pos_trace_ratio": 1.0,
        },
        profile_cfg={
            "soft_fake_len": 2,
            "hard_fake_len": 4,
            "soft_low_score_ratio": 0.35,
            "hard_low_score_ratio": 0.60,
            "soft_pos_trace_ratio": 1.8,
            "hard_pos_trace_ratio": 2.6,
            "cost_penalty_start": 0.9,
        },
    )
    self.assertAlmostEqual(suppress["penalty"], 1.0)
    self.assertFalse(suppress["hard_reject"])


def test_dirty_suppressor_soft_penalizes_but_does_not_reject_moderate_dirty_track():
    suppress = dirty_track_suppressor(
        features={
            "recent_fake_len": 2,
            "fake_ratio": 0.4,
            "recent_low_score_match_count": 1,
            "low_score_ratio": 0.45,
            "recent_match_cost_mean": 1.0,
            "current_det_score": 0.35,
            "pos_trace_ratio": 2.0,
        },
        profile_cfg={
            "soft_fake_len": 2,
            "hard_fake_len": 4,
            "soft_low_score_ratio": 0.35,
            "hard_low_score_ratio": 0.60,
            "soft_pos_trace_ratio": 1.8,
            "hard_pos_trace_ratio": 2.6,
            "cost_penalty_start": 0.9,
        },
    )
    self.assertLess(suppress["penalty"], 1.0)
    self.assertGreaterEqual(suppress["penalty"], 0.5)
    self.assertFalse(suppress["hard_reject"])


def test_dirty_suppressor_hard_rejects_extreme_dirty_track():
    suppress = dirty_track_suppressor(
        features={
            "recent_fake_len": 5,
            "fake_ratio": 0.8,
            "recent_low_score_match_count": 4,
            "low_score_ratio": 0.75,
            "recent_match_cost_mean": 1.5,
            "current_det_score": 0.05,
            "pos_trace_ratio": 3.2,
        },
        profile_cfg={
            "soft_fake_len": 2,
            "hard_fake_len": 4,
            "soft_low_score_ratio": 0.35,
            "hard_low_score_ratio": 0.60,
            "soft_pos_trace_ratio": 1.8,
            "hard_pos_trace_ratio": 2.6,
            "cost_penalty_start": 0.9,
        },
    )
    self.assertTrue(suppress["hard_reject"])
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: failures complaining that `map_class_to_dirty_profile` and `dirty_track_suppressor` do not exist yet.

- [ ] **Step 3: Add minimal helper surface in `tracker/compat_utils.py`**

Add:

```python
DIRTY_PROFILE_STABLE_LARGE = "stable_large"
DIRTY_PROFILE_AGILE_WEAK = "agile_weak"
DIRTY_PROFILE_HEAVY_LONG = "heavy_long"
DIRTY_PROFILE_HUMAN = "human"

_DIRTY_CLASS_TO_PROFILE = {
    0: DIRTY_PROFILE_STABLE_LARGE,
    4: DIRTY_PROFILE_STABLE_LARGE,
    2: DIRTY_PROFILE_AGILE_WEAK,
    3: DIRTY_PROFILE_AGILE_WEAK,
    5: DIRTY_PROFILE_HEAVY_LONG,
    6: DIRTY_PROFILE_HEAVY_LONG,
    1: DIRTY_PROFILE_HUMAN,
}


def map_class_to_dirty_profile(class_id: int):
    return _DIRTY_CLASS_TO_PROFILE.get(int(class_id))


def dirty_track_suppressor(*, features: dict, profile_cfg: dict) -> dict:
    profile_cfg = profile_cfg or {}
    soft_fake_len = int(profile_cfg.get("soft_fake_len", 99))
    hard_fake_len = int(profile_cfg.get("hard_fake_len", 999))
    soft_low_score_ratio = float(profile_cfg.get("soft_low_score_ratio", 1.0))
    hard_low_score_ratio = float(profile_cfg.get("hard_low_score_ratio", 2.0))
    soft_pos_trace_ratio = float(profile_cfg.get("soft_pos_trace_ratio", 999.0))
    hard_pos_trace_ratio = float(profile_cfg.get("hard_pos_trace_ratio", 9999.0))
    cost_penalty_start = float(profile_cfg.get("cost_penalty_start", 999.0))

    recent_fake_len = int(features.get("recent_fake_len", 0))
    low_score_ratio = float(features.get("low_score_ratio", 0.0))
    pos_trace_ratio = float(features.get("pos_trace_ratio", 1.0))
    recent_match_cost_mean = float(features.get("recent_match_cost_mean", 0.0))
    current_det_score = float(features.get("current_det_score", 1.0))

    fake_penalty = 1.0
    if recent_fake_len >= soft_fake_len:
        fake_penalty = min(fake_penalty, 0.85)
    if low_score_ratio >= soft_low_score_ratio:
        fake_penalty = min(fake_penalty, 0.8)
    if pos_trace_ratio >= soft_pos_trace_ratio:
        fake_penalty = min(fake_penalty, 0.8)
    if recent_match_cost_mean >= cost_penalty_start:
        fake_penalty = min(fake_penalty, 0.85)
    if current_det_score <= 0.1:
        fake_penalty = min(fake_penalty, 0.75)

    hard_reject = (
        recent_fake_len >= hard_fake_len
        and low_score_ratio >= hard_low_score_ratio
        and pos_trace_ratio >= hard_pos_trace_ratio
    )

    return {
        "penalty": max(0.5, float(fake_penalty)),
        "hard_reject": bool(hard_reject),
    }
```

- [ ] **Step 4: Run the tests again**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: the new suppressor tests pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/compat_utils.py tests/test_mctrack_compat_utils.py
git commit -m "feat: add dirty track suppressor helpers"
```

### Task 2: Add dirty feature extraction for output-stage use

**Files:**
- Modify: `tracker/compat_utils.py`
- Modify: `tests/test_mctrack_compat_utils.py`

- [ ] **Step 1: Write failing tests for feature extraction**

Add tests that verify a helper like `collect_dirty_track_features(...)` can derive stable fields from a trajectory-like stub:

```python
def test_collect_dirty_track_features_uses_recent_fake_and_trace_ratio():
    traj = make_traj_stub(
        fake_history=[False, True, True],
        low_score_history=[False, True, False],
        recent_match_costs=[0.4, 1.1],
        current_score=0.3,
        pos_trace=4.0,
        pos_trace_prior=2.0,
    )
    features = collect_dirty_track_features(traj, base_score=0.3, pos_trace_prior=2.0)
    self.assertEqual(features["recent_fake_len"], 2)
    self.assertAlmostEqual(features["low_score_ratio"], 1 / 3)
    self.assertAlmostEqual(features["recent_match_cost_mean"], 0.75)
    self.assertAlmostEqual(features["current_det_score"], 0.3)
    self.assertAlmostEqual(features["pos_trace_ratio"], 2.0)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: failure because `collect_dirty_track_features` does not exist.

- [ ] **Step 3: Implement minimal feature extraction**

Add a helper like:

```python
def collect_dirty_track_features(traj, *, base_score: float, pos_trace_prior: float) -> dict:
    recent_flags = list(getattr(traj, "debug_fake_history", []))
    low_score_flags = list(getattr(traj, "debug_low_score_history", []))
    recent_costs = list(getattr(traj, "debug_match_cost_history", []))
    pos_trace = float(getattr(traj, "debug_pos_trace", 0.0))
    prior = max(float(pos_trace_prior), 1e-6)

    recent_fake_len = 0
    for flag in reversed(recent_flags):
        if not flag:
            break
        recent_fake_len += 1

    fake_ratio = (sum(1 for flag in recent_flags if flag) / len(recent_flags)) if recent_flags else 0.0
    low_score_ratio = (sum(1 for flag in low_score_flags if flag) / len(low_score_flags)) if low_score_flags else 0.0
    recent_low_score_match_count = sum(1 for flag in low_score_flags if flag)
    recent_match_cost_mean = (sum(recent_costs) / len(recent_costs)) if recent_costs else 0.0

    return {
        "recent_fake_len": recent_fake_len,
        "fake_ratio": fake_ratio,
        "recent_low_score_match_count": recent_low_score_match_count,
        "low_score_ratio": low_score_ratio,
        "recent_match_cost_mean": recent_match_cost_mean,
        "current_det_score": float(base_score),
        "pos_trace": pos_trace,
        "pos_trace_ratio": pos_trace / prior,
    }
```

- [ ] **Step 4: Run tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: feature extraction tests pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/compat_utils.py tests/test_mctrack_compat_utils.py
git commit -m "feat: add dirty track feature collection"
```

### Task 3: Wire suppressor into output stage behind config gate

**Files:**
- Modify: `tracker/base_tracker.py`
- Modify: `tracker/compat_utils.py`
- Modify: `tests/test_mctrack_compat_utils.py`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor.yaml`

- [ ] **Step 1: Write failing integration-style tests for config-gated score suppression**

Add a focused test that exercises a small output helper instead of the entire tracker loop. The helper should accept:
- `base_score`
- `class_id`
- `traj`
- suppressor config

Test expectations:

```python
def test_apply_dirty_track_suppressor_is_identity_when_disabled():
    result = apply_dirty_track_suppressor_to_output(
        base_score=0.8,
        class_id=0,
        traj=make_clean_traj_stub(),
        suppressor_cfg={"ENABLED": False},
        pos_trace_prior=2.0,
    )
    self.assertAlmostEqual(result["final_score"], 0.8)
    self.assertFalse(result["hard_reject"])


def test_apply_dirty_track_suppressor_softly_downweights_dirty_track():
    result = apply_dirty_track_suppressor_to_output(
        base_score=0.4,
        class_id=5,
        traj=make_dirty_traj_stub(),
        suppressor_cfg={
            "ENABLED": True,
            "PROFILES": {
                "heavy_long": {
                    "soft_fake_len": 2,
                    "hard_fake_len": 4,
                    "soft_low_score_ratio": 0.35,
                    "hard_low_score_ratio": 0.60,
                    "soft_pos_trace_ratio": 1.8,
                    "hard_pos_trace_ratio": 2.6,
                    "cost_penalty_start": 0.9,
                }
            },
        },
        pos_trace_prior=2.0,
    )
    self.assertLess(result["final_score"], 0.4)
    self.assertFalse(result["hard_reject"])
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: failure because `apply_dirty_track_suppressor_to_output` does not exist.

- [ ] **Step 3: Implement output helper and wire config lookup**

In `tracker/compat_utils.py`, add:

```python
def get_dirty_track_profile_cfg(class_id: int, suppressor_cfg: dict) -> dict:
    profile_name = map_class_to_dirty_profile(class_id)
    if not profile_name:
        return {}
    return ((suppressor_cfg or {}).get("PROFILES") or {}).get(profile_name, {})


def apply_dirty_track_suppressor_to_output(
    *,
    base_score: float,
    class_id: int,
    traj,
    suppressor_cfg: dict,
    pos_trace_prior: float,
) -> dict:
    if not bool((suppressor_cfg or {}).get("ENABLED", False)):
        return {"final_score": float(base_score), "hard_reject": False, "penalty": 1.0}

    profile_cfg = get_dirty_track_profile_cfg(class_id, suppressor_cfg)
    features = collect_dirty_track_features(traj, base_score=base_score, pos_trace_prior=pos_trace_prior)
    suppress = dirty_track_suppressor(features=features, profile_cfg=profile_cfg)
    return {
        "final_score": float(base_score) * float(suppress["penalty"]),
        "hard_reject": bool(suppress["hard_reject"]),
        "penalty": float(suppress["penalty"]),
        "features": features,
    }
```

Then in `tracker/base_tracker.py`, near the final output stage, wrap the existing score emission with:

```python
suppressor_cfg = self.cfg.get("DIRTY_TRACK_SUPPRESSOR", {})
base_score = output_bbox.det_score if output_bbox.det_score is not None else output_bbox.score
suppress_result = apply_dirty_track_suppressor_to_output(
    base_score=base_score,
    class_id=track_bbox.category,
    traj=trajectory,
    suppressor_cfg=suppressor_cfg,
    pos_trace_prior=1.0,
)
if suppress_result["hard_reject"]:
    continue
output_bbox.score = suppress_result["final_score"]
```

If the exact output path uses a different bbox/trajectory variable naming scheme, adapt the wiring but keep the same semantics.

- [ ] **Step 4: Create branch evaluation config**

Create `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor.yaml` by copying the frozen baseline and appending:

```yaml
DIRTY_TRACK_SUPPRESSOR:
  ENABLED: true
  RECENT_WINDOW: 5
  PROFILES:
    stable_large:
      soft_fake_len: 2
      hard_fake_len: 4
      soft_low_score_ratio: 0.35
      hard_low_score_ratio: 0.60
      soft_pos_trace_ratio: 1.8
      hard_pos_trace_ratio: 2.6
      cost_penalty_start: 0.9
    agile_weak:
      soft_fake_len: 2
      hard_fake_len: 3
      soft_low_score_ratio: 0.30
      hard_low_score_ratio: 0.55
      soft_pos_trace_ratio: 1.6
      hard_pos_trace_ratio: 2.2
      cost_penalty_start: 0.85
    heavy_long:
      soft_fake_len: 2
      hard_fake_len: 4
      soft_low_score_ratio: 0.35
      hard_low_score_ratio: 0.60
      soft_pos_trace_ratio: 1.7
      hard_pos_trace_ratio: 2.4
      cost_penalty_start: 0.9
    human:
      soft_fake_len: 2
      hard_fake_len: 3
      soft_low_score_ratio: 0.30
      hard_low_score_ratio: 0.55
      soft_pos_trace_ratio: 1.7
      hard_pos_trace_ratio: 2.4
      cost_penalty_start: 0.85
```

Do not modify the frozen baseline config.

- [ ] **Step 5: Run tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
python -m py_compile tracker/compat_utils.py tracker/base_tracker.py
```

Expected: pass.

- [ ] **Step 6: Commit**

```bash
git add tracker/compat_utils.py tracker/base_tracker.py tests/test_mctrack_compat_utils.py config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor.yaml
git commit -m "feat: wire dirty track suppressor into output path"
```

### Task 4: Add trajectory-side debug fields needed by suppressor

**Files:**
- Modify: `tracker/trajectory.py`
- Modify: `tests/test_mctrack_compat_utils.py` or existing trajectory-related tests if present

- [ ] **Step 1: Write failing tests for suppressor-facing trajectory debug history**

Add tests that expect the trajectory object to expose enough recent state for suppressor feature collection:

```python
def test_trajectory_updates_fake_and_low_score_debug_histories():
    traj = make_real_trajectory_stub()
    traj.debug_fake_history = []
    traj.debug_low_score_history = []
    traj.debug_match_cost_history = []

    record_dirty_track_step(traj, is_fake=False, is_low_score=True, match_cost=1.2, pos_trace=3.0)
    record_dirty_track_step(traj, is_fake=True, is_low_score=False, match_cost=None, pos_trace=4.0)

    self.assertEqual(traj.debug_fake_history[-2:], [False, True])
    self.assertEqual(traj.debug_low_score_history[-2:], [True, False])
    self.assertEqual(traj.debug_match_cost_history[-1], 1.2)
    self.assertEqual(traj.debug_pos_trace, 4.0)
```

- [ ] **Step 2: Run the failing tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: failure because the recorder/helper does not exist.

- [ ] **Step 3: Add a tiny recorder helper and callsites**

In `tracker/compat_utils.py` add:

```python
def record_dirty_track_step(traj, *, is_fake: bool, is_low_score: bool, match_cost, pos_trace: float, max_window: int = 5):
    if not hasattr(traj, "debug_fake_history"):
        traj.debug_fake_history = []
    if not hasattr(traj, "debug_low_score_history"):
        traj.debug_low_score_history = []
    if not hasattr(traj, "debug_match_cost_history"):
        traj.debug_match_cost_history = []

    traj.debug_fake_history.append(bool(is_fake))
    traj.debug_low_score_history.append(bool(is_low_score))
    if match_cost is not None:
        traj.debug_match_cost_history.append(float(match_cost))
    traj.debug_fake_history = traj.debug_fake_history[-max_window:]
    traj.debug_low_score_history = traj.debug_low_score_history[-max_window:]
    traj.debug_match_cost_history = traj.debug_match_cost_history[-max_window:]
    traj.debug_pos_trace = float(pos_trace)
```

Then add the minimal callsites in trajectory update/unmatch paths so the needed debug lists are maintained. Keep this purely observational; it must not change tracking behavior.

- [ ] **Step 4: Re-run tests**

Run:

```bash
python -m unittest tests.test_mctrack_compat_utils -v
```

Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/trajectory.py tracker/compat_utils.py tests/test_mctrack_compat_utils.py
git commit -m "feat: record dirty track debug histories"
```

### Task 5: Branch evaluation and documentation

**Files:**
- Modify: `README.md` (only if implementation is stable enough to document)

- [ ] **Step 1: Run branch config evaluation**

Run:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor.yaml -p 12
```

Expected: a complete evaluation run with no tracker crashes.

- [ ] **Step 2: Compare against frozen baseline**

Compare the suppressor branch against:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- frozen `AMOTA 0.737`

Inspect especially:

- aggregated `AMOTA`
- `bicycle`
- `trailer`
- `truck`
- whether strong classes remain stable

- [ ] **Step 3: Only if stable, add a short README note**

Add a brief note near the exact-hybrid experiment section describing:

- the branch config name
- that the suppressor is output-stage only
- that it is a denoiser, not a reorderer

- [ ] **Step 4: Commit final branch changes**

```bash
git add README.md
 git commit -m "docs: document dirty track suppressor branch"
```

If README is not updated because the branch is not yet stable, skip this commit and keep documentation unchanged.

## Self-review

- Spec coverage: all approved sections are covered:
  - output-only placement
  - online-visible features
  - soft-first / rare hard rejection
  - profile-based parameters
  - frozen baseline protection
- Placeholder scan: no unresolved placeholders remain.
- Type consistency: helper names, config keys, and branch config names are consistent across tasks.
