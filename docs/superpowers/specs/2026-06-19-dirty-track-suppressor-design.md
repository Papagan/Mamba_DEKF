# Dirty Track Suppressor Design

Date: 2026-06-19
Status: Approved design, pending implementation plan
Scope: Replace the current TRACK_SCORE-as-reorderer direction with a conservative dirty-track suppressor that protects the frozen `AMOTA 0.737` baseline.

## Context

The current best frozen baseline is:

- Config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Aggregated `AMOTA`: `0.737`

Earlier experiments showed that the existing `TRACK_SCORE=true` path behaves like a post-hoc reorderer rather than a denoiser. It changes final ranking semantics without synchronizing the main matching chain, which makes it prone to degrading globally stable results, especially for weak or short-lived categories.

The goal of this redesign is not to redefine what a “good trajectory” is. The goal is narrower:

- keep the current matching / lifecycle chain intact
- suppress obviously dirty trajectories at output time
- avoid disturbing categories that already perform well
- only keep this line if it can eventually coexist with or improve on the frozen baseline

## Goals

1. Convert `TRACK_SCORE` from a ranking-oriented quality score into a dirty-track suppressor.
2. Keep the frozen `0.737` baseline untouched when the suppressor is disabled.
3. Limit first-pass logic to online-visible features only.
4. Default to soft suppression; only use hard rejection for clearly corrupted trajectories.
5. Make the first version explainable and profile-based rather than fully per-class hand-tuned.

## Non-goals

This design does not:

- change the main matching logic
- change birth logic
- change KF predict/update math
- change lifecycle state transitions
- introduce a learned head in the first version
- replace the ongoing `mamba` closure work

## Recommended approach

Use a **rule-based dirty-track suppressor** inserted at final output time.

The suppressor:

- preserves the existing base score semantics
- applies multiplicative penalties only when a trajectory shows clear dirty signals
- optionally hard-rejects only extreme dirty trajectories

This replaces the previous “quality score reorderer” idea with a constrained denoising layer.

## Alternatives considered

### Option A: Rule-based dirty-track suppressor

Recommended.

Pros:

- lowest risk to the frozen baseline
- directly aligned with “denoise, don’t reorder”
- easy to audit and tune
- easy to bypass for non-regression checks

Cons:

- lower ceiling than a learned suppressor
- requires careful threshold design

### Option B: Lightweight learned cleanliness head

Pros:

- more expressive
- can later integrate with `mamba` / uncertainty features

Cons:

- adds training/inference contract complexity
- too risky before the inference closure line is fully settled

### Option C: Weaken the old TRACK_SCORE logic

Pros:

- least code movement

Cons:

- preserves the same architectural mistake
- still acts like a reorderer
- likely to repeat previous regressions

## Architecture

### Placement

The suppressor will run only in the final output stage.

It will not affect:

- association
- birth
- matched update
- unmatched update
- lifecycle transitions

It will only affect whether an already-built trajectory:

- keeps its base output score
- gets softly down-weighted
- gets hard-rejected in extreme cases

### Units

The implementation should be split into three focused units.

#### 1. Dirty feature collector

Location: `tracker/compat_utils.py`

Responsibility:

- extract online-visible dirty-track features from the current trajectory state

Inputs:

- trajectory object
- class id / profile
- base score
- current KF covariance state

Outputs:

- normalized feature dictionary

#### 2. Dirty suppressor evaluator

Location: `tracker/compat_utils.py`

Responsibility:

- map features plus class profile config to:
  - `penalty`
  - `hard_reject`
  - optional suppression reason/debug info

Outputs:

- `penalty` in `[0, 1]`
- `hard_reject: bool`

#### 3. Output-stage integration

Location: `tracker/base_tracker.py`

Responsibility:

- preserve current output path when suppressor is disabled
- when enabled:
  - compute base score
  - collect dirty features
  - evaluate suppressor
  - emit `final_score = base_score * penalty`
  - skip output if `hard_reject` is true

## Features used in v1

The first version will only use online-visible features.

### Trajectory dirtiness features

- `recent_fake_len`
  - consecutive recent fake/coast length
- `fake_ratio`
  - fraction of fake/coast steps in a recent window
- `recent_low_score_match_count`
  - count of recent low-score real matches
- `low_score_ratio`
  - ratio of low-score real matches in a recent window
- `recent_match_cost_mean`
  - average recent real-match association cost
- `current_det_score`
  - current real detection score; if current frame is fake, treat as very low

### KF uncertainty features

- `pos_trace`
  - current position covariance trace
- `pos_trace_ratio`
  - ratio of current `P_pos` trace relative to the category baseline prior

### Deferred features

The first version will not actively use:

- orientation uncertainty features
- size uncertainty features
- learned cleanliness predictions

These may be added only after v1 proves safe.

## Behavior

### Core rule

Default behavior is identity:

- if no strong dirty signal is present, `penalty = 1.0`
- no trajectory is improved or rewarded

This system is strictly one-directional:

- it may keep score unchanged
- it may reduce score
- it may hard-reject extreme cases
- it never boosts a trajectory

### Soft suppression

Soft suppression applies a multiplicative penalty:

`final_score = base_score * penalty`

The penalty is factored across dirty dimensions, for example:

- fake/coast penalty
- low-score penalty
- match-cost penalty
- uncertainty penalty

Each factor should stay in a conservative range, such as:

- `[0.7, 1.0]` for mild penalties
- `[0.5, 1.0]` only for stronger dirty evidence

The intent is to gradually demote suspicious tracks, not to abruptly reorder the whole score distribution.

### Hard rejection

Hard rejection must be rare.

It should trigger only when multiple strong dirty conditions co-occur, such as:

- long consecutive fake/coast span and high `pos_trace_ratio`
- very high `low_score_ratio` and very poor recent association cost
- current fake state combined with clear covariance drift

Single mild anomalies should never hard-reject a trajectory.

## Class parameterization

The first version will use **profiles**, not full per-class free tuning.

Profiles:

- `stable_large`
  - `car`, `bus`
- `agile_weak`
  - `bicycle`, `motorcycle`
- `heavy_long`
  - `trailer`, `truck`
- `human`
  - `pedestrian`

Each profile will define:

- `soft_fake_len`
- `hard_fake_len`
- `soft_low_score_ratio`
- `hard_low_score_ratio`
- `soft_pos_trace_ratio`
- `hard_pos_trace_ratio`
- `cost_penalty_start`

This keeps parameter count low, aligns with earlier noise-profile grouping, and reduces regression risk.

## Config structure

Add a new independent config block rather than reusing the old `TRACK_SCORE.ENABLED` semantics.

Proposed shape:

```yaml
DIRTY_TRACK_SUPPRESSOR:
  ENABLED: false
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
      # same field structure, with profile-specific thresholds
    heavy_long:
      # same field structure, with profile-specific thresholds
    human:
      # same field structure, with profile-specific thresholds
```

This avoids overloading the previous `TRACK_SCORE` meaning while keeping the feature close to the same conceptual area.

## Rollout strategy

### Stage 1: non-regression

When suppressor is disabled:

- output must remain identical to the frozen baseline path

This protects the current `0.737` result.

### Stage 2: conservative enablement

Enable the suppressor in a dedicated branch config with:

- conservative thresholds
- soft penalties dominant
- hard rejection only for obvious dirty cases

The first acceptance criterion is not immediate gain. It is:

- no clear regression versus `0.737`
- no major drop in `bicycle`, `trailer`, or `truck`

### Stage 3: focused tuning

Only if Stage 2 is stable:

- tune soft/hard fake thresholds
- tune low-score thresholds
- tune covariance thresholds

Still no reordering logic.

## Error handling and safety

- Missing suppressor config should behave as disabled.
- Missing trajectory-side debug fields should degrade gracefully to “no suppression”.
- If covariance-derived features are unavailable or invalid, uncertainty penalty should be skipped instead of forcing rejection.
- Debug output should be available in development mode to inspect:
  - penalty
  - hard reject flag
  - triggered conditions

## Testing strategy

### Unit tests

- feature collector returns expected fields for:
  - clean real trajectories
  - fake/coast-heavy trajectories
  - low-score rescue-heavy trajectories
- suppressor evaluator:
  - returns identity penalty for clean inputs
  - returns soft penalty for moderate dirty inputs
  - returns hard reject only for extreme dirty combinations
- unknown/missing data paths degrade safely

### Integration tests

- when suppressor is disabled, output behavior matches current baseline path
- when suppressor is enabled, only output-stage score/reject behavior changes
- no matching / lifecycle side effects

### Evaluation gates

The frozen baseline remains:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- `AMOTA 0.737`

The suppressor line must first demonstrate non-regression before it is considered for wider tuning.

## Success criteria

The first implementation is successful if:

1. It leaves the frozen baseline unchanged when disabled.
2. It does not disturb the main tracking pipeline.
3. It provides a safe branch for suppressing clearly dirty trajectories.
4. It maintains or improves overall `AMOTA` without the broad regressions seen from the old reorderer-style `TRACK_SCORE`.

The final success threshold for keeping this line is:

- it must coexist with the protected baseline and show a clear path to improving over `0.737`
- otherwise it remains an experimental branch only
