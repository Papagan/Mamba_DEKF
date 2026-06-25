# Prior-Conditioned Multi-Head Mamba Closure Design

**Date:** 2026-06-24

**Status:** Draft for review

## Goal

Redesign the `mamba` branch so it becomes a **training/inference-closed uncertainty controller** for the existing 3DMOT pipeline, rather than an unconstrained generator of absolute covariance tensors.

The redesign must preserve the current protected baseline and only replace it if the new branch demonstrates a real metric gain:

- aggregated `AMOTA > 0.740`
- with special attention to weak classes:
  - `bicycle`
  - `motorcycle`
  - `pedestrian`
  - `trailer`

## Frozen Baseline

The protected reference line is:

- config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml`
- protected aggregated result: `AMOTA 0.739`
- protected trailer reference in that run: `AMOTA 0.533`

This baseline remains untouched during the redesign. All Mamba-closure experiments must be introduced as new branch configs and evaluated against this frozen line.

## Problem Statement

The current tracker shell is already reasonably strong, but the `mamba` uncertainty path still has a fundamental contract problem.

The key failure is not minor instability. It is **semantic mismatch**:

1. inference still lets the model behave too much like an unconstrained covariance generator;
2. training and runtime do not yet fully share the same uncertainty language;
3. weak classes and `unmatched` states are especially sensitive to this mismatch.

The completed noise audit shows the concrete failure pattern:

- `bicycle`, `motorcycle`, `pedestrian`, and parts of `car` exhibit large learned-vs-prior scale divergence;
- the worst explosions concentrate in:
  - `R_pos`
  - `R_siz`
  - `R_ori`
- the risk is strongest in:
  - `unmatched`
  - low-history
  - low-confidence conditions.

This means the next architecture should not ask one shared head to jointly solve:

- all categories,
- all lifecycle states,
- all physical state families,
- and all covariance scales

with one undifferentiated output space.

## Design Principles

The redesign is guided by six rules.

1. **Same runtime contract in training and inference**
   - the model must be trained on the same uncertainty semantics that the tracker consumes.

2. **Prior-relative control, not absolute covariance generation**
   - `mamba` should adjust trusted physical priors, not replace them.

3. **Category-specific conditioning**
   - all 7 nuScenes categories must be treated explicitly, not merged into coarse profiles in the first pass.

4. **State-family factorization**
   - position, velocity, orientation, and size must not share one generic output head.

5. **Lifecycle-aware conservatism**
   - `matched`, `unmatched`, and `coast` must have distinct behavior.

6. **Non-regression discipline**
   - if a new design does not exceed the frozen baseline, it does not become the new mainline.

## Adopted Ideas From `docs/restructure_paln.md`

The following ideas from `docs/restructure_paln.md` are retained because they are directionally correct:

- stop predicting absolute covariance values as the primary learning target;
- move the model closer to residual / innovation space instead of raw global coordinate space;
- separate state families rather than using one monolithic head;
- use strong physical priors and bounded outputs;
- explicitly degrade to prior-only behavior during coasting;
- use warmup, anchor regularization, and progressive release of learned freedom.

The following parts are revised for this codebase:

- the model cannot use the **current-frame innovation** at `predict_before_associate()` time because the current frame is not matched yet;
- therefore the input must use **historical matched residual tokens**, not current-frame innovation;
- orientation should no longer keep a separate long-term `kappa -> 1/kappa` semantic path if the new branch is defined around prior-relative `R` ratios;
- the first pass should not rely on coarse class profiles alone; it should use all 7 categories explicitly.

## High-Level Architecture

The first-pass architecture is:

- one shared temporal trunk;
- 7 category-specific head groups;
- each category has independent state-family heads for:
  - `position`
  - `velocity`
  - `orientation`
  - `size`
- all heads emit **prior-relative bounded ratios**;
- the tracker consumes those ratios under a unified lifecycle contract.

Conceptually:

`historical residual tokens -> shared temporal trunk -> class-specific state heads -> bounded ratios -> prior-scaled Q/R -> KF predict/update`

This is deliberately not a full MoE. It borrows the useful idea of conditional specialization while keeping the system simple enough to train and validate safely.

## Input Representation

### Why Current History Is Not Enough

The current `track_history` is already partially normalized and masked, but it is still dominated by historical state values rather than explicit filter-error evolution.

That is insufficient for the new goal. The new branch should learn:

- how residuals change,
- when detections become unreliable,
- how lifecycle degradation affects uncertainty,
- and how these patterns differ by category and state family.

### New Historical Residual Tokens

Each historical step should be represented by a token dominated by closure-relevant signals:

- matched-update position residual
- matched-update velocity residual
- matched-update size residual
- matched-update orientation residual
- detection score
- match flag
- time since last real match
- age within history window
- range
- speed
- yaw rate
- valid-token mask

For `unmatched` / `coast` steps:

- residual fields are zero-filled;
- the mask and lifecycle flags remain explicit.

This prevents fake zeros from being misread as confident measurements.

### Residual Cache Requirement

The tracker must store historical matched residuals inside trajectory state so that training caches and live inference use the same semantic input.

This requires explicit residual bookkeeping during matched updates, not implicit reconstruction from bbox values later.

## Shared Temporal Trunk

The first pass retains a single temporal trunk:

- `Mamba` if available;
- same sequential backbone style already used in the repo.

The trunk learns only shared temporal dynamics:

- residual persistence
- quality decay
- match-to-coast degradation
- category-agnostic motion uncertainty patterns

The trunk does **not** emit final covariances directly.

## Category-Conditioned Multi-Head Output

### Category Granularity

The first pass uses all 7 nuScenes categories explicitly:

- `car`
- `pedestrian`
- `bicycle`
- `motorcycle`
- `bus`
- `trailer`
- `truck`

The user explicitly requested per-class conditioning because empirical results already show substantial inter-class noise differences, especially between classes such as `bus` and `truck`.

### Head Structure

Each category owns an independent head group:

- `pos head`
- `vel head`
- `ori head`
- `size head`

This gives 7 independent category branches over a shared trunk.

### Output Semantics

Heads do not emit absolute covariance matrices.

They emit bounded ratios relative to the existing `pure_dekf` prior:

- `Q_pos_final = Q_pos_prior * gamma_q_pos`
- `Q_vel_final = Q_vel_prior * gamma_q_vel`
- `R_pos_final = R_pos_prior * gamma_r_pos`
- `R_vel_final = R_vel_prior * gamma_r_vel`
- `R_siz_final = R_siz_prior * gamma_r_siz`
- `R_ori_final = R_ori_prior * gamma_r_ori`

The exact tensor grouping can remain factorized rather than fully free:

- position process noise:
  - `xyz`
  - `vxyz`
- position measurement noise:
  - `xyz`
  - `vxy`
- size measurement noise:
  - `lw`
  - `h`
- orientation measurement noise:
  - `yaw`

This keeps the representation expressive but much safer than full unconstrained matrix regression.

### Bounded Ratio Parameterization

Each ratio should be emitted in bounded log-space, for example:

- head predicts `u`
- ratio computed as `gamma = exp(alpha * tanh(u))`

where:

- `alpha` is family-specific and class-specific;
- `gamma = 1` means full fallback to the prior.

This preserves the strongest idea from `restructure_paln.md`: the model should be a dynamic tuner, not the owner of the physical scale.

## Lifecycle Contract

The branch must use exactly three runtime states:

- `matched`
- `unmatched`
- `coast`

### Matched

- full ratio prediction is allowed;
- still bounded by per-class per-family limits.

### Unmatched

- bounds are tighter than matched;
- history is shortened;
- ratios should be biased back toward 1.

### Coast

- ratios are not learned;
- `gamma = 1` is enforced directly;
- tracker falls back to pure prior physics.

This is not just an inference heuristic. It is a core contract rule and must also hold in training.

## Adaptive Windowing

### Engineering Rule

The tensor shape stays fixed for batching convenience, using a global upper bound such as:

- `HISTORY_LEN = 8`

### Modeling Rule

The effective history is adaptive.

Different categories have different default history and rollout ranges:

| Class | min_history | max_history | min_rollout | max_rollout |
| --- | ---: | ---: | ---: | ---: |
| car | 4 | 8 | 1 | 3 |
| pedestrian | 4 | 7 | 1 | 3 |
| bicycle | 3 | 6 | 1 | 2 |
| motorcycle | 3 | 6 | 1 | 2 |
| bus | 4 | 8 | 1 | 3 |
| trailer | 5 | 8 | 1 | 3 |
| truck | 4 | 8 | 1 | 3 |

### Dynamic Shrink Rules

The default window is further reduced by runtime quality:

- `matched`
  - use the category-allowed effective window
- `unmatched`
  - shorten history relative to the category maximum
- `coast`
  - use only a very short tail or pure prior-only behavior
- low-quality history
  - if recent match ratio is poor or recent scores are persistently low, further shorten the effective history

This keeps training and inference aligned while respecting the empirical fact that categories have different track persistence and noise dynamics.

## Loss Design

Loss design is part of the architecture contract, not a late training detail.

The first-pass loss stack is designed around one central principle:

> optimize the same uncertainty semantics that the runtime consumes.

### 1. Main State Likelihood Loss

The main objective remains rollout-based Kalman likelihood, because the runtime still consumes Gaussian-KF uncertainty.

The recommended primary terms are:

- position posterior Gaussian NLL
- velocity posterior Gaussian NLL
- size posterior Gaussian NLL
- orientation posterior wrapped Gaussian NLL

Orientation is intentionally moved away from the legacy `Von Mises/kappa` main path for this redesign because the new branch wants orientation uncertainty to obey the same prior-relative ratio contract as other families.

The role of this main loss is:

- teach the model whether its `Q/R` ratios make the filter posterior statistically sensible;
- preserve strict training/inference closure.

### 2. Ratio Identity Anchor Loss

A dedicated log-ratio anchor is required:

- penalize deviation of `log(gamma)` from zero;
- computed separately for each family;
- category-aware and lifecycle-aware.

This implements the safe fallback rule:

- no confidence -> return to prior.

### 3. Ratio Bound Violation Loss

Bounds used at inference must also exist in training.

The training loss should penalize:

- upper-bound violations strongly;
- lower-bound violations more gently.

This asymmetry is justified by the audit:

- catastrophic failures are driven more by noise explosion than by mild overconfidence.

### 4. Robust Auxiliary State Loss

`Student-t` / `Huber` ideas from `restructure_paln.md` are useful, but not as a replacement for the main KF likelihood.

They should appear only as auxiliary stabilizers:

- Huber on posterior position error
- Huber or log-space smooth L1 on size
- small circular auxiliary on orientation

This improves robustness to noisy detections without breaking the Gaussian-KF runtime contract.

### 5. Association / Contrastive Loss

If semantic embedding is still used for association:

- retain contrastive loss at low weight;
- never allow it to dominate the trunk over the uncertainty objective.

### 6. Warmup and Progressive Release

The training schedule should preserve the useful idea already identified in `restructure_paln.md`:

- early epochs:
  - strong anchor weight
  - near-prior output freedom
- middle epochs:
  - gradual relaxation of bounds and anchor weight
- late epochs:
  - learned ratios allowed to act more freely, but still inside contract bounds

### 7. Old Losses To Retire Or Downweight

The redesign should remove or strongly demote legacy pieces that conflict with the new contract:

- legacy `kappa`-centric main orientation path
- `kappa_reg` as a central safety device
- aggressive direct `delta_pos` shortcut supervision

If a direct state-delta branch is retained at all, it should be heavily downweighted or disabled in the first pass to prevent the model from learning two inconsistent control roles:

- directly moving states
- and shaping uncertainty.

## Runtime Consumption Rules

The tracker consumes the new branch under these rules:

1. compute base prior noise from the frozen physical configuration;
2. build residual-token history;
3. run shared trunk;
4. select the class-specific head group;
5. emit bounded ratios for the current lifecycle state;
6. if `coast`, override ratios to 1;
7. reconstruct final `Q/R`;
8. pass only reconstructed `Q/R` to KF predict/update.

No post-hoc semantic reinterpretation is allowed after that point.

## Implementation Boundaries

The redesign primarily affects:

- `tracker/base_tracker.py`
  - historical token construction
  - residual caching
  - lifecycle bucket plumbing
- `tracker/trajectory.py`
  - storage of matched residual history
- `kalmanfilter/mamba_adaptive_kf.py`
  - shared trunk output redesign
  - class-specific head groups
  - ratio reconstruction path
- `training/train.py`
  - aligned residual-token input
  - aligned lifecycle training states
  - new loss stack and schedule
- `training/losses.py`
  - wrapped-Gaussian orientation likelihood
  - ratio anchor / bound losses
- `config/train_nuscenes.yaml`
  - 7-class windows
  - per-class per-family ratio bounds
  - warmup and loss weights
- new inference branch config(s)
  - isolated from the protected baseline config

## Non-Goals For The First Pass

The following are explicitly deferred:

- full generic MoE with learned routing over many experts;
- replacing the proven tracker shell;
- redesigning matching geometry as part of this step;
- claiming success via audit only without full nuScenes evaluation.

## Validation Plan

The redesign is only successful if it passes three layers.

### 1. Contract Validation

- training and inference use the same token semantics;
- `matched / unmatched / coast` are consistent end-to-end;
- coast behavior hard-degrades to prior-only.

### 2. Audit Validation

Compared with current `mamba`, the new branch should materially reduce:

- weak-class ratio explosions;
- `unmatched` `R_pos` inflation;
- `R_siz` and `R_ori` runaway behavior.

### 3. Metric Validation

The branch is considered viable only if:

- aggregated `AMOTA > 0.740`

with no unacceptable collapse in the strong classes and with improved weak-class behavior relative to the frozen `0.739` baseline.

## Recommendation

This design is the recommended first-pass Mamba redesign because it combines:

- the strongest ideas from `docs/restructure_paln.md`,
- the user's requirement for category-specific specialization,
- explicit lifecycle closure,
- adaptive windows,
- and a loss stack that is mathematically aligned with the live KF runtime.

It is intentionally narrower than a full MoE system, but much more likely to produce a real, safe metric gain.
