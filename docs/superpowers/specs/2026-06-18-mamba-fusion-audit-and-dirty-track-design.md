# Mamba/Fusion Numeric Audit And Dirty-Track Suppressor Design

**Date:** 2026-06-18

**Status:** Draft for user review

## Goal

Protect the frozen `AMOTA 0.737` baseline while preparing two higher-value follow-up changes:

1. close the `mamba` inference loop so `mamba`, `pure_dekf`, and `fusion` operate on explicit, auditable numerical semantics rather than loosely coupled behavior; and
2. replace the current `TRACK_SCORE` ranking override idea with a dirty-track suppressor that only penalizes obviously noisy trajectories.

This spec only covers the **audit and instrumentation phase** required before either behavioral change is attempted.

## Baseline And Non-Goals

### Frozen baseline

The immutable reference baseline is:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Aggregated `AMOTA 0.737`

All work in this spec must preserve the behavior of that baseline unless an opt-in audit flag is explicitly enabled.

### Non-goals for this phase

This phase does **not**:

- change tracking decisions
- introduce new losses
- retrain models
- enable a new `TRACK_SCORE` mode
- revise ByteTrack further

The output of this phase is evidence, not a new metric claim.

## Problem Statement

Current evidence suggests that:

- `mamba` affects tracking indirectly through `Q/R`, but the main association chain remains geometric and lifecycle-driven.
- `fusion` likely mixes a stable hand-crafted prior (`pure_dekf`) with a learned residual (`mamba`) whose numerical scale may drift across classes, windows, and matched/unmatched states.
- the current `TRACK_SCORE=true` path behaves like a reranker, not a true denoiser, because it modifies output ordering without synchronizing the matching chain.

Before changing inference behavior or training objectives, the project needs a single, aligned view of:

- what `Q/R` values are actually consumed at inference time
- how those values differ from the `pure_dekf` prior
- whether the differences are class-dependent or state-dependent
- whether `fusion` introduces large ratio or scale mismatches

## Scope

This phase adds a unified numerical audit path with matched/unmatched state awareness and shared summary format for both training and inference.

The first implementation must support:

- per-class aggregation
- matched vs unmatched state split
- `mamba`, `pure_dekf`, and `fusion` mode comparison
- ratio-to-prior statistics
- history/window metadata where available

## Design Summary

### Recommended approach

Implement a shared audit schema and two opt-in producers:

1. an inference-side producer that records the effective `Q/R` values actually consumed by the tracker/KF path
2. a training-side producer that records the same families of values during model training/logging

Both producers write the same logical structure so later analysis can compare training and inference without ad hoc scripts.

### Alternatives considered

#### Option A: inference-only audit

Pros:

- lowest implementation effort
- directly answers whether inference-time scales are unstable

Cons:

- later explicit training constraints would still need a second instrumentation pass
- no clean way to compare train/infer scale drift

#### Option B: training-only audit

Pros:

- useful for future loss design

Cons:

- does not tell us what the tracker actually consumes at runtime
- cannot diagnose fusion instability in the live chain

#### Option C: shared train + inference audit

Pros:

- single schema, single analysis path
- supports later bounded-residual fusion design
- supports later class-conditional and window-aware constraints
- best fits the project’s “do not break current baseline” requirement

Cons:

- slightly more implementation work

**Recommendation:** Option C.

## Data To Capture

### Common metadata

Every audit record or aggregate bucket must be attributable by:

- mode: `mamba`, `pure_dekf`, or `fusion`
- class id / class name
- state: `matched` or `unmatched`
- split source: `train` or `infer`

Where available, also capture:

- history length / effective window length
- scene/run identifier or iteration identifier

### Noise statistics

The first version only needs summaries for the families already meaningful to the project:

- position/process noise
- position/measurement noise
- size/measurement noise
- orientation/measurement noise

For each family, aggregate:

- count
- mean
- median
- p90
- min
- max

### Prior-relative statistics

To answer whether `mamba` and `fusion` diverge numerically from `pure_dekf`, capture ratio-to-prior summaries in log-safe form:

- `value / prior`
- or equivalently `log(value / prior)`

The summary should retain enough information to identify:

- classes that are systematically over-confident
- classes that are systematically under-confident
- state splits that spike only in unmatched/coast-like situations

## Inference-Side Design

### Behavior

Inference audit is opt-in and must be fully disabled by default.

When enabled, the tracker/KF path records the **effective** noise values actually used by the live run, not merely raw model outputs. This is important because later fusion logic may clip, mix, or fall back.

### Collection points

Audit should be emitted from the point where `predict_with_mamba()` or equivalent live noise selection has already resolved:

- which mode is active
- what prior is used
- what final `Q/R` values are consumed

The design must capture both:

- the final consumed values
- the reference prior used to compute ratios

### State split

The minimum required split is:

- matched
- unmatched

This is intentionally smaller than a full `matched/unmatched/coast/rescue` taxonomy so the first implementation stays cheap and low risk. If future dirty-track work needs more detail, the schema can extend without changing current fields.

### Output format

The first implementation should prefer lightweight JSON artifacts in a deterministic location under `debug/` or a caller-specified output path. The format should be machine-readable and stable enough to diff across runs.

## Training-Side Design

### Behavior

Training-side audit is also opt-in and must not affect optimization.

It only records the same logical statistics so later work can compare:

- train distribution vs inference distribution
- short-window vs long-window behavior
- class-specific scale drift

### Minimum metadata

Training aggregates must include:

- class id / class name
- mode / branch identity where relevant
- window/history length
- prior-relative ratios

### Why this matters

If `fusion` scale mismatch is real, later explicit constraints should be evidence-driven. The training audit provides the evidence needed to decide whether to add:

- ratio clamp / anchor loss
- class-conditional anchors
- window-aware consistency penalties

This phase does not implement those constraints yet.

## Dirty-Track Suppressor Preparation

This spec does not implement the suppressor, but the audit must be designed to support it.

The future suppressor will use online-visible signals only and should penalize obviously dirty tracks rather than reranking all tracks. Likely candidate features include:

- repeated low-score matches
- repeated fake/coast updates
- abnormally high recent match cost
- uncertainty spikes relative to class prior

The current audit therefore needs to preserve uncertainty summaries in a way that can later be joined to track cleanliness heuristics.

## Error Handling And Safety

Audit must fail safe:

- if disabled, it adds effectively no behavior change
- if enabled but output writing fails, tracking should continue unless the caller explicitly requests strict mode
- missing metadata such as window length should degrade gracefully to `null`/omitted fields, not crash tracking or training

## Testing Strategy

The first implementation requires:

1. unit tests for shared statistic aggregation
2. unit tests that disabled audit is a no-op
3. unit tests that inference-side records preserve class/mode/state metadata
4. unit tests that training-side logging emits the same schema family
5. one lightweight integration test or fixture-level check that ratio-to-prior fields are present and numerically sane

No benchmark or full re-evaluation is required in this phase, because behavior must remain unchanged.

## Success Criteria

This phase is successful when:

- the frozen `0.737` baseline configuration behaves identically with audit disabled
- both training and inference can emit aligned numerical summaries
- we can answer, from recorded artifacts, whether `fusion` exhibits class- or state-dependent scale mismatch relative to `pure_dekf`
- the recorded outputs are sufficient to design the next two phases:
  - bounded or constrained `mamba/fusion` inference semantics
  - a dirty-track suppressor that penalizes only obviously noisy trajectories

## Follow-Up Phases Enabled By This Spec

Once this audit exists, the next phases can proceed with evidence:

1. redefine `fusion` as bounded residual noise correction if the audit shows scale drift
2. add explicit training constraints if ratios are unstable by class or window length
3. redesign `TRACK_SCORE` into a dirty-track suppressor using online-visible uncertainty and cleanliness signals
