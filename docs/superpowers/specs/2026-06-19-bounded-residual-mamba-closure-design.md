# Bounded Residual Mamba Closure Design

**Date:** 2026-06-19

**Status:** Draft for review

## Goal

Redefine the `mamba` filter path so that inference and training consume the same noise semantics: `mamba` must no longer act as an unconstrained generator of final `Q/R` tensors, but instead as a **bounded residual corrector around the `pure_dekf` prior**.

The design must preserve the current frozen baseline while creating a new branch that can only be considered successful if:

- aggregated `AMOTA >= 0.740`

Anything below that threshold does **not** replace the current mainline.

## Frozen Baseline

The protected reference line remains:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- aggregated `AMOTA 0.737`

This baseline must stay untouched throughout the closure work.

All `mamba` closure experiments must be derived from new branch configs and compared back to this baseline.

## Problem Statement

The completed audit phase showed that current `mamba` behavior is not numerically well aligned with the live tracker/KF chain.

The dominant failure mode is not general mild drift; it is **state-specific scale explosion**, concentrated especially in:

- `unmatched` state
- `R_pos`

and secondarily in:

- `R_siz`
- `Q_pos`

The effect is class dependent. The most unstable classes are:

- `bicycle`
- `motorcycle`
- `trailer`
- `truck`
- `car`

Meanwhile, `fusion` generally stays much closer to `pure_dekf` than raw `mamba` does, which suggests:

- the baseline prior is still the right anchor, and
- the learned signal should be treated as a residual correction, not as an unconstrained replacement.

## Design Objectives

The closure redesign must satisfy four conditions simultaneously:

1. **Inference/Training semantic alignment**
   - the same conceptual quantity must be trained and consumed at inference time;
   - no post-hoc clamp should exist that the model was never trained to anticipate.

2. **Per-class sensitivity control**
   - different categories can tolerate different residual ranges.

3. **Per-state sensitivity control**
   - `matched` and `unmatched` trajectories must not share the same noise freedom;
   - `unmatched` must be more conservative.

4. **Non-regression discipline**
   - the current `0.737` baseline remains the production-safe path;
   - closure changes live behind dedicated configs until they hit `AMOTA >= 0.740`.

## Proposed Approach

### Core Idea

Replace the current free-form learned noise path with:

- `prior noise` from `pure_dekf`
- multiplied by a **bounded residual ratio** predicted by `mamba`

Conceptually:

- `Q_final = Q_prior * ratio_q`
- `R_final = R_prior * ratio_r`

where each ratio is:

- learned,
- class-aware,
- state-aware,
- and explicitly bounded.

This means `mamba` no longer defines the absolute scale of the filter by itself. It only says how much to adjust the prior, within a trusted envelope.

### Families In Scope

The first closure pass applies this bounded-residual treatment to:

- `Q_pos`
- `R_pos`
- `R_siz`
- `R_ori`

Although `R_ori` was not the strongest `mamba`-specific failure signal in the audit, it must be brought into the same semantic framework so that the model and inference path do not treat orientation noise as a separate legacy special case.

### State Buckets

The closure design uses exactly two state buckets:

- `matched`
  - track had `unmatch_length == 0` before prediction
- `unmatched`
  - track had `unmatch_length > 0` before prediction

This matches the existing tracker structure and the audit outputs already produced.

The system must enforce stricter residual bounds for:

- `unmatched`

than for:

- `matched`

especially for:

- `R_pos`

## Class Profiles

The first implementation should not require one hand-written bound table per class for every family. Instead it should group classes into stable profiles:

- `stable_large`
  - `car`
  - `bus`
- `agile_weak`
  - `bicycle`
  - `motorcycle`
- `heavy_long`
  - `trailer`
  - `truck`
- `human`
  - `pedestrian`

Each profile must define separate residual envelopes for:

- `matched`
- `unmatched`

and for each family:

- `q_pos`
- `r_pos`
- `r_siz`
- `r_ori`

### Envelope Direction

The audit already implies the first-pass policy:

- `unmatched r_pos` gets the strictest caps
- `unmatched r_siz` gets the second strictest caps
- `q_pos` gets moderate caps
- `r_ori` is bounded too, but with looser first-pass envelopes than `r_pos`

The exact bound values should be data-driven from the audit summary, not chosen ad hoc.

## Inference Path Redesign

### Current Behavior

Right now the filter path behaves roughly like this:

1. `mamba` predicts full noise tensors
2. `pure_dekf` provides optional priors
3. `fusion` applies a trace-gate blend after the fact

This allows `mamba` to produce values far outside the prior scale and forces `fusion` to act as a rescue mechanism.

### New Behavior

The inference path should be redefined as:

1. compute `pure_dekf` prior noise tensors
2. compute raw `mamba` outputs
3. convert raw outputs into residual ratios relative to the prior
4. map each sample into:
   - class profile
   - state bucket
5. bound the ratios according to that profile/state/family envelope
6. reconstruct final `Q/R` from:
   - `prior × bounded_ratio`
7. pass only these bounded tensors into KF predict/update

### Mode Semantics

The runtime modes should become:

- `pure_dekf`
  - unchanged reference prior-only mode
- `mamba`
  - bounded residual mode
- `fusion`
  - bounded residual mode plus a conservative blend around the same prior anchor

This removes the current ambiguity where `mamba` and `fusion` behave like two partially overlapping but differently interpreted learned-noise modes.

## Training Alignment

Inference-side bounding alone is not enough. The model must be trained against the same semantics it will face at inference.

### Required Training Change

Training must introduce explicit residual-scale regularization around the same prior anchor used at inference.

The first pass should add:

1. **Ratio anchor loss**
   - regularize the learned residual ratios for:
     - `Q_pos`
     - `R_pos`
     - `R_siz`
     - `R_ori`
   - use a log-space penalty relative to the prior

2. **Class-conditioned margins**
   - `agile_weak` and `heavy_long` classes must use tighter tolerated ranges than `stable_large`

3. **State-conditioned margins**
   - `unmatched` must use stricter penalties than `matched`

4. **Window-aware consistency**
   - shorter valid history windows must not be allowed to become more overconfident than longer-history examples

This training work does **not** require a MoE backbone in the first pass. It only requires the model to learn bounded residual behavior.

## Why This Is Better Than The Alternatives

### Better Than Pure Post-Hoc Clamp

A hard clamp at inference time would prevent catastrophic explosions, but it would not align the model with the runtime contract. The model would continue learning outputs that the runtime later discards.

### Better Than Immediate MoE / Multi-Head Rewrite

A full class-specialized or state-specialized architecture would increase complexity too early. The current evidence does not yet justify that level of model change. The system first needs a disciplined residual contract.

### Better Than Redesigning `TRACK_SCORE` First

`TRACK_SCORE` is downstream of the filter path. If the noise semantics are still unstable, a dirty-track suppressor would be compensating for an unresolved primary instability. The main filter contract must be fixed first.

## Validation Plan

### Numeric Success

The first validation layer is audit-based, not metric-based.

The redesign should eliminate the largest audit anomalies, especially:

- `unmatched r_pos`

for:

- `truck`
- `car`
- `trailer`
- `motorcycle`

The expected first-pass outcome is:

- raw `mamba/prior` ratio medians and p90 values move substantially closer to `fusion/prior`
- `unmatched` no longer exhibits multi-x uncontrolled `R_pos` inflation

### Metric Success

The closure branch is only considered successful if:

- aggregated `AMOTA >= 0.740`

Anything below that threshold fails the success gate and does not replace the current frozen baseline.

### Non-Regression Rules

During development:

- the `0.737` frozen baseline stays untouched
- all closure work stays in separate configs/branches
- no dirty-track suppressor work begins until the bounded-residual closure experiment is complete

## Follow-Up Order

Once bounded-residual `mamba` closure is implemented and evaluated:

1. if `AMOTA >= 0.740`
   - keep the closure branch
   - then redesign `TRACK_SCORE` as a dirty-track suppressor
2. if numeric audit improves but `AMOTA < 0.740`
   - inspect whether the remaining issue is:
     - lifecycle/update timing
     - over-conservative envelopes
     - training regularization strength
   - do not yet move to `TRACK_SCORE`
3. if the closure branch fails both numerically and metrically
   - revert to frozen baseline
   - reassess whether stronger architectural specialization is needed

## Scope Guardrails

This design does **not** include:

- MoE backbone changes
- ByteTrack revival
- global threshold retuning
- reranking-style `TRACK_SCORE`

Those are separate downstream efforts and must remain blocked until the `mamba` closure branch is resolved against the `0.740` success gate.
