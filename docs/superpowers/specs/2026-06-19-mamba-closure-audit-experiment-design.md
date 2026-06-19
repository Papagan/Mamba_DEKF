# Mamba Closure Audit Experiment Design

**Date:** 2026-06-19

**Status:** Draft for execution

## Goal

Run a controlled, audit-driven comparison of `pure_dekf`, `fusion`, and `mamba` under the same frozen tracking baseline so the project can answer, with evidence, whether current `mamba` inference behavior is numerically well-aligned with the live tracker/KF chain.

This experiment is a prerequisite for two later changes:

1. closing the `mamba` inference loop in a numerically disciplined way; and
2. redesigning `TRACK_SCORE` as a dirty-track suppressor rather than a ranking override.

## Frozen Baseline

The experiment must not modify the protected Route-A reference line:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- aggregated `AMOTA 0.737`

All experiment configs must be derived from this baseline and may only differ in:

- `FILTER_MODE`
- opt-in audit output paths

No threshold, lifecycle, noise prior, or exact-update settings may change during this experiment.

## Questions To Answer

The experiment must produce enough evidence to answer:

1. Does `mamba` generate effective `Q/R` values whose scale differs materially from `pure_dekf` priors?
2. Does `fusion` damp or amplify those scale differences?
3. Are the differences concentrated in specific:
   - classes
   - matched vs unmatched states
   - noise families (`q_pos`, `r_pos`, `r_siz`, `r_ori`)?
4. Are the differences large enough to justify explicit bounded-residual fusion or training-time scale constraints?

## Scope

This phase only covers:

- experiment configs for the three modes
- inference-side audit output generation under matched/unmatched split
- optional training-side audit preparation already completed in the previous phase
- post-run summary/compare tooling for the audit JSON files

This phase does **not**:

- change live tracking decisions
- change thresholds
- change lifecycle logic
- change `TRACK_SCORE`
- retrain models

## Experiment Design

### Modes

The experiment compares three runs derived from the same frozen baseline:

- `pure_dekf`
- `fusion`
- `mamba`

Each run must use:

- the same exact-hybrid baseline config values
- the same exact matched/unmatched compatibility settings
- the same noise priors and thresholds
- the same dataset split and evaluation route

The only allowed differences are:

- `FILTER_MODE`
- `AUDIT.NOISE_AUDIT.ENABLED`
- audit output paths

### Audit Outputs

Each mode must produce:

- one inference audit JSON

The outputs must remain schema-compatible with the shared audit module:

- top-level `schema_version`
- top-level `families`
- per-bucket:
  - `split`
  - `mode`
  - `class_id`
  - `class_name`
  - `state`
  - `history_len`
  - `count`
  - `families`
  - `ratios`

### Comparison Dimensions

The first analysis pass must report, for each mode:

- per-class `median` and `p90` for each family
- per-class `median` and `p90` for ratio-to-prior
- matched/unmatched split

The first cross-mode compare must emphasize:

- `bicycle`
- `motorcycle`
- `bus`
- `trailer`
- `truck`

because prior tuning suggests these classes are most sensitive to noise-scale behavior and lifecycle coupling.

## Recommended Execution Order

### Step 1: Build fair experiment configs

Create three configs that inherit all effective settings from the frozen baseline and only flip `FILTER_MODE` plus audit output destinations.

### Step 2: Run a fast subset first

Use a fixed, representative scene subset for the first pass so the project can see numeric drift quickly before spending a full validation run.

The subset should remain fixed across the three modes.

### Step 3: Summarize the audit outputs

Summarize mode-by-mode and cross-mode differences in a stable text or JSON report so later design decisions do not depend on manual inspection.

### Step 4: Decide the next control strategy

Use the findings to choose one of:

- bounded residual fusion
- training-time ratio anchors
- class-conditioned noise constraints
- no constraint, if scales are already well aligned

## Success Criteria

The experiment succeeds if it produces a trustworthy answer to:

- whether `mamba` and `fusion` differ materially in scale from `pure_dekf`
- where the mismatch lives
- whether the mismatch is systematic enough to justify explicit constraints

Success in this phase is **evidence quality**, not an immediate `AMOTA` gain.

## Non-Regression Contract

The experiment must preserve these guarantees:

- the existing frozen baseline config stays untouched
- audit remains opt-in
- no live tracking behavior changes when audit is disabled
- any new helper scripts or configs operate on copies/derived configs only

## Follow-Up Decision Rule

After the three-mode audit results are available:

- if `mamba / prior` and `fusion / prior` ratios are tightly centered and stable, focus next on tracker update/lifecycle semantics rather than scale constraints
- if `mamba` drifts but `fusion` stabilizes it, consider bounded residual fusion before changing training
- if `fusion` amplifies class-dependent scale mismatch, prioritize explicit bounded fusion and training-side ratio anchors
- only after that should `TRACK_SCORE` be redesigned as a dirty-track suppressor
