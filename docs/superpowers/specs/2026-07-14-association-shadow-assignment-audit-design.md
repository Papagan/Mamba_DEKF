# Association Shadow Assignment Audit Design

## Objective

Build a decision-level audit for the class-conditioned Mamba association head without changing the frozen tracking result. The first implementation phase must preserve the current geometric assignment path exactly while computing, in shadow, what the learned association correction would have changed.

The audit must answer three questions that the current pair-level summary cannot answer:

1. Which Hungarian assignments would change after the learned correction?
2. Did each changed assignment improve or damage the GT-consistent association?
3. Which class/state buckets have enough calibrated evidence to justify a later A/B activation?

This phase does not rebuild pair caches, retrain the association head, change the geometric cost, or activate new classes in the production assignment path.

## Baseline Safety Contract

The frozen reference is the configuration and checkpoint combination that produces `AMOTA 0.740`. Association experiments must use a separate experimental configuration. The reference configuration must remain unchanged.

Shadow mode has the following invariants:

- The official matcher receives the original geometric cost matrix.
- The official match result and trajectory lifecycle are computed only from the original matrix.
- The corrected cost matrix is used only by a second, side-effect-free assignment call.
- Disabling the audit or enabling shadow mode must produce identical `results.json` content for the same input and process count.
- The audit must fail closed: missing scores, malformed shapes, missing metadata, or audit export failures must not modify matching.

## Correct Semantics

`track_state_bucket` describes the trajectory lifecycle before the current association step:

- `matched`: the trajectory was not coasting before this frame.
- `unmatched`: the trajectory had one or more previous unmatched frames.

It is not a pair label. Pair correctness is represented separately:

- `pair_label=1`: the track and detection refer to the same GT instance.
- `pair_label=0`: both identities are known and differ.
- `pair_label=null`: GT identity is unavailable or ambiguous.

The comparison utility must not infer pair correctness from `track_state_bucket`.

## Runtime Manifest

Every merged audit JSON must contain a `runtime_manifest` with:

- normalized absolute evaluation config path;
- SHA-256 of the loaded YAML content;
- git commit hash and dirty-worktree flag;
- association checkpoint path and SHA-256;
- association-head architecture fields used to load the checkpoint;
- effective `ACTIVE_CLASS_STATES` after YAML parsing;
- `APPLY_MODE`, `COST_MARGIN_EPS`, `MIN_SCORE`, `ALPHA`, and `MAX_DELTA`;
- filter mode, tracker compatibility mode, BEV matching mode, and per-class BEV cost modes;
- process count and audit schema version.

Scene states with different runtime manifests must not be merged. In strict mode this is an error; otherwise the merged audit records the mismatch and omits decision-level conclusions.

## Shadow Assignment Data Flow

For each BEV matching call:

1. Build the original geometric cost matrix exactly as today.
2. Compute association scores using the existing online track and detection embeddings.
3. Build a corrected matrix with the existing bounded association correction.
4. Run the configured assignment algorithm once on the original matrix for production.
5. Run the same assignment algorithm once on the corrected matrix for shadow analysis.
6. Compare assignments per class and emit one decision record per affected track, plus aggregate counters for unchanged tracks.

The shadow assignment helper must be pure with respect to trajectories, detections, filters, lifecycle counters, and output results.

## Decision Records

Each decision record contains:

- scene and sample token;
- class id and class name;
- track id and `track_state_bucket`;
- geometric selected detection index or `null`;
- shadow selected detection index or `null`;
- geometric and shadow selected costs;
- selected association scores;
- geometric candidate rank, shadow candidate rank, and original cost gap;
- whether the assignment changed;
- whether a match crossed the class threshold in either direction;
- track GT instance token, detection GT instance tokens, and derived pair labels when available;
- outcome: `improved`, `damaged`, `neutral`, or `unknown`.

An `improved` decision changes an incorrect or absent geometric selection into the GT-consistent detection. A `damaged` decision changes a GT-consistent geometric selection into an incorrect or absent selection. Unknown GT identities must never be counted as improved or damaged.

## GT Identity Attachment

The initial audit supports optional GT identity lookup. The tracker remains detector-driven; GT metadata is audit-only and must not enter cost calculation or lifecycle updates.

The lookup interface is keyed by scene/sample token and detection index. If a direct detection-to-GT mapping is unavailable, the audit records `pair_label=null` rather than reconstructing identities with a new heuristic inside the matcher. A later runtime-replay cache phase may provide the authoritative mapping generated by the aligned CenterPoint tracklet cache.

## Aggregates

The merged JSON reports per class and track state:

- matching calls and candidate pairs;
- active and penalized pairs;
- assignment changes;
- changed-to-match, changed-to-unmatched, and changed-detection counts;
- threshold crossings;
- known-GT decisions;
- improved, damaged, and neutral decisions;
- net improvement (`improved - damaged`);
- harm precision (`damaged / known changed decisions`);
- score and cost-gap summaries for geometric winners, shadow winners, improved decisions, and damaged decisions.

Raw decision samples remain bounded by configuration, but aggregate counters are exact. Multiprocess merging must preserve configured samples instead of silently dropping them.

## Activation Gate

No class/state bucket is eligible for real matching intervention unless all conditions hold:

- shadow mode reproduces the frozen production result exactly;
- at least 100 known-GT changed decisions are observed, unless the complete validation split contains fewer;
- `improved > damaged` and net improvement is positive;
- damaged decisions are below 20% of known changed decisions;
- the evidence is reproduced in a second run with the same manifest;
- a one-class/state A/B evaluation raises aggregate AMOTA above the exact frozen baseline, not only the rounded display value.

Car remains disabled in the first activation experiment because the latest run regressed its AMOTA from `0.865` to `0.856`. Class activation order is determined by shadow net improvement, not by offline AUC.

## Comparison Utility Correction

The existing train/inference comparison remains useful for feature-distribution summaries, but its diagnoses must be renamed and narrowed:

- `matched` and `unmatched` are reported as lifecycle states only;
- no `likely_distribution_mismatch` diagnosis may treat matched-state candidates as positive pairs;
- safe-activation recommendations require decision-level GT outcomes from the shadow audit;
- absent GT outcomes produce `insufficient_decision_evidence`.

## Testing Strategy

Unit tests cover:

- shadow mode returns the unmodified production matrix and production assignment;
- corrected assignment is computed independently and can differ in the audit result;
- lifecycle state and pair label remain separate;
- assignment outcome classification handles match changes, threshold crossings, unknown GT, and no-match cases;
- runtime manifests are deterministic and reject incompatible merges;
- multiprocess audit merging preserves bounded samples;
- comparison diagnostics no longer interpret lifecycle matched as a positive pair.

An integration test uses a small synthetic two-track/two-detection cost matrix where the learned correction changes one Hungarian assignment. It verifies that the production assignment remains geometric while the shadow audit reports the change.

## Follow-On Phases

After this audit is validated, a separate design and plan will cover exact runtime-replay cache generation and listwise association-head training. That phase will dump the actual online pair vectors and exact `RO_GDIOU_3D` candidate ordering instead of reconstructing fusion histories offline.
