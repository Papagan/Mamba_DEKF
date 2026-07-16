# Inference-Style Association Cache Design

**Date:** 2026-07-16

## Goal

Build association training data directly from the real pre-association runtime
path. The exported cache must contain exactly the track and detection embeddings,
pair vectors, geometric costs, candidate ranks, gates, lifecycle states, and
assignment outcomes seen by matching. Enabling export must not change the frozen
production assignment or tracking result.

This is the second stage of the association optimization route. It produces data
for later retraining but does not retrain the association head and does not enable
learned association intervention.

## Problem

The current offline builder reconstructs predicted tracks and candidates from a
tracklet cache. It approximates runtime history, candidate geometry, and margin
mining. The current audit shows a strong train/inference distribution mismatch:
training positives are separable while runtime scores are often saturated near
zero or one, and most runtime candidates fall below the global score threshold.

The reconstruction cannot guarantee equality with:

- the state after `predict_before_associate()`;
- the actual residual-token or track-history input selected by the filter mode;
- the Mamba embeddings used during evaluation;
- the final seven-dimensional pair geometry vector;
- the real RO_GDIOU_3D cost matrix, geometric gate, and Hungarian candidate set;
- the lifecycle state and no-match cases present online.

The new cache therefore records runtime values instead of deriving substitutes.

## Non-Goals

- Do not change geometric cost calculation, gating, Hungarian assignment, track
  lifecycle updates, or output scores.
- Do not retrain Mamba or the association heads in this stage.
- Do not use the association score to alter production matching.
- Do not treat the production geometric assignment as ground truth.
- Do not inject nuScenes GT into candidate generation, scoring, or matching.
- Do not replace the existing offline cache builder; retain it for comparison.

## Approaches Considered

### Offline reconstruction

Extending `build_pairwise_association_cache.py` is minimally invasive, but it
cannot reproduce runtime embeddings, exact predicted states, matching gates, or
no-match decisions. This preserves the distribution mismatch and is rejected.

### Model-forward capture

Capturing tensors inside the association model provides exact embeddings and pair
vectors, but the model does not own geometric costs, gates, assignments, scene
context, or GT sidecars. Correlating these later would be fragile and is rejected.

### Matching-boundary read-only callback

The selected design stages pair tensors in `Base3DTracker` and emits complete
candidate groups at the matching boundary, where the untouched geometric cost
matrix and production assignment are available. This captures the real inference
distribution while keeping export observational.

## Architecture

### Shared pair feature composition

Move pair-vector composition into a small pure helper used by runtime scoring and
tests. Given track embeddings, detection embeddings, and the seven geometry
features, it returns:

```text
[track_embedding,
 detection_embedding,
 abs(track_embedding - detection_embedding),
 track_embedding * detection_embedding,
 center_distance,
 yaw_difference,
 size_l1,
 track_detection_score,
 candidate_detection_score,
 state_id,
 class_id]
```

The runtime association score and cache exporter consume the same returned pair
vector. No second implementation may reconstruct it independently.

### Runtime staging

`Base3DTracker` creates a frame-local payload after
`predict_before_associate()` and detection embedding extraction. It contains:

- track and detection indices;
- stable track id and frame context;
- class id/name and lifecycle state;
- track and detection embeddings;
- the exact pair vector for every same-class pair;
- detector scores and optional audit-only GT tokens.

The payload is passed into matching as optional audit metadata. When export is
disabled, no pair vectors are copied for cache storage and no callback is passed.

### Matching-boundary enrichment

After `cost_calculate_general()` and the existing baseline-prior calculation,
but before pairwise learned correction, matching enriches the staged payload with:

- real base geometric cost;
- finite/gated status;
- same-class candidate rank;
- best candidate cost and cost gap;
- class-specific geometric threshold;
- production-selected detection index after the unchanged solver;
- production match cost and unmatched outcome.

The callback runs only after the production assignment has been computed. Its
return value is ignored. It receives copies or immutable values and cannot mutate
the cost matrix.

### Accumulator and writer

`InferencePairCacheAccumulator` owns validation, bounded buffering, summaries,
state export, and deterministic multiprocess merging. Scene workers return
accumulator state through the existing worker-result path. The parent process
validates a runtime manifest before merging and atomically writes the final cache
and JSON summary.

## Cache Schema

The primary unit is a candidate group, not an independent flat pair. Grouped data
is required by the later listwise objective and preserves no-match examples.

Each group contains:

```text
schema_version
scene_id, sample_token, frame_id
track_id, class_id, class_name, state_bucket
track_embedding
track_gt_token                         # audit-only, nullable
production_det_index                  # nullable for no-match
production_match_cost                 # nullable for no-match
candidate_count, finite_candidate_count
candidates[]
```

Each candidate contains:

```text
det_index
det_embedding
pair_vector
geometric_cost
geometric_threshold
finite
inside_geometric_gate
candidate_rank
best_cost
cost_gap
association_score                     # observational only, nullable
production_selected
det_gt_token                           # audit-only, nullable
gt_label                               # 1, 0, or null when GT is unavailable
```

Group-level `gt_target_det_index` is set only when the track GT token and a unique
same-class candidate GT token agree. It is `null` for a true no-match target or
unknown GT, distinguished by `gt_target_status`:

- `matched`: a unique GT-positive candidate exists;
- `no_match`: track GT is known and no candidate carries that token;
- `unknown`: required GT identity is unavailable or ambiguous.

Production selection and GT labels remain separate fields. Training code must
not infer `gt_label` from `production_selected`.

## Configuration

Add a default-off section to inference configuration:

```yaml
INFERENCE_PAIR_CACHE:
  ENABLED: false
  OUTPUT_PATH: debug/inference_pair_cache.pkl
  SUMMARY_PATH: debug/inference_pair_cache_summary.json
  MAX_GROUPS_PER_CLASS_STATE: 0
  INCLUDE_EMBEDDINGS: true
  INCLUDE_PAIR_VECTORS: true
  REQUIRE_AUDIT_GT: false
  STRICT: false
```

`MAX_GROUPS_PER_CLASS_STATE: 0` means unlimited. A positive value applies a
deterministic per-class/state cap. `REQUIRE_AUDIT_GT: true` rejects unknown target
groups from the output but still counts them in the summary. `STRICT: false`
records export errors and preserves tracking; strict mode raises only after the
production assignment has been computed.

The runtime manifest records config hash, git commit, checkpoint hash, filter
mode, tracker compatibility mode, cost mode, pair-vector dimension, embedding
dimension, process count, and class map. States with incompatible manifests may
not merge.

## Data Flow

1. The tracker predicts active trajectories through the configured filter path.
2. The tracker computes track and detection embeddings exactly as used by the
   association head.
3. The pure pair composer creates the exact runtime pair vectors.
4. Matching computes the untouched base geometric cost matrix.
5. The existing production solver returns the frozen assignment.
6. Matching creates read-only candidate-group records from staged tensors, base
   costs, gates, assignments, and audit-only GT sidecars.
7. The scene accumulator validates and buffers groups.
8. Worker states merge deterministically in the parent process.
9. The parent atomically writes the pickle cache and JSON summary.

No exporter output feeds back into steps 1-5.

## Error Handling

- Missing embeddings, dimension mismatches, non-finite pair vectors, invalid
  class ids, duplicate track ids, or malformed GT sidecars are explicit errors.
- With `STRICT: false`, the affected group is skipped, an error counter and
  bounded sample are recorded, and production tracking continues.
- With `STRICT: true`, the exporter raises after production assignment exists.
- Cache writes use a temporary file in the destination directory followed by
  `os.replace`; partial files must not replace a previous valid cache.
- Merge rejects schema or manifest mismatches rather than silently mixing runs.
- The summary records seen, accepted, skipped, matched, no-match, unknown-GT,
  positive, negative, finite, gated, and selected counts by class and state.

## Production Invariance

The exporter is audit-only. The following invariants are mandatory:

- `INFERENCE_PAIR_CACHE.ENABLED: false` follows the existing path without
  constructing an accumulator or callback.
- Enabling export cannot change the cost matrix passed to production matching.
- Callback failures in non-strict mode cannot replace or modify matching output.
- Identical input with export disabled and enabled must produce identical matched
  indices, match costs, track lifecycle state, and final `results.json`.
- GT sidecars are read only after candidate generation and production assignment.

## Testing

### Unit tests

- Pair composition equals the vector consumed by the association head.
- Candidate rank, cost gap, finite/gated flags, and no-match status are correct.
- GT labels distinguish matched, no-match, unknown, and ambiguous cases.
- Invalid tensors and malformed metadata fail closed.
- Accumulator caps, summaries, atomic writes, and deterministic merges work.

### Integration tests

- Export-disabled and export-enabled matching return identical assignments and
  costs for Hungarian and compatibility paths.
- Callback mutation attempts cannot alter production matrices.
- Callback exceptions preserve production output in non-strict mode.
- Real tracker history and detection embeddings reach the exported pair vector.
- Multiprocess scene states merge to the same canonical cache independent of
  worker completion order.

### Regression tests

- Existing frozen matching, association shadow audit, checkpoint contract, and
  0.740 baseline configuration tests remain green.
- Existing offline pairwise builder and training loader remain supported.

## Acceptance Criteria

- A remote validation run produces a non-empty inference-style cache containing
  all seven classes when present in the evaluated scenes.
- Every exported pair vector has the checkpoint-declared dimension and can be
  passed directly to `ClassConditionedAssociationHeadBank`.
- Summary counts reconcile exactly with serialized groups and candidates.
- Matched, unmatched, production no-match, GT no-match, and unknown-GT groups are
  represented and separately counted.
- Export-enabled and export-disabled `results.json` files are byte-identical for
  the same code, config, checkpoint, process count, and input data.
- This stage does not alter AMOTA; its success is exact data capture and proven
  production invariance. AMOTA improvement is evaluated only after stage-three
  retraining and stage-four gated activation.

