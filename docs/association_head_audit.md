# Association Head Audit

The frozen baseline keeps `MAMBA_ASSOCIATION_HEAD.ENABLED: false` and must not
be edited. The learned association head must first run in shadow mode, where it
is scored and audited without changing production assignments.

## Why Audit First

The current pairwise association head can achieve very low training loss while
still hurting AMOTA, because the training pair features and inference pair
features are not yet fully aligned. In particular, the old cache uses a
future-detection to candidate-detection geometry, while inference applies the
head to track-state to candidate-detection pairs.

## Safe Shadow Run

Create a separate remote experimental YAML from the frozen evaluation YAML and
add this block. Do not change the frozen YAML itself:

```yaml
MAMBA_ASSOCIATION_HEAD:
  ENABLED: true

ASSOCIATION_HEAD_AUDIT:
  ENABLED: true
  SHADOW_MODE: true
  INFER_OUTPUT_PATH: "debug/association_head_shadow_audit.json"
  MAX_SAMPLES_PER_BUCKET: 32
  STRICT: true
```

Run the normal evaluation remotely with the separate shadow config:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_association_head_shadow_audit.yaml \
  -p 12
```

With `SHADOW_MODE: true`, the produced `results.json` must reproduce the frozen
0.740 baseline `results.json` byte-for-byte and therefore have the same hash.
Check both before interpreting the audit:

```bash
cmp /path/to/frozen/results.json /path/to/shadow/results.json
sha256sum /path/to/frozen/results.json /path/to/shadow/results.json
```

Copy the remote audit JSON and compare it with the rebuilt training cache:

```bash
python tools/compare_pairwise_assoc_train_infer.py \
  --train-cache /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --infer-audit debug/association_head_shadow_audit.json \
  --output debug/association_shadow_decision_report.json \
  --min-known-changed 100
```

Do not overwrite the historical JSON files under `docs/`. In particular, the
old `docs/pairwise_assoc_train_infer_compare.json` is not schema v2 decision
evidence and cannot authorize an activation experiment.

The compare report preserves train-cache class balance, hard-negative, and
distance summaries. Its `track_state_scores` section reports per-class,
per-track-state score and cost distributions only:

- `pair_count`: same-class finite pairs scored by the head.
- `active_pair_count`: pairs in configured active class/state buckets.
- `penalized_pair_count`: active pairs that received a positive cost delta.
- `penalized_ratio`: `penalized_pair_count / active_pair_count`.
- `avg_score`: mean association score.
- `avg_delta`: mean cost increase applied to the geometric cost.
- `avg_cost_before` and `avg_cost_after`: cost shift caused by the head.
- `score_quantiles`, `delta_quantiles`, `cost_before_quantiles`, and
  `cost_after_quantiles`: observed distribution summaries.

Track lifecycle states such as `matched` and `unmatched` are not positive or
negative labels. `track_state_scores` must never be used to infer a distribution
mismatch or to recommend activation.

## Schema v2 Decision Gates

Activation diagnostics use only schema v2 `decision_buckets`. The activation
unit is a `(class_id, class_name, track_state_bucket)` class/state tuple, not a
class pooled across `matched` and `unmatched`. Buckets for the same class/state
are aggregated, then each state independently applies:

- `known_changed_count == 0`: `insufficient_decision_evidence`.
- Fewer than 100 known changed decisions: `shadow_decisions_not_safe`.
- At least 100 known changed decisions, `improved_count > damaged_count`, and
  `harm_precision <= 0.20`: `candidate_for_shadow_ab_activation`.
- Any other complete schema v2 result: `shadow_decisions_not_safe`.

The report exposes these results as `decision_evidence_by_state` and
`diagnosis_by_state`. Class-level activation scope comes only from
`runtime_manifest.active_class_states`; observed buckets do not define or
expand production activation. `runtime_manifest.class_id_to_name` is the
canonical class mapping. Its keys and `active_class_states` keys must be
canonical decimal class IDs in the bounded range `0..2147483647`; each
configured state must be exactly `matched` or `unmatched`.

The comparator checks every training-cache `(class_id, category)`, pair-bucket
`(class_id, class_name)`, and decision-bucket `(class_id, class_name)` against
that canonical mapping. Unknown IDs, mismatched names, malformed or oversized
numeric keys, duplicate identities, and missing mappings are reported in
`class_identity_validation`. Any such error makes decision evidence and
class-level activation unusable. The CLI still writes a fail-closed report
rather than authorizing a candidate or crashing on an oversized key.

Every manifest-active class and state is required for a class-level candidate.
The report includes active classes even when train, pair, and decision inputs
never observed them. A configured state with no decision bucket receives an
explicit zero-count `decision_evidence_by_state` entry and
`insufficient_decision_evidence` in `diagnosis_by_state`. The class is a
candidate only when every configured active state independently reaches the
known-changed minimum and is a candidate. An observed state that is not active
in the manifest remains visible in the state-level report, but it neither
blocks nor extends the configured production activation scope. Class-level
pooled counts remain descriptive and are never used to bypass a failed state
gate.

Missing or malformed `active_class_states` fails closed only at class level:
valid observed states may still retain state-level conclusions, but no class
can be a candidate. Empty class state lists and classes omitted from the map
also cannot produce a class-level candidate. The report records normalized
scope and validation failures in `class_state_activation`. Safely parsed active
entries remain visible with insufficient evidence even when another scope key
is malformed; retaining them for reporting never restores class-level use.

The 100 known-changed minimum is a hard safety floor. The
`--min-known-changed` option may raise it for a stricter experiment but rejects
values below 100.

Schema v1 and malformed schema v2 input fail closed as
`insufficient_decision_evidence`; no class/state can produce a candidate. A
usable top level requires native integer `schema_version: 2`, native boolean
`true` values for availability, completeness, and manifest compatibility, a
mapping `runtime_manifest`, an optional `manifest_initialized` that is exactly
`true`, an empty list of incomplete reasons, and a non-empty decision bucket
list. `audit_errors` must contain non-negative native integer `count` and
`sample_count` values plus a `samples` list, with
`sample_count == len(samples) <= count`; an optional top-level
`audit_error_count` must agree. Any recorded audit error also fails closed.

Each decision bucket requires a bounded non-negative native integer `class_id`,
the exact canonical `class_name`, a `matched` or `unmatched` state, and every
production counter as a non-negative native integer. Outcome counts, known-GT
counts, changed-assignment categories, threshold crossings, and known-changed
counts must satisfy the schema v2 accounting identities. Optional
`net_improvement` and `harm_precision` must be finite and consistent with those
counters. Numeric strings, booleans used as integers, negative values, and
inconsistent derived metrics are rejected rather than coerced or clipped.
Rejection reasons are reported in `decision_evidence.incomplete_reasons`.

Unknown GT identities are retained as unknown evidence and cannot count as
`improved` or `damaged`, so sparse or unavailable GT can prevent the minimum
known-changed gate from being reached.

### Audit-only GT enrichment

GT enrichment is implemented as an explicit, default-off nuScenes preprocessing
mode. Generate a separate audit dataset rather than replacing the normal
converted detections:

```bash
python preprocess/convert_nuscenes.py \
  --raw_data_path /path/to/nuscenes/raw_data \
  --dets_path data/nuscenes/detectors \
  --detector centerpoint \
  --save_path data/base_version/nuscenes_audit_gt \
  --split val \
  --audit_gt_enrichment
```

The converter matches detections to same-class GT annotations within the audit
distance gate and writes aligned `audit_gt_tokens` as a sidecar. Filtering and
NMS keep that sidecar aligned; GT identities are not copied into detection or
`BBox` payloads. The tracker carries the sidecar only into the shadow decision
comparison so known changed assignments can be classified as improved,
damaged, or neutral. Running conversion without `--audit_gt_enrichment` keeps
the prior behavior and normally leaves `known_changed_count` at zero.

GT must never be supplied to candidate generation, scoring, matching, or
assignment selection. It only annotates outcomes after production and shadow
decisions have both been made. Use the separately converted audit dataset only
for the shadow audit, never for baseline or intervention measurements.

A candidate authorizes only a separate A/B experiment, not default activation.
Real assignment intervention requires another experimental config with
`ASSOCIATION_HEAD_AUDIT.SHADOW_MODE: false`. Compare that run against both the
frozen 0.740 baseline and the byte-identical shadow run; never reuse or modify
the frozen config for intervention.

## Rebuild Pairwise Cache After Audit Failure

If the audit shows nearly all pairs being penalized, rebuild the pairwise cache
with inference-aligned features:

```bash
python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --summary-output docs/train_pairwise_assoc_train_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000

python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/val_pairwise_assoc.pkl \
  --summary-output docs/train_pairwise_assoc_val_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000
```

This mode changes the training pair features from the old
future-detection/candidate-detection geometry to predicted-track/candidate
geometry. It also stores `candidate_history_12` in the same detection-token
format used by inference (`x=y=0` for the one-frame detection token).
The `inference_margin` negative miner first samples negatives from the same
near-best candidate set used by `MAMBA_ASSOCIATION_HEAD.APPLY_MODE:
margin_tiebreak`. If that set is empty, `--min-hard-negatives` falls back to the
nearest same-class candidates. This avoids both previous failure modes:
easy-negative shortcuts where negatives were meters away, and all-positive
caches where large classes had no useful negatives.
