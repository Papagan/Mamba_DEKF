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
  --output docs/pairwise_assoc_train_infer_compare.json \
  --min-known-changed 100
```

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

Activation diagnostics use only schema v2 `decision_buckets`. The report states
the input `infer_audit_schema_version`, source evidence status, effective
usability, manifest compatibility, audit error count, and configured gates.
Per class, decision buckets are combined across track states before applying:

- `known_changed_count == 0`: `insufficient_decision_evidence`.
- Fewer than 100 known changed decisions: `shadow_decisions_not_safe`.
- At least 100 known changed decisions, `improved_count > damaged_count`, and
  `harm_precision <= 0.20`: `candidate_for_shadow_ab_activation`.
- Any other complete schema v2 result: `shadow_decisions_not_safe`.

The 100 known-changed minimum is a hard safety floor. The
`--min-known-changed` option may raise it for a stricter experiment but rejects
values below 100.

Schema v1, `decision_evidence_available: false`, incomplete decision evidence,
missing decision buckets, incompatible manifests, or any `audit_errors` fail
closed as `insufficient_decision_evidence`; none can produce a candidate.
`audit_errors` must be an object whose `count` is a non-negative integer.
Malformed structures or counts set `audit_errors_valid: false`, add
`invalid_audit_errors` to `incomplete_reasons`, and fail closed.
Unknown GT identities are retained as unknown evidence and cannot count as
`improved` or `damaged`, so sparse or unavailable GT can prevent the minimum
known-changed gate from being reached.

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
