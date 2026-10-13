# Neural Affinity Association

This implementation adopts the useful part of the neural-enhanced Bayesian
association idea without pretending that the current cache contains labels it
does not have. The learned head supplies a bounded association log-factor; the
existing geometric gate and Hungarian assignment remain authoritative.

## Runtime correction

For an active same-class candidate with head probability `p`, runtime computes

```text
log_factor = B * tanh(logit(p) / T)
delta_cost = clip(-alpha * log_factor, -MAX_DELTA, MAX_DELTA)
```

Only candidates within `COST_MARGIN_EPS` of that track row's best geometric
cost are corrected. With `PRESERVE_ROW_MIN: true`, a promoted candidate cannot
fall below the original row minimum. The head can therefore reorder a genuine
geometric ambiguity, but cannot create a match outside the existing gate or
make a track row globally easier to match than before.

The active experiment uses:

```yaml
MAMBA_ASSOCIATION_HEAD:
  APPLY_MODE: bounded_log_factor
  COST_MARGIN_EPS: 0.05
  ALPHA: 0.05
  MAX_DELTA: 0.03
  LOG_FACTOR_BOUND: 1.0
  LOGIT_TEMPERATURE: 1.0
  PRESERVE_ROW_MIN: true

ASSOCIATION_HEAD_AUDIT:
  SHADOW_MODE: true
```

Shadow mode computes the alternative Hungarian result and audit evidence while
leaving the production assignment unchanged.

## Training alignment

Every candidate set for one track and frame is kept intact by the batch
sampler. The category-specific head is optimized with three complementary
terms:

- balanced BCE for absolute positive/negative calibration;
- margin ranking for positive-versus-hard-negative separation;
- listwise loss over the complete candidate set for the actual top-1 decision.

Positive and negative BCE total weight is equalized after hard-negative
weighting. This matters because one track normally has one positive and several
negatives: without balancing, the learned intercept would make `p=0.5` a
non-neutral operating point. The shared temporal backbone stays frozen, pair
embeddings are computed once, and every category head is trained and selected
independently.

## Detection reliability and remaining geometry approximation

The CenterPoint cache now retains unmatched detections that are safely outside
every same-class GT matching radius. These can serve as association negatives.
Detections near a GT remain ambiguous and are excluded. A negative association
label still does not by itself establish a calibrated false-positive or birth
probability.

The offline `predicted_track_candidate` geometry extrapolates the current
fused track state using its velocity. Online geometry uses the tracker KF and
the learned motion residual, and its near-best rule compares geometric matching
costs rather than center distances. Thus `inference_margin` mining is a distance
proxy, not an exact replay of runtime intervention candidates. Check the
shadow decision audit before activating the head.

## Remote acceptance sequence

Rebuild both caches with relative track-history coordinates and a predicted
fused-track geometry proxy:

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
  --max-easy-negatives 0

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
  --max-easy-negatives 0
```

Run the configured main training. Stage B will load the motion checkpoint,
freeze the shared backbone, and write the merged association checkpoint. The
association checkpoint records the motion checkpoint hash, and inference
rejects a head paired with a different motion checkpoint:

```bash
python training/train.py --config config/train_nuscenes.yaml
```

Then run evaluation with
`config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml`. Keep
`SHADOW_MODE: true`, enable audit GT enrichment as described in
`docs/association_head_audit.md`, and generate the decision report:

```bash
python tools/compare_pairwise_assoc_train_infer.py \
  --train-cache /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --infer-audit debug/association_head_audit.json \
  --output debug/association_shadow_decision_report.json \
  --min-known-changed 100
```

Do not activate the intervention merely because validation loss or pairwise AUC
improves. A class/state is eligible only when the schema-v2 decision audit has
at least 100 known changed decisions, more improvements than damages, harm
precision at most 0.20, a valid runtime manifest, and no incomplete-evidence
reason. Even then, first run a separate non-shadow A/B experiment; do not edit
the protected baseline configuration.
