# Association Head Audit

The frozen baseline keeps `MAMBA_ASSOCIATION_HEAD.ENABLED: false` to preserve
the 0.740 AMOTA path. The learned association head must be audited before it is
used in matching.

## Why Audit First

The current pairwise association head can achieve very low training loss while
still hurting AMOTA, because the training pair features and inference pair
features are not yet fully aligned. In particular, the old cache uses a
future-detection to candidate-detection geometry, while inference applies the
head to track-state to candidate-detection pairs.

## Audit Run

Enable both blocks only for an audit A/B run:

```yaml
MAMBA_ASSOCIATION_HEAD:
  ENABLED: true

ASSOCIATION_HEAD_AUDIT:
  ENABLED: true
  INFER_OUTPUT_PATH: "debug/association_head_audit.json"
  STRICT: false
```

Then run the normal evaluation command:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml \
  -p 12
```

The audit output contains per-class and per-state buckets:

- `pair_count`: same-class finite pairs scored by the head.
- `active_pair_count`: pairs in configured active class/state buckets.
- `penalized_pair_count`: active pairs that received a positive cost delta.
- `penalized_ratio`: `penalized_pair_count / active_pair_count`.
- `avg_score`: mean association score.
- `avg_delta`: mean cost increase applied to the geometric cost.
- `avg_cost_before` and `avg_cost_after`: cost shift caused by the head.

## Decision Rule

Do not enable the head by default if any stable class has a high penalized
ratio or AMOTA drops versus the 0.740 baseline. The next valid step is to
rebuild pairwise training samples so that pair geometry and history tokens match
the exact inference path.

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
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000
```

This mode changes the training pair features from the old
future-detection/candidate-detection geometry to predicted-track/candidate
geometry. It also stores `candidate_history_12` in the same detection-token
format used by inference (`x=y=0` for the one-frame detection token).
The `inference_margin` negative miner further restricts negatives to the same
near-best candidate set used by `MAMBA_ASSOCIATION_HEAD.APPLY_MODE:
margin_tiebreak`, avoiding the previous easy-negative shortcut where negatives
were meters away from the positive candidate.
