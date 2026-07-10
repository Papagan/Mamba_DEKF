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
