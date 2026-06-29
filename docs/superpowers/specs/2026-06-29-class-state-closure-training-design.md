# Class-State Closure Training Design

## Goal

Redesign the `mamba_multihead_closure` training and inference workflow so the model can train all nuScenes classes while inference safely activates only selected `class_id + state_bucket` gates. The workflow must preserve the protected `AMOTA 0.739` pure-DEKF baseline and avoid the previously confirmed closure regression where forced-prior inference was not numerically equivalent to `pure_dekf`.

## Baseline Contract

The protected baseline remains:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml`
- aggregate `AMOTA 0.739`
- trailer AMOTA around `0.533-0.534`

The closure branch must first reproduce this baseline before any Mamba gate is reopened. The equivalence settings are:

```yaml
MAMBA_CLOSURE:
  USE_CONDITIONAL_PRIOR: false
  FORCE_COAST_PRIOR_ONLY: true
  FORCE_PRIOR_STATES: ["matched", "unmatched"]
  ACTIVE_CLASS_STATES: {}
```

`USE_CONDITIONAL_PRIOR=false` is mandatory for baseline equivalence because `pure_dekf` uses static `_get_base_noise()` semantics and does not apply history-dependent conditional scaling. Forced-prior `R_ori` must also bypass the `R_ori -> 1/kappa -> clamp` path and keep the raw prior value.

## Training Contract

Training should support all seven class heads and both lifecycle states, but monitoring and selection must be bucketed by:

```text
class_id + state_bucket
```

The default training closure prior must match the safe inference prior:

```yaml
MAMBA_CLOSURE:
  USE_CONDITIONAL_PRIOR: false
  FORCE_COAST_PRIOR_ONLY: true
  FORCE_PRIOR_STATES: ["matched"]
  TRAIN_ALL_CLASS_STATES: true
  ACTIVE_CLASS_STATES: {}
```

This means training can learn all class/state heads, while inference remains all-prior by default. `ACTIVE_CLASS_STATES` is treated as an inference activation map, not as the full training coverage list.

## Monitoring

The current single global validation loss is insufficient because high-volume classes can dominate optimization. Training must aggregate metrics per class and per state:

- `loss_real`
- `loss_pos`
- `loss_siz`
- `loss_ori`
- `loss_vel`
- `loss_ratio_anchor`
- `loss_ratio_bound`
- ratio means for `q_pos`, `r_pos`, `r_siz`, `r_ori`
- ratio edge-hit rates for lower and upper bounds
- sample count

Metrics must be emitted with stable names such as:

```text
train/class_2/unmatched/loss_real
val/class_5/unmatched/loss_ratio_bound
val/class_3/matched/r_ori_ratio_mean
```

The first implementation stage should not change model parameter count. Class balancing can be added later after per-class metrics show whether weak classes are under-optimized.

## Checkpoint Selection

The training loop should continue saving:

```text
best.pt
checkpoint_epoch{N}.pt
```

It should also save per-class best checkpoints:

```text
best_class_0.pt
best_class_1.pt
best_class_2.pt
best_class_3.pt
best_class_4.pt
best_class_5.pt
best_class_6.pt
```

Each per-class best is selected by that class's validation `loss_real`, with a finite `min_samples` guard. The first implementation should save full checkpoints for simplicity. Later work may add a head-bank hybrid assembler that extracts only `head_bank.family_heads.*.{class_id}` parameters from per-class best checkpoints and combines them with a stable shared backbone.

## Runtime Contract

Every checkpoint must persist enough closure configuration to detect train/eval mismatch:

```python
runtime_contract = {
    "filter_mode": "mamba_multihead_closure",
    "tracker_compat_mode": "mctrack",
    "history_source": "fusion",
    "init_state_source": "fusion",
    "expected_bev_cost_mode": "geometric",
    "closure_use_conditional_prior": False,
    "closure_force_prior_states": ["matched"],
    "closure_active_class_states": {},
    "closure_train_all_class_states": True,
}
```

Inference should warn when the checkpoint was trained with `closure_use_conditional_prior=true` but the eval config uses `false`, or when force-prior semantics are incompatible.

## Inference Activation

Inference remains configuration driven. The all-prior guard config is the baseline. Mamba experiments should reopen exactly one or a small number of gates:

```yaml
MAMBA_CLOSURE:
  USE_CONDITIONAL_PRIOR: false
  FORCE_PRIOR_STATES: ["matched"]
  ACTIVE_CLASS_STATES:
    2: ["unmatched"]
```

A class/state pair not listed in `ACTIVE_CLASS_STATES` must be forced to ratio `1.0`, including `R_ori`. Any future change to this logic must pass tests proving forced prior is numerically equivalent to the static prior.

## Evaluation Protocol

After training, evaluate in this order:

1. all-prior closure equivalence
2. `bicycle unmatched`
3. `motorcycle unmatched`
4. `trailer unmatched`
5. `truck unmatched`
6. `bicycle + motorcycle`
7. `trailer + truck`

A gate is accepted only if:

- aggregate AMOTA is at least the all-prior baseline
- target class AMOTA improves or the overall AMOTA gain is clear
- strong classes do not show meaningful regression
- ratio distributions are not stuck on lower or upper bounds

## Non-Goals

- No model parameter increase in this stage.
- No ByteTrack or TRACK_SCORE changes.
- No conditional prior re-enable for baseline equivalence.
- No automatic activation of all trained heads in inference.

## Implementation Risks

The main risk is introducing a new train/eval mismatch. To prevent this, the implementation must add tests for:

- training config and eval config closure contract fields
- checkpoint runtime contract persistence
- per-class metric aggregation correctness
- forced-prior `R_ori` and ratio behavior
- warning when checkpoint and inference closure contracts disagree
