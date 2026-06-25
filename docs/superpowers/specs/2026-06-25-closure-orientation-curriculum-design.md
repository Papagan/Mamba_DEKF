# Closure Orientation Curriculum Design

## Goal

Stabilize orientation training for the `mamba_multihead_closure` branch without reverting to the legacy `kappa`-centric main loss path.

The redesign must preserve the closure branch's core contract:

- orientation uncertainty is still represented as prior-relative `R_ori`
- training and inference must optimize the same effective orientation uncertainty semantics
- the frozen `AMOTA 0.739` baseline remains untouched

The practical target is to stop the current long-term orientation saturation pattern visible in training logs, where `mean_kappa` and `std_kappa` remain in the hundreds while the runtime path clamps orientation confidence to a much smaller effective range.

## Current Problem

The current closure branch uses wrapped Gaussian orientation NLL as the main orientation uncertainty objective, but the runtime path still maps orientation through:

`raw_R_ori -> kappa_ori_unc -> clamp(max=5) -> effective R_ori`

This creates a failure mode:

- the model keeps pushing `raw_R_ori` smaller
- `kappa_ori_unc` grows very large
- runtime clamps the effective orientation confidence
- training logs show large `mean_kappa/std_kappa`, but inference only sees the clamped value

As a result, part of the optimization pressure is spent in a numerically invalid region that no longer changes the filter's effective behavior.

## Design Principles

1. Do not restore the old `Von Mises + kappa` path as the closure branch's main orientation objective.
2. Reuse the useful part of the old strategy: stable angle-state learning first, uncertainty freedom later.
3. Keep training and inference aligned on effective orientation uncertainty.
4. Penalize orientation saturation lightly, not aggressively.
5. Optimize for final nuScenes AMOTA, not for lower training loss alone.

## Proposed Design

### 1. Two-Stage Orientation Curriculum

The closure branch will use a dedicated orientation curriculum layered on top of the existing uncertainty warmup.

#### Stage A: Orientation State First

Purpose:

- learn stable yaw state prediction
- avoid early collapse of `raw_R_ori`
- keep the orientation uncertainty head near its prior

Behavior:

- the dominant orientation objective is circular state error:
  - `loss_ori_state = 1 - cos(wrap(pred_yaw - gt_yaw))`
- wrapped Gaussian orientation NLL is disabled or heavily down-weighted
- `r_ori` freedom is limited through prior anchoring and bound constraints

Expected effect:

- yaw prediction becomes stable before the model is allowed to express strong orientation confidence

#### Stage B: Effective-Uncertainty Alignment

Purpose:

- transition from pure angle-state supervision to full closure semantics
- make the orientation uncertainty head useful without entering saturation

Behavior:

- gradually increase wrapped Gaussian NLL weight
- gradually reduce pure angle-state loss weight
- compute orientation uncertainty loss using the same effective `R_ori` semantics seen by inference

Expected effect:

- orientation uncertainty starts to carry signal only after yaw state prediction is already stable

### 2. Effective `R_ori` Supervision

The orientation uncertainty objective must be tied to the effective uncertainty used by runtime filtering, not only the pre-clamp latent quantity.

For closure training:

- the orientation loss should use effective `R_ori`
- the orientation regularizer should also reference closure-runtime priors
- train-side audit should record the same runtime priors used by the closure branch

This keeps:

- train loss
- train audit
- inference runtime

on the same orientation uncertainty contract.

### 3. Lightweight Orientation Saturation Penalty

A small saturation-aware regularizer will be added only for orientation.

Purpose:

- discourage the model from pushing `kappa_ori_unc` far above the runtime-effective zone
- reduce wasted optimization in clamp-saturated regions

Behavior:

- if orientation confidence goes beyond the configured effective ceiling, add a mild penalty
- the penalty must be weaker than the main state losses
- it is a safety term, not the primary learning signal

This is intentionally different from the old central `kappa_reg` design. The new penalty is local, branch-specific, and subordinate to the closure objective.

### 4. Separate Orientation Schedule Knobs

Orientation needs its own schedule instead of sharing the generic uncertainty warmup completely.

New training controls:

- `ORI_WARMUP_EPOCHS`
- `ORI_TRANSITION_EPOCHS`
- `ORI_STATE_WEIGHT`
- `ORI_WRAPPED_NLL_WEIGHT`
- `ORI_SATURATION_REG_WEIGHT`
- `ORI_MAX_EFFECTIVE_KAPPA`

These knobs control only the closure branch orientation path.

## Files Affected

### `training/train.py`

Responsibilities:

- add orientation curriculum scheduling
- blend state-first and wrapped-NLL orientation objectives
- log orientation-specific curriculum signals
- add orientation saturation penalty

New logging targets should include:

- `loss_ori_state`
- `loss_ori_wrapped`
- `loss_ori_saturation_reg`
- `effective_r_ori_mean`
- `effective_kappa_mean`
- `effective_kappa_std`

### `training/losses.py`

Responsibilities:

- keep existing wrapped Gaussian helper
- expose a stable circular state-loss helper
- expose a lightweight saturation-penalty helper

The closure branch will combine these helpers in a schedule-controlled way.

### `config/train_nuscenes.yaml`

Responsibilities:

- add orientation curriculum config
- keep closure branch as the default experimental training mode

### Tests

Add or extend tests to verify:

- orientation curriculum weights change as expected across epochs
- saturation penalty is zero below threshold and positive above threshold
- closure orientation regularization uses closure-runtime priors, not unrelated fallback priors
- validation continues to use the same closure branch mode as training

## Training Behavior Expectations

If the redesign works, training should change in the following way:

- early epochs:
  - `loss_ori_state` drops first
  - `loss_ori_wrapped` contributes little
  - `mean_kappa/std_kappa` stop exploding upward

- transition period:
  - `loss_ori_wrapped` gradually becomes meaningful
  - saturation penalty remains small but non-zero when needed

- later epochs:
  - orientation confidence becomes expressive without living permanently in the clamp-saturated region
  - train and validation losses remain stable
  - checkpoint selection becomes more predictive of real nuScenes eval

## Evaluation Protocol

Success is not declared from validation loss alone.

Each candidate run should be checked with:

1. training logs
   - verify `mean_kappa/std_kappa` are materially reduced from the current hundreds-level pattern
2. audit config
   - inspect closure orientation ratios and effective uncertainty behavior
3. official nuScenes evaluation
   - evaluate at least several checkpoints, not only `best.pt`

Recommended checkpoints for the first pass:

- `epoch10`
- `epoch20`
- `epoch30`
- `best.pt`

## Success Criteria

The redesign is considered successful only if all of the following hold:

1. orientation saturation is materially reduced in logs
2. closure-train and closure-infer orientation semantics stay aligned
3. no regression is introduced into the protected `0.739` baseline path
4. the closure branch reaches at least baseline-level tracking quality and is a credible candidate to exceed `AMOTA 0.740`

## Non-Goals

This redesign does not:

- re-enable the legacy `Von Mises + kappa` main path for closure mode
- change the frozen pure-DEKF baseline config
- redesign the full position/size curriculum
- change matching geometry or dirty-suppressor logic

