# Mamba Closure Regression Notes

This note records a confirmed `mamba_multihead_closure` regression that reduced
nuScenes AMOTA from the protected `0.739` pure-DEKF baseline to `0.729-0.730`.

## Protected Baseline

- Baseline config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml`
- Closure equivalence config: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`
- Protected aggregate AMOTA: `0.739`
- Protected trailer AMOTA target: about `0.533-0.534`

The closure config must first reproduce the protected baseline before any Mamba
class/state gate is reopened.

## Root Cause

The apparent "force prior" closure path was not numerically equivalent to
`pure_dekf`.

Two issues caused this:

1. `MAMBA_CLOSURE.USE_CONDITIONAL_PRIOR=true` made closure priors call
   `build_base_covariances()` with history-dependent conditional scaling.
   The protected `pure_dekf` baseline uses `_get_base_noise()` and does not use
   this conditional prior scaling. As a result, forcing ratios to `1.0` still
   changed Q/R and reduced AMOTA to about `0.729-0.730`.
2. Forced-prior `R_ori` still went through the closure `R_ori -> 1/kappa`
   conversion and `kappa <= 5` clamp. That meant ratio `1.0` could still change
   orientation measurement noise. Forced-prior samples must keep the raw prior
   `R_ori` directly.

## Current Guardrail

For closure-to-baseline equivalence, keep:

```yaml
MAMBA_CLOSURE:
  USE_CONDITIONAL_PRIOR: false
  FORCE_COAST_PRIOR_ONLY: true
  FORCE_PRIOR_STATES: ["matched", "unmatched"]
  ACTIVE_CLASS_STATES: {}
```

This configuration is a regression guard, not a Mamba improvement experiment.
It verifies that the closure branch can no-op back to the `0.739` baseline.

## Do Not Reintroduce

- Do not enable `USE_CONDITIONAL_PRIOR` in the baseline equivalence config.
- Do not assume `ratio=1.0` is equivalent to pure-DEKF unless Q/R/R_ori are all
  checked against `_get_base_noise()` semantics.
- Do not reopen `ACTIVE_CLASS_STATES` before the all-prior closure config
  reproduces AMOTA `0.739`.
- Do not train against a broader closure target and evaluate with a narrower
  class/state gate without explicitly documenting that mismatch.

## Safe Ablation Order

1. Run all-prior closure equivalence and confirm AMOTA is about `0.739`.
2. Keep `USE_CONDITIONAL_PRIOR=false`.
3. Reopen exactly one small gate, for example:

   ```yaml
   FORCE_PRIOR_STATES: ["matched"]
   ACTIVE_CLASS_STATES:
     2: ["unmatched"]
     3: ["unmatched"]
   ```

4. Compare against the all-prior closure result, not only against historical
   pure-DEKF logs.
5. If the ablation drops AMOTA, close the gate and inspect noise audit before
   changing tracking thresholds.

## Required Local Checks

Run these after editing closure logic or configs:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config -v
python3 -m py_compile kalmanfilter/mamba_adaptive_kf.py tracker/base_tracker.py training/train.py
```

In the remote training/eval environment with torch installed, also run:

```bash
python -m unittest tests.test_prior_conditioned_heads -v
```

## Class/State Training Guardrail

Training may cover all class/state heads, but evaluation must remain all-prior until a gate-specific ablation is run. `ACTIVE_CLASS_STATES` is an inference activation map, not the full set of trained heads.

Per-class checkpoints are saved as `best_class_{id}.pt` using class/state validation loss. Treat them as candidates for one-gate-at-a-time ablations, not as permission to enable all trained heads at once.
