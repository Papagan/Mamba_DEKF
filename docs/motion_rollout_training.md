# Residual motion rollout

The active Mamba branch is a learned state transition, not only a Q/R noise adapter.

For every rollout step, training performs:

```text
CV predict -> bounded Mamba state correction -> state-mean loss -> optional measurement update
```

The corrected state is written back into the filter before the next step. Consequently, later losses backpropagate through earlier motion corrections. One predicted control is recursively applied across each synthetic coast horizon, so every rollout loss trains the same head used online; inference replans that control from updated history every frame. The correction layout is `[dx, dy, dz, dvx, dvy, dyaw]`; class-specific limits, `tanh` bounding, and frame-interval scaling are configured under `BASE_NOISE.MAMBA_STATE_RESIDUAL`.

`BASE_NOISE.MOTION_ROLLOUT` controls coast simulation and deterministic motion losses. Training randomly removes future updates for contiguous 2–8 frame horizons. Validation removes all future updates, so its motion loss cannot be improved by teacher forcing or covariance inflation.

Train on the remote machine:

```bash
python training/train.py \
  --config config/train_nuscenes.yaml \
  --skip-association-head
```

For the current pipeline smoke test, `DATA.VAL_TRACKLET_PKL` may be a copy of
the training fusion cache as requested. In that case the validation losses and
ADE/FDE gains only confirm the code path runs; they do not measure
generalization. Use a held-out cache before interpreting those metrics as model
performance.

The trailer and truck rollout now uses the same MCTrack exact matched and fake
update rules as the experimental inference config. The training entry point
checks those configured class IDs before starting.

Evaluate the resulting `checkpoints/mamba_motion/best.pt`:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml \
  -p 12
```

Monitor at least these TensorBoard values:

- `train/loss_motion_rollout` and `val/loss_motion_rollout`
- `train/loss_motion_pos`, `train/loss_motion_vel`, and `train/loss_motion_yaw`
- `train/forced_coast_ratio`, which should be close to the configured sampling regime
- `train/loss_delta_pos_reg`, to detect saturated or exploding raw controls
- `val/motion_ade_gain` and `val/motion_fde_gain`; positive values mean the learned transition beats CV in fully open-loop validation

Do not reuse an old `checkpoints/mamba_dekf/best.pt`. Its residual head was trained with one-step auxiliary semantics and is not a valid learned-transition checkpoint.
