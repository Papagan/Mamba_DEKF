# Closure Orientation Curriculum Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a state-first orientation curriculum and a light saturation-aware orientation regularizer to the `mamba_multihead_closure` branch while keeping training, audit, and inference aligned on effective `R_ori`.

**Architecture:** Keep the closure branch's prior-relative `R_ori` contract intact. Add small, well-scoped helpers in `training/losses.py`, schedule them from `training/train.py`, and extend the existing lightweight unit-test style used in `tests/test_noise_audit_train.py` and `tests/test_runtime_contract_checks.py`. The frozen `0.739` baseline path remains untouched because all new behavior is gated behind `filter_mode == "mamba_multihead_closure"`.

**Tech Stack:** Python, PyTorch, YAML, `unittest`, existing `training`, `kalmanfilter`, and `tests` modules.

---

## File Structure

- Modify: `training/losses.py`
  - Add a closure-oriented circular state loss helper and a lightweight saturation penalty helper.
- Modify: `training/train.py`
  - Add orientation curriculum config parsing, schedule helpers, closure-branch loss blending, effective-orientation logging, and saturation regularization.
- Modify: `config/train_nuscenes.yaml`
  - Add explicit orientation curriculum knobs under `BASE_NOISE.MAMBA_CLOSURE`.
- Modify: `tests/test_noise_audit_train.py`
  - Add helper-level tests for state loss, saturation penalty, and closure ratio math.
- Modify: `tests/test_runtime_contract_checks.py`
  - Add a lightweight AST-loaded test that verifies validation keeps the same closure `filter_mode`.
- Modify: `README.md`
  - Add one short note documenting the new orientation curriculum knobs in the closure training path.

## Task 1: Add orientation curriculum helper functions in `training/losses.py`

**Files:**
- Modify: `training/losses.py`
- Modify: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Write the failing helper tests**

Add these tests inside `NoiseAuditTrainTest` in `tests/test_noise_audit_train.py`:

```python
    def test_circular_orientation_state_loss_wraps_pi_boundary(self):
        helpers = self._load_loss_helpers()
        circular_orientation_state_loss = helpers["circular_orientation_state_loss"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        pred = tensor([[3.13]], dtype=torch.float32 if torch is not None else np.float32)
        gt = tensor([[-3.13]], dtype=torch.float32 if torch is not None else np.float32)
        loss = circular_orientation_state_loss(pred, gt)

        self.assertLess(float(loss.item()), 0.01)

    def test_orientation_saturation_penalty_is_zero_below_threshold(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        penalty = orientation_saturation_penalty(
            tensor([[3.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        self.assertAlmostEqual(float(penalty.item()), 0.0, places=6)

    def test_orientation_saturation_penalty_is_positive_above_threshold(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        penalty = orientation_saturation_penalty(
            tensor([[25.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        self.assertGreater(float(penalty.item()), 0.0)
```

- [ ] **Step 2: Run the helper tests to verify they fail**

Run:

```bash
python3 -m unittest tests.test_noise_audit_train.NoiseAuditTrainTest.test_circular_orientation_state_loss_wraps_pi_boundary tests.test_noise_audit_train.NoiseAuditTrainTest.test_orientation_saturation_penalty_is_zero_below_threshold tests.test_noise_audit_train.NoiseAuditTrainTest.test_orientation_saturation_penalty_is_positive_above_threshold -v
```

Expected: failures because the new helpers are not defined yet.

- [ ] **Step 3: Implement the minimal helper functions**

Add these functions to `training/losses.py` below `wrapped_orientation_nll`:

```python
def circular_orientation_state_loss(
    pred_yaw: torch.Tensor,
    gt_yaw: torch.Tensor,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    pred = pred_yaw.squeeze(-1)
    gt = gt_yaw.squeeze(-1)
    per_sample = 1.0 - torch.cos(wrap_to_pi_torch(pred - gt))
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()


def orientation_saturation_penalty(
    kappa_ori_unc: torch.Tensor,
    *,
    max_effective_kappa: float,
    sample_weights: torch.Tensor = None,
) -> torch.Tensor:
    per_sample = F.relu(kappa_ori_unc.squeeze(-1) - float(max_effective_kappa)).pow(2)
    if sample_weights is None:
        return per_sample.mean()
    w = sample_weights / (sample_weights.sum() + 1e-8)
    return (per_sample * w).sum()
```

- [ ] **Step 4: Extend the test loader to expose the new helpers**

Update `_load_loss_helpers()` in `tests/test_noise_audit_train.py` so it loads:

```python
            [
                "wrap_to_pi_torch",
                "wrapped_orientation_nll",
                "log_ratio_anchor_loss",
                "circular_orientation_state_loss",
                "orientation_saturation_penalty",
            ],
```

and include `F` in the extra namespace:

```python
            extra_namespace={
                "math": math,
                "torch": torch_namespace,
                "F": torch.nn.functional if torch is not None else _FunctionalStub(),
            },
```

- [ ] **Step 5: Run the helper tests to verify they pass**

Run:

```bash
python3 -m unittest tests.test_noise_audit_train.NoiseAuditTrainTest.test_circular_orientation_state_loss_wraps_pi_boundary tests.test_noise_audit_train.NoiseAuditTrainTest.test_orientation_saturation_penalty_is_zero_below_threshold tests.test_noise_audit_train.NoiseAuditTrainTest.test_orientation_saturation_penalty_is_positive_above_threshold -v
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add training/losses.py tests/test_noise_audit_train.py
git commit -m "feat: add closure orientation curriculum loss helpers"
```

## Task 2: Add orientation curriculum scheduling and closure-branch loss blending

**Files:**
- Modify: `training/train.py`
- Modify: `tests/test_runtime_contract_checks.py`

- [ ] **Step 1: Write the failing schedule and validation forwarding tests**

Add these tests to `tests/test_runtime_contract_checks.py`:

```python
    def test_orientation_curriculum_schedule_transitions_from_state_to_wrapped(self):
        helpers = _load_functions(
            REPO_ROOT / "training" / "train.py",
            ["resolve_orientation_curriculum_weights"],
        )
        resolve_orientation_curriculum_weights = helpers["resolve_orientation_curriculum_weights"]

        early = resolve_orientation_curriculum_weights(
            epoch=1,
            closure_cfg={
                "ORI_WARMUP_EPOCHS": 4,
                "ORI_TRANSITION_EPOCHS": 4,
                "ORI_STATE_WEIGHT": 1.0,
                "ORI_WRAPPED_NLL_WEIGHT": 1.0,
            },
        )
        late = resolve_orientation_curriculum_weights(
            epoch=9,
            closure_cfg={
                "ORI_WARMUP_EPOCHS": 4,
                "ORI_TRANSITION_EPOCHS": 4,
                "ORI_STATE_WEIGHT": 1.0,
                "ORI_WRAPPED_NLL_WEIGHT": 1.0,
            },
        )

        self.assertGreater(early["state_weight"], early["wrapped_weight"])
        self.assertGreater(late["wrapped_weight"], late["state_weight"])
```

Keep the existing `test_validate_forwards_filter_mode_into_training_step`.

- [ ] **Step 2: Run the new schedule test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_runtime_contract_checks.RuntimeContractChecksTest.test_orientation_curriculum_schedule_transitions_from_state_to_wrapped -v
```

Expected: failure because `resolve_orientation_curriculum_weights` does not exist yet.

- [ ] **Step 3: Add a schedule helper in `training/train.py`**

Add this helper above `training_step()` in `training/train.py`:

```python
def resolve_orientation_curriculum_weights(epoch: int, closure_cfg: dict) -> dict:
    cfg = closure_cfg or {}
    warmup_epochs = int(cfg.get("ORI_WARMUP_EPOCHS", 0))
    transition_epochs = int(cfg.get("ORI_TRANSITION_EPOCHS", 0))
    base_state = float(cfg.get("ORI_STATE_WEIGHT", 1.0))
    base_wrapped = float(cfg.get("ORI_WRAPPED_NLL_WEIGHT", 1.0))

    if epoch < warmup_epochs:
        alpha = 0.0
    elif transition_epochs <= 0:
        alpha = 1.0
    else:
        alpha = max(0.0, min(1.0, (epoch - warmup_epochs + 1) / float(transition_epochs)))

    return {
        "alpha": alpha,
        "state_weight": base_state * (1.0 - alpha),
        "wrapped_weight": base_wrapped * alpha,
    }
```

- [ ] **Step 4: Blend the orientation losses only on the closure branch**

In `training_step()` in `training/train.py`, after `closure_cfg` is defined, add:

```python
    ori_curriculum = resolve_orientation_curriculum_weights(epoch=epoch, closure_cfg=closure_cfg)
```

Then replace the closure-only loss call block with:

```python
        loss_k, detail_k = loss_fn(
            pos_x_pred[active_k], pos_P_pred[active_k],
            siz_x_pred[active_k], siz_P_pred[active_k],
            ori_x_pred[active_k], ori_P_pred[active_k],
            gt_future_pos[:, k, :][active_k], gt_future_siz[:, k, :][active_k], gt_future_ori[:, k, :][active_k],
            mamba_out["embedding"][active_k] if k == 0 else None,
            [tok for idx, tok in enumerate(instance_tokens) if bool(active_k[idx].item())] if k == 0 else None,
            R_pos=R_pos[active_k], R_siz=R_siz[active_k], R_ori=R_ori[active_k],
            kappa_ori=kappa_ori[active_k],
            gt_next_vel=vel_gt[active_k] if vel_active else None,
            in_warmup=in_warmup,
            ori_nll_alpha=unfreeze_alpha,
            class_ids=class_ids[active_k],
            use_wrapped_orientation_nll=use_closure_loss_path,
        )
```

and immediately below it, for closure mode only, add:

```python
        if use_closure_loss_path:
            state_w = ori_curriculum["state_weight"]
            wrapped_w = ori_curriculum["wrapped_weight"]
            loss_k = (
                loss_k
                - loss_fn.physics_scale * loss_fn.state_loss.w_ori * detail_k["loss_ori"]
                + loss_fn.physics_scale * loss_fn.state_loss.w_ori * (
                    state_w * detail_k["loss_ori_state_tensor"]
                    + wrapped_w * detail_k["loss_ori_wrapped_tensor"]
                )
            )
```

This step depends on Task 3 below exposing tensor-valued orientation components from `JointLoss`. Do not invent a second path elsewhere.

- [ ] **Step 5: Add closure-only logging fields in `training_step()`**

Add to `detail` in `training/train.py`:

```python
    detail["ori_curriculum_alpha"] = ori_curriculum["alpha"] if use_closure_loss_path else 0.0
    detail["ori_state_weight"] = ori_curriculum["state_weight"] if use_closure_loss_path else 0.0
    detail["ori_wrapped_weight"] = ori_curriculum["wrapped_weight"] if use_closure_loss_path else 0.0
    detail["effective_r_ori_mean"] = mamba_out["R_ori"].diagonal(dim1=-2, dim2=-1).sum(-1).mean().item()
    detail["effective_kappa_mean"] = kappa_ori.mean().item()
    detail["effective_kappa_std"] = kappa_ori.std(dim=0).mean().item()
```

- [ ] **Step 6: Run the schedule and validation tests**

Run:

```bash
python3 -m unittest tests.test_runtime_contract_checks -v
```

Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add training/train.py tests/test_runtime_contract_checks.py
git commit -m "feat: add closure orientation curriculum scheduling"
```

## Task 3: Expose closure orientation sub-loss tensors and add saturation regularization

**Files:**
- Modify: `training/losses.py`
- Modify: `training/train.py`
- Modify: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Write the failing closure regularization tests**

Add this test to `tests/test_noise_audit_train.py`:

```python
    def test_orientation_saturation_penalty_grows_with_kappa_excess(self):
        helpers = self._load_loss_helpers()
        orientation_saturation_penalty = helpers["orientation_saturation_penalty"]
        tensor = torch.tensor if torch is not None else _NumpyTorchStub.tensor

        small = orientation_saturation_penalty(
            tensor([[6.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )
        large = orientation_saturation_penalty(
            tensor([[20.0]], dtype=torch.float32 if torch is not None else np.float32),
            max_effective_kappa=5.0,
        )

        self.assertGreater(float(large.item()), float(small.item()))
```

- [ ] **Step 2: Run the saturation-growth test to verify it fails if the helper is incomplete**

Run:

```bash
python3 -m unittest tests.test_noise_audit_train.NoiseAuditTrainTest.test_orientation_saturation_penalty_grows_with_kappa_excess -v
```

Expected: failure if the helper does not yet support the desired monotonic behavior.

- [ ] **Step 3: Expose tensor-valued orientation components from `StatePredictionLoss` and `JointLoss`**

In `training/losses.py`, update the `detail` path so closure scheduling in `training_step()` can reuse the component tensors:

```python
        detail = {
            "loss_pos": loss_pos.item(),
            "loss_siz": loss_siz.item(),
            "loss_ori": loss_ori.item(),
            "loss_ori_angle": loss_ori_angle.item(),
            "loss_ori_vm": loss_ori_vm.item(),
            "loss_ori_wrapped": loss_ori_wrapped.item(),
            "loss_vel": loss_vel_val,
            "loss_nis": loss_nis.item(),
            "loss_ori_state_tensor": loss_ori_angle,
            "loss_ori_wrapped_tensor": loss_ori_wrapped,
        }
```

Then, in `JointLoss.forward()`, keep these tensor-valued keys in the returned detail dict:

```python
        detail = {
            **detail_state,
            **detail_contrast,
            "loss_state": loss_state.item(),
            "loss_total": loss_total.item(),
        }
```

Do not `.item()` the tensor-only keys in `detail_state`.

- [ ] **Step 4: Add closure-only saturation regularization in `training_step()`**

In `training/train.py`, import the new helper:

```python
from training.losses import (
    JointLoss,
    log_ratio_anchor_loss,
    log_ratio_bound_loss,
    wrap_to_pi_torch,
    orientation_saturation_penalty,
)
```

Then inside the `if use_closure_loss_path:` block, add:

```python
        ori_sat_weight = float(closure_cfg.get("ORI_SATURATION_REG_WEIGHT", 0.0))
        max_effective_kappa = float(closure_cfg.get("ORI_MAX_EFFECTIVE_KAPPA", 5.0))
        ori_saturation_reg = (
            orientation_saturation_penalty(
                mamba_out["kappa_ori_unc"],
                max_effective_kappa=max_effective_kappa,
            ) * ori_sat_weight
            if ori_sat_weight > 0.0 else torch.zeros((), device=device, dtype=history.dtype)
        )
```

Include it in `real_loss`:

```python
    real_loss = (
        total_loss_tensor / norm
        + total_aux_loss_tensor / norm
        + kappa_reg
        + delta_pos_reg
        + ratio_anchor_loss
        + ratio_bound_loss
        + ori_saturation_reg
    )
```

Log it:

```python
    detail["loss_ori_saturation_reg"] = ori_saturation_reg.item() if use_closure_loss_path else 0.0
```

- [ ] **Step 5: Run the focused training-loss tests**

Run:

```bash
python3 -m unittest tests.test_noise_audit_train -v
```

Expected: `OK` with existing torch-dependent skips if torch is unavailable in the interpreter.

- [ ] **Step 6: Commit**

```bash
git add training/losses.py training/train.py tests/test_noise_audit_train.py
git commit -m "feat: add closure orientation saturation regularization"
```

## Task 4: Add closure orientation curriculum config and document the knobs

**Files:**
- Modify: `config/train_nuscenes.yaml`
- Modify: `README.md`
- Modify: `tests/test_mamba_multihead_closure_config.py`

- [ ] **Step 1: Write the failing config-schema test**

Add this test to `tests/test_mamba_multihead_closure_config.py`:

```python
    def test_train_config_contains_orientation_curriculum_knobs(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        closure_cfg = cfg["BASE_NOISE"]["MAMBA_CLOSURE"]
        for key in [
            "ORI_WARMUP_EPOCHS",
            "ORI_TRANSITION_EPOCHS",
            "ORI_STATE_WEIGHT",
            "ORI_WRAPPED_NLL_WEIGHT",
            "ORI_SATURATION_REG_WEIGHT",
            "ORI_MAX_EFFECTIVE_KAPPA",
        ]:
            self.assertIn(key, closure_cfg)
```

- [ ] **Step 2: Run the config test to verify it fails**

Run:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config.MambaMultiheadClosureConfigTest.test_train_config_contains_orientation_curriculum_knobs -v
```

Expected: failure because the new keys are missing.

- [ ] **Step 3: Add the orientation curriculum knobs to `config/train_nuscenes.yaml`**

Under `BASE_NOISE.MAMBA_CLOSURE` in `config/train_nuscenes.yaml`, add:

```yaml
    ORI_WARMUP_EPOCHS: 8
    ORI_TRANSITION_EPOCHS: 8
    ORI_STATE_WEIGHT: 1.0
    ORI_WRAPPED_NLL_WEIGHT: 1.0
    ORI_SATURATION_REG_WEIGHT: 1.0e-5
    ORI_MAX_EFFECTIVE_KAPPA: 5.0
```

- [ ] **Step 4: Document the new knobs in `README.md`**

Add this short paragraph in the training section of `README.md` after the note about `FILTER_MODE`:

```markdown
- the closure branch now also has orientation-specific curriculum knobs under `BASE_NOISE.MAMBA_CLOSURE`:
  - `ORI_WARMUP_EPOCHS`
  - `ORI_TRANSITION_EPOCHS`
  - `ORI_STATE_WEIGHT`
  - `ORI_WRAPPED_NLL_WEIGHT`
  - `ORI_SATURATION_REG_WEIGHT`
  - `ORI_MAX_EFFECTIVE_KAPPA`
```

- [ ] **Step 5: Run the config tests**

Run:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config -v
```

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add config/train_nuscenes.yaml README.md tests/test_mamba_multihead_closure_config.py
git commit -m "feat: add closure orientation curriculum config"
```

## Task 5: Final focused verification and checkpoint-eval handoff

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Run the focused closure training test suite**

Run:

```bash
python3 -m unittest tests.test_runtime_contract_checks tests.test_mamba_multihead_closure_config tests.test_noise_audit_train tests.test_noise_audit_infer -v
```

Expected: `OK` with only existing torch-environment skips if applicable.

- [ ] **Step 2: Run syntax verification**

Run:

```bash
python3 -m py_compile training/losses.py training/train.py tests/test_noise_audit_train.py tests/test_runtime_contract_checks.py tests/test_mamba_multihead_closure_config.py
```

Expected: no output, exit code `0`

- [ ] **Step 3: Add an explicit checkpoint-eval note to `README.md`**

Append this note under the closure-branch workflow section:

```markdown
For the first curriculum run, do not trust only `best.pt`. Evaluate at least:

- `checkpoint_epoch10.pt`
- `checkpoint_epoch20.pt`
- `checkpoint_epoch30.pt`
- `best.pt`

and compare AMOTA against the frozen `0.739` baseline.
```

- [ ] **Step 4: Run final doc/config regression checks**

Run:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config tests.test_runtime_contract_checks -v
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: add closure orientation curriculum eval guidance"
```

