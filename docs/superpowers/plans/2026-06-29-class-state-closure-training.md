# Class-State Closure Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add class/state bucketed monitoring, per-class checkpoint selection, and explicit closure runtime contracts so all class heads can be trained while inference safely activates only selected class/state gates.

**Architecture:** Keep the existing `TemporalMamba` and `PriorConditionedHeadBank` parameter count unchanged. Add small helper functions in `training/train.py` for class/state metric aggregation and checkpoint selection, extend runtime contract fields, and harden tests around closure prior equivalence. Inference remains config-driven through `MAMBA_CLOSURE.ACTIVE_CLASS_STATES`.

**Tech Stack:** Python, PyTorch, YAML configs, `unittest`, TensorBoard `SummaryWriter`.

---

## File Structure

- Modify `config/train_nuscenes.yaml`: align training closure prior semantics with the safe eval contract and add class/state monitoring/checkpoint knobs.
- Modify `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`: keep all-prior equivalence guard explicit.
- Modify `training/train.py`: add bucket aggregation, per-class validation metrics, per-class checkpoint saving, and closure runtime contract fields.
- Modify `tracker/base_tracker.py`: warn when checkpoint closure contract conflicts with inference config.
- Modify `tests/test_mamba_multihead_closure_config.py`: assert config guardrails.
- Modify `tests/test_runtime_contract_checks.py`: assert closure runtime contract warnings.
- Add `tests/test_class_state_metrics.py`: test pure helper functions without requiring nuScenes data.
- Modify `README.md`: add the new training/eval workflow.
- Existing reference: `docs/mamba_closure_regression_notes.md`.

---

### Task 1: Lock Closure Config Contract

**Files:**
- Modify: `config/train_nuscenes.yaml`
- Modify: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`
- Modify: `tests/test_mamba_multihead_closure_config.py`

- [ ] **Step 1: Write failing config tests**

Add these tests to `tests/test_mamba_multihead_closure_config.py`:

```python
    def test_train_config_uses_static_closure_prior_for_training(self):
        cfg = yaml.safe_load((REPO_ROOT / "config" / "train_nuscenes.yaml").read_text(encoding="utf-8"))
        closure_cfg = cfg["BASE_NOISE"]["MAMBA_CLOSURE"]
        self.assertFalse(closure_cfg["USE_CONDITIONAL_PRIOR"])
        self.assertTrue(closure_cfg["TRAIN_ALL_CLASS_STATES"])
        self.assertEqual(closure_cfg["FORCE_PRIOR_STATES"], ["matched"])
        self.assertEqual(closure_cfg["ACTIVE_CLASS_STATES"], {})

    def test_eval_closure_config_is_all_prior_equivalence_guard(self):
        cfg = yaml.safe_load(
            (REPO_ROOT / "config" / "nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml")
            .read_text(encoding="utf-8")
        )
        closure_cfg = cfg["DEKF_BASE_NOISE"]["MAMBA_CLOSURE"]
        self.assertFalse(closure_cfg["USE_CONDITIONAL_PRIOR"])
        self.assertEqual(closure_cfg["FORCE_PRIOR_STATES"], ["matched", "unmatched"])
        self.assertEqual(closure_cfg["ACTIVE_CLASS_STATES"], {})
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config -v
```

Expected: fail because `config/train_nuscenes.yaml` currently does not define `USE_CONDITIONAL_PRIOR` / `TRAIN_ALL_CLASS_STATES` and may still list active classes.

- [ ] **Step 3: Update training config**

Under `BASE_NOISE.MAMBA_CLOSURE` in `config/train_nuscenes.yaml`, set:

```yaml
    USE_CONDITIONAL_PRIOR: false
    FORCE_COAST_PRIOR_ONLY: true
    FORCE_PRIOR_STATES: ["matched"]
    TRAIN_ALL_CLASS_STATES: true
    ACTIVE_CLASS_STATES: {}
```

Keep `MATCHED_KF_BAND`, `RATIO_*`, and orientation curriculum knobs unchanged.

- [ ] **Step 4: Update eval closure config**

Ensure `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml` contains:

```yaml
    USE_CONDITIONAL_PRIOR: false
    FORCE_COAST_PRIOR_ONLY: true
    FORCE_PRIOR_STATES: ["matched", "unmatched"]
    ACTIVE_CLASS_STATES: {}
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest tests.test_mamba_multihead_closure_config -v
```

Expected: pass.

---

### Task 2: Add Class/State Metric Aggregation Helpers

**Files:**
- Modify: `training/train.py`
- Add: `tests/test_class_state_metrics.py`

- [ ] **Step 1: Write helper tests**

Create `tests/test_class_state_metrics.py`:

```python
import unittest

from training.train import (
    class_state_bucket_key,
    init_class_state_metric_accumulator,
    update_class_state_metric_accumulator,
    finalize_class_state_metric_accumulator,
)


class ClassStateMetricTest(unittest.TestCase):
    def test_class_state_bucket_key_is_stable(self):
        self.assertEqual(class_state_bucket_key(2, "unmatched"), "class_2/unmatched")
        self.assertEqual(class_state_bucket_key("5", "matched"), "class_5/matched")

    def test_metric_accumulator_averages_by_sample_count(self):
        acc = init_class_state_metric_accumulator()
        update_class_state_metric_accumulator(
            acc,
            class_ids=[2, 2, 5],
            state_buckets=["unmatched", "unmatched", "matched"],
            metrics={
                "loss_real": [1.0, 3.0, 9.0],
                "q_pos_ratio_mean": [1.1, 1.3, 0.9],
            },
        )
        out = finalize_class_state_metric_accumulator(acc)
        self.assertEqual(out["class_2/unmatched/count"], 2)
        self.assertAlmostEqual(out["class_2/unmatched/loss_real"], 2.0)
        self.assertAlmostEqual(out["class_2/unmatched/q_pos_ratio_mean"], 1.2)
        self.assertEqual(out["class_5/matched/count"], 1)
        self.assertAlmostEqual(out["class_5/matched/loss_real"], 9.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m unittest tests.test_class_state_metrics -v
```

Expected: fail because helper functions do not exist.

- [ ] **Step 3: Implement helpers**

Add near the top-level helper section of `training/train.py`:

```python
def class_state_bucket_key(class_id, state_bucket) -> str:
    return f"class_{int(class_id)}/{str(state_bucket)}"


def init_class_state_metric_accumulator() -> dict:
    return {}


def update_class_state_metric_accumulator(acc: dict, *, class_ids, state_buckets, metrics: dict) -> None:
    class_id_list = [int(v.item()) if hasattr(v, "item") else int(v) for v in class_ids]
    state_list = [str(v) for v in state_buckets]
    if len(class_id_list) != len(state_list):
        raise ValueError("class_ids and state_buckets length mismatch")
    for metric_name, values in metrics.items():
        if len(values) != len(class_id_list):
            raise ValueError(f"metric {metric_name} length mismatch")
    for idx, (class_id, state_bucket) in enumerate(zip(class_id_list, state_list)):
        key = class_state_bucket_key(class_id, state_bucket)
        bucket = acc.setdefault(key, {"count": 0, "sums": {}})
        bucket["count"] += 1
        for metric_name, values in metrics.items():
            value = values[idx]
            if hasattr(value, "item"):
                value = value.item()
            bucket["sums"][metric_name] = bucket["sums"].get(metric_name, 0.0) + float(value)


def finalize_class_state_metric_accumulator(acc: dict) -> dict:
    out = {}
    for key, bucket in sorted(acc.items()):
        count = int(bucket.get("count", 0))
        out[f"{key}/count"] = count
        denom = max(count, 1)
        for metric_name, total in sorted(bucket.get("sums", {}).items()):
            out[f"{key}/{metric_name}"] = float(total) / float(denom)
    return out
```

- [ ] **Step 4: Run helper tests**

Run:

```bash
python3 -m unittest tests.test_class_state_metrics -v
```

Expected: pass.

---

### Task 3: Emit Per-Class/State Validation Metrics

**Files:**
- Modify: `training/train.py`
- Test: `tests/test_class_state_metrics.py`

- [ ] **Step 1: Extend `training_step` detail output**

In `training_step`, after `class_ids` and `state_buckets` are created, preserve them in `detail` before return:

```python
    detail["_class_ids"] = [int(v.item()) for v in class_ids.detach().cpu()]
    detail["_state_buckets"] = list(state_buckets)
```

Also add per-sample ratio means before return:

```python
    if use_closure_loss_path and "ratios" in mamba_out:
        ratios = mamba_out["ratios"]
        for ratio_name in ["q_pos_xyz", "q_pos_vxyz", "r_pos_xyz", "r_pos_vxy", "r_siz_lw", "r_siz_h", "r_ori"]:
            if ratio_name in ratios:
                detail[f"_sample_{ratio_name}"] = ratios[ratio_name].detach().view(-1).cpu().tolist()
```

- [ ] **Step 2: Update `validate` to aggregate class/state metrics**

In `validate`, create an accumulator before the loop:

```python
    class_state_acc = init_class_state_metric_accumulator()
```

Inside the loop, after `training_step`:

```python
        class_ids_detail = detail.get("_class_ids")
        state_buckets_detail = detail.get("_state_buckets")
        if class_ids_detail is not None and state_buckets_detail is not None:
            per_sample_metrics = {}
            for name in ["q_pos_xyz", "q_pos_vxyz", "r_pos_xyz", "r_pos_vxy", "r_siz_lw", "r_siz_h", "r_ori"]:
                sample_key = f"_sample_{name}"
                if sample_key in detail:
                    per_sample_metrics[f"{name}_mean"] = detail[sample_key]
            if per_sample_metrics:
                update_class_state_metric_accumulator(
                    class_state_acc,
                    class_ids=class_ids_detail,
                    state_buckets=state_buckets_detail,
                    metrics=per_sample_metrics,
                )
```

After averaging totals:

```python
    averaged = {k: v / n_batches for k, v in totals.items() if not str(k).startswith("_")}
    averaged.update({f"class_state/{k}": v for k, v in finalize_class_state_metric_accumulator(class_state_acc).items()})
    return averaged
```

- [ ] **Step 3: Prevent private detail keys from epoch sums**

In the training loop where `epoch_detail` is updated, skip keys starting with `_`:

```python
            for k, v in detail.items():
                if str(k).startswith("_"):
                    continue
                epoch_detail[k] = epoch_detail.get(k, 0.0) + v
```

- [ ] **Step 4: Run tests and compile**

Run:

```bash
python3 -m unittest tests.test_class_state_metrics tests.test_mamba_multihead_closure_config -v
python3 -m py_compile training/train.py
```

Expected: pass.

---

### Task 4: Save Per-Class Best Checkpoints

**Files:**
- Modify: `training/train.py`
- Test: `tests/test_class_state_metrics.py`

- [ ] **Step 1: Add helper tests**

Append to `tests/test_class_state_metrics.py`:

```python
from training.train import extract_class_validation_losses


class ClassCheckpointSelectionTest(unittest.TestCase):
    def test_extract_class_validation_losses_uses_weighted_state_average(self):
        avg_val = {
            "class_state/class_2/matched/loss_real": 4.0,
            "class_state/class_2/matched/count": 1,
            "class_state/class_2/unmatched/loss_real": 2.0,
            "class_state/class_2/unmatched/count": 3,
            "class_state/class_5/unmatched/loss_real": 10.0,
            "class_state/class_5/unmatched/count": 2,
        }
        losses = extract_class_validation_losses(avg_val, min_samples=2)
        self.assertAlmostEqual(losses[2], 2.5)
        self.assertAlmostEqual(losses[5], 10.0)
```

- [ ] **Step 2: Run test to verify failure**

Run:

```bash
python3 -m unittest tests.test_class_state_metrics -v
```

Expected: fail because `extract_class_validation_losses` does not exist.

- [ ] **Step 3: Implement helper**

Add to `training/train.py`:

```python
def extract_class_validation_losses(avg_val: dict, *, min_samples: int = 1) -> dict:
    class_totals = {}
    class_counts = {}
    prefix = "class_state/class_"
    for key, value in avg_val.items():
        if not key.startswith(prefix) or not key.endswith("/loss_real"):
            continue
        parts = key.split("/")
        class_id = int(parts[1].replace("class_", ""))
        state = parts[2]
        count_key = f"class_state/class_{class_id}/{state}/count"
        count = int(avg_val.get(count_key, 0))
        if count <= 0:
            continue
        class_totals[class_id] = class_totals.get(class_id, 0.0) + float(value) * count
        class_counts[class_id] = class_counts.get(class_id, 0) + count
    out = {}
    for class_id, total in class_totals.items():
        count = class_counts.get(class_id, 0)
        if count >= int(min_samples):
            out[class_id] = total / float(count)
    return out
```

- [ ] **Step 4: Add per-class best tracking**

In `main()`, before the epoch loop:

```python
    best_class_val_loss = {}
    per_class_min_samples = int(train_cfg.get("PER_CLASS_BEST_MIN_SAMPLES", 16))
```

After `val_total` is computed:

```python
        class_val_losses = extract_class_validation_losses(avg_val, min_samples=per_class_min_samples)
        for class_id, class_loss in sorted(class_val_losses.items()):
            prev = best_class_val_loss.get(class_id, float("inf"))
            if class_loss < prev:
                best_class_val_loss[class_id] = class_loss
                class_path = os.path.join(save_dir, f"best_class_{class_id}.pt")
                torch.save({
                    "epoch": epoch,
                    "class_id": class_id,
                    "model_state_dict": mamba.state_dict(),
                    "val_loss": avg_val,
                    "class_val_loss": class_loss,
                    "runtime_contract": runtime_contract,
                }, class_path)
                logger.info(f"  New best class {class_id} model -> {class_path} (class_val_loss={class_loss:.4f})")
```

- [ ] **Step 5: Run tests and compile**

Run:

```bash
python3 -m unittest tests.test_class_state_metrics -v
python3 -m py_compile training/train.py
```

Expected: pass.

---

### Task 5: Persist Closure Runtime Contract

**Files:**
- Modify: `training/train.py`
- Modify: `tracker/base_tracker.py`
- Modify: `tests/test_runtime_contract_checks.py`

- [ ] **Step 1: Add runtime contract fields in training**

In `training/train.py`, before constructing `runtime_contract`, create:

```python
    closure_runtime_cfg = (cfg.get("BASE_NOISE", {}) or {}).get("MAMBA_CLOSURE", {}) or {}
```

Add these fields to `runtime_contract`:

```python
        "closure_use_conditional_prior": bool(closure_runtime_cfg.get("USE_CONDITIONAL_PRIOR", True)),
        "closure_force_prior_states": list(closure_runtime_cfg.get("FORCE_PRIOR_STATES", ["matched"])),
        "closure_active_class_states": dict(closure_runtime_cfg.get("ACTIVE_CLASS_STATES", {}) or {}),
        "closure_train_all_class_states": bool(closure_runtime_cfg.get("TRAIN_ALL_CLASS_STATES", False)),
```

- [ ] **Step 2: Add runtime warning test**

In `tests/test_runtime_contract_checks.py`, add:

```python
    def test_warns_on_closure_conditional_prior_mismatch(self):
        warnings = build_runtime_contract_warnings(
            runtime_contract={
                "tracker_compat_mode": "mctrack",
                "filter_mode": "mamba_multihead_closure",
                "expected_bev_cost_mode": "geometric",
                "closure_use_conditional_prior": True,
                "closure_force_prior_states": ["matched"],
                "closure_active_class_states": {2: ["unmatched"]},
            },
            tracker_compat_mode="mctrack",
            filter_mode="mamba_multihead_closure",
            current_cost_mode="geometric",
            current_history_source="fusion",
            current_init_state_source="fusion",
            current_closure_cfg={
                "USE_CONDITIONAL_PRIOR": False,
                "FORCE_PRIOR_STATES": ["matched", "unmatched"],
                "ACTIVE_CLASS_STATES": {},
            },
        )
        self.assertTrue(any("closure_use_conditional_prior" in item for item in warnings))
```

- [ ] **Step 3: Update warning function signature**

In `tracker/base_tracker.py`, update `build_runtime_contract_warnings(...)` to accept:

```python
    current_closure_cfg=None,
```

Add mismatch checks:

```python
    current_closure_cfg = current_closure_cfg or {}
    if "closure_use_conditional_prior" in runtime_contract:
        expected = bool(runtime_contract.get("closure_use_conditional_prior"))
        current = bool(current_closure_cfg.get("USE_CONDITIONAL_PRIOR", True))
        if expected != current:
            warnings.append(
                "[Base3DTracker] WARNING: checkpoint runtime_contract closure_use_conditional_prior="
                f"{expected}, but current config uses {current}"
            )
```

Add similar checks for `closure_force_prior_states` and `closure_active_class_states` by comparing normalized string/int representations.

- [ ] **Step 4: Pass closure config from tracker**

Where `build_runtime_contract_warnings` is called in `tracker/base_tracker.py`, pass:

```python
                        current_closure_cfg=(self.cfg.get("DEKF_BASE_NOISE", {}) or {}).get("MAMBA_CLOSURE", {}),
```

- [ ] **Step 5: Run tests**

Run:

```bash
python3 -m unittest tests.test_runtime_contract_checks -v
python3 -m py_compile tracker/base_tracker.py training/train.py
```

Expected: pass.

---

### Task 6: TensorBoard Logging for Class/State Metrics

**Files:**
- Modify: `training/train.py`

- [ ] **Step 1: Write class/state val scalar logging**

After existing validation TensorBoard scalar writes, add:

```python
        for key, value in avg_val.items():
            if str(key).startswith("class_state/") and isinstance(value, (int, float)):
                writer.add_scalar(f"val/{key}", float(value), step)
```

- [ ] **Step 2: Log class best values**

After class best checkpoint update logic:

```python
        for class_id, class_loss in sorted(class_val_losses.items()):
            writer.add_scalar(f"val/class_{class_id}/selected_loss", float(class_loss), step)
```

- [ ] **Step 3: Compile**

Run:

```bash
python3 -m py_compile training/train.py
```

Expected: pass.

---

### Task 7: Documentation Update

**Files:**
- Modify: `README.md`
- Modify: `docs/mamba_closure_regression_notes.md`

- [ ] **Step 1: Update README training section**

Add a subsection after the Mamba training command:

```markdown
### Class/State Closure Training

The closure branch trains all seven class heads while inference remains all-prior by default. The training contract is:

```yaml
MAMBA_CLOSURE:
  USE_CONDITIONAL_PRIOR: false
  FORCE_PRIOR_STATES: ["matched"]
  TRAIN_ALL_CLASS_STATES: true
  ACTIVE_CLASS_STATES: {}
```

Per-class checkpoints are saved as `best_class_{id}.pt`. Do not activate a class/state gate in eval until the all-prior closure config reproduces AMOTA `0.739`.
```

- [ ] **Step 2: Update regression note**

Append to `docs/mamba_closure_regression_notes.md`:

```markdown
## Class/State Training Guardrail

Training may cover all class/state heads, but evaluation must remain all-prior until a gate-specific ablation is run. `ACTIVE_CLASS_STATES` is an inference activation map, not the full set of trained heads.
```

- [ ] **Step 3: Run markdown sanity grep**

Run:

```bash
rg -n "USE_CONDITIONAL_PRIOR|TRAIN_ALL_CLASS_STATES|best_class_" README.md docs/mamba_closure_regression_notes.md
```

Expected: all three terms appear in docs.

---

### Task 8: Final Verification

**Files:**
- All modified files

- [ ] **Step 1: Run local unit tests**

Run:

```bash
python3 -m unittest \
  tests.test_class_state_metrics \
  tests.test_mamba_multihead_closure_config \
  tests.test_runtime_contract_checks \
  -v
```

Expected: pass.

- [ ] **Step 2: Run compile checks**

Run:

```bash
python3 -m py_compile \
  training/train.py \
  tracker/base_tracker.py \
  kalmanfilter/mamba_adaptive_kf.py
```

Expected: pass.

- [ ] **Step 3: Remote torch test**

On the remote environment with PyTorch installed, run:

```bash
python -m unittest tests.test_prior_conditioned_heads -v
```

Expected: pass.

- [ ] **Step 4: Baseline eval guard**

Run:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml \
  -p 12
```

Expected: all-prior closure remains about `AMOTA 0.739`.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-29-class-state-closure-training.md`.

Two execution options:

1. **Subagent-Driven (recommended)** - dispatch a fresh subagent per task, review between tasks, fast iteration.
2. **Inline Execution** - execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
