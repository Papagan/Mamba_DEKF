# Bounded State Residual Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route Mamba's learned state residual into the inference geometry path in a bounded, class/state-gated way while preserving the frozen 0.739 baseline by default.

**Architecture:** Reuse the existing `delta_pos` head as the first residual source, add a small focused helper for class/state activation and numeric clipping, then apply the residual only inside the Mamba closure inference path before bbox prediction is written. The default config keeps the feature disabled, so existing baseline behavior remains unchanged unless an experiment explicitly activates it.

**Tech Stack:** Python, PyTorch tensors, unittest, YAML configuration.

---

### Task 1: State Residual Helper

**Files:**
- Create: `kalmanfilter/state_residual.py`
- Test: `tests/test_state_residual.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_state_residual.py` with tests for disabled behavior, class/state gating, per-class bounds, and yaw wrapping:

```python
import math
import unittest

import torch

from kalmanfilter.state_residual import apply_bounded_state_residuals


class StateResidualTest(unittest.TestCase):
    def test_disabled_returns_original_tensors(self):
        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        residual = torch.ones(2, 6)

        out_pos, out_siz, out_ori, mask = apply_bounded_state_residuals(
            pos, siz, ori, residual,
            class_ids=torch.tensor([2, 3]),
            state_buckets=["unmatched", "unmatched"],
            cfg={"ENABLED": False},
        )

        self.assertTrue(torch.equal(out_pos, pos))
        self.assertTrue(torch.equal(out_siz, siz))
        self.assertTrue(torch.equal(out_ori, ori))
        self.assertFalse(bool(mask.any().item()))

    def test_only_active_class_state_is_modified(self):
        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        residual = torch.tensor([[2.0, -2.0, 1.0, 5.0, -5.0, 0.4], [2.0, 2.0, 1.0, 5.0, 5.0, 0.4]])

        out_pos, out_siz, out_ori, mask = apply_bounded_state_residuals(
            pos, siz, ori, residual,
            class_ids=torch.tensor([2, 5]),
            state_buckets=["unmatched", "unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {2: ["unmatched"]},
                "DEFAULT_BOUNDS": {"POS_XY": 0.3, "POS_Z": 0.1, "VEL_XY": 0.5, "YAW": 0.2},
            },
        )

        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.3)
        self.assertAlmostEqual(float(out_pos[0, 1, 0]), -0.3)
        self.assertAlmostEqual(float(out_pos[0, 2, 0]), 0.1)
        self.assertAlmostEqual(float(out_pos[0, 3, 0]), 0.5)
        self.assertAlmostEqual(float(out_pos[0, 4, 0]), -0.5)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.2)
        self.assertTrue(torch.equal(out_pos[1], pos[1]))
        self.assertTrue(torch.equal(out_ori[1], ori[1]))

    def test_class_bounds_override_default_bounds(self):
        pos = torch.zeros(1, 6, 1)
        siz = torch.zeros(1, 3, 1)
        ori = torch.zeros(1, 2, 1)
        residual = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])

        out_pos, _, out_ori, _ = apply_bounded_state_residuals(
            pos, siz, ori, residual,
            class_ids=torch.tensor([5]),
            state_buckets=["unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {5: ["unmatched"]},
                "DEFAULT_BOUNDS": {"POS_XY": 0.3, "POS_Z": 0.2, "VEL_XY": 0.4, "YAW": 0.2},
                "CLASS_BOUNDS": {5: {"unmatched": {"POS_XY": 0.8, "POS_Z": 0.4, "VEL_XY": 0.2, "YAW": 0.05}}},
            },
        )

        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.8)
        self.assertAlmostEqual(float(out_pos[0, 2, 0]), 0.4)
        self.assertAlmostEqual(float(out_pos[0, 3, 0]), 0.2)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.05)

    def test_yaw_is_wrapped_after_residual(self):
        pos = torch.zeros(1, 6, 1)
        siz = torch.zeros(1, 3, 1)
        ori = torch.tensor([[[math.pi - 0.01], [0.0]]])
        residual = torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.2]])

        _, _, out_ori, _ = apply_bounded_state_residuals(
            pos, siz, ori, residual,
            class_ids=torch.tensor([3]),
            state_buckets=["unmatched"],
            cfg={
                "ENABLED": True,
                "ACTIVE_CLASS_STATES": {3: ["unmatched"]},
                "DEFAULT_BOUNDS": {"POS_XY": 0.3, "POS_Z": 0.2, "VEL_XY": 0.4, "YAW": 0.2},
            },
        )

        self.assertLess(float(out_ori[0, 0, 0]), -3.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_state_residual -v`

Expected: FAIL with `ModuleNotFoundError` for `kalmanfilter.state_residual`.

- [ ] **Step 3: Implement helper**

Create `kalmanfilter/state_residual.py` with:

```python
from __future__ import annotations

import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch import Tensor


def wrap_to_pi(values: Tensor) -> Tensor:
    return values - 2.0 * math.pi * torch.round(values / (2.0 * math.pi))


def _normalize_active_class_states(values: Optional[dict]) -> Dict[int, set]:
    out = {}
    for key, states in (values or {}).items():
        class_id = int(key)
        out[class_id] = {str(state).strip().lower() for state in (states or [])}
    return out


def _bound_for(cfg: dict, class_id: int, state_bucket: str, name: str) -> float:
    default_bounds = cfg.get("DEFAULT_BOUNDS") or {}
    class_bounds = cfg.get("CLASS_BOUNDS") or {}
    class_cfg = class_bounds.get(class_id, class_bounds.get(str(class_id), {})) or {}
    state_cfg = class_cfg.get(str(state_bucket), {}) or {}
    return float(state_cfg.get(name, default_bounds.get(name, 0.0)))


def _active_mask(class_ids: Tensor, state_buckets: Sequence[str], cfg: dict) -> Tensor:
    active_cfg = _normalize_active_class_states(cfg.get("ACTIVE_CLASS_STATES") or {})
    if not active_cfg:
        return torch.zeros(class_ids.shape[0], device=class_ids.device, dtype=torch.bool)
    values = []
    for class_id, state in zip(class_ids.detach().cpu().tolist(), state_buckets):
        values.append(str(state).strip().lower() in active_cfg.get(int(class_id), set()))
    return torch.tensor(values, device=class_ids.device, dtype=torch.bool)


def apply_bounded_state_residuals(
    pos_x: Tensor,
    siz_x: Tensor,
    ori_x: Tensor,
    residual: Optional[Tensor],
    *,
    class_ids: Optional[Tensor],
    state_buckets: Optional[Sequence[str]],
    cfg: Optional[dict],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    cfg = cfg or {}
    batch = pos_x.shape[0]
    device = pos_x.device
    if (
        not bool(cfg.get("ENABLED", False))
        or residual is None
        or class_ids is None
        or state_buckets is None
    ):
        return pos_x, siz_x, ori_x, torch.zeros(batch, device=device, dtype=torch.bool)

    if len(state_buckets) != batch:
        raise ValueError("state_buckets length must match batch size")
    residual = residual.to(device=device, dtype=pos_x.dtype)
    if residual.shape[0] != batch or residual.shape[1] < 6:
        raise ValueError("state residual must have shape [B, >=6]")

    class_ids = class_ids.to(device=device, dtype=torch.long)
    mask = _active_mask(class_ids, state_buckets, cfg)
    if not bool(mask.any().item()):
        return pos_x, siz_x, ori_x, mask

    out_pos = pos_x.clone()
    out_siz = siz_x.clone()
    out_ori = ori_x.clone()
    residual = residual.clone()

    for idx in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        class_id = int(class_ids[idx].item())
        state_bucket = str(state_buckets[idx]).strip().lower()
        pos_xy_bound = _bound_for(cfg, class_id, state_bucket, "POS_XY")
        pos_z_bound = _bound_for(cfg, class_id, state_bucket, "POS_Z")
        vel_xy_bound = _bound_for(cfg, class_id, state_bucket, "VEL_XY")
        yaw_bound = _bound_for(cfg, class_id, state_bucket, "YAW")

        out_pos[idx, 0:2, 0] = out_pos[idx, 0:2, 0] + torch.clamp(residual[idx, 0:2], -pos_xy_bound, pos_xy_bound)
        out_pos[idx, 2, 0] = out_pos[idx, 2, 0] + torch.clamp(residual[idx, 2], -pos_z_bound, pos_z_bound)
        out_pos[idx, 3:5, 0] = out_pos[idx, 3:5, 0] + torch.clamp(residual[idx, 3:5], -vel_xy_bound, vel_xy_bound)
        out_ori[idx, 0, 0] = wrap_to_pi(out_ori[idx, 0, 0] + torch.clamp(residual[idx, 5], -yaw_bound, yaw_bound))

    return out_pos, out_siz, out_ori, mask
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_state_residual -v`

Expected: PASS.

### Task 2: Inference Integration

**Files:**
- Modify: `tracker/base_tracker.py`
- Test: `tests/test_state_residual_infer.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_state_residual_infer.py` with tests that call the helper through the tracker path only where possible and directly test the closure config extraction:

```python
import unittest
from types import SimpleNamespace

import torch

from tracker.base_tracker import Base3DTracker


class StateResidualInferTest(unittest.TestCase):
    def test_tracker_extracts_state_residual_config(self):
        tracker = object.__new__(Base3DTracker)
        tracker.cfg = {"DEKF_BASE_NOISE": {"MAMBA_STATE_RESIDUAL": {"ENABLED": True}}}
        self.assertEqual(tracker._state_residual_cfg(), {"ENABLED": True})

    def test_apply_state_residual_to_prediction_changes_only_active_track(self):
        tracker = object.__new__(Base3DTracker)
        tracker.cfg = {
            "DEKF_BASE_NOISE": {
                "MAMBA_STATE_RESIDUAL": {
                    "ENABLED": True,
                    "ACTIVE_CLASS_STATES": {2: ["unmatched"]},
                    "DEFAULT_BOUNDS": {"POS_XY": 0.5, "POS_Z": 0.2, "VEL_XY": 0.4, "YAW": 0.1},
                }
            }
        }

        pos = torch.zeros(2, 6, 1)
        siz = torch.zeros(2, 3, 1)
        ori = torch.zeros(2, 2, 1)
        mamba_out = {"delta_pos": torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 0.5], [1.0, 1.0, 1.0, 1.0, 1.0, 0.5]])}
        class_ids = torch.tensor([2, 3])
        state_buckets = ["unmatched", "unmatched"]

        out_pos, out_siz, out_ori, mask = tracker._apply_state_residual_to_prediction(
            pos, siz, ori, mamba_out, class_ids, state_buckets
        )

        self.assertEqual(mask.tolist(), [True, False])
        self.assertAlmostEqual(float(out_pos[0, 0, 0]), 0.5)
        self.assertAlmostEqual(float(out_ori[0, 0, 0]), 0.1)
        self.assertTrue(torch.equal(out_pos[1], pos[1]))
        self.assertTrue(torch.equal(out_siz, siz))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_state_residual_infer -v`

Expected: FAIL because `Base3DTracker._state_residual_cfg` does not exist.

- [ ] **Step 3: Integrate helper**

Modify `tracker/base_tracker.py`:

```python
from kalmanfilter.state_residual import apply_bounded_state_residuals
```

Add methods inside `Base3DTracker`:

```python
    def _state_residual_cfg(self) -> Dict:
        return ((self.cfg.get("DEKF_BASE_NOISE", {}) or {}).get("MAMBA_STATE_RESIDUAL", {}) or {})

    def _apply_state_residual_to_prediction(
        self,
        pos_x: torch.Tensor,
        siz_x: torch.Tensor,
        ori_x: torch.Tensor,
        mamba_out: Dict,
        class_ids: torch.Tensor,
        state_buckets: List[str],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        return apply_bounded_state_residuals(
            pos_x,
            siz_x,
            ori_x,
            mamba_out.get("delta_pos"),
            class_ids=class_ids,
            state_buckets=state_buckets,
            cfg=self._state_residual_cfg(),
        )
```

In `predict_before_associate`, after `predict_with_mamba(...)` and before `_unbatch_kf_states(...)`, apply:

```python
            px, sx, ox, state_residual_mask = self._apply_state_residual_to_prediction(
                px, sx, ox, mamba_out, class_ids, state_buckets
            )
            mamba_out["state_residual_active_mask"] = state_residual_mask
```

- [ ] **Step 4: Run tests to verify pass**

Run: `python3 -m unittest tests.test_state_residual tests.test_state_residual_infer -v`

Expected: PASS.

### Task 3: Configuration Guardrails

**Files:**
- Modify: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_mamba_multihead_closure.yaml`
- Modify: `config/train_nuscenes.yaml`
- Test: `tests/test_mamba_multihead_closure_config.py`

- [ ] **Step 1: Add tests for disabled-by-default config**

Add assertions that `MAMBA_STATE_RESIDUAL.ENABLED` is false and `ACTIVE_CLASS_STATES` is empty in the closure eval config.

- [ ] **Step 2: Run tests to verify failure**

Run: `python3 -m unittest tests.test_mamba_multihead_closure_config -v`

Expected: FAIL because the new config block is absent.

- [ ] **Step 3: Add guarded config block**

Add this under `DEKF_BASE_NOISE` in both configs:

```yaml
  MAMBA_STATE_RESIDUAL:
    ENABLED: false
    ACTIVE_CLASS_STATES: {}
    DEFAULT_BOUNDS:
      POS_XY: 0.30
      POS_Z: 0.10
      VEL_XY: 0.50
      YAW: 0.08
    CLASS_BOUNDS:
      2:
        unmatched: {POS_XY: 0.35, POS_Z: 0.08, VEL_XY: 0.60, YAW: 0.10}
      3:
        unmatched: {POS_XY: 0.40, POS_Z: 0.08, VEL_XY: 0.70, YAW: 0.12}
      5:
        unmatched: {POS_XY: 0.25, POS_Z: 0.08, VEL_XY: 0.35, YAW: 0.05}
      6:
        unmatched: {POS_XY: 0.30, POS_Z: 0.08, VEL_XY: 0.45, YAW: 0.06}
```

- [ ] **Step 4: Run config tests**

Run: `python3 -m unittest tests.test_mamba_multihead_closure_config -v`

Expected: PASS.

### Task 4: Verification

**Files:**
- No new files.

- [ ] **Step 1: Run targeted test suite**

Run:

```bash
python3 -m unittest \
  tests.test_state_residual \
  tests.test_state_residual_infer \
  tests.test_mamba_multihead_closure_config \
  tests.test_runtime_contract_checks \
  tests.test_noise_audit_infer -v
```

Expected: PASS.

- [ ] **Step 2: Compile changed Python files**

Run:

```bash
python3 -m py_compile \
  kalmanfilter/state_residual.py \
  tracker/base_tracker.py
```

Expected: no output and exit code 0.

- [ ] **Step 3: Confirm baseline guard**

Inspect configs and confirm `MAMBA_STATE_RESIDUAL.ENABLED: false` in the default closure eval config. This ensures the frozen baseline path is not changed unless explicitly activated.
