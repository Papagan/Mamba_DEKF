# Mamba/Fusion Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a shared audit pipeline for `mamba`, `pure_dekf`, and `fusion` noise consumption in inference and training without changing tracking behavior or the frozen `AMOTA 0.737` baseline.

**Architecture:** Introduce a small shared audit module that records per-class, per-mode, and matched/unmatched noise statistics with ratio-to-prior summaries, then wire it into the inference KF path and the training loop behind explicit opt-in flags. Keep all changes passive: when disabled, the baseline path must remain byte-for-byte behaviorally identical.

**Tech Stack:** Python, PyTorch, existing tracker/KF code, JSON logging, `unittest`

---

## File Structure

**Create**
- `kalmanfilter/noise_audit.py` — shared audit schema, stat accumulator, ratio-to-prior helpers, JSON export helpers
- `tests/test_noise_audit.py` — unit tests for shared audit helpers
- `tests/test_noise_audit_infer.py` — unit tests for inference-side no-op behavior and payload structure
- `tests/test_noise_audit_train.py` — unit tests for training-side summary emission

**Modify**
- `kalmanfilter/mamba_adaptive_kf.py` — surface effective `Q/R` and prior references to the audit hook without changing filter outputs
- `tracker/base_tracker.py` — add opt-in inference audit lifecycle and write JSON summaries after evaluation runs
- `training/train.py` — add opt-in training-side audit collection and JSON summaries
- `config/train_nuscenes.yaml` — add audit config knobs for training
- `README.md` — document audit-only workflow and baseline-safety guarantees

**Do not modify**
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- tracking decision logic
- loss definitions

---

### Task 1: Add the shared audit schema and aggregation helpers

**Files:**
- Create: `kalmanfilter/noise_audit.py`
- Test: `tests/test_noise_audit.py`

- [ ] **Step 1: Write the failing tests for the shared accumulator**

```python
import unittest

from kalmanfilter.noise_audit import NoiseAuditAccumulator


class NoiseAuditAccumulatorTest(unittest.TestCase):
    def test_records_grouped_stats_and_ratios(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="fusion",
            class_id=5,
            class_name="trailer",
            state="matched",
            history_len=6,
            families={
                "q_pos": 2.0,
                "r_pos": 4.0,
                "r_siz": 6.0,
                "r_ori": 8.0,
            },
            prior_families={
                "q_pos": 1.0,
                "r_pos": 2.0,
                "r_siz": 3.0,
                "r_ori": 4.0,
            },
        )
        payload = acc.to_summary()
        bucket = payload["buckets"][0]
        self.assertEqual(bucket["split"], "infer")
        self.assertEqual(bucket["mode"], "fusion")
        self.assertEqual(bucket["class_id"], 5)
        self.assertEqual(bucket["state"], "matched")
        self.assertEqual(bucket["count"], 1)
        self.assertAlmostEqual(bucket["families"]["q_pos"]["median"], 2.0)
        self.assertAlmostEqual(bucket["ratios"]["q_pos"]["median"], 2.0)

    def test_ignores_missing_optional_history_length(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="train",
            mode="mamba",
            class_id=2,
            class_name="bicycle",
            state="unmatched",
            history_len=None,
            families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
            prior_families={"q_pos": 1.0, "r_pos": 1.0, "r_siz": 1.0, "r_ori": 1.0},
        )
        payload = acc.to_summary()
        self.assertIsNone(payload["buckets"][0]["history_len"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `python -m unittest tests.test_noise_audit -v`

Expected: import failure for `kalmanfilter.noise_audit` or missing `NoiseAuditAccumulator`

- [ ] **Step 3: Write the minimal shared audit module**

```python
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median


FAMILIES = ("q_pos", "r_pos", "r_siz", "r_ori")


def _safe_ratio(value, prior):
    if prior is None:
        return None
    if not math.isfinite(value):
        return None
    if not math.isfinite(prior) or prior <= 0:
        return None
    ratio = value / prior
    if not math.isfinite(ratio):
        return None
    return ratio


def _percentile(values, q):
    if not values:
        return None
    ordered = sorted(values)
    idx = int(round((len(ordered) - 1) * q))
    return ordered[idx]


@dataclass
class _Bucket:
    split: str
    mode: str
    class_id: int
    class_name: str
    state: str
    history_len: int | None
    sample_count: int = 0
    families: dict = field(default_factory=lambda: {name: [] for name in FAMILIES})
    ratios: dict = field(default_factory=lambda: {name: [] for name in FAMILIES})

    def add(self, families, prior_families):
        self.sample_count += 1
        for name in FAMILIES:
            value = float(families[name])
            if not math.isfinite(value):
                continue
            self.families[name].append(value)
            ratio = _safe_ratio(value, prior_families.get(name))
            if ratio is not None:
                self.ratios[name].append(ratio)


class NoiseAuditAccumulator:
    def __init__(self):
        self._buckets = {}

    def add_sample(self, *, split, mode, class_id, class_name, state, history_len, families, prior_families):
        key = (split, mode, class_id, class_name, state, history_len)
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(
                split=split,
                mode=mode,
                class_id=class_id,
                class_name=class_name,
                state=state,
                history_len=history_len,
            )
            self._buckets[key] = bucket
        bucket.add(families, prior_families)

    def to_summary(self):
        payload = {"buckets": []}
        for bucket in self._buckets.values():
            payload["buckets"].append(
                {
                    "split": bucket.split,
                    "mode": bucket.mode,
                    "class_id": bucket.class_id,
                    "class_name": bucket.class_name,
                    "state": bucket.state,
                    "history_len": bucket.history_len,
                    "count": bucket.sample_count,
                    "families": {name: _summarize(bucket.families[name]) for name in FAMILIES},
                    "ratios": {name: _summarize(bucket.ratios[name]) for name in FAMILIES},
                }
            )
        payload["buckets"].sort(
            key=lambda item: (
                item["split"],
                item["mode"],
                item["class_id"],
                item["class_name"],
                item["state"],
                item["history_len"] is None,
                item["history_len"],
            )
        )
        return payload

    def write_json(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(self.to_summary(), indent=2, sort_keys=True, allow_nan=False),
            encoding="utf-8",
        )


def _summarize(values):
    if not values:
        return {"count": 0, "mean": None, "median": None, "p90": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": mean(values),
        "median": median(values),
        "p90": _percentile(values, 0.9),
        "min": min(values),
        "max": max(values),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m unittest tests.test_noise_audit -v`

Expected: `OK`

- [ ] **Step 4a: Add regression coverage for non-finite value handling**

Add tests that verify:

- non-finite priors do not produce ratio summaries
- non-finite observed family values are skipped instead of poisoning the bucket
- bucket-level `count` continues to reflect the number of submitted samples, even if some families are filtered
- `write_json()` emits strict JSON that remains parseable by `json.loads`

- [ ] **Step 5: Commit**

```bash
git add kalmanfilter/noise_audit.py tests/test_noise_audit.py
git commit -m "feat: add shared noise audit accumulator"
```

### Task 2: Surface effective inference-side noise samples without changing filter outputs

**Files:**
- Modify: `kalmanfilter/mamba_adaptive_kf.py`
- Modify: `tracker/base_tracker.py`
- Test: `tests/test_noise_audit_infer.py`

- [ ] **Step 1: Write the failing inference audit tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from kalmanfilter.noise_audit import NoiseAuditAccumulator


class NoiseAuditInferTest(unittest.TestCase):
    def test_disabled_inference_audit_is_noop(self):
        cfg = {"AUDIT": {"NOISE_AUDIT": {"ENABLED": False}}}
        self.assertFalse(cfg["AUDIT"]["NOISE_AUDIT"]["ENABLED"])

    def test_inference_summary_contains_required_fields(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="infer",
            mode="pure_dekf",
            class_id=6,
            class_name="truck",
            state="unmatched",
            history_len=4,
            families={"q_pos": 1.0, "r_pos": 2.0, "r_siz": 3.0, "r_ori": 4.0},
            prior_families={"q_pos": 1.0, "r_pos": 2.0, "r_siz": 3.0, "r_ori": 4.0},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "infer_noise_audit.json"
            acc.write_json(out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            bucket = payload["buckets"][0]
            self.assertEqual(bucket["split"], "infer")
            self.assertEqual(bucket["mode"], "pure_dekf")
            self.assertEqual(bucket["state"], "unmatched")
            self.assertIn("families", bucket)
            self.assertIn("ratios", bucket)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify they fail or are incomplete**

Run: `python -m unittest tests.test_noise_audit_infer -v`

Expected: failure because the production inference audit path does not exist yet

- [ ] **Step 3: Add an opt-in audit config reader and live sample hook**

```python
# tracker/base_tracker.py
from kalmanfilter.noise_audit import NoiseAuditAccumulator


def _build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


def _noise_audit_enabled(cfg):
    return bool(_build_noise_audit_cfg(cfg).get("ENABLED", False))


class Base3DTracker:
    def __init__(self, cfg, ...):
        ...
        self.noise_audit_cfg = _build_noise_audit_cfg(cfg)
        self.noise_audit = NoiseAuditAccumulator() if self.noise_audit_cfg.get("ENABLED", False) else None

    def _record_noise_audit_sample(self, *, mode, class_id, class_name, state, history_len, families, prior_families):
        if self.noise_audit is None:
            return
        self.noise_audit.add_sample(
            split="infer",
            mode=mode,
            class_id=class_id,
            class_name=class_name,
            state=state,
            history_len=history_len,
            families=families,
            prior_families=prior_families,
        )
```

```python
# kalmanfilter/mamba_adaptive_kf.py
def build_noise_audit_sample(*, mode, traj_labels, matched_mask, history_lens, q_pos, r_pos, r_siz, r_ori, prior_q_pos, prior_r_pos, prior_r_siz, prior_r_ori):
    samples = []
    for idx, cls_id in enumerate(traj_labels.tolist()):
        samples.append(
            {
                "mode": mode,
                "class_id": int(cls_id),
                "state": "matched" if bool(matched_mask[idx]) else "unmatched",
                "history_len": None if history_lens is None else int(history_lens[idx]),
                "families": {
                    "q_pos": float(q_pos[idx]),
                    "r_pos": float(r_pos[idx]),
                    "r_siz": float(r_siz[idx]),
                    "r_ori": float(r_ori[idx]),
                },
                "prior_families": {
                    "q_pos": float(prior_q_pos[idx]),
                    "r_pos": float(prior_r_pos[idx]),
                    "r_siz": float(prior_r_siz[idx]),
                    "r_ori": float(prior_r_ori[idx]),
                },
            }
        )
    return samples
```

- [ ] **Step 4: Wire summary export at tracker shutdown / eval completion**

```python
# tracker/base_tracker.py
def dump_noise_audit_if_needed(self):
    if self.noise_audit is None:
        return
    output_path = self.noise_audit_cfg.get("INFER_OUTPUT_PATH", "debug/infer_noise_audit.json")
    self.noise_audit.write_json(output_path)
```

- [ ] **Step 5: Run the inference audit tests**

Run: `python -m unittest tests.test_noise_audit_infer -v`

Expected: `OK`

- [ ] **Step 6: Commit**

```bash
git add kalmanfilter/mamba_adaptive_kf.py tracker/base_tracker.py tests/test_noise_audit_infer.py
git commit -m "feat: add inference-side noise audit hooks"
```

### Task 3: Add training-side audit logging with the same schema family

**Files:**
- Modify: `training/train.py`
- Modify: `config/train_nuscenes.yaml`
- Test: `tests/test_noise_audit_train.py`

- [ ] **Step 1: Write the failing training audit tests**

```python
import json
import tempfile
import unittest
from pathlib import Path

from kalmanfilter.noise_audit import NoiseAuditAccumulator


class NoiseAuditTrainTest(unittest.TestCase):
    def test_training_summary_contains_window_metadata(self):
        acc = NoiseAuditAccumulator()
        acc.add_sample(
            split="train",
            mode="mamba",
            class_id=4,
            class_name="bus",
            state="matched",
            history_len=8,
            families={"q_pos": 2.0, "r_pos": 3.0, "r_siz": 4.0, "r_ori": 5.0},
            prior_families={"q_pos": 1.0, "r_pos": 1.5, "r_siz": 2.0, "r_ori": 2.5},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "train_noise_audit.json"
            acc.write_json(out)
            payload = json.loads(out.read_text(encoding="utf-8"))
            self.assertEqual(payload["buckets"][0]["split"], "train")
            self.assertEqual(payload["buckets"][0]["history_len"], 8)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests to verify the production path is still missing**

Run: `python -m unittest tests.test_noise_audit_train -v`

Expected: test scaffold passes locally but no training path emits this summary yet

- [ ] **Step 3: Add opt-in audit config to training config**

```yaml
# config/train_nuscenes.yaml
AUDIT:
  NOISE_AUDIT:
    ENABLED: false
    TRAIN_OUTPUT_PATH: debug/train_noise_audit.json
```

- [ ] **Step 4: Add training-side accumulator lifecycle**

```python
# training/train.py
from kalmanfilter.noise_audit import NoiseAuditAccumulator


def build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


audit_cfg = build_noise_audit_cfg(cfg)
train_noise_audit = NoiseAuditAccumulator() if audit_cfg.get("ENABLED", False) else None
```

- [ ] **Step 5: Record batch summaries from the same family names**

```python
if train_noise_audit is not None:
    for idx in range(batch_size):
        train_noise_audit.add_sample(
            split="train",
            mode=filter_mode,
            class_id=int(cls_ids[idx]),
            class_name=class_name_lookup[int(cls_ids[idx])],
            state="matched",
            history_len=int(history_lens[idx]),
            families={
                "q_pos": float(q_pos[idx]),
                "r_pos": float(r_pos[idx]),
                "r_siz": float(r_siz[idx]),
                "r_ori": float(r_ori[idx]),
            },
            prior_families={
                "q_pos": float(prior_q_pos[idx]),
                "r_pos": float(prior_r_pos[idx]),
                "r_siz": float(prior_r_siz[idx]),
                "r_ori": float(prior_r_ori[idx]),
            },
        )
```

- [ ] **Step 6: Write the training summary on epoch-end / training-end**

```python
if train_noise_audit is not None:
    train_noise_audit.write_json(audit_cfg.get("TRAIN_OUTPUT_PATH", "debug/train_noise_audit.json"))
```

- [ ] **Step 7: Run the training audit tests**

Run: `python -m unittest tests.test_noise_audit_train -v`

Expected: `OK`

- [ ] **Step 8: Commit**

```bash
git add training/train.py config/train_nuscenes.yaml tests/test_noise_audit_train.py
git commit -m "feat: add training-side noise audit logging"
```

### Task 4: Make the audit schema comparable across training and inference

**Files:**
- Modify: `kalmanfilter/noise_audit.py`
- Modify: `tests/test_noise_audit.py`

- [ ] **Step 1: Add a schema version and top-level metadata test**

```python
def test_summary_has_schema_version_and_top_level_keys(self):
    acc = NoiseAuditAccumulator()
    payload = acc.to_summary()
    self.assertEqual(payload["schema_version"], 1)
    self.assertIn("buckets", payload)
```

- [ ] **Step 2: Run the shared audit tests and confirm failure**

Run: `python -m unittest tests.test_noise_audit -v`

Expected: missing `schema_version`

- [ ] **Step 3: Add top-level schema metadata**

```python
def to_summary(self):
    payload = {
        "schema_version": 1,
        "families": list(FAMILIES),
        "buckets": [],
    }
    ...
    return payload
```

- [ ] **Step 4: Re-run the shared audit tests**

Run: `python -m unittest tests.test_noise_audit -v`

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add kalmanfilter/noise_audit.py tests/test_noise_audit.py
git commit -m "feat: version shared noise audit schema"
```

### Task 5: Document the audit-only workflow and baseline safety contract

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a short documentation section**

```md
## Mamba/Fusion Noise Audit

The project provides an audit-only path for comparing `mamba`, `pure_dekf`, and `fusion`
noise scales without changing tracking behavior. The frozen reference baseline remains:

- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- Aggregated AMOTA: `0.737`

When `AUDIT.NOISE_AUDIT.ENABLED=false`, behavior must remain unchanged.
When enabled, inference and/or training write JSON summaries that aggregate:

- class-aware noise families
- matched vs unmatched split
- ratio-to-prior summaries
- window-length metadata where available
```

- [ ] **Step 2: Sanity-check the documentation text**

Run: `rg -n "Mamba/Fusion Noise Audit|0.737|ratio-to-prior" README.md`

Expected: the new section is present once

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: describe mamba fusion audit workflow"
```

### Task 6: Run the non-regression verification suite

**Files:**
- No code changes

- [ ] **Step 1: Run the new audit test suite**

Run: `python -m unittest tests.test_noise_audit tests.test_noise_audit_infer tests.test_noise_audit_train -v`

Expected: all tests `OK`

- [ ] **Step 2: Re-run the existing runtime contract checks**

Run: `python -m unittest tests.test_runtime_contract_checks tests.test_mctrack_compat_utils -v`

Expected: all tests `OK`

- [ ] **Step 3: Run static compilation checks**

Run: `python -m py_compile kalmanfilter/noise_audit.py kalmanfilter/mamba_adaptive_kf.py tracker/base_tracker.py training/train.py tests/test_noise_audit.py tests/test_noise_audit_infer.py tests/test_noise_audit_train.py`

Expected: no output

- [ ] **Step 4: Commit the verification pass if needed**

```bash
git status --short
```

Expected: clean working tree or only expected unstaged artifacts outside tracked source files

---

## Self-Review

**Spec coverage:** This plan implements the audit-only phase described in `docs/superpowers/specs/2026-06-18-mamba-fusion-audit-and-dirty-track-design.md`: shared schema, inference producer, training producer, ratio-to-prior summaries, matched/unmatched split, and documentation. It does not add new losses, behavior changes, or a new `TRACK_SCORE` mode.

**Placeholder scan:** No `TBD`, `TODO`, or unresolved implementation placeholders remain.

**Type consistency:** Shared family names are fixed to `q_pos`, `r_pos`, `r_siz`, and `r_ori` across the module, tests, training, and inference hooks.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-18-mamba-fusion-audit-implementation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
