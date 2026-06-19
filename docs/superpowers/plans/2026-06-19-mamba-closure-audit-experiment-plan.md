# Mamba Closure Audit Experiment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Create a fair three-mode audit experiment (`pure_dekf`, `fusion`, `mamba`) on top of the frozen `0.737` baseline and add a small comparison utility that highlights cross-mode scale drift without changing tracking behavior.

**Architecture:** Derive three experiment configs from the frozen exact-hybrid baseline, enable audit output with per-mode destinations, and add a small analysis tool that reads the shared noise-audit JSON schema and produces compact class/state/family comparisons. Keep the live baseline untouched and keep all new behavior opt-in and sidecar-only.

**Tech Stack:** YAML configs, Python, JSON, existing audit schema in `kalmanfilter/noise_audit.py`, `unittest`

---

## File Structure

**Create**
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_pure_dekf.yaml` — frozen baseline clone with `FILTER_MODE: pure_dekf` and audit output path
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_fusion.yaml` — frozen baseline clone with `FILTER_MODE: fusion` and audit output path
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba.yaml` — frozen baseline clone with `FILTER_MODE: mamba` and audit output path
- `tools/summarize_noise_audit.py` — compare one or more audit JSON files and emit a compact report
- `tests/test_summarize_noise_audit.py` — focused tests for the summarizer

**Modify**
- `README.md` — add a short “three-mode audit experiment” usage note if needed

**Do not modify**
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`
- tracking logic
- KF logic
- loss definitions

---

### Task 1: Create the three frozen-baseline audit configs

**Files:**
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_pure_dekf.yaml`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_fusion.yaml`
- Create: `config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba.yaml`

- [ ] **Step 1: Copy the frozen baseline config as the source of truth**

Read:
- `config/nuscenes_single_stage_mctrack_exact_noise_hybrid.yaml`

Verify these must stay identical across all three experiment configs:
- thresholds
- exact matched/unmatched update settings
- noise priors
- tracker compat mode
- evaluation settings

- [ ] **Step 2: Create the `pure_dekf` audit config**

Write a new config whose only intended semantic changes are:

```yaml
FILTER_MODE: pure_dekf
AUDIT:
  NOISE_AUDIT:
    ENABLED: true
    STRICT: false
    INFER_OUTPUT_PATH: debug/noise_audit/pure_dekf/infer_noise_audit.json
```

- [ ] **Step 3: Create the `fusion` audit config**

Write a new config whose only intended semantic changes are:

```yaml
FILTER_MODE: fusion
AUDIT:
  NOISE_AUDIT:
    ENABLED: true
    STRICT: false
    INFER_OUTPUT_PATH: debug/noise_audit/fusion/infer_noise_audit.json
```

- [ ] **Step 4: Create the `mamba` audit config**

Write a new config whose only intended semantic changes are:

```yaml
FILTER_MODE: mamba
AUDIT:
  NOISE_AUDIT:
    ENABLED: true
    STRICT: false
    INFER_OUTPUT_PATH: debug/noise_audit/mamba/infer_noise_audit.json
```

- [ ] **Step 5: Parse-check the three configs**

Run:

```bash
python - <<'PY'
import yaml
for path in [
    'config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_pure_dekf.yaml',
    'config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_fusion.yaml',
    'config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_mamba.yaml',
]:
    with open(path, 'r', encoding='utf-8') as f:
        yaml.safe_load(f)
    print(path, 'ok')
PY
```

Expected: all three print `ok`

---

### Task 2: Add a small audit summarizer

**Files:**
- Create: `tools/summarize_noise_audit.py`
- Test: `tests/test_summarize_noise_audit.py`

- [ ] **Step 1: Write the failing summarizer tests**

Add focused tests that verify:

- the tool can load a single audit JSON and emit a compact per-bucket summary
- the tool can compare two or more modes on the same `(class_name, state, family)`
- the tool highlights ratio medians and p90 values

Example fixture shape:

```python
sample = {
    "schema_version": 1,
    "families": ["q_pos", "r_pos", "r_siz", "r_ori"],
    "buckets": [
        {
            "split": "infer",
            "mode": "fusion",
            "class_id": 5,
            "class_name": "trailer",
            "state": "matched",
            "history_len": 6,
            "count": 10,
            "families": {"q_pos": {"median": 2.0, "p90": 3.0}},
            "ratios": {"q_pos": {"median": 1.4, "p90": 1.9}},
        }
    ],
}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```bash
python -m unittest tests.test_summarize_noise_audit -v
```

Expected: import failure or missing CLI/helper

- [ ] **Step 3: Implement the minimal summarizer**

Implement a small tool that:

- loads one or more audit JSON files
- groups rows by `(class_name, state, history_len optional, family)`
- exposes compact comparison output using:
  - family median
  - family p90
  - ratio median
  - ratio p90

Keep the first version simple:
- no plotting
- no pandas dependency
- text or JSON summary only

- [ ] **Step 4: Re-run the tests**

Run:

```bash
python -m unittest tests.test_summarize_noise_audit -v
```

Expected: `OK`

---

### Task 3: Document the experiment entry points

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add a concise usage note**

Document:

- the three experiment configs
- the fact that they are fair derivatives of the frozen `0.737` baseline
- a sample evaluation command:

```bash
python main.py --dataset nuscenes --eval --config config/nuscenes_single_stage_mctrack_exact_noise_hybrid_audit_fusion.yaml -p 12
```

- the summarizer entry point, for example:

```bash
python tools/summarize_noise_audit.py \
  --inputs debug/noise_audit/pure_dekf/infer_noise_audit.json \
           debug/noise_audit/fusion/infer_noise_audit.json \
           debug/noise_audit/mamba/infer_noise_audit.json
```

- [ ] **Step 2: Verify the README anchors exist once**

Run:

```bash
rg -n "audit_pure_dekf|audit_fusion|audit_mamba|summarize_noise_audit" README.md
```

Expected: one short block for the experiment instructions

---

### Task 4: Run non-regression checks for the experiment scaffolding

**Files:**
- No additional code changes

- [ ] **Step 1: Run the existing audit suite**

Run:

```bash
python -m unittest tests.test_noise_audit tests.test_noise_audit_infer tests.test_noise_audit_train tests.test_summarize_noise_audit -v
```

Expected: all tests `OK`

- [ ] **Step 2: Run static compilation checks**

Run:

```bash
python -m py_compile \
  kalmanfilter/noise_audit.py \
  kalmanfilter/mamba_adaptive_kf.py \
  tracker/base_tracker.py \
  training/train.py \
  tools/summarize_noise_audit.py \
  tests/test_noise_audit.py \
  tests/test_noise_audit_infer.py \
  tests/test_noise_audit_train.py \
  tests/test_summarize_noise_audit.py \
  main.py
```

Expected: no output

---

## Self-Review

**Spec coverage:** This plan only implements the experiment scaffolding required to run the three-mode audit and summarize the results. It does not introduce any bounded fusion, scale constraints, or `TRACK_SCORE` redesign yet.

**Placeholder scan:** No intentional placeholders remain; every task has concrete files, commands, and expected outcomes.

**Type consistency:** The experiment relies on the existing audit schema with `schema_version`, `families`, and bucket-level `families/ratios` summaries and does not invent a second format.

---

Plan complete and saved to `docs/superpowers/plans/2026-06-19-mamba-closure-audit-experiment-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
