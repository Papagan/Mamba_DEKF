# Association Shadow Assignment Audit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a decision-level shadow audit that measures how the learned association head would change assignments while guaranteeing that production tracking continues to use the frozen geometric assignment.

**Architecture:** Keep pair scoring and bounded correction unchanged, but split production and shadow assignment after the geometric cost matrix is built. A pure comparison module converts the two assignment results into decision records, while an audit accumulator merges exact counters, bounded samples, and a deterministic runtime manifest across scene workers.

**Tech Stack:** Python 3, NumPy, existing `lap` Hungarian wrapper, YAML configuration, `unittest`, JSON audit output.

---

## File Structure

- Create `tracker/association_runtime_manifest.py`: deterministic config/checkpoint/git fingerprints and manifest compatibility checks.
- Create `tracker/association_shadow_audit.py`: pure assignment comparison and GT outcome classification.
- Modify `tracker/association_head_audit.py`: schema v2 decision buckets, runtime manifest, exact merge counters, bounded samples.
- Modify `tracker/matching.py`: reusable assignment solver and side-effect-free shadow branch.
- Modify `tracker/base_tracker.py`: decision callback and frame context propagation.
- Modify `tracker/frame.py`: carry scene id as audit-only metadata.
- Modify `dataset/baseversion_dataset.py`: populate frame scene id.
- Modify `main.py`: build runtime manifest once and merge audit states with the configured sample bound.
- Modify `tools/compare_pairwise_assoc_train_infer.py`: stop treating lifecycle `matched` as a positive pair label.
- Modify `docs/association_head_audit.md`: document shadow mode, schema, and verification commands.
- Create `tests/test_association_runtime_manifest.py`: manifest determinism and compatibility tests.
- Create `tests/test_association_shadow_audit.py`: decision comparison tests.
- Modify `tests/test_association_head_audit.py`: schema v2 and merge tests.
- Modify `tests/test_mamba_association_prior.py`: production/shadow assignment integration tests.
- Modify `tests/test_compare_pairwise_assoc_train_infer.py`: corrected diagnostic semantics.

### Task 1: Deterministic Runtime Manifest

**Files:**
- Create: `tracker/association_runtime_manifest.py`
- Create: `tests/test_association_runtime_manifest.py`
- Modify: `main.py:317-335`

- [ ] **Step 1: Write failing manifest tests**

```python
class AssociationRuntimeManifestTest(unittest.TestCase):
    def test_manifest_normalizes_effective_association_contract(self):
        manifest = build_association_runtime_manifest(
            cfg={
                "FILTER_MODE": "mamba_multihead_closure",
                "TRACKER_COMPAT_MODE": "mctrack",
                "MATCHING": {"BEV": {"MATCHING_MODE": "Hungarian", "COST_MODE": {0: "RO_GDIOU_3D"}}},
                "MAMBA_ASSOCIATION_HEAD": {
                    "CHECKPOINT_PATH": "",
                    "ACTIVE_CLASS_STATES": {2: ["unmatched", "matched"]},
                    "APPLY_MODE": "margin_tiebreak",
                    "COST_MARGIN_EPS": 0.05,
                    "MIN_SCORE": 0.6,
                    "ALPHA": 0.05,
                    "MAX_DELTA": 0.03,
                },
            },
            config_path=self.config_path,
            process_count=12,
            git_root=self.root,
        )
        self.assertEqual(manifest["active_class_states"], {"2": ["matched", "unmatched"]})
        self.assertEqual(manifest["process_count"], 12)
        self.assertEqual(len(manifest["config_sha256"]), 64)

    def test_manifest_compatibility_rejects_changed_active_classes(self):
        left = {"config_sha256": "a", "active_class_states": {"2": ["matched"]}}
        right = {"config_sha256": "a", "active_class_states": {"3": ["matched"]}}
        self.assertFalse(runtime_manifests_compatible(left, right))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_association_runtime_manifest -v
```

Expected: import failure because `tracker.association_runtime_manifest` does not exist.

- [ ] **Step 3: Implement manifest helpers**

Implement these public functions:

```python
def sha256_file(path: str | Path) -> str | None: ...

def build_association_runtime_manifest(
    *, cfg: dict, config_path: str, process_count: int, git_root: str | Path
) -> dict: ...

def runtime_manifests_compatible(left: dict | None, right: dict | None) -> bool:
    return canonical_manifest(left) == canonical_manifest(right)
```

The builder must sort state names, stringify class keys, hash the YAML before `SAVE_PATH` is mutated, resolve the checkpoint relative to the git root, and record `git_commit` plus `git_dirty`. Hash failures return `null` checkpoint hashes rather than changing tracker behavior.

- [ ] **Step 4: Attach the manifest before multiprocessing**

Immediately after loading YAML in `main.py`, add:

```python
cfg["_ASSOCIATION_AUDIT_RUNTIME_MANIFEST"] = build_association_runtime_manifest(
    cfg=cfg,
    config_path=cfg_path,
    process_count=args.process,
    git_root=os.path.dirname(os.path.abspath(__file__)),
)
```

This field is audit-only and must not be included in matching calculations.

- [ ] **Step 5: Run tests and commit**

Run the new test and `python -m py_compile tracker/association_runtime_manifest.py main.py`. Expected: all pass.

```bash
git add tracker/association_runtime_manifest.py tests/test_association_runtime_manifest.py main.py
git commit -m "feat: record association audit runtime manifest"
```

### Task 2: Pure Shadow Decision Comparison

**Files:**
- Create: `tracker/association_shadow_audit.py`
- Create: `tests/test_association_shadow_audit.py`

- [ ] **Step 1: Write failing decision tests**

Cover unchanged assignment, changed detection, changed-to-unmatched, unknown GT, improved GT, and damaged GT. The core test is:

```python
def test_compare_assignments_classifies_improved_gt_decision(self):
    records = compare_assignment_results(
        production_matches=np.array([[0, 0]], dtype=int),
        production_costs=np.array([0.20]),
        shadow_matches=np.array([[0, 1]], dtype=int),
        shadow_costs=np.array([0.21]),
        base_cost_matrix=np.array([[0.20, 0.21]]),
        corrected_cost_matrix=np.array([[0.23, 0.21]]),
        association_scores=np.array([[0.1, 0.9]]),
        track_ids=[10],
        class_ids=[5],
        class_names=["trailer"],
        state_buckets=["unmatched"],
        track_gt_tokens=["gt-b"],
        det_gt_tokens=["gt-a", "gt-b"],
        context={"scene_id": "scene-1", "sample_token": "sample-1"},
        class_thresholds={5: 1.26},
    )
    self.assertTrue(records[0]["assignment_changed"])
    self.assertEqual(records[0]["outcome"], "improved")
    self.assertEqual(records[0]["track_state_bucket"], "unmatched")
```

- [ ] **Step 2: Run and verify RED**

Run `python -m unittest tests.test_association_shadow_audit -v`. Expected: missing module failure.

- [ ] **Step 3: Implement pure comparison functions**

Expose:

```python
def assignment_map(matches: np.ndarray, costs: np.ndarray) -> dict[int, dict]: ...

def classify_assignment_outcome(
    track_gt_token, production_det_token, shadow_det_token
) -> str: ...

def compare_assignment_results(
    *, production_matches, production_costs, shadow_matches, shadow_costs,
    base_cost_matrix, corrected_cost_matrix, association_scores,
    track_ids, class_ids, class_names, state_buckets,
    track_gt_tokens=None, det_gt_tokens=None, context=None,
    class_thresholds=None,
) -> list[dict]: ...
```

Return one record per track. Never infer pair correctness when either identity is unavailable. Compute cost ranks from the original same-row finite candidates and record threshold crossing separately from assignment change.

- [ ] **Step 4: Run tests and commit**

Run the new tests and `py_compile`. Expected: all pass.

```bash
git add tracker/association_shadow_audit.py tests/test_association_shadow_audit.py
git commit -m "feat: compare production and shadow assignments"
```

### Task 3: Audit Schema v2 and Safe Multiprocess Merge

**Files:**
- Modify: `tracker/association_head_audit.py`
- Modify: `tests/test_association_head_audit.py`

- [ ] **Step 1: Write failing schema and merge tests**

Add tests asserting:

```python
acc = AssociationHeadAuditAccumulator(
    max_samples_per_bucket=2,
    runtime_manifest={"config_sha256": "abc"},
)
acc.add_decision({
    "class_id": 5,
    "class_name": "trailer",
    "track_state_bucket": "unmatched",
    "assignment_changed": True,
    "production_det_index": 0,
    "shadow_det_index": 1,
    "threshold_crossing": False,
    "outcome": "improved",
})
summary = acc.to_summary()
self.assertEqual(summary["schema_version"], 2)
self.assertEqual(summary["decision_buckets"][0]["improved_count"], 1)
self.assertEqual(summary["runtime_manifest"]["config_sha256"], "abc")
```

Also merge two scene states and assert that two bounded samples survive when the merged accumulator is configured with a bound of two. Add a strict incompatible-manifest test that raises `ValueError`.

- [ ] **Step 2: Run and verify RED**

Run `python -m unittest tests.test_association_head_audit -v`. Expected: constructor/add-decision failures.

- [ ] **Step 3: Add decision buckets and schema v2**

Keep existing pair buckets for backward-compatible score diagnostics. Add `_DecisionBucket` counters for total decisions, changed decisions, changed detection, changed-to-match, changed-to-unmatched, threshold crossings, known-GT decisions, improved, damaged, neutral, and unknown. Compute:

```python
net_improvement = improved_count - damaged_count
harm_precision = damaged_count / max(known_changed_count, 1)
```

Store bounded raw decisions and score/cost-gap lists. `merge_state()` must validate manifests before merging and must preserve samples up to `max_samples_per_bucket`.

- [ ] **Step 4: Preserve schema v1 input compatibility**

When merging an old scene state without `decision_buckets`, merge pair buckets and leave decision buckets empty. Mark the summary with `decision_evidence_available: false` instead of fabricating outcomes.

- [ ] **Step 5: Run tests and commit**

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_association_head_audit -v
git add tracker/association_head_audit.py tests/test_association_head_audit.py
git commit -m "feat: add decision-level association audit schema"
```

### Task 4: Side-Effect-Free Shadow Matching

**Files:**
- Modify: `tracker/matching.py`
- Modify: `tests/test_mamba_association_prior.py`

- [ ] **Step 1: Write a failing production-invariance test**

Create a two-track/two-detection matrix where the learned correction flips the shadow assignment. Patch only `cost_calculate_general` in the test and assert:

```python
production_matches, _ = match_trajs_and_dets(..., association_shadow_callback=records.extend)
self.assertEqual(production_matches.tolist(), [[0, 0], [1, 1]])
self.assertTrue(any(item["assignment_changed"] for item in records))
```

Run the same call with `SHADOW_MODE=false` and assert existing intervention behavior remains available for explicit A/B experiments.

- [ ] **Step 2: Run and verify RED**

Run the specific new test. Expected: unexpected keyword argument or changed production assignment.

- [ ] **Step 3: Extract one reusable assignment solver**

Move the existing category mask, transpose, Hungarian/Greedy call, and result conversion into:

```python
def solve_matching_cost_matrix(cost_matrix, trajs_category, dets_category, cfg, match_type):
    """Return matched_indices and selected costs without mutating tracker state."""
```

Before adding shadow behavior, run all existing matching tests and confirm this refactor preserves outputs.

- [ ] **Step 4: Add shadow branch**

In BEV matching:

```python
base_cost_matrix = apply_mamba_association_prior_to_cost_matrix(...)
corrected_cost_matrix = apply_pairwise_association_head_to_cost_matrix(
    np.array(base_cost_matrix, copy=True), ...
)
shadow_mode = bool((cfg.get("ASSOCIATION_HEAD_AUDIT", {}) or {}).get("SHADOW_MODE", False))
production_cost_matrix = base_cost_matrix if shadow_mode else corrected_cost_matrix
production = solve_matching_cost_matrix(production_cost_matrix, ...)
if shadow_mode:
    shadow = solve_matching_cost_matrix(corrected_cost_matrix, ...)
    association_shadow_callback(compare_assignment_results(...))
```

Do not run shadow assignment for RV matching. Missing scores or shape mismatches leave `corrected_cost_matrix` equal to the base matrix.

- [ ] **Step 5: Run tests and commit**

Run `tests.test_mamba_association_prior` and `tests.test_association_shadow_audit`. Expected: all pass.

```bash
git add tracker/matching.py tests/test_mamba_association_prior.py
git commit -m "feat: add non-mutating association shadow matching"
```

### Task 5: Tracker Context and Main-Process Merge

**Files:**
- Modify: `tracker/frame.py`
- Modify: `dataset/baseversion_dataset.py`
- Modify: `tracker/base_tracker.py`
- Modify: `main.py`
- Modify: `tests/test_association_head_audit.py`

- [ ] **Step 1: Write failing context and merge tests**

Test that `BaseVersionTrackingDataset` places `scene_id` and `cur_sample_token` on each frame. Test `_write_merged_association_head_audit()` with two scene states and `MAX_SAMPLES_PER_BUCKET=2`, asserting samples are retained in the output JSON.

- [ ] **Step 2: Run and verify RED**

Run the new tests. Expected: missing scene id or zero merged samples.

- [ ] **Step 3: Wire context into matching**

Add `scene_id` to `Frame`. Pass this context from `Base3DTracker`:

```python
association_context = {
    "scene_id": getattr(frame_info, "scene_id", None),
    "sample_token": getattr(frame_info, "cur_sample_token", None),
}
```

Add `_record_association_shadow_decisions(records)` to forward each record to `AssociationHeadAuditAccumulator.add_decision()`.

- [ ] **Step 4: Construct and merge accumulators with the runtime contract**

Initialize scene trackers with:

```python
AssociationHeadAuditAccumulator(
    max_samples_per_bucket=int(audit_cfg.get("MAX_SAMPLES_PER_BUCKET", 0)),
    runtime_manifest=cfg.get("_ASSOCIATION_AUDIT_RUNTIME_MANIFEST"),
    strict_manifest=bool(audit_cfg.get("STRICT", False)),
)
```

Use the same arguments for the merged accumulator in `main.py`; this fixes the current loss of bounded samples.

- [ ] **Step 5: Add fail-closed handling**

If the shadow callback raises and `STRICT=false`, record an `audit_errors` counter and continue with the production assignment. If `STRICT=true`, re-raise after the production result has been computed. No audit error may substitute a different match result.

- [ ] **Step 6: Run tests and commit**

Run association audit, shadow audit, matching, and dataset tests. Expected: all pass.

```bash
git add tracker/frame.py dataset/baseversion_dataset.py tracker/base_tracker.py main.py tests/test_association_head_audit.py
git commit -m "feat: wire shadow decisions into merged audit"
```

### Task 6: Correct Diagnostics and Document the A/B Procedure

**Files:**
- Modify: `tools/compare_pairwise_assoc_train_infer.py`
- Modify: `tests/test_compare_pairwise_assoc_train_infer.py`
- Modify: `docs/association_head_audit.md`

- [ ] **Step 1: Write failing diagnostic tests**

For schema v1 input with low matched-state scores, assert the result contains `insufficient_decision_evidence` and does not contain `likely_distribution_mismatch`. For schema v2 input with 120 known changed decisions, 100 improved and 20 damaged, assert `candidate_for_shadow_ab_activation` is emitted.

- [ ] **Step 2: Run and verify RED**

Run `python -m unittest tests.test_compare_pairwise_assoc_train_infer -v`. Expected: old diagnosis names fail assertions.

- [ ] **Step 3: Replace lifecycle-based diagnoses**

Keep train-cache class balance and distance summaries. Rename inference sections to `track_state_scores`. Generate activation diagnoses only from decision buckets:

```python
if known_changed_count == 0:
    diagnoses.append("insufficient_decision_evidence")
elif improved_count > damaged_count and harm_precision <= 0.20:
    diagnoses.append("candidate_for_shadow_ab_activation")
else:
    diagnoses.append("shadow_decisions_not_safe")
```

- [ ] **Step 4: Update audit documentation**

Document the safe experimental block without changing the frozen config:

```yaml
MAMBA_ASSOCIATION_HEAD:
  ENABLED: true

ASSOCIATION_HEAD_AUDIT:
  ENABLED: true
  SHADOW_MODE: true
  INFER_OUTPUT_PATH: "debug/association_head_shadow_audit.json"
  MAX_SAMPLES_PER_BUCKET: 32
  STRICT: true
```

State explicitly that `SHADOW_MODE=true` must reproduce the frozen `results.json`, and that real intervention requires `SHADOW_MODE=false` in a separate A/B config.

- [ ] **Step 5: Run tests and commit**

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_compare_pairwise_assoc_train_infer -v
git add tools/compare_pairwise_assoc_train_infer.py tests/test_compare_pairwise_assoc_train_infer.py docs/association_head_audit.md
git commit -m "docs: define association shadow audit decision gates"
```

### Task 7: Verification and Handoff

**Files:**
- Verify all files changed by Tasks 1-6.

- [ ] **Step 1: Run focused regression tests**

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_association_runtime_manifest \
  tests.test_association_shadow_audit \
  tests.test_association_head_audit \
  tests.test_mamba_association_prior \
  tests.test_compare_pairwise_assoc_train_infer -v
```

Expected: all tests pass with no warnings.

- [ ] **Step 2: Run syntax and whitespace checks**

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m py_compile \
  tracker/association_runtime_manifest.py \
  tracker/association_shadow_audit.py \
  tracker/association_head_audit.py \
  tracker/matching.py \
  tracker/base_tracker.py \
  main.py \
  tools/compare_pairwise_assoc_train_infer.py
git diff --check
```

Expected: exit code zero.

- [ ] **Step 3: Verify the baseline safety invariant locally**

Run the synthetic integration test once with the association head disabled and once with shadow mode enabled. Assert byte-equivalent production match arrays and costs.

- [ ] **Step 4: Prepare the remote verification commands**

Remote sequence:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo_assoc_shadow.yaml \
  -p 12

sha256sum /path/to/frozen/results.json /path/to/shadow/results.json

python tools/compare_pairwise_assoc_train_infer.py \
  --train-cache /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --infer-audit debug/association_head_shadow_audit.json \
  --output docs/pairwise_assoc_train_infer_compare.json
```

The repository does not create the experimental full YAML automatically; derive it from the frozen remote YAML and change only `ASSOCIATION_HEAD_AUDIT.SHADOW_MODE`, output path, and strictness. The frozen YAML remains untouched.

- [ ] **Step 5: Request code review before integration**

Use `superpowers:requesting-code-review`, address findings, rerun all verification, then use `superpowers:finishing-a-development-branch` to merge or hand off the isolated branch.
