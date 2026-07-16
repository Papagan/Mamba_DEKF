# Inference-Style Association Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Export grouped association training samples directly from the real pre-association inference path without changing production matching or tracking output.

**Architecture:** A pure shared pair composer guarantees that runtime scoring and cache export use the same vector. `Base3DTracker` stages exact embeddings and pair vectors, matching enriches them with untouched geometric costs and production assignments through a read-only callback, and a dedicated accumulator validates and deterministically merges scene states before atomic pickle/JSON writes.

**Tech Stack:** Python 3, NumPy, PyTorch tensors converted to bounded CPU arrays, YAML/argparse configuration, pickle/JSON serialization, multiprocessing manager state, `unittest`.

---

## File Structure

- Create `training/association_features.py`: pure pair geometry and pair-vector composition shared by training/runtime code.
- Create `tracker/inference_pair_cache.py`: candidate-group construction, GT labeling, validation, deterministic accumulation, manifest construction, atomic writers.
- Create `tests/test_association_features.py`: exact vector composition and shape/error tests.
- Create `tests/test_inference_pair_cache.py`: group schema, GT/no-match semantics, accumulator, merge, and atomic-write tests.
- Create `tests/test_inference_pair_cache_matching.py`: matching callback and production-invariance tests.
- Create `tests/test_inference_pair_cache_wiring.py`: `Base3DTracker` payload and `main.py` multiprocess/CLI wiring tests.
- Modify `training/association_model.py`: call the shared pair composer for batched model input.
- Modify `tracker/base_tracker.py`: build exact runtime pair payload, own scene accumulator, expose state, and pass read-only callback options.
- Modify `tracker/matching.py`: emit candidate groups after production assignment without changing cost/solver inputs.
- Modify `main.py`: CLI overrides, per-scene state collection, deterministic merge, and atomic output.
- Modify `docs/association_head_audit.md`: add remote export and invariance verification commands.

The existing `tools/build_pairwise_association_cache.py` and
`training/pairwise_association_cache.py` remain unchanged for old-cache comparison.

### Task 1: Shared Pair Vector Composition

**Files:**
- Create: `training/association_features.py`
- Create: `tests/test_association_features.py`
- Modify: `training/association_model.py:106-125`

- [ ] **Step 1: Write failing tests for exact composition**

Create `tests/test_association_features.py` with tests that define the required
vector order and reject shape mismatches:

```python
import unittest

import torch

from training.association_features import compose_pair_vectors


class AssociationFeaturesTest(unittest.TestCase):
    def test_compose_pair_vectors_matches_runtime_contract(self):
        track = torch.tensor([[1.0, 2.0]])
        det = torch.tensor([[3.0, 5.0]])
        geometry = torch.tensor([[0.4, 0.2, 1.5, 0.8, 0.9, 1.0, 5.0]])

        actual = compose_pair_vectors(track, det, geometry)

        expected = torch.tensor([[1.0, 2.0, 3.0, 5.0, 2.0, 3.0,
                                  3.0, 10.0, 0.4, 0.2, 1.5, 0.8,
                                  0.9, 1.0, 5.0]])
        torch.testing.assert_close(actual, expected)

    def test_compose_pair_vectors_rejects_misaligned_batches(self):
        with self.assertRaisesRegex(ValueError, "batch dimension"):
            compose_pair_vectors(
                torch.zeros(2, 4), torch.zeros(3, 4), torch.zeros(2, 7)
            )

    def test_compose_pair_vectors_requires_seven_geometry_features(self):
        with self.assertRaisesRegex(ValueError, "seven"):
            compose_pair_vectors(
                torch.zeros(2, 4), torch.zeros(2, 4), torch.zeros(2, 6)
            )
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_association_features -v
```

Expected: import failure because `training.association_features` does not exist.

- [ ] **Step 3: Implement the pure composer**

Create `training/association_features.py`:

```python
from __future__ import annotations

import torch
from torch import Tensor


PAIR_GEOMETRY_DIM = 7


def compose_pair_vectors(
    track_embeddings: Tensor,
    detection_embeddings: Tensor,
    pair_geometry: Tensor,
) -> Tensor:
    if track_embeddings.ndim != 2 or detection_embeddings.ndim != 2:
        raise ValueError("embeddings must have shape [B, E]")
    if pair_geometry.ndim != 2:
        raise ValueError("pair geometry must have shape [B, 7]")
    if not (
        track_embeddings.shape[0]
        == detection_embeddings.shape[0]
        == pair_geometry.shape[0]
    ):
        raise ValueError("pair inputs must share the batch dimension")
    if track_embeddings.shape[1] != detection_embeddings.shape[1]:
        raise ValueError("track and detection embedding dimensions must match")
    if pair_geometry.shape[1] != PAIR_GEOMETRY_DIM:
        raise ValueError("pair geometry must contain exactly seven features")
    return torch.cat(
        [
            track_embeddings,
            detection_embeddings,
            torch.abs(track_embeddings - detection_embeddings),
            track_embeddings * detection_embeddings,
            pair_geometry,
        ],
        dim=-1,
    )
```

Modify `PairwiseAssociationModel.forward()` to replace its local `torch.cat`
with `compose_pair_vectors(track_embedding, det_embedding, pair_features)`.

- [ ] **Step 4: Run focused model tests**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_association_features tests.test_multihead_association -v
```

Expected: all tests pass and the model head input dimension is unchanged.

- [ ] **Step 5: Commit**

```bash
git add training/association_features.py training/association_model.py \
  tests/test_association_features.py
git commit -m "refactor: share association pair composition"
```

### Task 2: Pure Inference Candidate Group Builder

**Files:**
- Create: `tracker/inference_pair_cache.py`
- Create: `tests/test_inference_pair_cache.py`

- [ ] **Step 1: Write failing schema and label tests**

Add tests for one matched group, one GT no-match group, and unknown GT:

```python
import unittest

import numpy as np

from tracker.inference_pair_cache import build_candidate_groups


class InferencePairCacheGroupTest(unittest.TestCase):
    def _payload(self):
        return {
            "scene_id": "scene-1",
            "sample_token": "sample-1",
            "frame_id": 3,
            "track_ids": [17],
            "class_ids": [5],
            "class_names": ["trailer"],
            "state_buckets": ["unmatched"],
            "track_gt_tokens": ["gt-b"],
            "det_gt_tokens": ["gt-a", "gt-b"],
            "pair_indices": [(0, 0), (0, 1)],
            "pair_vectors": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "track_embeddings": np.asarray([[0.1, 0.2]], dtype=np.float32),
            "det_embeddings": np.asarray([[0.3, 0.4], [0.5, 0.6]], dtype=np.float32),
            "association_scores": np.asarray([[0.1, 0.9]], dtype=np.float32),
        }

    def test_group_keeps_production_and_gt_labels_separate(self):
        groups = build_candidate_groups(
            payload=self._payload(),
            base_cost_matrix=np.asarray([[0.2, 0.21]], dtype=np.float32),
            production_matches=np.asarray([[0, 0]], dtype=np.int64),
            production_costs=np.asarray([0.2], dtype=np.float32),
            thresholds={5: 1.26},
        )

        group = groups[0]
        self.assertEqual(group["production_det_index"], 0)
        self.assertEqual(group["gt_target_det_index"], 1)
        self.assertEqual(group["gt_target_status"], "matched")
        self.assertTrue(group["candidates"][0]["production_selected"])
        self.assertEqual(group["candidates"][0]["gt_label"], 0)
        self.assertEqual(group["candidates"][1]["gt_label"], 1)

    def test_known_track_without_gt_candidate_is_no_match(self):
        payload = self._payload()
        payload["track_gt_tokens"] = ["missing-gt"]
        group = build_candidate_groups(
            payload=payload,
            base_cost_matrix=np.asarray([[0.2, 0.21]], dtype=np.float32),
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,), dtype=np.float32),
            thresholds={5: 1.26},
        )[0]
        self.assertEqual(group["gt_target_status"], "no_match")
        self.assertIsNone(group["gt_target_det_index"])

    def test_missing_track_gt_is_unknown_not_negative(self):
        payload = self._payload()
        payload["track_gt_tokens"] = [None]
        group = build_candidate_groups(
            payload=payload,
            base_cost_matrix=np.asarray([[0.2, 0.21]], dtype=np.float32),
            production_matches=np.empty((0, 2), dtype=np.int64),
            production_costs=np.empty((0,), dtype=np.float32),
            thresholds={5: 1.26},
        )[0]
        self.assertEqual(group["gt_target_status"], "unknown")
        self.assertTrue(all(item["gt_label"] is None for item in group["candidates"]))
```

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache.InferencePairCacheGroupTest -v
```

Expected: import failure because `tracker.inference_pair_cache` does not exist.

- [ ] **Step 3: Implement validation and group construction**

Create `tracker/inference_pair_cache.py` with:

```python
INFERENCE_PAIR_CACHE_SCHEMA_VERSION = 1
TRACK_STATES = {"matched", "unmatched"}


def build_candidate_groups(
    *,
    payload,
    base_cost_matrix,
    production_matches,
    production_costs,
    thresholds,
    include_embeddings=True,
    include_pair_vectors=True,
):
    """Build JSON/pickle-safe candidate groups without mutating inputs."""
    # Validate matrix shapes, stable non-negative unique track ids, aligned
    # metadata, finite pair vectors, canonical states, and pair-index coverage.
    # Copy all arrays before converting them to Python lists.
    # Rank only finite same-class candidates by the untouched base cost.
    # Keep production_selected independent from GT-derived labels.
```

Use an assignment map keyed by track index. For each track, sort finite candidates
by `(geometric_cost, det_index)`, assign one-based ranks, and retain non-finite
candidates with `candidate_rank=None`. Determine GT status as follows:

```python
matching_gt_indices = [
    index for index, token in enumerate(det_gt_tokens)
    if track_gt_token is not None and token == track_gt_token
]
if track_gt_token is None:
    target_status = "unknown"
elif len(matching_gt_indices) == 0:
    target_status = "no_match"
elif len(matching_gt_indices) == 1:
    target_status = "matched"
else:
    target_status = "unknown"
```

An ambiguous duplicate token must never create two positives.
Create one group for every input track, including tracks with no same-class
candidate. Such a group has `candidates=[]`, no production detection, and either
`gt_target_status="no_match"` or `"unknown"` according to audit-GT availability.
When `include_embeddings` or `include_pair_vectors` is false, omit only the
corresponding serialized fields; candidate identity, cost, gate, rank, assignment,
and GT labels remain unchanged.

- [ ] **Step 4: Add malformed-input and rank/gate tests**

Add tests that assert:

```python
payload = self._payload()
payload["pair_indices"] = [(0, 0), (0, 0)]
with self.assertRaisesRegex(ValueError, "pair_indices"):
    build_candidate_groups(
        payload=payload,
        base_cost_matrix=np.asarray([[0.2, 0.21]], dtype=np.float32),
        production_matches=np.empty((0, 2), dtype=np.int64),
        production_costs=np.empty((0,), dtype=np.float32),
        thresholds={5: 1.26},
    )

self.assertEqual(group["candidates"][0]["candidate_rank"], 1)
self.assertAlmostEqual(group["candidates"][1]["cost_gap"], 0.01)
self.assertTrue(group["candidates"][0]["inside_geometric_gate"])
```

Cover boolean/negative/duplicate track ids, wrong matrix shape, non-finite pair
vectors, invalid state, duplicate GT positives, missing score matrix, and a track
with zero same-class candidates. Parameterize the serialization tests over all
four `include_embeddings`/`include_pair_vectors` combinations and assert that the
default keeps both exact runtime inputs.

- [ ] **Step 5: Run and commit**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_inference_pair_cache -v
```

Expected: all group tests pass.

```bash
git add tracker/inference_pair_cache.py tests/test_inference_pair_cache.py
git commit -m "feat: build inference association candidate groups"
```

### Task 3: Deterministic Accumulator and Atomic Writers

**Files:**
- Modify: `tracker/inference_pair_cache.py`
- Modify: `tests/test_inference_pair_cache.py`

- [ ] **Step 1: Write failing accumulator and writer tests**

Add tests for deterministic caps, merge order, manifest mismatch, and atomic
replacement:

```python
def test_merge_order_does_not_change_canonical_payload(self):
    left = InferencePairCacheAccumulator(runtime_manifest=self.manifest)
    right = InferencePairCacheAccumulator(runtime_manifest=self.manifest)
    left.add_group(self.group(frame_id=2, track_id=20))
    right.add_group(self.group(frame_id=1, track_id=10))

    first = InferencePairCacheAccumulator(runtime_manifest=self.manifest)
    first.merge_state(left.export_state())
    first.merge_state(right.export_state())
    second = InferencePairCacheAccumulator(runtime_manifest=self.manifest)
    second.merge_state(right.export_state())
    second.merge_state(left.export_state())

    self.assertEqual(first.to_payload(), second.to_payload())

def test_atomic_pickle_replaces_only_after_success(self):
    atomic_write_pickle(self.output_path, {"schema_version": 1, "groups": []})
    self.assertEqual(load_pickle(self.output_path)["schema_version"], 1)
```

Also assert a positive per-bucket cap retains the lexicographically smallest
stable group keys independent of insertion order. Manifest tests must reject a
worker state when either the main Mamba checkpoint hash, association-head
checkpoint hash, or effective-config hash differs.

- [ ] **Step 2: Run the tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache.InferencePairCacheAccumulatorTest -v
```

Expected: missing class/function failures.

- [ ] **Step 3: Implement accumulator, manifest, and writers**

Add:

```python
import copy
import hashlib
import json
import os
import pickle
import subprocess
import tempfile
from collections import defaultdict
from pathlib import Path


class InferencePairCacheAccumulator:
    def __init__(
        self,
        *,
        runtime_manifest=None,
        max_groups_per_class_state=0,
        require_audit_gt=False,
        max_error_samples=32,
    ):
        self.runtime_manifest = copy.deepcopy(runtime_manifest or {})
        self.max_groups_per_class_state = int(max_groups_per_class_state)
        self.require_audit_gt = bool(require_audit_gt)
        self.max_error_samples = int(max_error_samples)
        self._groups = {}
        self._error_count = 0
        self._error_samples = []

    @staticmethod
    def _group_key(group):
        return (
            str(group["scene_id"]),
            str(group["sample_token"]),
            int(group["frame_id"]),
            int(group["track_id"]),
        )

    def add_group(self, group):
        normalized = validate_candidate_group(group)
        key = self._group_key(normalized)
        existing = self._groups.get(key)
        if existing is not None and existing != normalized:
            raise ValueError(f"conflicting inference pair group: {key}")
        self._groups[key] = normalized
        return not (
            self.require_audit_gt and normalized["gt_target_status"] == "unknown"
        )

    def add_error(self, record):
        normalized = validate_error_record(record)
        self._error_count += 1
        if len(self._error_samples) < self.max_error_samples:
            self._error_samples.append(normalized)

    def export_state(self):
        return {
            "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "runtime_manifest": copy.deepcopy(self.runtime_manifest),
            "groups": [copy.deepcopy(self._groups[key]) for key in sorted(self._groups)],
            "error_count": self._error_count,
            "error_samples": copy.deepcopy(self._error_samples),
        }

    def merge_state(self, state):
        normalized = validate_accumulator_state(state)
        if canonical_json(normalized["runtime_manifest"]) != canonical_json(self.runtime_manifest):
            raise ValueError("inference pair cache runtime manifest mismatch")
        for group in normalized["groups"]:
            self.add_group(group)
        self._error_count += normalized["error_count"]
        remaining = max(0, self.max_error_samples - len(self._error_samples))
        self._error_samples.extend(normalized["error_samples"][:remaining])

    def _retained_groups(self):
        groups = [self._groups[key] for key in sorted(self._groups)]
        if self.require_audit_gt:
            groups = [group for group in groups if group["gt_target_status"] != "unknown"]
        if self.max_groups_per_class_state <= 0:
            return groups
        buckets = defaultdict(list)
        for group in groups:
            buckets[(group["class_id"], group["state_bucket"])].append(group)
        retained = []
        for bucket in sorted(buckets):
            retained.extend(buckets[bucket][: self.max_groups_per_class_state])
        return sorted(retained, key=self._group_key)

    def to_payload(self):
        return {
            "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
            "runtime_manifest": copy.deepcopy(self.runtime_manifest),
            "groups": copy.deepcopy(self._retained_groups()),
        }

    def to_summary(self):
        return summarize_candidate_groups(
            observed_groups=[self._groups[key] for key in sorted(self._groups)],
            retained_groups=self._retained_groups(),
            error_count=self._error_count,
            error_samples=self._error_samples,
            runtime_manifest=self.runtime_manifest,
        )


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_inference_pair_cache_manifest(
    *, cfg, config_path, process_count, git_root, pair_vector_dim, embedding_dim
):
    config_file = Path(config_path).expanduser().resolve()

    def resolve_checkpoint(section_name):
        checkpoint_value = (cfg.get(section_name, {}) or {}).get("CHECKPOINT_PATH")
        if not checkpoint_value:
            return None
        checkpoint = Path(checkpoint_value).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = Path(git_root).resolve() / checkpoint
        return checkpoint.resolve()

    mamba_checkpoint = resolve_checkpoint("MAMBA")
    association_checkpoint = resolve_checkpoint("MAMBA_ASSOCIATION_HEAD")
    try:
        git_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(Path(git_root).resolve()),
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        git_commit = None
    return {
        "schema_version": INFERENCE_PAIR_CACHE_SCHEMA_VERSION,
        "config_path": str(config_file),
        "config_sha256": sha256_file(config_file),
        "effective_config_sha256": hashlib.sha256(
            canonical_json(cfg).encode("utf-8")
        ).hexdigest(),
        "git_commit": git_commit,
        "mamba_checkpoint_path": (
            None if mamba_checkpoint is None else str(mamba_checkpoint)
        ),
        "mamba_checkpoint_sha256": (
            None if mamba_checkpoint is None else sha256_file(mamba_checkpoint)
        ),
        "association_checkpoint_path": (
            None if association_checkpoint is None else str(association_checkpoint)
        ),
        "association_checkpoint_sha256": (
            None
            if association_checkpoint is None
            else sha256_file(association_checkpoint)
        ),
        "filter_mode": str(cfg.get("FILTER_MODE", "")).strip().lower(),
        "tracker_compat_mode": str(cfg.get("TRACKER_COMPAT_MODE", "default")).strip().lower(),
        "bev_cost_mode": copy.deepcopy(((cfg.get("MATCHING", {}) or {}).get("BEV", {}) or {}).get("COST_MODE", {})),
        "pair_vector_dim": int(pair_vector_dim),
        "embedding_dim": int(embedding_dim),
        "process_count": int(process_count),
        "class_map": copy.deepcopy(cfg.get("CATEGORY_MAP_TO_NUMBER", {})),
        "export_options": {
            "include_embeddings": bool(
                (cfg.get("INFERENCE_PAIR_CACHE", {}) or {}).get(
                    "INCLUDE_EMBEDDINGS", True
                )
            ),
            "include_pair_vectors": bool(
                (cfg.get("INFERENCE_PAIR_CACHE", {}) or {}).get(
                    "INCLUDE_PAIR_VECTORS", True
                )
            ),
            "require_audit_gt": bool(
                (cfg.get("INFERENCE_PAIR_CACHE", {}) or {}).get(
                    "REQUIRE_AUDIT_GT", False
                )
            ),
        },
    }


def _atomic_write(path, write_value):
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(destination.parent), prefix=f".{destination.name}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file_obj:
            write_value(file_obj)
            file_obj.flush()
            os.fsync(file_obj.fileno())
        os.replace(temporary, destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def atomic_write_pickle(path, payload):
    _atomic_write(path, lambda file_obj: pickle.dump(payload, file_obj))


def atomic_write_json(path, payload):
    def write_json(file_obj):
        file_obj.write(
            (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
        )
    _atomic_write(path, write_json)
```

Define the referenced validators in the same module, not as permissive pass-through
helpers:

- `validate_candidate_group()` deep-copies to Python built-ins and requires the
  stable group key, class/state metadata, production assignment, GT target status,
  and candidate list. It rejects non-finite costs/vectors, booleans used as integer
  ids, duplicate candidate detection indices, more than one positive, and a
  `matched` status without exactly one positive.
- `validate_error_record()` requires `scene_id`, `sample_token`, `frame_id`,
  `stage`, `error_type`, and `message`, truncating only the stored message to a
  deterministic 1,024-character bound.
- `validate_accumulator_state()` requires the exact schema version, a mapping
  manifest, a list of validated groups, a non-negative integer `error_count`, and
  validated bounded error samples.
- `summarize_candidate_groups()` reports observed, rejected-unknown, retained, and
  capped group counts plus candidate/positive/negative/production-selected/gated
  counts overall and by class/state. Its `accepted_groups` value is exactly
  `len(retained_groups)` and every candidate count is recomputed from serialized
  retained groups rather than incrementally accumulated worker counters.

Use stable group key:

```python
(
    str(group["scene_id"]),
    str(group["sample_token"]),
    int(group["frame_id"]),
    int(group["track_id"]),
)
```

Store groups by key and serialize them sorted. Apply caps by sorting each
`(class_id, state_bucket)` bucket and retaining the first `K`; this makes merged
worker completion order irrelevant. Reject duplicate keys with unequal contents.
Use `tempfile.mkstemp(dir=destination.parent)` plus `os.replace` for both writers.

- [ ] **Step 4: Verify exact summary reconciliation**

Add a test that recomputes group/candidate/positive/negative/selected/gated counts
from `to_payload()["groups"]` and compares every retained-data field with
`to_summary()`. Add an unknown-GT group under `require_audit_gt=True` and assert it
increments `observed_groups` and `rejected_unknown_groups` but is absent from the
pickle payload and `accepted_groups`.

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest tests.test_inference_pair_cache -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add tracker/inference_pair_cache.py tests/test_inference_pair_cache.py
git commit -m "feat: accumulate inference association cache"
```

### Task 4: Read-Only Matching Export Callback

**Files:**
- Modify: `tracker/matching.py:459-590`
- Modify: `tracker/matching.py:735-1010`
- Create: `tests/test_inference_pair_cache_matching.py`

- [ ] **Step 1: Write production-invariance tests**

Create tests that patch `cost_calculate_general()` with a known matrix, invoke
matching with and without export, and compare results and input bytes:

```python
def test_export_callback_observes_but_cannot_change_production(self):
    observed = []
    original_cost = np.asarray([[0.10, 0.11], [0.11, 0.10]], dtype=np.float32)
    payload = make_pair_payload()

    with patch.object(
        matching,
        "cost_calculate_general",
        return_value=(original_cost, self.categories, self.categories),
    ):
        expected = matching.match_trajs_and_dets(self.trajs, self.dets, self.cfg)
        actual = matching.match_trajs_and_dets(
            self.trajs,
            self.dets,
            self.cfg,
            inference_pair_cache_payload=payload,
            inference_pair_cache_callback=observed.extend,
        )

    np.testing.assert_array_equal(actual[0], expected[0])
    np.testing.assert_array_equal(actual[1], expected[1])
    np.testing.assert_array_equal(original_cost, self.original_cost_copy)
    self.assertEqual(len(observed), 2)
```

Add a callback that mutates its received lists and prove the original cost,
payload arrays, matches, and costs remain unchanged. Add a callback that raises
and prove non-strict mode reports one error but returns production output. Add a
GT provider that asserts the patched production solver has already completed
before it is invoked; when export is disabled, assert the provider is never
called.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_matching -v
```

Expected: unexpected keyword argument failures.

- [ ] **Step 3: Add the optional callback interface**

Extend matching entry points with defaulted keyword arguments:

```python
inference_pair_cache_payload=None,
inference_pair_cache_callback=None,
inference_pair_cache_error_callback=None,
inference_pair_cache_gt_provider=None,
inference_pair_cache_strict=False,
```

After the production solver returns, call `build_candidate_groups()` with a
copy of the base cost matrix and copied assignment arrays. Only after that solver
result exists, call the optional GT provider and attach its copied
`track_gt_tokens`/`det_gt_tokens` to an exporter-local payload copy. Invoke the
callback with the returned list. Catch exporter and GT-provider exceptions only;
in non-strict mode route an error record to
`inference_pair_cache_error_callback` and return the already computed production
result. In strict mode re-raise after computation.

Preserve the existing immediate return when there are no tracks. For tracks with
no detections, preserve the exact empty production assignment but, only when both
the payload and callback are supplied, call the builder with an `[N, 0]` copied
cost matrix and empty assignment/cost arrays before returning. This is required to
capture real no-candidate groups and must not invoke the Hungarian solver. For a
non-empty detection list, the builder still emits every track even when class
filtering leaves that track with zero candidates.

For shadow mode, emit after the base production solver and before the shadow
correction. For non-shadow mode, emit after the actual production solver while
still recording the untouched base geometric costs. Forward the same arguments
through `match_trajs_and_dets_uncertainty_aware()` and its fallback branches.

- [ ] **Step 4: Run matching regressions**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_matching \
  tests.test_mamba_association_prior \
  tests.test_association_shadow_audit -v
```

Expected: all tests pass; existing shadow behavior remains unchanged.

- [ ] **Step 5: Commit**

```bash
git add tracker/matching.py tests/test_inference_pair_cache_matching.py
git commit -m "feat: export association groups from matching"
```

### Task 5: Base3DTracker Exact Pair Payload Wiring

**Files:**
- Modify: `tracker/base_tracker.py:1138-1205`
- Modify: `tracker/base_tracker.py:2430-2545`
- Create: `tests/test_inference_pair_cache_wiring.py`

- [ ] **Step 1: Write failing runtime payload tests**

Use AST-isolated methods following `tests/test_association_head_audit.py` to
avoid loading GPU dependencies. Assert that exact pair vectors are returned even
when the association head is absent but cache export is enabled:

```python
def test_pair_payload_uses_exact_shared_composer_without_loaded_head(self):
    tracker = make_tracker(pairwise_association_head=None, cache_enabled=True)
    scores, payload = compute_payload(
        tracker,
        trajs=[make_traj("trailer", track_id=17)],
        dets=[make_det("trailer")],
        trk_embeddings=np.asarray([[1.0, 2.0]], dtype=np.float32),
        det_embeddings=np.asarray([[3.0, 5.0]], dtype=np.float32),
        state_buckets=["unmatched"],
    )
    self.assertIsNone(scores)
    np.testing.assert_allclose(
        payload["pair_vectors"][0],
        np.asarray([1.0, 2.0, 3.0, 5.0, 2.0, 3.0, 3.0, 10.0,
                    payload["pair_vectors"][0][-7],
                    *payload["pair_vectors"][0][-6:]], dtype=np.float32),
    )
```

Also assert export-disabled behavior returns no payload and does not force
detection embedding inference.

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_wiring -v
```

Expected: missing method/config wiring failures.

- [ ] **Step 3: Refactor runtime scoring into payload plus optional scores**

Replace `_compute_pairwise_association_scores()` internals with a method returning:

```python
def _compute_pairwise_association_payload(
    self, trajs, dets, trk_embeddings, det_embeddings, state_buckets
):
    return association_scores, {
        "scene_id": self.scene_id,
        "sample_token": sample_token,
        "frame_id": frame_id,
        "track_ids": track_ids,
        "class_ids": class_ids,
        "class_names": class_names,
        "state_buckets": state_buckets,
        "pair_indices": pair_indices,
        "pair_vectors": pair_vectors,
        "track_embeddings": copied_track_embeddings,
        "det_embeddings": copied_det_embeddings,
        "association_scores": copied_score_matrix_or_none,
    }
```

Build vectors through `compose_pair_vectors()`. Run the head only when it is
loaded; payload construction is independent. Extend detection-embedding gating
so `INFERENCE_PAIR_CACHE.ENABLED` requests real embeddings even when both learned
association modes are disabled.

Read `INFERENCE_PAIR_CACHE` with code defaults rather than modifying the frozen
evaluation YAML: `ENABLED=False`, `MAX_GROUPS_PER_CLASS_STATE=0`,
`INCLUDE_EMBEDDINGS=True`, `INCLUDE_PAIR_VECTORS=True`,
`REQUIRE_AUDIT_GT=False`, and `STRICT=False`. Pass both include flags to
`build_candidate_groups()`. Pair-vector construction still occurs when
`INCLUDE_PAIR_VECTORS=False` if a loaded head needs it; the flag controls only
serialization.

- [ ] **Step 4: Add tracker accumulator and callback options**

In `Base3DTracker.__init__`, construct `InferencePairCacheAccumulator` only when
enabled. Add:

```python
def _record_inference_pair_groups(self, groups):
    for group in groups:
        self.inference_pair_cache.add_group(group)

def _record_inference_pair_cache_error(self, record):
    self.inference_pair_cache.add_error(record)

def export_inference_pair_cache_state(self):
    return None if self.inference_pair_cache is None else self.inference_pair_cache.export_state()
```

Build frame context in the payload. Wrap the existing sidecar helpers in a lazy
GT-provider callable that returns aligned copied track/detection token lists;
matching invokes it only after the production assignment exists. Pass the
payload/callback/error/GT-provider/strict options into both compatibility and
default matching paths. Do not construct or invoke the provider unless the
exporter is enabled.

- [ ] **Step 5: Run tracker and matching tests**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_wiring \
  tests.test_inference_pair_cache_matching \
  tests.test_association_head_audit \
  tests.test_mamba_association_prior -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add tracker/base_tracker.py tests/test_inference_pair_cache_wiring.py
git commit -m "feat: stage exact inference association pairs"
```

### Task 6: Multiprocess Merge, CLI Overrides, and Atomic Output

**Files:**
- Modify: `main.py:1-210`
- Modify: `main.py:317-445`
- Modify: `tests/test_inference_pair_cache_wiring.py`

- [ ] **Step 1: Write failing main wiring tests**

AST-load the new main helpers and assert sorted scene merge plus atomic writes:

```python
def test_main_merges_scene_cache_states_in_scene_order(self):
    write_merged = load_main_function("_write_merged_inference_pair_cache")
    with patch("tracker.inference_pair_cache.atomic_write_pickle") as write_pickle:
        write_merged(
            self.cfg,
            {"scene-b": self.state_b, "scene-a": self.state_a},
        )
    groups = write_pickle.call_args.args[1]["groups"]
    self.assertEqual([item["scene_id"] for item in groups], ["scene-a", "scene-b"])

def test_cli_output_override_enables_cache_without_yaml_edit(self):
    cfg = {"INFERENCE_PAIR_CACHE": {"ENABLED": False}}
    apply_cli_overrides(
        cfg,
        output_path="debug/runtime_pairs.pkl",
        summary_path="debug/runtime_pairs.json",
        strict=False,
    )
    self.assertTrue(cfg["INFERENCE_PAIR_CACHE"]["ENABLED"])
```

- [ ] **Step 2: Run tests and verify RED**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_wiring -v
```

Expected: missing helper failures.

- [ ] **Step 3: Add CLI flags and apply overrides before manifests**

Add:

```python
parser.add_argument("--inference-pair-cache-output", default=None)
parser.add_argument("--inference-pair-cache-summary", default=None)
parser.add_argument("--inference-pair-cache-strict", action="store_true")
parser.add_argument("--inference-pair-cache-require-gt", action="store_true")
parser.add_argument("--inference-pair-cache-omit-embeddings", action="store_true")
parser.add_argument("--inference-pair-cache-omit-pair-vectors", action="store_true")
```

When output is supplied, set `ENABLED`, `OUTPUT_PATH`, optional `SUMMARY_PATH`,
`STRICT`, `REQUIRE_AUDIT_GT`, `INCLUDE_EMBEDDINGS`, and `INCLUDE_PAIR_VECTORS`
before building either audit runtime manifest. The omit flags invert only their
named serialization option. Default behavior with no CLI flags must leave the
loaded YAML unchanged.

- [ ] **Step 4: Wire worker state collection and parent merge**

Add `scene_pair_cache_states = manager.dict()`, pass it into `run()`, collect
`tracker.export_inference_pair_cache_state()` at scene completion, convert the
manager dict to a normal dict, and call:

```python
_write_merged_inference_pair_cache(cfg, scene_pair_cache_states)
```

The writer constructs an accumulator from the effective config, merges states in
sorted scene-id order, writes pickle payload and JSON summary atomically, and
preserves tracking output when non-strict writing fails.

- [ ] **Step 5: Run main wiring and compile checks**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_inference_pair_cache_wiring \
  tests.test_association_head_audit -v
/home/alvin/anaconda3/envs/HiVT/bin/python -m py_compile \
  main.py tracker/base_tracker.py tracker/matching.py \
  tracker/inference_pair_cache.py training/association_features.py
```

Expected: tests and compilation pass.

- [ ] **Step 6: Commit**

```bash
git add main.py tests/test_inference_pair_cache_wiring.py
git commit -m "feat: write merged inference association cache"
```

### Task 7: Documentation and End-to-End Safety Verification

**Files:**
- Modify: `docs/association_head_audit.md`
- Modify: `tests/test_inference_pair_cache_matching.py`
- Modify: `tests/test_inference_pair_cache_wiring.py`

- [ ] **Step 1: Add the remote export command**

Document:

```bash
python main.py \
  --dataset nuscenes \
  --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml \
  -p 12 \
  --inference-pair-cache-output debug/inference_pair_cache.pkl \
  --inference-pair-cache-summary debug/inference_pair_cache_summary.json
```

Document that audit-GT conversion is required for supervised stage-three labels,
and add the stricter command variant with `--inference-pair-cache-require-gt`.

- [ ] **Step 2: Add disabled/enabled full matching invariance coverage**

Extend integration tests to cover:

```python
for shadow_mode in (False, True):
    for callback_enabled in (False, True):
        # Run identical Hungarian inputs and compare matches, costs, and matrices.
```

Cover normal matching, uncertainty-aware fallback, empty tracks, empty detections,
missing embeddings, missing GT, callback exception, and strict exception after
production solve. Load one serialized pair vector from the test cache into
`ClassConditionedAssociationHeadBank` and assert its last dimension equals the
head's declared `pair_input_dim` without reshaping or reconstruction.

- [ ] **Step 3: Run the focused second-stage suite**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_association_features \
  tests.test_inference_pair_cache \
  tests.test_inference_pair_cache_matching \
  tests.test_inference_pair_cache_wiring \
  tests.test_mamba_association_prior \
  tests.test_association_shadow_audit \
  tests.test_association_head_audit \
  tests.test_association_runtime_manifest \
  tests.test_association_checkpoint_audit -q
```

Expected: zero failures.

- [ ] **Step 4: Run the frozen association regression suite**

Run:

```bash
/home/alvin/anaconda3/envs/HiVT/bin/python -m unittest \
  tests.test_train_pairwise_association_cli \
  tests.test_train_association_stage \
  tests.test_compare_pairwise_assoc_train_infer \
  tests.test_multihead_association -q
git diff --check
```

Expected: zero failures and no whitespace errors in the feature worktree.

- [ ] **Step 5: Verify production code invariants by diff**

Inspect:

```bash
git diff 0900d45..HEAD -- tracker/matching.py tracker/base_tracker.py main.py
```

Confirm that all new behavior is gated by `INFERENCE_PAIR_CACHE.ENABLED`, callback
return values are unused, base cost matrices are copied before export, and GT is
read only for post-assignment labels.

- [ ] **Step 6: Commit documentation and final tests**

```bash
git add docs/association_head_audit.md \
  tests/test_inference_pair_cache_matching.py \
  tests/test_inference_pair_cache_wiring.py
git commit -m "docs: document inference association cache export"
```

## Remote Acceptance Run

After merging the implementation branch, run the frozen evaluation twice on the
remote machine with the same process count and checkpoint.

Baseline:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml -p 12
```

Export-enabled:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml -p 12 \
  --inference-pair-cache-output debug/inference_pair_cache.pkl \
  --inference-pair-cache-summary debug/inference_pair_cache_summary.json
```

Verify:

```bash
sha256sum /path/to/baseline/results.json /path/to/export/results.json
python - <<'PY'
import json
from pathlib import Path

summary = json.loads(Path("debug/inference_pair_cache_summary.json").read_text())
assert summary["overall"]["accepted_groups"] > 0
assert summary["overall"]["candidate_count"] > 0
assert set(summary["per_class"]) == {
    "bicycle", "bus", "car", "motorcycle", "pedestrian", "trailer", "truck"
}
print(summary["overall"])
PY
```

Acceptance requires identical `results.json` hashes and reconciled non-empty
cache counts. AMOTA is expected to remain exactly at the frozen result in this
stage; any score change is a blocker.
