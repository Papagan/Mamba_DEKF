# Association Head Audit

The frozen baseline keeps `MAMBA_ASSOCIATION_HEAD.ENABLED: false` and must not
be edited. The learned association head must first run in shadow mode, where it
is scored and audited without changing production assignments.

## Why Audit First

The current pairwise association head can achieve very low training loss while
still hurting AMOTA, because the training pair features and inference pair
features are not yet fully aligned. In particular, the old cache uses a
future-detection to candidate-detection geometry, while inference applies the
head to track-state to candidate-detection pairs.

## Safe Shadow Run

Create a separate remote experimental YAML from the frozen evaluation YAML and
add this block. Do not change the frozen YAML itself:

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

Run the normal evaluation remotely with the separate shadow config:

```bash
python main.py --dataset nuscenes --eval \
  --config config/nuscenes_association_head_shadow_audit.yaml \
  -p 12
```

With `SHADOW_MODE: true`, the produced `results.json` must reproduce the frozen
0.740 baseline `results.json` byte-for-byte and therefore have the same hash.
Check both before interpreting the audit:

```bash
cmp /path/to/frozen/results.json /path/to/shadow/results.json
sha256sum /path/to/frozen/results.json /path/to/shadow/results.json
```

Copy the remote audit JSON and compare it with the rebuilt training cache:

```bash
python tools/compare_pairwise_assoc_train_infer.py \
  --train-cache /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --infer-audit debug/association_head_shadow_audit.json \
  --output debug/association_shadow_decision_report.json \
  --min-known-changed 100
```

Do not overwrite the historical JSON files under `docs/`. In particular, the
old `docs/pairwise_assoc_train_infer_compare.json` is not schema v2 decision
evidence and cannot authorize an activation experiment.

The compare report preserves train-cache class balance, hard-negative, and
distance summaries. Its `track_state_scores` section reports per-class,
per-track-state score and cost distributions only:

- `pair_count`: same-class finite pairs scored by the head.
- `active_pair_count`: pairs in configured active class/state buckets.
- `penalized_pair_count`: active pairs that received a positive cost delta;
  bounded-log-factor promotions (negative deltas) are not included in this count.
- `penalized_ratio`: `penalized_pair_count / active_pair_count`.
- `avg_score`: mean association score.
- `avg_delta`: signed mean correction applied to the geometric cost; it can be
  negative in `bounded_log_factor` mode.
- `avg_cost_before` and `avg_cost_after`: cost shift caused by the head.
- `score_quantiles`, `delta_quantiles`, `cost_before_quantiles`, and
  `cost_after_quantiles`: observed distribution summaries.

Track lifecycle states such as `matched` and `unmatched` are not positive or
negative labels. `track_state_scores` must never be used to infer a distribution
mismatch or to recommend activation.

## Schema v2 Decision Gates

Activation diagnostics use only schema v2 `decision_buckets`. The activation
unit is a `(class_id, class_name, track_state_bucket)` class/state tuple, not a
class pooled across `matched` and `unmatched`. Buckets for the same class/state
are aggregated, then each state independently applies:

- `known_changed_count == 0`: `insufficient_decision_evidence`.
- Fewer than 100 known changed decisions: `shadow_decisions_not_safe`.
- At least 100 known changed decisions, `improved_count > damaged_count`, and
  `harm_precision <= 0.20`: `candidate_for_shadow_ab_activation`.
- Any other complete schema v2 result: `shadow_decisions_not_safe`.

The report exposes these results as `decision_evidence_by_state` and
`diagnosis_by_state`. Class-level activation scope comes only from
`runtime_manifest.active_class_states`; observed buckets do not define or
expand production activation. `runtime_manifest.class_id_to_name` is the
canonical class mapping. Its keys and `active_class_states` keys must be
canonical decimal class IDs in the bounded range `0..2147483647`; each
configured state must be exactly `matched` or `unmatched`.

The comparator checks every training-cache `(class_id, category)`, pair-bucket
`(class_id, class_name)`, and decision-bucket `(class_id, class_name)` against
that canonical mapping. Unknown IDs, mismatched names, malformed or oversized
numeric keys, duplicate identities, and missing mappings are reported in
`class_identity_validation`. Any such error makes decision evidence and
class-level activation unusable. The CLI still writes a fail-closed report
rather than authorizing a candidate or crashing on an oversized key.

Every manifest-active class and state is required for a class-level candidate.
The report includes active classes even when train, pair, and decision inputs
never observed them. A configured state with no decision bucket receives an
explicit zero-count `decision_evidence_by_state` entry and
`insufficient_decision_evidence` in `diagnosis_by_state`. The class is a
candidate only when every configured active state independently reaches the
known-changed minimum and is a candidate. An observed state that is not active
in the manifest remains visible in the state-level report, but it neither
blocks nor extends the configured production activation scope. Class-level
pooled counts remain descriptive and are never used to bypass a failed state
gate.

Missing or malformed `active_class_states` fails closed only at class level:
valid observed states may still retain state-level conclusions, but no class
can be a candidate. Empty class state lists and classes omitted from the map
also cannot produce a class-level candidate. The report records normalized
scope and validation failures in `class_state_activation`. Safely parsed active
entries remain visible with insufficient evidence even when another scope key
is malformed; retaining them for reporting never restores class-level use.

The 100 known-changed minimum is a hard safety floor. The
`--min-known-changed` option may raise it for a stricter experiment but rejects
values below 100.

Schema v1 and malformed schema v2 input fail closed as
`insufficient_decision_evidence`; no class/state can produce a candidate. A
usable top level requires native integer `schema_version: 2`, native boolean
`true` values for availability, completeness, and manifest compatibility, a
mapping `runtime_manifest`, `manifest_initialized` present and exactly `true`,
an empty list of incomplete reasons, and a non-empty decision bucket
list. `audit_errors` must contain non-negative native integer `count` and
`sample_count` values plus a `samples` list, with
`sample_count == len(samples) <= count`; an optional top-level
`audit_error_count` must agree. Any recorded audit error also fails closed.

Before decision evidence is usable, `runtime_manifest` must also prove the
runtime that produced it. Its `audit_schema_version` must be native integer
`3`, while the enclosing decision audit remains schema `2`. `config_sha256`
must be a non-empty string. The manifest must contain native boolean `true` values for
`association_head_enabled`, `association_head_audit_enabled`, and
`association_head_audit_shadow_mode`. If any of these flags is missing,
malformed, or false, the report is not candidate evidence. Checkpoint fields
are required only when all three flags prove that the association head was
actually enabled in a shadow audit; an inactive/non-shadow report is still
insufficient evidence but is not diagnosed as a missing-checkpoint report.

An enabled shadow audit requires this complete checkpoint contract:

- `association_checkpoint_configured` and `association_checkpoint_loaded` are
  native boolean `true`.
- `association_checkpoint_path` is a non-empty normalized absolute resolved
  path, and `association_checkpoint_sha256` is a 64-digit hexadecimal digest.
- `association_checkpoint_loadable` is native boolean `true`,
  `association_checkpoint_head_key_count` is a positive native integer, and
  `association_checkpoint_key_validation_status` is `valid`.
- `association_checkpoint_validation_error` is exactly `null`.
- `association_checkpoint_runtime_contract_expected` and
  `association_checkpoint_runtime_contract_actual` are non-empty mappings
  with identical JSON values and types, and
  `association_checkpoint_runtime_contract_validation_status` is `valid`.

Missing fields are recorded as explicit
`missing_runtime_manifest_field:<field>` entries in
`decision_evidence.incomplete_reasons`. A null checkpoint hash, false
loadability, configured-but-not-loaded state, non-`valid` validation status,
or expected/actual mismatch also makes evidence unusable. The comparator
recursively rejects every present `*_validation_error` that is not `null`,
every present `*_validation_status` that is not `valid`, and every future
`*_expected`/`*_actual` pair that differs. These rules prevent schema v1 or a
partially extended schema v2 manifest from becoming a candidate by omission.

The CLI writes the comparison report through a temporary file in the output
directory, flushes and `fsync`s it, preserves an existing report's permission
mode, and publishes it with `os.replace`. A failed write therefore does not
expose a partially written JSON report.

Each decision bucket requires a bounded non-negative native integer `class_id`,
the exact canonical `class_name`, a `matched` or `unmatched` state, and every
production counter as a non-negative native integer. Outcome counts, known-GT
counts, changed-assignment categories, threshold crossings, and known-changed
counts must satisfy the schema v2 accounting identities. Optional
`net_improvement` and `harm_precision` must be finite and consistent with those
counters. Numeric strings, booleans used as integers, negative values, and
inconsistent derived metrics are rejected rather than coerced or clipped.
Rejection reasons are reported in `decision_evidence.incomplete_reasons`.

Unknown GT identities are retained as unknown evidence and cannot count as
`improved` or `damaged`, so sparse or unavailable GT can prevent the minimum
known-changed gate from being reached.

### Audit-only GT enrichment

GT enrichment is implemented as an explicit, default-off nuScenes preprocessing
mode. Generate a separate audit dataset rather than replacing the normal
converted detections:

```bash
python preprocess/convert_nuscenes.py \
  --raw_data_path /path/to/nuscenes/raw_data \
  --dets_path data/nuscenes/detectors \
  --detector centerpoint \
  --save_path data/base_version/nuscenes_audit_gt \
  --split val \
  --audit_gt_enrichment
```

The converter matches detections to same-class GT annotations within the audit
distance gate and writes aligned `audit_gt_tokens` as a sidecar. Filtering and
NMS keep that sidecar aligned; GT identities are not copied into detection or
`BBox` payloads. The tracker carries the sidecar only into the shadow decision
comparison so known changed assignments can be classified as improved,
damaged, or neutral. Running conversion without `--audit_gt_enrichment` keeps
the prior behavior and normally leaves `known_changed_count` at zero.

GT must never be supplied to candidate generation, scoring, matching, or
assignment selection. It only annotates outcomes after production and shadow
decisions have both been made. Use the separately converted audit dataset only
for the shadow audit, never for baseline or intervention measurements.

A candidate authorizes only a separate A/B experiment, not default activation.
Real assignment intervention requires another experimental config with
`ASSOCIATION_HEAD_AUDIT.SHADOW_MODE: false`. Compare that run against both the
frozen 0.740 baseline and the byte-identical shadow run; never reuse or modify
the frozen config for intervention.

## Inference-Style Pair Cache Export

The pair cache exporter is audit-only and defaults to disabled. The reference
run must use the frozen 0.740 configuration without editing its YAML:

```bash
/usr/bin/time -v -o debug/inference_pair_cache.time \
python main.py \
  --dataset nuscenes \
  --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml \
  -p 12 \
  --inference-pair-cache-output debug/inference_pair_cache.pkl \
  --inference-pair-cache-summary debug/inference_pair_cache_summary.json
```

This command records the exact track-to-detection candidate vectors seen by the
primary BEV matching stage. It does not activate the association head and must
not change matching. Record the timestamped result directory printed by
`[SAVE]`; it is needed for the hash comparison below.

Supervised stage-three labels require the audit-only GT sidecar described in
`Audit-only GT enrichment`. Do not point the frozen configuration at enriched
data by editing it. Create a temporary audit configuration copied from the
frozen file and change only its detection-data root to the separately converted
audit-GT dataset. Before running, inspect several frames and confirm that
`audit_gt_tokens` exists, is aligned one-to-one with detections after filtering
and NMS, and contains nuScenes instance tokens or `null`.

Create that local-only configuration explicitly:

```bash
cp config/nuscenes_single_stage_mctrack_motion_residual_combo.yaml \
  config/nuscenes_single_stage_mctrack_motion_residual_combo_audit_gt.yaml
```

In the copied file, change only the detection data root/input so it resolves to
the separately generated audit-GT detections. Do not alter tracking, matching,
noise, Mamba, or checkpoint settings. Do not commit this audit-only copy, do not
overwrite the frozen source configuration, and verify with `git diff` that the
frozen file remains unchanged.

With that prerequisite satisfied, use the fail-closed variant:

```bash
/usr/bin/time -v -o debug/inference_pair_cache_require_gt.time \
python main.py \
  --dataset nuscenes \
  --eval \
  --config config/nuscenes_single_stage_mctrack_motion_residual_combo_audit_gt.yaml \
  -p 12 \
  --inference-pair-cache-output debug/inference_pair_cache_require_gt.pkl \
  --inference-pair-cache-summary debug/inference_pair_cache_require_gt_summary.json \
  --inference-pair-cache-require-gt \
  --inference-pair-cache-strict
```

`--inference-pair-cache-require-gt` excludes groups whose target remains
unknown. `--inference-pair-cache-strict` makes payload, GT, merge, and write
failures fatal. The strict run is invalid training evidence if the summary has
any cache error or unknown retained label.

### Output contract

The pickle contains `schema_version`, `runtime_manifest`, and `groups`. Each
group identifies one runtime track by scene, sample, frame, track ID, class, and
matched/unmatched state. It stores the production assignment, optional track
embedding, GT status, and a candidate list. Each candidate records its
detection index, untouched geometric cost and gate, rank and cost gap,
production-selection flag, optional association score, optional exact
`pair_vector`, and GT label.

The JSON summary contains the same schema version and runtime manifest plus
`overall`, `per_class`, `per_class_state`, and `error_samples`. Check that the
retained group and candidate counts reconcile with the pickle:

```bash
python - <<'PY'
import json
import pickle
from pathlib import Path

cache = pickle.loads(Path("debug/inference_pair_cache.pkl").read_bytes())
summary = json.loads(
    Path("debug/inference_pair_cache_summary.json").read_text()
)
assert cache["schema_version"] == summary["schema_version"] == 1
assert cache["runtime_manifest"] == summary["runtime_manifest"]
assert summary["overall"]["accepted_groups"] == len(cache["groups"]) > 0
assert summary["overall"]["candidate_count"] == sum(
    len(group["candidates"]) for group in cache["groups"]
)
assert summary["overall"]["candidate_count"] > 0
assert summary["overall"]["error_count"] == 0
assert summary["error_samples"] == []
assert set(summary["per_class"]) == {
    "bicycle", "bus", "car", "motorcycle",
    "pedestrian", "trailer", "truck",
}
print(summary["overall"])
PY
```

For the require-GT output, repeat the check with its paths and additionally
require `rejected_unknown_groups >= 0`, `unknown_groups == 0`, and
`unknown_label_count == 0`. Rejected unknown observations may be nonzero; they
must not appear among retained groups.

### Frozen-result acceptance

Run the same frozen configuration, process count, checkpoint, detector input,
and environment once without export and once with export. Use the two result
directories printed by `[SAVE]`:

```bash
BASE_RESULT=/path/to/baseline/results.json
EXPORT_RESULT=/path/to/export-enabled/results.json
sha256sum "$BASE_RESULT" "$EXPORT_RESULT"
cmp -s "$BASE_RESULT" "$EXPORT_RESULT"
```

Acceptance requires identical SHA-256 values and `cmp` exit status zero. A
matching rounded AMOTA is insufficient; any byte-level `results.json`
difference blocks use of the cache. Also compare `Maximum resident set size`
and elapsed time in the two `/usr/bin/time -v` reports. During the full run,
monitor the timestamped result directory and its
`.inference_pair_cache_spool.*` child with `du -sh`; the uncapped default can
still produce a large final pickle even though per-scene spooling avoids
holding all worker states in `Manager.dict()`.

### Local verification

The Task 7 local verification on 2026-07-28 produced:

- Focused second-stage suite: 312 tests passed, 1 skipped. The skipped test is
  the real `multiprocessing.Manager` round trip because the restricted local
  sandbox denied IPC startup.
- Frozen association regression suite: 41 tests passed, 0 skipped.
- Runtime contract checks: 13 tests passed, 0 skipped.
- `py_compile` passed for `main.py`, the cache/matching/tracker modules, and the
  two Task 7 test modules.
- `git diff --check` reported no whitespace errors.

These are synthetic/unit checks. They do not replace the remote nuScenes
baseline/export hash comparison.

### Residual operational risks

- The pickle/JSON publisher rolls back ordinary exceptions, but a process
  `SIGKILL` between its two final `os.replace` calls cannot execute rollback.
  Preserve the previous pair and summary until both new files pass the manifest
  and count checks.
- The real `multiprocessing.Manager` round-trip test is skipped when the local
  sandbox forbids IPC startup. This does not validate the remote 12-process
  manager, so inspect worker completion, spool cleanup, peak RSS, and disk use
  during the remote acceptance run.
- Detection JSON parsing and `Manager()` initialization failures now clean an
  already-created spool and are covered by unit tests. Abrupt host loss or
  kernel termination can still leave the hidden spool directory for manual
  removal after confirming no evaluation process is using it.
- No local nuScenes dataset or remote checkpoint is available in this
  worktree. The commands above are required remote acceptance steps, not a
  claim that end-to-end nuScenes export has been run locally.

## Rebuild Pairwise Cache After Audit Failure

If the audit shows nearly all pairs being penalized, rebuild the pairwise cache
with inference-aligned features:

```bash
python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/train_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/train_pairwise_assoc.pkl \
  --summary-output docs/train_pairwise_assoc_train_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000

python tools/build_pairwise_association_cache.py \
  --input /root/autodl-tmp/data/training_cache/nuscenes/mini/val_fusion.pkl \
  --output /root/autodl-tmp/data/training_cache/nuscenes/mini/val_pairwise_assoc.pkl \
  --summary-output docs/train_pairwise_assoc_val_summary.json \
  --train-config config/train_nuscenes.yaml \
  --pair-geometry-source predicted_track_candidate \
  --negative-mining-mode inference_margin \
  --cost-margin-eps 0.05 \
  --min-hard-negatives 2 \
  --max-hard-negatives 8 \
  --max-easy-negatives 0 \
  --max-pairs-per-class car=80000,pedestrian=70000
```

This mode changes the training pair features from the old
future-detection/candidate-detection geometry to predicted-track/candidate
geometry. It also stores `candidate_history_12` in the same detection-token
format used by inference (`x=y=0` for the one-frame detection token).
The `inference_margin` negative miner first samples negatives from the same
near-best candidate set used by `MAMBA_ASSOCIATION_HEAD.APPLY_MODE:
margin_tiebreak`. If that set is empty, `--min-hard-negatives` falls back to the
nearest same-class candidates. This avoids both previous failure modes:
easy-negative shortcuts where negatives were meters away, and all-positive
caches where large classes had no useful negatives.
