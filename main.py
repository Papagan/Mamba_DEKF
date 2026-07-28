# ------------------------------------------------------------------------
# Copyright (c) 2024 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------

import json, yaml
import logging
import copy
import argparse
import hashlib
import os
import pickle
import shutil
import tempfile
import time
import multiprocessing
import math
import uuid
from tqdm import tqdm
from datetime import datetime
from functools import partial
from pathlib import Path
from kalmanfilter.noise_audit import NoiseAuditAccumulator
from tracker.base_tracker import Base3DTracker
from tracker.dirty_suppressor_audit import DirtySuppressorAuditAccumulator
from tracker.association_head_audit import AssociationHeadAuditAccumulator
from tracker.association_runtime_manifest import build_association_runtime_manifest
from tracker.inference_pair_cache import (
    InferencePairCacheAccumulator,
    atomic_write_pickle,
    atomic_write_pickle_json_pair,
    build_inference_pair_cache_manifest,
    paths_refer_to_same_file,
)
from dataset.baseversion_dataset import BaseVersionTrackingDataset
from evaluation.static_evaluation.kitti.evaluation_HOTA.scripts.run_kitti import (
    eval_kitti,
)
from evaluation.static_evaluation.nuscenes.eval import eval_nusc
from evaluation.static_evaluation.waymo.eval import eval_waymo
from utils.kitti_utils import save_results_kitti
from utils.nusc_utils import save_results_nuscenes, save_results_nuscenes_for_motion
from utils.waymo_utils.convert_result import save_results_waymo


def _print_nuscenes_result_diagnostics(results_path):
    with open(results_path, "r", encoding="utf-8") as f:
        saved = json.load(f)

    results = saved.get("results", {})
    sample_count = len(results)
    nonempty_samples = sum(1 for boxes in results.values() if boxes)
    total_boxes = sum(len(boxes) for boxes in results.values())
    print(
        f"[SAVE DIAG] samples={sample_count} nonempty_samples={nonempty_samples} "
        f"boxes={total_boxes}"
    )

    class_counts = {}
    invalid_counts = {
        "score_nonfinite": 0,
        "translation_nonfinite": 0,
        "size_nonfinite": 0,
        "rotation_nonfinite": 0,
        "velocity_nonfinite": 0,
    }
    score_stats = {}
    first_examples = []

    for sample_token, boxes in results.items():
        for box in boxes:
            cls = box.get("tracking_name", "UNKNOWN")
            class_counts[cls] = class_counts.get(cls, 0) + 1

            score = box.get("tracking_score", None)
            if not isinstance(score, (int, float)) or not math.isfinite(score):
                invalid_counts["score_nonfinite"] += 1
            else:
                stat = score_stats.setdefault(cls, {"min": score, "max": score})
                stat["min"] = min(stat["min"], score)
                stat["max"] = max(stat["max"], score)

            for key, invalid_key in [
                ("translation", "translation_nonfinite"),
                ("size", "size_nonfinite"),
                ("rotation", "rotation_nonfinite"),
                ("velocity", "velocity_nonfinite"),
            ]:
                values = box.get(key, [])
                if (
                    not isinstance(values, list)
                    or any((not isinstance(v, (int, float)) or not math.isfinite(v)) for v in values)
                ):
                    invalid_counts[invalid_key] += 1

            if len(first_examples) < 5:
                first_examples.append(
                    {
                        "sample_token": sample_token,
                        "tracking_name": cls,
                        "tracking_id": box.get("tracking_id"),
                        "tracking_score": box.get("tracking_score"),
                        "translation": box.get("translation"),
                        "size": box.get("size"),
                    }
                )

    print(f"[SAVE DIAG] class_counts={class_counts}")
    print(f"[SAVE DIAG] invalid_counts={invalid_counts}")
    print(f"[SAVE DIAG] score_stats={score_stats}")
    for idx, example in enumerate(first_examples):
        print(f"[SAVE DIAG] example_{idx}={example}")


def _build_noise_audit_cfg(cfg):
    return (((cfg or {}).get("AUDIT") or {}).get("NOISE_AUDIT") or {})


def _collect_scene_inference_audit_state(scene_id, tracker, cfg, scene_audit_states):
    audit_cfg = _build_noise_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return
    state = tracker.export_noise_audit_state()
    if state is not None:
        scene_audit_states[scene_id] = state


def _write_merged_infer_noise_audit(cfg, scene_audit_states):
    audit_cfg = _build_noise_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return

    merged = NoiseAuditAccumulator()
    for scene_id in sorted(scene_audit_states.keys()):
        merged.merge_state(scene_audit_states[scene_id])

    output_path = audit_cfg.get("INFER_OUTPUT_PATH", "debug/infer_noise_audit.json")
    try:
        merged.write_json(output_path)
    except Exception as exc:
        if audit_cfg.get("STRICT", False):
            raise
        print(f"[main] WARNING: failed to write merged inference noise audit to {output_path}: {exc}")


def _build_dirty_suppressor_audit_cfg(cfg):
    return ((((cfg or {}).get("DIRTY_TRACK_SUPPRESSOR") or {}).get("AUDIT") or {}))


def _collect_scene_dirty_suppressor_audit_state(scene_id, tracker, cfg, scene_dirty_states):
    audit_cfg = _build_dirty_suppressor_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return
    state = tracker.export_dirty_suppressor_audit_state()
    if state is not None:
        scene_dirty_states[scene_id] = state


def _write_merged_dirty_suppressor_audit(cfg, scene_dirty_states):
    audit_cfg = _build_dirty_suppressor_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return

    merged = DirtySuppressorAuditAccumulator()
    for scene_id in sorted(scene_dirty_states.keys()):
        merged.merge_state(scene_dirty_states[scene_id])

    output_path = audit_cfg.get(
        "INFER_OUTPUT_PATH",
        "debug/dirty_track_suppressor_audit.json",
    )
    try:
        merged.write_json(output_path)
    except Exception as exc:
        if audit_cfg.get("STRICT", False):
            raise
        print(
            "[main] WARNING: failed to write merged dirty suppressor audit "
            f"to {output_path}: {exc}"
        )


def _build_association_head_audit_cfg(cfg):
    return (((cfg or {}).get("ASSOCIATION_HEAD_AUDIT") or {}))


def _collect_scene_association_head_audit_state(scene_id, tracker, cfg, scene_assoc_states):
    audit_cfg = _build_association_head_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return
    state = tracker.export_association_head_audit_state()
    if state is not None:
        scene_assoc_states[scene_id] = state


def _write_merged_association_head_audit(cfg, scene_assoc_states):
    audit_cfg = _build_association_head_audit_cfg(cfg)
    if not audit_cfg.get("ENABLED", False):
        return

    strict = bool(audit_cfg.get("STRICT", False))
    merged = AssociationHeadAuditAccumulator(
        max_samples_per_bucket=int(
            audit_cfg.get("MAX_SAMPLES_PER_BUCKET", 0)
        ),
        runtime_manifest=cfg.get("_ASSOCIATION_AUDIT_RUNTIME_MANIFEST"),
        strict_manifest=strict,
    )
    for scene_id in sorted(scene_assoc_states.keys()):
        try:
            merged.merge_state(scene_assoc_states[scene_id])
        except Exception as exc:
            if strict:
                raise
            merged.add_error(
                stage="worker_merge",
                error=exc,
                context={"scene_id": scene_id},
            )
            print(
                "[main] WARNING: failed to merge association head audit "
                f"for scene {scene_id}: {exc}"
            )

    output_path = audit_cfg.get(
        "INFER_OUTPUT_PATH",
        "debug/association_head_audit.json",
    )
    try:
        merged.write_json(output_path)
    except Exception as exc:
        if strict:
            raise
        print(
            "[main] WARNING: failed to write merged association head audit "
            f"to {output_path}: {exc}"
        )


def _inference_pair_cache_enabled(cfg):
    cache_cfg = ((cfg or {}).get("INFERENCE_PAIR_CACHE") or {})
    runtime_disabled = bool(
        (cfg or {}).get("_INFERENCE_PAIR_CACHE_RUNTIME_DISABLED", False)
    )
    return bool(cache_cfg.get("ENABLED", False)) and not runtime_disabled


def _default_inference_pair_cache_summary_path(output_path):
    root, extension = os.path.splitext(output_path)
    summary_path = f"{root}.json" if extension else f"{output_path}.json"
    if os.path.abspath(summary_path) == os.path.abspath(output_path):
        summary_path = f"{root}.summary.json"
    return summary_path


def _validate_inference_pair_cache_output_paths(output_path, summary_path):
    if paths_refer_to_same_file(output_path, summary_path):
        raise ValueError(
            "inference pair cache output and summary paths must differ"
        )


def _apply_inference_pair_cache_cli_overrides(
    cfg,
    *,
    output_path=None,
    summary_path=None,
    strict=False,
    require_gt=False,
    omit_embeddings=False,
    omit_pair_vectors=False,
):
    """Apply explicit cache CLI options without changing YAML-only runs."""
    if output_path is None or output_path == "":
        return cfg
    if not isinstance(output_path, str) or not output_path.strip():
        raise ValueError("inference pair cache output path must be nonblank")
    if summary_path is not None and (
        not isinstance(summary_path, str) or not summary_path.strip()
    ):
        raise ValueError("inference pair cache summary path must be nonblank")
    resolved_summary_path = (
        summary_path
        if summary_path is not None
        else _default_inference_pair_cache_summary_path(output_path)
    )
    _validate_inference_pair_cache_output_paths(
        output_path, resolved_summary_path
    )

    cache_cfg = cfg.setdefault("INFERENCE_PAIR_CACHE", {})
    if not isinstance(cache_cfg, dict):
        raise TypeError("INFERENCE_PAIR_CACHE must be a mapping")
    cache_cfg.update(
        {
            "ENABLED": True,
            "OUTPUT_PATH": output_path,
            "SUMMARY_PATH": resolved_summary_path,
            "STRICT": bool(strict),
            "REQUIRE_AUDIT_GT": bool(require_gt),
            "INCLUDE_EMBEDDINGS": not bool(omit_embeddings),
            "INCLUDE_PAIR_VECTORS": not bool(omit_pair_vectors),
        }
    )
    return cfg


def _effective_cfg_for_runtime_manifest(cfg):
    """Remove injected runtime identity before hashing effective configuration."""
    effective_cfg = copy.deepcopy(cfg)
    for key in list(effective_cfg):
        if (
            key.startswith("_")
            and (
                key.endswith("RUNTIME_MANIFEST")
                or key.startswith("_INFERENCE_PAIR_CACHE_")
            )
        ):
            del effective_cfg[key]
    return effective_cfg


def _build_runtime_manifests(cfg, *, config_path, process_count, git_root):
    """Build each parent-owned manifest once from the same effective config."""
    effective_cfg = _effective_cfg_for_runtime_manifest(cfg)
    pair_manifest = None
    if _inference_pair_cache_enabled(effective_cfg):
        mamba_cfg = effective_cfg.get("MAMBA", {}) or {}
        embedding_dim = int(mamba_cfg["EMBED_DIM"])
        pair_manifest = build_inference_pair_cache_manifest(
            cfg=effective_cfg,
            config_path=config_path,
            process_count=process_count,
            git_root=git_root,
            pair_vector_dim=4 * embedding_dim + 7,
            embedding_dim=embedding_dim,
        )
    association_manifest = build_association_runtime_manifest(
        cfg=effective_cfg,
        config_path=config_path,
        process_count=process_count,
        git_root=git_root,
    )

    if pair_manifest is None:
        cfg.pop("_INFERENCE_PAIR_CACHE_RUNTIME_MANIFEST", None)
    else:
        cfg["_INFERENCE_PAIR_CACHE_RUNTIME_MANIFEST"] = pair_manifest
    cfg["_ASSOCIATION_AUDIT_RUNTIME_MANIFEST"] = association_manifest


def _create_inference_pair_cache_spool(cfg, parent_dir):
    if not _inference_pair_cache_enabled(cfg):
        return None
    spool_dir = tempfile.mkdtemp(
        prefix=".inference_pair_cache_spool.",
        dir=str(parent_dir),
    )
    cfg["_INFERENCE_PAIR_CACHE_SPOOL_DIR"] = spool_dir
    return spool_dir


def _initialize_inference_pair_cache_spool(cfg, parent_dir):
    if not _inference_pair_cache_enabled(cfg):
        return None
    strict = bool(
        ((cfg.get("INFERENCE_PAIR_CACHE") or {}).get("STRICT", False))
    )
    try:
        spool_dir = _create_inference_pair_cache_spool(cfg, parent_dir)
    except Exception as exc:
        if strict:
            raise
        cfg["_INFERENCE_PAIR_CACHE_RUNTIME_DISABLED"] = True
        cfg.pop("_INFERENCE_PAIR_CACHE_SPOOL_DIR", None)
        cfg.pop("_INFERENCE_PAIR_CACHE_RUNTIME_MANIFEST", None)
        print(
            "[main] WARNING: failed to initialize inference pair cache "
            f"spool; cache disabled for this run: {exc}"
        )
        return None
    cfg.pop("_INFERENCE_PAIR_CACHE_RUNTIME_DISABLED", None)
    return spool_dir


def _cleanup_inference_pair_cache_spool(cfg):
    spool_value = cfg.get("_INFERENCE_PAIR_CACHE_SPOOL_DIR")
    if not spool_value:
        return
    spool_path = Path(spool_value).expanduser().resolve(strict=False)
    if not spool_path.name.startswith(".inference_pair_cache_spool."):
        raise ValueError(f"refusing to remove unsafe spool path: {spool_path}")
    shutil.rmtree(spool_path, ignore_errors=False)
    if cfg.get("_INFERENCE_PAIR_CACHE_SPOOL_DIR") == spool_value:
        cfg.pop("_INFERENCE_PAIR_CACHE_SPOOL_DIR", None)


def _scene_inference_pair_cache_spool_path(scene_id, cfg):
    spool_value = cfg.get("_INFERENCE_PAIR_CACHE_SPOOL_DIR")
    if not isinstance(spool_value, str) or not spool_value.strip():
        raise ValueError("inference pair cache spool directory is unavailable")
    scene_text = str(scene_id)
    scene_digest = hashlib.sha256(scene_text.encode("utf-8")).hexdigest()
    filename = f"{scene_digest}.{uuid.uuid4().hex}.pkl"
    return Path(spool_value) / filename


def _publish_scene_inference_pair_cache_marker(
    scene_id, scene_pair_cache_states, marker
):
    if scene_pair_cache_states is None:
        raise ValueError("inference pair cache manager state is unavailable")
    scene_pair_cache_states[scene_id] = marker


def _spool_scene_inference_pair_cache_state(
    scene_id, tracker, cfg, scene_pair_cache_states
):
    if (
        not _inference_pair_cache_enabled(cfg)
        or scene_pair_cache_states is None
    ):
        return
    strict = bool(
        ((cfg.get("INFERENCE_PAIR_CACHE") or {}).get("STRICT", False))
    )
    try:
        state = tracker.export_inference_pair_cache_state()
    except Exception as exc:
        marker = {
            "kind": "error",
            "scene_id": str(scene_id),
            "error": _inference_pair_cache_error_record(
                scene_id, "worker_export", exc
            ),
        }
        try:
            _publish_scene_inference_pair_cache_marker(
                scene_id, scene_pair_cache_states, marker
            )
        except Exception as manager_exc:
            print(
                "[main] WARNING: failed to publish inference pair cache "
                f"error marker for scene {scene_id}: {manager_exc}"
            )
        if strict:
            raise
        print(
            "[main] WARNING: failed to export inference pair cache "
            f"for scene {scene_id}: {exc}"
        )
        return
    if state is None:
        return

    spool_path = None
    try:
        spool_path = _scene_inference_pair_cache_spool_path(scene_id, cfg)
        atomic_write_pickle(spool_path, state)
        marker = {
            "kind": "state",
            "scene_id": str(scene_id),
            "state_path": str(spool_path),
        }
        _publish_scene_inference_pair_cache_marker(
            scene_id, scene_pair_cache_states, marker
        )
    except Exception as exc:
        if spool_path is not None:
            try:
                spool_path.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                pass
        marker = {
            "kind": "error",
            "scene_id": str(scene_id),
            "error": _inference_pair_cache_error_record(
                scene_id, "worker_spool", exc
            ),
        }
        try:
            _publish_scene_inference_pair_cache_marker(
                scene_id, scene_pair_cache_states, marker
            )
        except Exception as manager_exc:
            print(
                "[main] WARNING: failed to publish inference pair cache "
                f"spool error for scene {scene_id}: {manager_exc}"
            )
        if strict:
            raise
        print(
            "[main] WARNING: failed to spool inference pair cache "
            f"for scene {scene_id}: {exc}"
        )


def _inference_pair_cache_error_record(scene_id, stage, error):
    normalized_scene_id = (
        scene_id.strip()
        if isinstance(scene_id, str) and scene_id.strip()
        else "unknown"
    )
    message = str(error).strip() or type(error).__name__
    return {
        "scene_id": normalized_scene_id,
        "sample_token": "unknown",
        "frame_id": 0,
        "stage": stage,
        "error_type": type(error).__name__,
        "message": message,
    }


def _write_merged_inference_pair_cache_impl(cfg, scene_pair_cache_states):
    if not _inference_pair_cache_enabled(cfg):
        return None

    cache_cfg = cfg.get("INFERENCE_PAIR_CACHE", {}) or {}
    strict = bool(cache_cfg.get("STRICT", False))
    try:
        output_path = cache_cfg.get("OUTPUT_PATH")
        if not isinstance(output_path, str) or not output_path.strip():
            raise ValueError(
                "INFERENCE_PAIR_CACHE.OUTPUT_PATH must be a nonblank string"
            )
        summary_path = cache_cfg.get("SUMMARY_PATH")
        if summary_path is None:
            summary_path = _default_inference_pair_cache_summary_path(
                output_path
            )
        if not isinstance(summary_path, str) or not summary_path.strip():
            raise ValueError(
                "INFERENCE_PAIR_CACHE.SUMMARY_PATH must be a nonblank string"
            )
        _validate_inference_pair_cache_output_paths(
            output_path, summary_path
        )
    except Exception as error:
        if strict:
            raise
        print(f"[main] WARNING: inference pair cache disabled at write: {error}")
        try:
            _cleanup_inference_pair_cache_spool(cfg)
        except Exception as cleanup_error:
            print(
                "[main] WARNING: failed to clean inference pair cache spool: "
                f"{cleanup_error}"
            )
        return None

    merged = InferencePairCacheAccumulator(
        runtime_manifest=cfg.get(
            "_INFERENCE_PAIR_CACHE_RUNTIME_MANIFEST", {}
        ),
        max_groups_per_class_state=int(
            cache_cfg.get("MAX_GROUPS_PER_CLASS_STATE", 0)
        ),
        require_audit_gt=bool(cache_cfg.get("REQUIRE_AUDIT_GT", False)),
    )
    try:
        try:
            scene_ids = sorted(
                list(scene_pair_cache_states.keys())
                if scene_pair_cache_states is not None
                else [],
                key=str,
            )
        except Exception as exc:
            if strict:
                raise
            merged.add_error(
                _inference_pair_cache_error_record(
                    "parent", "manager_read", exc
                )
            )
            print(
                "[main] WARNING: failed to enumerate inference pair cache "
                f"worker markers: {exc}"
            )
            scene_ids = []

        spool_root_value = cfg.get("_INFERENCE_PAIR_CACHE_SPOOL_DIR")
        spool_root = (
            Path(spool_root_value).expanduser().resolve(strict=False)
            if isinstance(spool_root_value, str) and spool_root_value
            else None
        )
        for scene_id in scene_ids:
            spool_path = None
            try:
                marker = scene_pair_cache_states[scene_id]
                if not isinstance(marker, dict):
                    raise TypeError("worker cache marker must be a mapping")
                if marker.get("kind") == "error":
                    merged.add_error(marker["error"])
                    continue
                if marker.get("kind") != "state":
                    raise ValueError("worker cache marker kind is invalid")
                spool_path = Path(marker["state_path"]).expanduser().resolve(
                    strict=True
                )
                if (
                    spool_root is None
                    or spool_path.parent != spool_root
                ):
                    raise ValueError(
                        "worker cache state path escapes the spool directory"
                    )
                with spool_path.open("rb") as file_obj:
                    state = pickle.load(file_obj)
                merged.merge_state(state)
                del state
            except Exception as exc:
                if strict:
                    raise
                merged.add_error(
                    _inference_pair_cache_error_record(
                        scene_id, "worker_merge", exc
                    )
                )
                print(
                    "[main] WARNING: failed to merge inference pair cache "
                    f"for scene {scene_id}: {exc}"
                )
            finally:
                if spool_path is not None:
                    try:
                        spool_path.unlink()
                    except FileNotFoundError:
                        pass
                    except OSError as cleanup_error:
                        if strict:
                            raise
                        print(
                            "[main] WARNING: failed to remove inference pair "
                            f"cache spool {spool_path}: {cleanup_error}"
                        )

        summary = merged.to_summary()
        payload = merged.to_payload(copy_groups=False)
        try:
            atomic_write_pickle_json_pair(
                output_path,
                payload,
                summary_path,
                summary,
            )
            print(
                "[main] Wrote inference pair cache and summary to "
                f"{output_path} and {summary_path}"
            )
        except Exception as exc:
            if strict:
                raise
            merged.add_error(
                _inference_pair_cache_error_record(
                    "parent", "write_pair", exc
                )
            )
            summary = merged.to_summary()
            print(
                "[main] WARNING: failed to publish inference pair cache pair "
                f"to {output_path} and {summary_path}: {exc}"
            )
        return payload, summary
    finally:
        try:
            _cleanup_inference_pair_cache_spool(cfg)
        except Exception as cleanup_error:
            if strict:
                raise
            print(
                "[main] WARNING: failed to clean inference pair cache spool: "
                f"{cleanup_error}"
            )


def _write_merged_inference_pair_cache(cfg, scene_pair_cache_states):
    if not _inference_pair_cache_enabled(cfg):
        return None
    strict = bool(
        ((cfg.get("INFERENCE_PAIR_CACHE") or {}).get("STRICT", False))
    )
    try:
        return _write_merged_inference_pair_cache_impl(
            cfg, scene_pair_cache_states
        )
    except Exception as exc:
        try:
            _cleanup_inference_pair_cache_spool(cfg)
        except Exception as cleanup_error:
            print(
                "[main] WARNING: failed to clean inference pair cache spool: "
                f"{cleanup_error}"
            )
        if strict:
            raise
        print(
            "[main] WARNING: inference pair cache failed without affecting "
            f"tracking results: {exc}"
        )
        return None


def run(
    scene_id,
    scenes_data,
    cfg,
    args,
    tracking_results,
    scene_audit_states,
    scene_dirty_states,
    scene_assoc_states,
    scene_pair_cache_states,
    scene_perf_stats,
):
    """
    Info: This function tracks objects in a given scene, processes frame data, and stores tracking results.
    Parameters:
        input:
            scene_id: ID of the scene to process.
            scenes_data: Dictionary with scene data.
            cfg: Configuration settings for tracking.
            args: Additional arguments.
            tracking_results: Dictionary to store results.
            scene_audit_states: Dictionary to store per-scene audit state.
        output:
            tracking_results: Updated tracking results for the scene.
    """
    scene_data = scenes_data[scene_id]
    dataset = BaseVersionTrackingDataset(scene_id, scene_data, cfg=cfg)
    tracker = Base3DTracker(cfg=cfg)
    all_trajs = {}
    tracking_seconds = 0.0

    for index in tqdm(range(len(dataset)), desc=f"Processing {scene_id}"):
        frame_info = dataset[index]
        frame_id = frame_info.frame_id
        cur_sample_token = frame_info.cur_sample_token
        frame_start = time.perf_counter()
        all_traj = tracker.track_single_frame(frame_info)
        tracking_seconds += time.perf_counter() - frame_start
        result_info = {
            "frame_id": frame_id,
            "cur_sample_token": cur_sample_token,
            "trajs": copy.deepcopy(all_traj),
            "transform_matrix": frame_info.transform_matrix,
        }
        all_trajs[frame_id] = copy.deepcopy(result_info)
    if cfg["TRACKING_MODE"] == "GLOBAL":
        trajs = tracker.post_processing()
        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                for bbox in trajs[track_id].bboxes:
                    if (
                        bbox.frame_id == frame_id
                        and bbox.is_interpolation
                        and track_id not in all_trajs[frame_id]["trajs"].keys()
                    ):
                        all_trajs[frame_id]["trajs"][track_id] = bbox

        for index in tqdm(
            range(len(dataset)), desc=f"Trajectory Postprocessing {scene_id}"
        ):
            frame_id = dataset[index].frame_id
            for track_id in sorted(list(trajs.keys())):
                det_score = 0
                for bbox in trajs[track_id].bboxes:
                    det_score = bbox.det_score
                    break
                if (
                    track_id in all_trajs[frame_id]["trajs"].keys()
                    and det_score <= cfg["THRESHOLD"]["GLOBAL_TRACK_SCORE"]
                ):
                    del all_trajs[frame_id]["trajs"][track_id]

    scene_perf_stats[scene_id] = {
        "frames": len(dataset),
        "tracking_seconds": tracking_seconds,
        "tracking_fps": len(dataset) / tracking_seconds if tracking_seconds > 0 else 0.0,
    }
    tracking_results[scene_id] = all_trajs
    print(
        f"[PERF] scene={scene_id} frames={len(dataset)} "
        f"tracking_time={tracking_seconds:.3f}s "
        f"tracking_fps={scene_perf_stats[scene_id]['tracking_fps']:.2f}"
    )
    _collect_scene_inference_audit_state(scene_id, tracker, cfg, scene_audit_states)
    _collect_scene_dirty_suppressor_audit_state(scene_id, tracker, cfg, scene_dirty_states)
    _collect_scene_association_head_audit_state(scene_id, tracker, cfg, scene_assoc_states)
    _spool_scene_inference_pair_cache_state(
        scene_id, tracker, cfg, scene_pair_cache_states
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MCTrack")
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Optional config path. If set, overrides dataset default config path.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="kitti",
        help="Which Dataset: kitti/nuscenes/waymo",
    )
    parser.add_argument("--eval", "-e", action="store_true", help="evaluation")
    parser.add_argument("--load_image", "-lm", action="store_true", help="load_image")
    parser.add_argument("--load_point", "-lp", action="store_true", help="load_point")
    parser.add_argument("--debug", action="store_true", help="debug")
    parser.add_argument("--mode", "-m", action="store_true", help="online or offline")
    parser.add_argument("--process", "-p", type=int, default=1, help="multi-process!")
    parser.add_argument("--inference-pair-cache-output", default=None)
    parser.add_argument("--inference-pair-cache-summary", default=None)
    parser.add_argument("--inference-pair-cache-strict", action="store_true")
    parser.add_argument("--inference-pair-cache-require-gt", action="store_true")
    parser.add_argument(
        "--inference-pair-cache-omit-embeddings", action="store_true"
    )
    parser.add_argument(
        "--inference-pair-cache-omit-pair-vectors", action="store_true"
    )
    args = parser.parse_args()

    if args.config:
        cfg_path = args.config
    else:
        if args.dataset == "kitti":
            cfg_path = "./config/kitti.yaml"
        elif args.dataset == "nuscenes":
            cfg_path = "./config/nuscenes_single_stage_mctrack_exact_noise_hybrid_dirty_suppressor_tuned.yaml"
        elif args.dataset == "waymo":
            cfg_path = "./config/waymo.yaml"
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset}")
        if args.mode:
            cfg_path = cfg_path.replace(".yaml", "_offline.yaml")

    cfg = yaml.load(open(cfg_path, "r"), Loader=yaml.Loader)
    _apply_inference_pair_cache_cli_overrides(
        cfg,
        output_path=args.inference_pair_cache_output,
        summary_path=args.inference_pair_cache_summary,
        strict=args.inference_pair_cache_strict,
        require_gt=args.inference_pair_cache_require_gt,
        omit_embeddings=args.inference_pair_cache_omit_embeddings,
        omit_pair_vectors=args.inference_pair_cache_omit_pair_vectors,
    )
    _build_runtime_manifests(
        cfg,
        config_path=cfg_path,
        process_count=args.process,
        git_root=os.path.dirname(os.path.abspath(__file__)),
    )

    save_path = os.path.join(
        os.path.dirname(cfg["SAVE_PATH"]),
        cfg["DATASET"],
        datetime.now().strftime("%Y%m%d_%H%M%S"),
    )
    os.makedirs(save_path, exist_ok=True)
    cfg["SAVE_PATH"] = save_path
    _initialize_inference_pair_cache_spool(cfg, save_path)

    start_time = time.time()

    detections_root = os.path.join(
        cfg["DETECTIONS_ROOT"], cfg["DETECTOR"], cfg["SPLIT"] + ".json"
    )
    with open(detections_root, "r", encoding="utf-8") as file:
        print(f"Loading data from {detections_root}...")
        data = json.load(file)
        print("Data loaded successfully.")

    if args.debug:
        if args.dataset == "kitti":
            scene_lists = [str(scene_id).zfill(4) for scene_id in cfg["TRACKING_SEQS"]]
        elif args.dataset == "nuscenes":
            scene_lists = [scene_id for scene_id in data.keys()][:2]
        else:
            scene_lists = [scene_id for scene_id in data.keys()][:2]
    else:
        scene_lists = [scene_id for scene_id in data.keys()]

    manager = multiprocessing.Manager()
    tracking_results = manager.dict()
    scene_audit_states = manager.dict()
    scene_dirty_states = manager.dict()
    scene_assoc_states = manager.dict()
    scene_pair_cache_states = (
        manager.dict() if _inference_pair_cache_enabled(cfg) else None
    )
    scene_perf_stats = manager.dict()
    try:
        if args.process > 1:
            pool = multiprocessing.Pool(args.process)
            func = partial(
                run,
                scenes_data=data,
                cfg=cfg,
                args=args,
                tracking_results=tracking_results,
                scene_audit_states=scene_audit_states,
                scene_dirty_states=scene_dirty_states,
                scene_assoc_states=scene_assoc_states,
                scene_pair_cache_states=scene_pair_cache_states,
                scene_perf_stats=scene_perf_stats,
            )
            pool.map(func, scene_lists)
            pool.close()
            pool.join()
        else:
            for scene_id in tqdm(scene_lists, desc="Running scenes"):
                run(
                    scene_id,
                    data,
                    cfg,
                    args,
                    tracking_results,
                    scene_audit_states,
                    scene_dirty_states,
                    scene_assoc_states,
                    scene_pair_cache_states,
                    scene_perf_stats,
                )
    except BaseException:
        try:
            _cleanup_inference_pair_cache_spool(cfg)
        except Exception as cleanup_error:
            print(
                "[main] WARNING: failed to clean inference pair cache spool: "
                f"{cleanup_error}"
            )
        raise
    try:
        tracking_results = dict(tracking_results)
        scene_audit_states = dict(scene_audit_states)
        scene_dirty_states = dict(scene_dirty_states)
        scene_assoc_states = dict(scene_assoc_states)
        scene_perf_stats = dict(scene_perf_stats)
    except BaseException:
        try:
            _cleanup_inference_pair_cache_spool(cfg)
        except Exception as cleanup_error:
            print(
                "[main] WARNING: failed to clean inference pair cache spool: "
                f"{cleanup_error}"
            )
        raise
    total_tracking_frames = sum(int(item.get("frames", 0)) for item in scene_perf_stats.values())
    total_tracking_seconds = sum(float(item.get("tracking_seconds", 0.0)) for item in scene_perf_stats.values())
    if total_tracking_seconds > 0:
        print(
            f"[PERF] total_tracking_frames={total_tracking_frames} "
            f"total_tracking_time={total_tracking_seconds:.3f}s "
            f"tracking_fps={total_tracking_frames / total_tracking_seconds:.2f}"
        )
    _write_merged_inference_pair_cache(cfg, scene_pair_cache_states)
    _write_merged_infer_noise_audit(cfg, scene_audit_states)
    _write_merged_dirty_suppressor_audit(cfg, scene_dirty_states)
    _write_merged_association_head_audit(cfg, scene_assoc_states)

    if args.dataset == "kitti":
        save_results_kitti(tracking_results, cfg)
        if args.eval:
            eval_kitti(cfg)
    if args.dataset == "nuscenes":
        save_results_nuscenes(tracking_results, save_path, cfg=cfg)
        # --- quick sanity: count saved boxes ---
        _res_file = os.path.join(save_path, "results.json")
        with open(_res_file) as _f:
            _saved = json.load(_f)
        _n_samples = len(_saved.get("results", {}))
        _n_boxes = sum(len(v) for v in _saved.get("results", {}).values())
        print(f"[SAVE] samples={_n_samples} boxes={_n_boxes} path={_res_file}")
        if _n_boxes == 0:
            # dump first scene's first frame for debugging
            _scenes = list(tracking_results.keys())
            if _scenes:
                _s0 = _scenes[0]
                _frames = list(tracking_results[_s0].keys())
                if _frames:
                    _f0 = _frames[0]
                    _trajs = tracking_results[_s0][_f0].get("trajs", {})
                    print(f"[SAVE DEBUG] scene={_s0} frame={_f0} n_trajs={len(_trajs)}")
                    for _tid, _bb in list(_trajs.items())[:3]:
                        print(f"[SAVE DEBUG]   track={_tid} cat={_bb.category} "
                              f"score={_bb.det_score:.3f} "
                              f"xyz=[{_bb.global_xyz_lwh_yaw_fusion[0]:.1f},"
                              f"{_bb.global_xyz_lwh_yaw_fusion[1]:.1f},"
                              f"{_bb.global_xyz_lwh_yaw_fusion[2]:.1f}] "
                              f"yaw_fusion={getattr(_bb, 'global_yaw_fusion', 'MISSING')}")
        _print_nuscenes_result_diagnostics(_res_file)
        save_results_nuscenes_for_motion(tracking_results, save_path)
        if args.eval:
            eval_nusc(cfg)
    elif args.dataset == "waymo":
        save_results_waymo(tracking_results, save_path)
        if args.eval:
            eval_waymo(cfg, save_path)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time} seconds")
