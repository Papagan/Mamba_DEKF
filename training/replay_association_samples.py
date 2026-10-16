"""Consume real tracker candidate groups without reconstructing oracle histories."""

from __future__ import annotations

from tracker.inference_pair_cache import validate_candidate_group


def replay_groups_to_samples(payload):
    if not isinstance(payload, dict) or not isinstance(payload.get("groups"), list):
        raise TypeError("Replay cache must contain groups and a runtime_manifest")
    manifest = payload.get("runtime_manifest") or {}
    backbone_hash = manifest.get("mamba_checkpoint_sha256")
    if not isinstance(backbone_hash, str) or len(backbone_hash) != 64:
        raise ValueError("Replay cache needs an exact motion checkpoint SHA-256 binding")
    vector_dim = int(manifest.get("pair_vector_dim", 0))
    embed_dim = int(manifest.get("embedding_dim", 0))
    if embed_dim <= 0 or vector_dim != 4 * embed_dim + 7:
        raise ValueError("Replay pair vectors must use four embeddings plus seven geometry features")
    samples = []
    seen = set()
    for raw_group in payload["groups"]:
        group = validate_candidate_group(raw_group)
        identity = (group["scene_id"], group["sample_token"], group["track_id"])
        if identity in seen:
            raise ValueError(f"Duplicate replay anchor group: {identity}")
        seen.add(identity)
        # Fully unknown groups have no supervision. Partial groups retain
        # trusted pair labels; unknown candidates are masked independently.
        if group["gt_target_status"] == "unknown":
            continue
        for candidate in group["candidates"]:
            if not candidate["finite"] or candidate["gt_label"] is None:
                continue
            vector = candidate.get("pair_vector")
            if vector is None or len(vector) != vector_dim:
                raise ValueError("Replay export must include complete pair vectors")
            samples.append({
                "feature_schema_version": 3,
                "source": "tracker_replay",
                "category": group["class_name"], "class_id": group["class_id"],
                "state_bucket": group["state_bucket"],
                "scene_id": group["scene_id"],
                "anchor_instance_token": f"{group['scene_id']}::track:{group['track_id']}",
                "current_sample_token": group["sample_token"],
                "label": candidate["gt_label"],
                "negative_type": "positive" if candidate["gt_label"] else "hard",
                "pair_vector": vector,
            })
    if not samples:
        raise ValueError("Replay cache has no finite candidates with trusted GT labels")
    return samples, manifest
