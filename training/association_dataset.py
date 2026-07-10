from __future__ import annotations

import pickle
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
from torch.utils.data import Dataset


STATE_BUCKET_TO_ID = {"matched": 0, "unmatched": 1}


def _float_array(values: Any, shape, fill: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=np.float32)
    if arr.shape == shape:
        return arr
    out = np.full(shape, fill, dtype=np.float32)
    flat = arr.reshape(-1) if arr.size else np.asarray([], dtype=np.float32)
    out.reshape(-1)[: min(out.size, flat.size)] = flat[: min(out.size, flat.size)]
    return out


def _candidate_history(sample: Dict[str, Any], history_len: int) -> np.ndarray:
    if sample.get("candidate_history_12") is not None:
        return _float_array(sample.get("candidate_history_12"), (history_len, 12))
    history = np.zeros((history_len, 12), dtype=np.float32)
    history[-1] = _float_array(sample.get("candidate_obs_feature_12"), (12,))
    return history


def _pair_features(sample: Dict[str, Any]) -> np.ndarray:
    state_id = STATE_BUCKET_TO_ID.get(str(sample.get("state_bucket", "matched")).lower(), 0)
    return np.asarray(
        [
            float(sample.get("center_distance", 0.0)),
            float(sample.get("yaw_diff", 0.0)),
            float(sample.get("size_l1", 0.0)),
            float(sample.get("anchor_det_score", 0.0)),
            float(sample.get("candidate_det_score", 0.0)),
            float(state_id),
            float(sample.get("class_id", -1)),
        ],
        dtype=np.float32,
    )


class PairwiseAssociationDataset(Dataset):
    """Dataset for offline class-conditioned track-detection association."""

    def __init__(self, samples_or_path: str | List[Dict[str, Any]], history_len: int = 8) -> None:
        if isinstance(samples_or_path, str):
            with open(samples_or_path, "rb") as f:
                samples = pickle.load(f)
        else:
            samples = samples_or_path
        if not isinstance(samples, list):
            raise TypeError("PairwiseAssociationDataset expects a list of sample dictionaries")
        self.samples = samples
        self.history_len = int(history_len)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]
        class_id = int(sample.get("class_id", -1))
        anchor_key = f"{sample.get('anchor_instance_token')}::{sample.get('current_sample_token')}"
        return {
            "anchor_history": torch.from_numpy(
                _float_array(sample.get("anchor_history_12"), (self.history_len, 12))
            ),
            "candidate_history": torch.from_numpy(_candidate_history(sample, self.history_len)),
            "pair_features": torch.from_numpy(_pair_features(sample)),
            "class_id": torch.tensor(class_id, dtype=torch.long),
            "label": torch.tensor(float(sample.get("label", 0)), dtype=torch.float32),
            "anchor_key": anchor_key,
            "negative_type": str(sample.get("negative_type", "")),
            "state_bucket": str(sample.get("state_bucket", "matched")),
            "category": str(sample.get("category", "")),
        }


def pairwise_association_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    tensor_keys = ["anchor_history", "candidate_history", "pair_features", "class_id", "label"]
    out = {key: torch.stack([item[key] for item in batch], dim=0) for key in tensor_keys}
    out["anchor_keys"] = [item["anchor_key"] for item in batch]
    out["negative_types"] = [item["negative_type"] for item in batch]
    out["state_buckets"] = [item["state_bucket"] for item in batch]
    out["categories"] = [item["category"] for item in batch]
    return out


class PrecomputedPairwiseAssociationDataset(Dataset):
    """Dataset for precomputed pair vectors when the temporal backbone is frozen."""

    def __init__(self, samples: List[Dict[str, Any]] | Dict[str, Any]) -> None:
        if isinstance(samples, dict):
            self.samples = None
            self.pair_vectors = torch.as_tensor(samples["pair_vector"], dtype=torch.float32)
            self.class_ids = torch.as_tensor(samples["class_id"], dtype=torch.long)
            self.labels = torch.as_tensor(samples["label"], dtype=torch.float32)
            self.anchor_keys = list(samples["anchor_keys"])
            self.negative_types = list(samples["negative_types"])
            self.categories = list(samples["categories"])
        elif isinstance(samples, list):
            self.samples = samples
            self.pair_vectors = None
            self.class_ids = None
            self.labels = None
            self.anchor_keys = None
            self.negative_types = None
            self.categories = None
        else:
            raise TypeError("PrecomputedPairwiseAssociationDataset expects a list or tensor dictionary")

    def __len__(self) -> int:
        if self.samples is not None:
            return len(self.samples)
        return int(self.pair_vectors.shape[0])

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.samples is None:
            return {
                "pair_vector": self.pair_vectors[idx],
                "class_id": self.class_ids[idx],
                "label": self.labels[idx],
                "anchor_key": self.anchor_keys[idx],
                "negative_type": self.negative_types[idx],
                "category": self.categories[idx],
            }
        sample = self.samples[idx]
        return {
            "pair_vector": torch.tensor(sample["pair_vector"], dtype=torch.float32),
            "class_id": torch.tensor(int(sample.get("class_id", -1)), dtype=torch.long),
            "label": torch.tensor(float(sample.get("label", 0)), dtype=torch.float32),
            "anchor_key": str(sample.get("anchor_key", "")),
            "negative_type": str(sample.get("negative_type", "")),
            "category": str(sample.get("category", "")),
        }


def precomputed_pairwise_association_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {
        "pair_vector": torch.stack([item["pair_vector"] for item in batch], dim=0),
        "class_id": torch.stack([item["class_id"] for item in batch], dim=0),
        "label": torch.stack([item["label"] for item in batch], dim=0),
    }
    out["anchor_keys"] = [item["anchor_key"] for item in batch]
    out["negative_types"] = [item["negative_type"] for item in batch]
    out["categories"] = [item["category"] for item in batch]
    return out
