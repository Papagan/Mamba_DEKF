from __future__ import annotations

import torch
from torch import Tensor


PAIR_GEOMETRY_DIM = 7


def compose_pair_vectors(
    track_embeddings: Tensor,
    detection_embeddings: Tensor,
    pair_geometry: Tensor,
) -> Tensor:
    """Compose embeddings and pair geometry into association-head inputs."""
    tensors = (
        ("track_embeddings", track_embeddings),
        ("detection_embeddings", detection_embeddings),
        ("pair_geometry", pair_geometry),
    )
    for name, tensor in tensors:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank 2")

    batch_sizes = (
        track_embeddings.shape[0],
        detection_embeddings.shape[0],
        pair_geometry.shape[0],
    )
    if len(set(batch_sizes)) != 1:
        raise ValueError(
            "batch dimension mismatch: "
            f"track_embeddings={batch_sizes[0]}, "
            f"detection_embeddings={batch_sizes[1]}, "
            f"pair_geometry={batch_sizes[2]}"
        )

    if track_embeddings.shape[1] != detection_embeddings.shape[1]:
        raise ValueError(
            "embedding width mismatch: "
            f"track_embeddings={track_embeddings.shape[1]}, "
            f"detection_embeddings={detection_embeddings.shape[1]}"
        )

    if pair_geometry.shape[1] != PAIR_GEOMETRY_DIM:
        raise ValueError(
            "pair_geometry width mismatch: "
            f"got {pair_geometry.shape[1]}, expected {PAIR_GEOMETRY_DIM}"
        )

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
