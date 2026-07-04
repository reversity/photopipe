"""Density-based clustering of face embeddings.

Pure functions — no I/O, no model loading. Wraps HDBSCAN so the rest of
the codebase never imports it directly.
"""
from __future__ import annotations

import numpy as np

EMBEDDING_DIM = 512


def cluster_embeddings(
    embeddings: list[list[float]],
    min_cluster_size: int = 3,
    min_samples: int = 2,
) -> list[int]:
    """Cluster L2-normalized face embeddings.

    Returns one integer label per input embedding. ``-1`` means noise —
    a face that did not group with any cluster. Cluster labels are
    otherwise arbitrary non-negative integers.

    Raises ValueError if any embedding is not EMBEDDING_DIM long.
    """
    if not embeddings:
        return []
    for i, emb in enumerate(embeddings):
        if len(emb) != EMBEDDING_DIM:
            raise ValueError(
                f"embedding {i} has length {len(emb)}, expected {EMBEDDING_DIM}"
            )
    # HDBSCAN needs at least min_cluster_size points to form any cluster.
    if len(embeddings) < min_cluster_size:
        return [-1] * len(embeddings)

    import hdbscan

    matrix = np.asarray(embeddings, dtype=np.float64)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        # HDBSCAN defaults min_samples to min_cluster_size, which is too
        # conservative on 512-d ArcFace embeddings and dumps real faces
        # into noise; a low fixed value groups markedly better.
        min_samples=min_samples,
        metric="euclidean",  # inputs are L2-normalized, so euclidean ~ cosine
    )
    labels = clusterer.fit_predict(matrix)
    return [int(x) for x in labels]
