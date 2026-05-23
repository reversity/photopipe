"""Tests for photopipe.faces.clustering."""
import numpy as np

from photopipe.faces.clustering import cluster_embeddings


def _normalized(vec):
    arr = np.asarray(vec, dtype=np.float32)
    return (arr / np.linalg.norm(arr)).tolist()


def test_three_tight_clouds_yield_three_clusters():
    rng = np.random.default_rng(42)
    embeddings = []
    # Three distinct 512-d centroids, 8 tight samples each.
    for c in range(3):
        centroid = rng.normal(size=512)
        centroid[c * 100] += 50  # push centroids far apart
        for _ in range(8):
            embeddings.append(_normalized(centroid + rng.normal(scale=0.01, size=512)))
    labels = cluster_embeddings(embeddings, min_cluster_size=3)
    real = {lbl for lbl in labels if lbl != -1}
    assert len(real) == 3


def test_outliers_marked_as_noise():
    rng = np.random.default_rng(7)
    centroid = rng.normal(size=512)
    embeddings = [_normalized(centroid + rng.normal(scale=0.01, size=512)) for _ in range(8)]
    # Two scattered outliers far from the cloud and from each other.
    embeddings.append(_normalized(rng.normal(size=512) * 100))
    embeddings.append(_normalized(rng.normal(size=512) * -100))
    labels = cluster_embeddings(embeddings, min_cluster_size=3)
    assert labels[-1] == -1
    assert labels[-2] == -1


def test_too_few_embeddings_all_noise():
    labels = cluster_embeddings([[0.1] * 512, [0.2] * 512], min_cluster_size=3)
    assert labels == [-1, -1]


def test_empty_input_returns_empty():
    assert cluster_embeddings([], min_cluster_size=3) == []


def test_wrong_dimension_raises():
    import pytest
    with pytest.raises(ValueError, match="512"):
        cluster_embeddings([[0.1, 0.2, 0.3]], min_cluster_size=3)
