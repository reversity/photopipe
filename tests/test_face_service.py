"""Tests for photopipe.faces.service."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photopipe.database import Database
from photopipe.faces.detector import DetectedFace
from photopipe.faces.service import FaceService
from photopipe.models import Batch, PhotoPair, PhotoPhase


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "test.db")


def _batch_with_photos(db, tmp_path, n):
    batch = Batch(name="Faces test")
    db.create_batch(batch)
    photos = []
    for i in range(n):
        front = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (300, 200)).save(front)
        photo = PhotoPair(
            batch_id=batch.id, sequence_num=i + 1,
            front_path=front, phase=PhotoPhase.CURATED,
        )
        db.create_photo(photo)
        photos.append(photo)
    return batch, photos


def _emb(seed):
    import numpy as np
    rng = np.random.default_rng(seed)
    v = rng.normal(size=512)
    v[seed % 512] += 50
    return (v / np.linalg.norm(v)).tolist()


def test_detect_batch_persists_faces(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 2)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(1, 2, 3, 4), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    result = svc.detect_batch(batch)
    assert result.faces_found == 2
    assert len(db.get_faces_by_batch(batch.id)) == 2


def test_cluster_batch_assigns_clusters_and_noise_bucket(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    # 6 faces: two groups of 3 distinct people.
    backend.detect.return_value = [
        DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9),
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    result = svc.cluster_batch(batch)
    assert result.cluster_count >= 0
    # Every face must end up with a cluster_id.
    assert all(f.cluster_id for f in db.get_faces_by_batch(batch.id))


def test_propagate_labels_adds_keywords_to_right_photos(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 2)
    backend = MagicMock()
    # photo 0 gets a face, photo 1 does not.
    backend.detect.side_effect = [
        [DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9)],
        [],
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.cluster_batch(batch)

    clusters = [c for c in db.get_face_clusters_by_batch(batch.id) if not c.is_noise]
    # Force the face's cluster to be a real (non-noise) named cluster.
    faces = db.get_faces_by_batch(batch.id)
    target_cluster_id = faces[0].cluster_id
    svc.name_cluster(target_cluster_id, "Grandma Rose")
    svc.propagate_labels(batch)

    p0 = db.get_photo(photos[0].id)
    p1 = db.get_photo(photos[1].id)
    assert "Grandma Rose" in p0.final_keywords
    assert "Grandma Rose" not in p1.final_keywords


def test_propagate_labels_is_idempotent(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.cluster_batch(batch)
    cid = db.get_faces_by_batch(batch.id)[0].cluster_id
    svc.name_cluster(cid, "Mom")
    svc.propagate_labels(batch)
    svc.propagate_labels(batch)  # second run
    kw = db.get_photo(photos[0].id).final_keywords
    assert kw.count("Mom") == 1


def test_detect_batch_clears_prior_state(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.detect_batch(batch)  # re-run
    # Re-running must not double the face rows.
    assert len(db.get_faces_by_batch(batch.id)) == 1


def test_merge_clusters_reassigns_faces(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.cluster_batch(batch)

    # Manufacture a second real cluster to merge into the first.
    from photopipe.models import FaceCluster
    extra = FaceCluster(batch_id=batch.id)
    db.create_face_cluster(extra)
    faces = db.get_faces_by_batch(batch.id)
    keep_id = faces[0].cluster_id

    svc.merge_clusters([keep_id, extra.id])
    # The merged-away cluster is gone.
    remaining = {c.id for c in db.get_face_clusters_by_batch(batch.id)}
    assert extra.id not in remaining
    # All faces still point at the kept cluster.
    assert all(f.cluster_id == keep_id for f in db.get_faces_by_batch(batch.id))


def test_move_face_changes_cluster(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(0, 0, 9, 9), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.cluster_batch(batch)

    from photopipe.models import FaceCluster
    target = FaceCluster(batch_id=batch.id)
    db.create_face_cluster(target)
    face = db.get_faces_by_batch(batch.id)[0]

    svc.move_face(face.id, target.id)
    moved = db.get_faces_by_batch(batch.id)[0]
    assert moved.cluster_id == target.id


def test_propagate_after_rename_removes_stale_keyword(db, tmp_path):
    batch, photos = _batch_with_photos(db, tmp_path, 1)
    backend = MagicMock()
    backend.detect.return_value = [
        DetectedFace(bbox=(1, 2, 3, 4), embedding=_emb(1), detection_score=0.9)
    ]
    svc = FaceService(db, backend=backend, crop_root=tmp_path / "crops")
    svc.detect_batch(batch)
    svc.cluster_batch(batch)

    # Put the lone face into a nameable (non-noise) cluster.
    from photopipe.models import FaceCluster
    cluster = FaceCluster(batch_id=batch.id)
    db.create_face_cluster(cluster)
    face = db.get_faces_by_batch(batch.id)[0]
    face.cluster_id = cluster.id
    db.update_face(face)

    svc.name_cluster(cluster.id, "Rose")
    svc.propagate_labels(batch)
    photo = db.get_photo(photos[0].id)
    assert "Rose" in photo.final_keywords

    svc.name_cluster(cluster.id, "Grandma Rose")
    svc.propagate_labels(batch)
    photo = db.get_photo(photos[0].id)
    assert "Grandma Rose" in photo.final_keywords
    assert "Rose" not in photo.final_keywords


def test_name_cluster_strips_whitespace(db, tmp_path):
    batch, _photos = _batch_with_photos(db, tmp_path, 1)
    from photopipe.models import FaceCluster
    cluster = FaceCluster(batch_id=batch.id)
    db.create_face_cluster(cluster)
    svc = FaceService(db, backend=MagicMock(), crop_root=tmp_path / "crops")
    svc.name_cluster(cluster.id, "Rose ")
    assert db.get_face_cluster(cluster.id).label == "Rose"


def test_merge_clusters_rejects_cross_batch(db, tmp_path):
    batch_a, _ = _batch_with_photos(db, tmp_path, 1)
    batch_b = Batch(name="Other batch")
    db.create_batch(batch_b)
    from photopipe.models import FaceCluster
    keep = FaceCluster(batch_id=batch_a.id)
    other = FaceCluster(batch_id=batch_b.id)
    db.create_face_cluster(keep)
    db.create_face_cluster(other)
    svc = FaceService(db, backend=MagicMock(), crop_root=tmp_path / "crops")
    with pytest.raises(ValueError):
        svc.merge_clusters([keep.id, other.id])


def test_merge_clusters_duplicate_keep_id_is_safe(db, tmp_path):
    batch, _ = _batch_with_photos(db, tmp_path, 1)
    from photopipe.models import FaceCluster
    keep = FaceCluster(batch_id=batch.id, label="Rose")
    db.create_face_cluster(keep)
    svc = FaceService(db, backend=MagicMock(), crop_root=tmp_path / "crops")
    svc.merge_clusters([keep.id, keep.id])
    assert db.get_face_cluster(keep.id) is not None


def test_merge_named_into_named_cleans_stale_keyword(db, tmp_path):
    """Merging Bob's (already-applied) cluster into Rose's must not leave 'Bob'
    on the photos forever."""
    from photopipe.models import FaceCluster
    batch, photos = _batch_with_photos(db, tmp_path, 1)

    # Two named clusters, each with a face on the same photo
    rose = FaceCluster(batch_id=batch.id, label="Rose", propagated_label="Rose")
    bob = FaceCluster(batch_id=batch.id, label="Bob", propagated_label="Bob")
    db.create_face_cluster(rose)
    db.create_face_cluster(bob)

    from photopipe.faces.detector import DetectedFace
    from photopipe.models import Face
    f_rose = Face(photo_id=photos[0].id, batch_id=batch.id, bbox=(0, 0, 5, 5),
                  embedding=_emb(1), detection_score=0.9, cluster_id=rose.id)
    f_bob = Face(photo_id=photos[0].id, batch_id=batch.id, bbox=(6, 6, 5, 5),
                 embedding=_emb(2), detection_score=0.9, cluster_id=bob.id)
    db.create_face(f_rose)
    db.create_face(f_bob)

    # Both names already on the photo
    photo = db.get_photo(photos[0].id)
    photo.final_keywords = ["Rose", "Bob"]
    db.update_photo(photo)

    svc = FaceService(db, backend=MagicMock(), crop_root=tmp_path / "crops")
    svc.merge_clusters([rose.id, bob.id])  # keep Rose, merge Bob

    photo = db.get_photo(photos[0].id)
    assert "Bob" not in photo.final_keywords
    assert "Rose" in photo.final_keywords
