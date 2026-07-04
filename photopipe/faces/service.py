"""Orchestration for face detection, clustering, naming, and propagation.

Delegates detection to a `FaceBackend`, clustering math to
`clustering.cluster_embeddings`, and persistence to `Database`. Holds no
ML logic itself.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image

from photopipe.database import Database
from photopipe.faces.clustering import cluster_embeddings
from photopipe.faces.detector import FaceBackend, InsightFaceBackend
from photopipe.models import Batch, Face, FaceCluster


@dataclass
class DetectResult:
    faces_found: int
    photos_scanned: int
    errors: list[str] = field(default_factory=list)


@dataclass
class ClusterResult:
    cluster_count: int       # real (non-noise) clusters
    noise_count: int         # faces in the noise bucket


@dataclass
class PropagateResult:
    photos_tagged: int
    names_applied: int


class FaceService:
    """Detect, cluster, name, and propagate faces for one batch at a time."""

    def __init__(
        self,
        db: Database,
        backend: Optional[FaceBackend] = None,
        crop_root: Optional[Path] = None,
        min_cluster_size: int = 3,
    ):
        self.db = db
        self.backend = backend if backend is not None else InsightFaceBackend()
        self.crop_root = crop_root or (Path.home() / ".photopipe" / "face_crops")
        self.min_cluster_size = min_cluster_size

    # ---- detection ----

    def detect_batch(
        self,
        batch: Batch,
        progress: Optional[Callable[[int, int], None]] = None,
    ) -> DetectResult:
        """Run face detection over every front image in the batch.

        Clears any prior faces/clusters for the batch first so re-runs are
        clean. Saves a crop thumbnail per face and persists `faces` rows
        with `cluster_id` left NULL (clustering is a separate step).
        """
        self._clear_batch(batch.id)

        photos = self.db.get_photos_by_batch(batch.id)
        crop_dir = self.crop_root / batch.id
        crop_dir.mkdir(parents=True, exist_ok=True)

        faces_found = 0
        errors: list[str] = []
        for i, photo in enumerate(photos):
            if progress:
                progress(i + 1, len(photos))
            if not photo.front_path or not Path(photo.front_path).exists():
                continue
            try:
                detected = self.backend.detect(Path(photo.front_path))
            except Exception as e:  # corrupt image, etc. — skip, keep going
                errors.append(f"{photo.front_path}: {e}")
                continue
            for d in detected:
                face = Face(
                    photo_id=photo.id,
                    batch_id=batch.id,
                    bbox=d.bbox,
                    embedding=d.embedding,
                    detection_score=d.detection_score,
                )
                face.crop_path = self._save_crop(
                    Path(photo.front_path), d.bbox, crop_dir, face.id
                )
                self.db.create_face(face)
                faces_found += 1

        return DetectResult(
            faces_found=faces_found, photos_scanned=len(photos), errors=errors
        )

    def _save_crop(self, image_path, bbox, crop_dir, face_id) -> Optional[Path]:
        x, y, w, h = bbox
        try:
            with Image.open(image_path) as img:
                crop = img.convert("RGB").crop((x, y, x + w, y + h))
                out = crop_dir / f"{face_id}.jpg"
                crop.save(out, format="JPEG", quality=85)
            return out
        except Exception:
            return None

    def _clear_batch(self, batch_id: str) -> None:
        self.db.delete_faces_by_batch(batch_id)
        self.db.delete_face_clusters_by_batch(batch_id)
        crop_dir = self.crop_root / batch_id
        if crop_dir.exists():
            shutil.rmtree(crop_dir, ignore_errors=True)

    # ---- clustering ----

    def cluster_batch(self, batch: Batch) -> ClusterResult:
        """Cluster the batch's faces; create face_clusters rows; set cluster_id.

        Always creates exactly one noise bucket for the batch; faces HDBSCAN
        marks as -1 go there. Each real cluster's representative is the face
        with the highest detection score.
        """
        # Fresh cluster rows each run.
        self.db.delete_face_clusters_by_batch(batch.id)
        faces = self.db.get_faces_by_batch(batch.id)
        if not faces:
            return ClusterResult(cluster_count=0, noise_count=0)

        labels = cluster_embeddings(
            [f.embedding for f in faces], min_cluster_size=self.min_cluster_size
        )

        # Group faces by HDBSCAN label.
        by_label: dict[int, list[Face]] = {}
        for face, label in zip(faces, labels):
            by_label.setdefault(label, []).append(face)

        noise_cluster = FaceCluster(batch_id=batch.id, is_noise=True)
        self.db.create_face_cluster(noise_cluster)

        real_count = 0
        for label, group in by_label.items():
            if label == -1:
                target_id = noise_cluster.id
            else:
                rep = max(group, key=lambda f: f.detection_score or 0.0)
                cluster = FaceCluster(
                    batch_id=batch.id, representative_face_id=rep.id
                )
                self.db.create_face_cluster(cluster)
                target_id = cluster.id
                real_count += 1
            for face in group:
                face.cluster_id = target_id
                self.db.update_face(face)

        noise_faces = len(by_label.get(-1, []))
        return ClusterResult(cluster_count=real_count, noise_count=noise_faces)

    # ---- naming + propagation ----

    def name_cluster(self, cluster_id: str, label: str) -> None:
        cluster = self.db.get_face_cluster(cluster_id)
        if cluster is None:
            raise ValueError(f"face cluster {cluster_id} not found")
        cluster.label = label.strip() or None
        self.db.update_face_cluster(cluster)

    def merge_clusters(self, cluster_ids: list[str]) -> None:
        """Merge all given clusters into the first one; delete the rest.

        Every face in the merged-away clusters is reassigned to the kept
        cluster. No-op if fewer than two ids are given.
        """
        if len(cluster_ids) < 2:
            return
        keep_id, *merge_ids = cluster_ids
        keep = self.db.get_face_cluster(keep_id)
        if keep is None:
            raise ValueError(f"face cluster {keep_id} not found")
        # Guard against a duplicated keep_id in the tail (would delete the
        # kept cluster) and against merging across batches (would orphan
        # the other batch's faces).
        merge_set = {cid for cid in merge_ids if cid != keep_id}
        for cid in merge_set:
            other = self.db.get_face_cluster(cid)
            if other is not None and other.batch_id != keep.batch_id:
                raise ValueError(
                    f"cannot merge cluster {cid} from batch {other.batch_id} "
                    f"into cluster {keep_id} from batch {keep.batch_id}"
                )
        for face in self.db.get_faces_by_batch(keep.batch_id):
            if face.cluster_id in merge_set:
                face.cluster_id = keep_id
                self.db.update_face(face)
        for cid in merge_set:
            cluster = self.db.get_face_cluster(cid)
            if cluster:
                # A named cluster merged into an unnamed keeper donates its label.
                if cluster.label and not keep.label:
                    keep.label = cluster.label
                    self.db.update_face_cluster(keep)
                self.db.delete_face_cluster(cid)

    def move_face(self, face_id: str, target_cluster_id: str) -> None:
        """Reassign a single face to a different cluster."""
        target = self.db.get_face_cluster(target_cluster_id)
        if target is None:
            raise ValueError(f"face cluster {target_cluster_id} not found")
        face = self.db.get_face(face_id)
        if face is None:
            raise ValueError(f"face {face_id} not found")
        face.cluster_id = target_cluster_id
        self.db.update_face(face)

    def _batch_id_for_cluster(self, cluster_id: str) -> str:
        cluster = self.db.get_face_cluster(cluster_id)
        if cluster is None:
            raise ValueError(f"face cluster {cluster_id} not found")
        return cluster.batch_id

    def propagate_labels(self, batch: Batch) -> PropagateResult:
        """Sync each cluster's label into its photos' keywords.

        Idempotent, and rename-aware: each cluster remembers the label it
        last propagated (``propagated_label``); when the label has changed
        since, the stale keyword is removed from the cluster's photos
        before the current one is added.
        """
        all_clusters = self.db.get_face_clusters_by_batch(batch.id)
        clusters = {c.id: c for c in all_clusters if c.label or c.propagated_label}
        if not clusters:
            return PropagateResult(photos_tagged=0, names_applied=0)

        # photo_id -> (names to add, stale names to remove)
        additions: dict[str, set[str]] = {}
        removals: dict[str, set[str]] = {}
        for face in self.db.get_faces_by_batch(batch.id):
            cluster = clusters.get(face.cluster_id)
            if cluster is None:
                continue
            if cluster.label:
                additions.setdefault(face.photo_id, set()).add(cluster.label)
            if cluster.propagated_label and cluster.propagated_label != cluster.label:
                removals.setdefault(face.photo_id, set()).add(cluster.propagated_label)

        tagged = 0
        for photo_id in additions.keys() | removals.keys():
            photo = self.db.get_photo(photo_id)
            if photo is None:
                continue
            existing = list(photo.final_keywords)
            stale = removals.get(photo_id, set())
            names = additions.get(photo_id, set())
            merged = [k for k in existing if k not in stale or k in names]
            merged += [n for n in sorted(names) if n not in merged]
            if merged != existing:
                photo.final_keywords = merged
                self.db.update_photo(photo)
                tagged += 1

        for cluster in clusters.values():
            if cluster.propagated_label != cluster.label:
                cluster.propagated_label = cluster.label
                self.db.update_face_cluster(cluster)

        return PropagateResult(
            photos_tagged=tagged,
            names_applied=len({c.label for c in clusters.values() if c.label}),
        )
