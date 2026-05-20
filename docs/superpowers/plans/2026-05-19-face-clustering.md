# Face Clustering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect faces in a batch's photos, cluster them by person locally, let the owner name each cluster once, and propagate the name to the keywords of every photo that person appears in.

**Architecture:** A new `photopipe/faces/` package: `detector.py` (InsightFace behind a `FaceBackend` protocol), `clustering.py` (pure HDBSCAN function), `service.py` (orchestration + persistence). New `faces` and `face_clusters` SQLite tables via additive migration `002`. A new owner page `pages/5_faces.py` drives the detect → cluster → name → apply flow. All face data stays on-device — embeddings are stored as local SQLite blobs, nothing is sent to any API.

**Tech Stack:** Python 3.11+, SQLite, `insightface` + `onnxruntime` (face detection + ArcFace embeddings), `hdbscan` (density clustering), `numpy`, Streamlit, pytest.

**Companion spec:** `docs/superpowers/specs/2026-05-19-face-clustering-design.md` — read before starting any task.

**Environment note:** the project `.venv` is at `/Users/Scanning/Developer/photopipe/.venv`. Run tests with `.venv/bin/pytest`. The full suite is currently 62 tests passing; every task must keep it green.

---

## Task 1: DB migration 002 + Face/FaceCluster models

**Files:**
- Create: `photopipe/migrations/_002_faces.py`
- Modify: `photopipe/migrations/__init__.py` — register the migration
- Modify: `photopipe/models.py` — add `Face`, `FaceCluster`
- Test: `tests/test_migrations.py` — add migration-002 cases

- [ ] **Step 1: Write the failing tests** (append to `tests/test_migrations.py`)

```python
def test_migration_creates_faces_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='faces'"
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()


def test_migration_creates_face_clusters_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='face_clusters'"
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()


def test_migration_002_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        run_all_migrations(conn)  # must not raise
        cur = conn.execute("SELECT COUNT(*) FROM faces")
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_migrations.py -v`
Expected: 3 new tests FAIL (no `faces` / `face_clusters` table).

- [ ] **Step 3: Create the migration**

```python
# photopipe/migrations/_002_faces.py
"""Migration 002: add faces and face_clusters tables."""

import sqlite3

MIGRATION_ID = "002_faces"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 002."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS face_clusters (
            id TEXT PRIMARY KEY,
            batch_id TEXT REFERENCES batches(id),
            label TEXT,
            representative_face_id TEXT,
            is_noise INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            photo_id TEXT REFERENCES photos(id),
            batch_id TEXT REFERENCES batches(id),
            bbox JSON NOT NULL,
            embedding BLOB NOT NULL,
            crop_path TEXT,
            cluster_id TEXT REFERENCES face_clusters(id),
            detection_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_faces_batch_id ON faces(batch_id);
        CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_face_clusters_batch ON face_clusters(batch_id);
        """
    )
```

- [ ] **Step 4: Register the migration**

In `photopipe/migrations/__init__.py`, change the imports and `MIGRATIONS` list:

```python
from photopipe.migrations import _001_phase_and_buckets, _002_faces

MIGRATIONS = [_001_phase_and_buckets, _002_faces]
```

- [ ] **Step 5: Add the models** (append to `photopipe/models.py`)

```python
class Face(BaseModel):
    """A single detected face in a photo, with its ArcFace embedding."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    photo_id: str
    batch_id: str
    bbox: tuple[int, int, int, int]  # x, y, w, h in front-image pixels
    embedding: list[float]            # 512-d, L2-normalized
    crop_path: Optional[Path] = None
    cluster_id: Optional[str] = None
    detection_score: Optional[float] = None
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True


class FaceCluster(BaseModel):
    """A group of faces believed to be the same person within one batch."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    batch_id: str
    label: Optional[str] = None
    representative_face_id: Optional[str] = None
    is_noise: bool = False
    created_at: datetime = Field(default_factory=datetime.now)

    class Config:
        use_enum_values = True
```

- [ ] **Step 6: Run tests, commit**

```bash
.venv/bin/pytest tests/test_migrations.py -v
.venv/bin/pytest tests/ -q
git add photopipe/migrations/_002_faces.py photopipe/migrations/__init__.py photopipe/models.py tests/test_migrations.py
git commit -m "Add faces and face_clusters schema migration"
```

---

## Task 2: clustering.py (pure HDBSCAN)

**Files:**
- Create: `photopipe/faces/__init__.py` (empty)
- Create: `photopipe/faces/clustering.py`
- Modify: `pyproject.toml` — add `hdbscan>=0.8.40`
- Test: `tests/test_face_clustering.py`

- [ ] **Step 1: Add the dependency**

In `pyproject.toml` `dependencies`, after the `numpy` line, add:

```
    "hdbscan>=0.8.40",       # faces/clustering.py: density-based face grouping
```

Then install it: `.venv/bin/pip install "hdbscan>=0.8.40"`

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_face_clustering.py
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
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_face_clustering.py -v`
Expected: ImportError on `photopipe.faces.clustering`.

- [ ] **Step 4: Implement**

```python
# photopipe/faces/__init__.py
"""Local face detection, embedding, and clustering."""
```

```python
# photopipe/faces/clustering.py
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
        metric="euclidean",  # inputs are L2-normalized, so euclidean ~ cosine
    )
    labels = clusterer.fit_predict(matrix)
    return [int(x) for x in labels]
```

- [ ] **Step 5: Run tests, commit**

```bash
.venv/bin/pytest tests/test_face_clustering.py -v
.venv/bin/pytest tests/ -q
git add photopipe/faces/__init__.py photopipe/faces/clustering.py pyproject.toml tests/test_face_clustering.py
git commit -m "Add pure HDBSCAN face-embedding clustering"
```

---

## Task 3: detector.py (FaceBackend + InsightFaceBackend)

**Files:**
- Create: `photopipe/faces/detector.py`
- Modify: `pyproject.toml` — add `insightface` + `onnxruntime`
- Test: `tests/test_face_detector.py`

- [ ] **Step 1: Add dependencies**

In `pyproject.toml` `dependencies`, after the `hdbscan` line:

```
    "insightface>=0.7.3",    # faces/detector.py: SCRFD detect + ArcFace embed
    "onnxruntime>=1.17.0",   # insightface inference runtime
```

Install: `.venv/bin/pip install "insightface>=0.7.3" "onnxruntime>=1.17.0"`

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_face_detector.py
"""Tests for photopipe.faces.detector."""
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
from PIL import Image

from photopipe.faces.detector import DetectedFace, InsightFaceBackend


def _fake_insightface_face(x, y, w, h, score):
    """Build an object shaped like an insightface Face."""
    f = MagicMock()
    f.bbox = np.array([x, y, x + w, y + h], dtype=np.float32)  # x1,y1,x2,y2
    f.det_score = np.float32(score)
    f.normed_embedding = np.ones(512, dtype=np.float32) / np.sqrt(512)
    return f


def test_detect_returns_detected_faces(tmp_path):
    img = tmp_path / "p.jpg"
    Image.new("RGB", (400, 300)).save(img)

    backend = InsightFaceBackend()
    fake_app = MagicMock()
    fake_app.get.return_value = [_fake_insightface_face(10, 20, 50, 60, 0.99)]
    with patch.object(backend, "_load_app", return_value=fake_app):
        faces = backend.detect(img)

    assert len(faces) == 1
    face = faces[0]
    assert isinstance(face, DetectedFace)
    assert face.bbox == (10, 20, 50, 60)  # converted x1y1x2y2 -> x,y,w,h
    assert len(face.embedding) == 512
    assert abs(face.detection_score - 0.99) < 1e-4


def test_detect_no_faces_returns_empty(tmp_path):
    img = tmp_path / "p.jpg"
    Image.new("RGB", (400, 300)).save(img)
    backend = InsightFaceBackend()
    fake_app = MagicMock()
    fake_app.get.return_value = []
    with patch.object(backend, "_load_app", return_value=fake_app):
        assert backend.detect(img) == []


def test_app_loaded_once(tmp_path):
    img = tmp_path / "p.jpg"
    Image.new("RGB", (400, 300)).save(img)
    backend = InsightFaceBackend()
    fake_app = MagicMock()
    fake_app.get.return_value = []
    with patch.object(backend, "_load_app", return_value=fake_app) as loader:
        backend.detect(img)
        backend.detect(img)
    loader.assert_called_once()
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_face_detector.py -v`
Expected: ImportError on `photopipe.faces.detector`.

- [ ] **Step 4: Implement**

```python
# photopipe/faces/detector.py
"""Face detection + embedding.

`InsightFaceBackend` is the default `FaceBackend`. It lazily loads the
InsightFace `buffalo_l` model pack (SCRFD detector + ArcFace recognizer)
on first use — the ~300 MB download happens then, not at import.

The `FaceBackend` protocol keeps clustering / persistence / UI decoupled
from InsightFace so InspireFace or Apple Vision can be swapped later.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol


@dataclass
class DetectedFace:
    """One detected face: pixel bbox, embedding, detector confidence."""
    bbox: tuple[int, int, int, int]   # x, y, w, h
    embedding: list[float]            # 512-d, L2-normalized
    detection_score: float


class FaceBackend(Protocol):
    """A face detector + embedder."""

    def detect(self, image_path: Path) -> list[DetectedFace]:
        ...


class InsightFaceBackend:
    """Default backend: InsightFace buffalo_l via onnxruntime."""

    def __init__(self) -> None:
        self._app = None

    def _load_app(self):
        """Construct the InsightFace app. Triggers the model download."""
        from insightface.app import FaceAnalysis

        app = FaceAnalysis(name="buffalo_l")
        # ctx_id=0 selects the first provider; onnxruntime picks CoreML on
        # Apple Silicon and falls back to CPU automatically.
        app.prepare(ctx_id=0, det_size=(640, 640))
        return app

    def detect(self, image_path: Path) -> list[DetectedFace]:
        """Detect every face in the image and return embeddings."""
        if self._app is None:
            self._app = self._load_app()

        import numpy as np
        from PIL import Image

        with Image.open(image_path) as img:
            rgb = np.asarray(img.convert("RGB"))
        # insightface expects BGR
        bgr = rgb[:, :, ::-1]

        results: list[DetectedFace] = []
        for f in self._app.get(bgr):
            x1, y1, x2, y2 = (int(v) for v in f.bbox)
            results.append(
                DetectedFace(
                    bbox=(x1, y1, x2 - x1, y2 - y1),
                    embedding=[float(v) for v in f.normed_embedding],
                    detection_score=float(f.det_score),
                )
            )
        return results
```

- [ ] **Step 5: Run tests, commit**

```bash
.venv/bin/pytest tests/test_face_detector.py -v
.venv/bin/pytest tests/ -q
git add photopipe/faces/detector.py pyproject.toml tests/test_face_detector.py
git commit -m "Add InsightFace detection backend behind a FaceBackend protocol"
```

---

## Task 4: faces DB CRUD + FaceService detect/cluster/name/propagate

**Files:**
- Create: `photopipe/faces/service.py`
- Modify: `photopipe/database.py` — add face / face_cluster CRUD
- Test: `tests/test_face_service.py`

- [ ] **Step 1: Add DB CRUD to `Database`** (append methods to `photopipe/database.py`)

First add `Face`, `FaceCluster` to the `photopipe.models` import block at the top of the file. Then add these methods to the `Database` class. They mirror the existing bucket CRUD style (`with self.connection() as conn:`):

```python
import numpy as np  # add to the imports at the top of database.py

# ---- Face operations ----

def create_face(self, face: Face) -> Face:
    with self.connection() as conn:
        conn.execute(
            """
            INSERT INTO faces(id, photo_id, batch_id, bbox, embedding,
                              crop_path, cluster_id, detection_score, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                face.id, face.photo_id, face.batch_id,
                json.dumps(list(face.bbox)),
                np.asarray(face.embedding, dtype=np.float32).tobytes(),
                str(face.crop_path) if face.crop_path else None,
                face.cluster_id, face.detection_score,
                face.created_at.isoformat(),
            ),
        )
    return face

def get_faces_by_batch(self, batch_id: str) -> list[Face]:
    with self.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM faces WHERE batch_id = ? ORDER BY created_at", (batch_id,)
        ).fetchall()
    return [self._row_to_face(r) for r in rows]

def update_face(self, face: Face) -> None:
    with self.connection() as conn:
        conn.execute(
            "UPDATE faces SET cluster_id = ?, crop_path = ? WHERE id = ?",
            (face.cluster_id, str(face.crop_path) if face.crop_path else None, face.id),
        )

def delete_faces_by_batch(self, batch_id: str) -> None:
    with self.connection() as conn:
        conn.execute("DELETE FROM faces WHERE batch_id = ?", (batch_id,))

def _row_to_face(self, row) -> Face:
    return Face(
        id=row["id"], photo_id=row["photo_id"], batch_id=row["batch_id"],
        bbox=tuple(json.loads(row["bbox"])),
        embedding=np.frombuffer(row["embedding"], dtype=np.float32).tolist(),
        crop_path=Path(row["crop_path"]) if row["crop_path"] else None,
        cluster_id=row["cluster_id"], detection_score=row["detection_score"],
        created_at=datetime.fromisoformat(row["created_at"]),
    )

# ---- Face cluster operations ----

def create_face_cluster(self, cluster: FaceCluster) -> FaceCluster:
    with self.connection() as conn:
        conn.execute(
            """
            INSERT INTO face_clusters(id, batch_id, label,
                                      representative_face_id, is_noise, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                cluster.id, cluster.batch_id, cluster.label,
                cluster.representative_face_id, int(cluster.is_noise),
                cluster.created_at.isoformat(),
            ),
        )
    return cluster

def get_face_clusters_by_batch(self, batch_id: str) -> list[FaceCluster]:
    with self.connection() as conn:
        rows = conn.execute(
            "SELECT * FROM face_clusters WHERE batch_id = ? ORDER BY created_at",
            (batch_id,),
        ).fetchall()
    return [self._row_to_face_cluster(r) for r in rows]

def get_face_cluster(self, cluster_id: str) -> Optional[FaceCluster]:
    with self.connection() as conn:
        row = conn.execute(
            "SELECT * FROM face_clusters WHERE id = ?", (cluster_id,)
        ).fetchone()
    return self._row_to_face_cluster(row) if row else None

def update_face_cluster(self, cluster: FaceCluster) -> None:
    with self.connection() as conn:
        conn.execute(
            """UPDATE face_clusters SET label = ?, representative_face_id = ?,
               is_noise = ? WHERE id = ?""",
            (cluster.label, cluster.representative_face_id,
             int(cluster.is_noise), cluster.id),
        )

def delete_face_clusters_by_batch(self, batch_id: str) -> None:
    with self.connection() as conn:
        conn.execute("DELETE FROM face_clusters WHERE batch_id = ?", (batch_id,))

def _row_to_face_cluster(self, row) -> FaceCluster:
    return FaceCluster(
        id=row["id"], batch_id=row["batch_id"], label=row["label"],
        representative_face_id=row["representative_face_id"],
        is_noise=bool(row["is_noise"]),
        created_at=datetime.fromisoformat(row["created_at"]),
    )
```

- [ ] **Step 2: Write the failing tests**

```python
# tests/test_face_service.py
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
```

- [ ] **Step 3: Run to verify failure**

Run: `.venv/bin/pytest tests/test_face_service.py -v`
Expected: ImportError on `photopipe.faces.service`.

- [ ] **Step 4: Implement `FaceService`**

```python
# photopipe/faces/service.py
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

    def propagate_labels(self, batch: Batch) -> PropagateResult:
        """Add each named, non-noise cluster's label to its photos' keywords.

        Idempotent: a keyword already present is not duplicated.
        """
        clusters = {
            c.id: c
            for c in self.db.get_face_clusters_by_batch(batch.id)
            if c.label and not c.is_noise
        }
        if not clusters:
            return PropagateResult(photos_tagged=0, names_applied=0)

        # photo_id -> set of names to add
        additions: dict[str, set[str]] = {}
        for face in self.db.get_faces_by_batch(batch.id):
            cluster = clusters.get(face.cluster_id)
            if cluster:
                additions.setdefault(face.photo_id, set()).add(cluster.label)

        tagged = 0
        for photo_id, names in additions.items():
            photo = self.db.get_photo(photo_id)
            if photo is None:
                continue
            existing = list(photo.final_keywords)
            merged = existing + [n for n in sorted(names) if n not in existing]
            if merged != existing:
                photo.final_keywords = merged
                self.db.update_photo(photo)
                tagged += 1

        return PropagateResult(
            photos_tagged=tagged,
            names_applied=len({c.label for c in clusters.values()}),
        )
```

- [ ] **Step 5: Run tests, commit**

```bash
.venv/bin/pytest tests/test_face_service.py -v
.venv/bin/pytest tests/ -q
git add photopipe/faces/service.py photopipe/database.py tests/test_face_service.py
git commit -m "Add FaceService: detect, cluster, name, propagate"
```

---

## Task 5: Cluster correction — merge and move

**Files:**
- Modify: `photopipe/faces/service.py` — add `merge_clusters`, `move_face`
- Test: `tests/test_face_service.py` — add correction tests

- [ ] **Step 1: Write the failing tests** (append to `tests/test_face_service.py`)

```python
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
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_face_service.py -k "merge or move" -v`
Expected: AttributeError — `FaceService` has no `merge_clusters` / `move_face`.

- [ ] **Step 3: Implement** (append these methods to the `FaceService` class)

```python
def merge_clusters(self, cluster_ids: list[str]) -> None:
    """Merge all given clusters into the first one; delete the rest.

    Every face in the merged-away clusters is reassigned to the kept
    cluster. No-op if fewer than two ids are given.
    """
    if len(cluster_ids) < 2:
        return
    keep_id, *merge_ids = cluster_ids
    merge_set = set(merge_ids)
    for face in self.db.get_faces_by_batch(
        self._batch_id_for_cluster(keep_id)
    ):
        if face.cluster_id in merge_set:
            face.cluster_id = keep_id
            self.db.update_face(face)
    for cid in merge_ids:
        cluster = self.db.get_face_cluster(cid)
        if cluster:
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
```

`merge_clusters` and `move_face` need two more `Database` methods. Add them to `photopipe/database.py`:

```python
def get_face(self, face_id: str) -> Optional[Face]:
    with self.connection() as conn:
        row = conn.execute(
            "SELECT * FROM faces WHERE id = ?", (face_id,)
        ).fetchone()
    return self._row_to_face(row) if row else None

def delete_face_cluster(self, cluster_id: str) -> None:
    with self.connection() as conn:
        conn.execute("DELETE FROM face_clusters WHERE id = ?", (cluster_id,))
```

- [ ] **Step 4: Run tests, commit**

```bash
.venv/bin/pytest tests/test_face_service.py -v
.venv/bin/pytest tests/ -q
git add photopipe/faces/service.py photopipe/database.py tests/test_face_service.py
git commit -m "Add cluster merge and single-face move corrections"
```

---

## Task 6: Faces page (pages/5_faces.py)

**Files:**
- Create: `pages/5_faces.py`

- [ ] **Step 1: Build the page**

```python
# pages/5_faces.py
"""Owner Faces page: detect → cluster → name → apply, one batch at a time."""
import streamlit as st

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.faces.service import FaceService
from photopipe.models import PhotoPhase

st.set_page_config(page_title="Faces - PhotoPipe", page_icon="🧑", layout="wide")


def init_state() -> None:
    if "db" not in st.session_state:
        get_config().ensure_directories()
        st.session_state.db = Database()
    st.session_state.setdefault("faces_min_cluster_size", 3)


def main() -> None:
    init_state()
    db = st.session_state.db

    st.title("🧑 Faces")
    st.caption(
        "Detect and group faces, then name each person once. "
        "All face data stays on this machine."
    )

    batches = [
        b for b in db.get_all_batches()
        if any(p.phase != PhotoPhase.FINALIZED for p in db.get_photos_by_batch(b.id))
    ]
    if not batches:
        st.info("No batches to work on. Convert a bucket on the Buckets page first.")
        return

    chosen = st.selectbox("Batch", [b.name for b in batches])
    batch = next(b for b in batches if b.name == chosen)

    min_size = st.slider(
        "Minimum faces per person", 2, 8,
        st.session_state.faces_min_cluster_size,
        help="A person needs at least this many face crops to form a group.",
    )
    st.session_state.faces_min_cluster_size = min_size
    svc = FaceService(db, min_cluster_size=min_size)

    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("1 · Detect faces", use_container_width=True):
            bar = st.progress(0.0, text="Detecting…")
            result = svc.detect_batch(
                batch, progress=lambda c, t: bar.progress(c / t, text=f"{c}/{t}")
            )
            st.success(
                f"Found {result.faces_found} faces in {result.photos_scanned} photos"
            )
            for err in result.errors:
                st.warning(err)

    with col2:
        if st.button("2 · Group faces", use_container_width=True):
            result = svc.cluster_batch(batch)
            st.success(
                f"{result.cluster_count} people, {result.noise_count} unsorted faces"
            )

    with col3:
        if st.button("3 · Apply names", type="primary", use_container_width=True):
            result = svc.propagate_labels(batch)
            st.success(
                f"Tagged {result.photos_tagged} photos with "
                f"{result.names_applied} names"
            )

    st.markdown("---")

    clusters = db.get_face_clusters_by_batch(batch.id)
    faces = db.get_faces_by_batch(batch.id)
    faces_by_cluster: dict[str, list] = {}
    for f in faces:
        faces_by_cluster.setdefault(f.cluster_id, []).append(f)

    if not clusters:
        st.info("Run Detect then Group to see people here.")
        return

    real = [c for c in clusters if not c.is_noise]
    noise = [c for c in clusters if c.is_noise]

    for cluster in real:
        members = faces_by_cluster.get(cluster.id, [])
        st.subheader(f"Person · {len(members)} faces")
        new_label = st.text_input(
            "Name", value=cluster.label or "", key=f"name_{cluster.id}",
            placeholder="e.g., Grandma Rose",
        )
        if new_label != (cluster.label or ""):
            svc.name_cluster(cluster.id, new_label)
            st.rerun()
        cols = st.columns(10)
        for i, face in enumerate(members[:10]):
            with cols[i]:
                if face.crop_path and face.crop_path.exists():
                    st.image(str(face.crop_path))

    for cluster in noise:
        members = faces_by_cluster.get(cluster.id, [])
        if members:
            st.subheader(f"Unsorted faces · {len(members)}")
            st.caption("Faces that didn't group with anyone.")
            cols = st.columns(10)
            for i, face in enumerate(members[:10]):
                with cols[i]:
                    if face.crop_path and face.crop_path.exists():
                        st.image(str(face.crop_path))


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke test, commit**

```bash
.venv/bin/python -c "import ast; ast.parse(open('pages/5_faces.py').read()); print('parse OK')"
git add pages/5_faces.py
git commit -m "Add owner Faces page"
```

---

## Task 7: doctor check + README

**Files:**
- Modify: `photopipe/cli/doctor.py` — add `check_face_model`
- Modify: `tests/test_doctor.py` — add tests
- Modify: `README.md` — Faces workflow section

- [ ] **Step 1: Write the failing tests** (append to `tests/test_doctor.py`)

```python
def test_check_face_model_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"x")
    c = check_face_model()
    assert c.ok


def test_check_face_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = check_face_model()
    assert not c.ok
    assert "downloaded" in c.detail.lower() or "first" in (c.fix or "").lower()
```

Add `check_face_model` to the import line at the top of `tests/test_doctor.py`.

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest tests/test_doctor.py -k face_model -v`
Expected: ImportError — `check_face_model` does not exist.

- [ ] **Step 3: Implement the check** (add to `photopipe/cli/doctor.py`, and append `check_face_model` to the `CHECKS` list)

```python
def check_face_model() -> Check:
    """Report whether the InsightFace buffalo_l model pack is downloaded.

    The pack (~300 MB) downloads automatically on the first face
    detection. This check just tells the owner whether that has
    happened yet — a missing pack is not an error.
    """
    from pathlib import Path

    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    present = model_dir.exists() and any(model_dir.glob("*.onnx"))
    return Check(
        "Face model (InsightFace buffalo_l)", present,
        detail="downloaded" if present else "not downloaded yet",
        fix=(
            "Downloads automatically (~300 MB) the first time you run "
            "Detect on the Faces page."
        ) if not present else None,
    )
```

Note: `check_face_model` reporting `ok=False` will make `run_doctor` exit non-zero. That is acceptable — a missing optional model is worth surfacing. If the team prefers doctor to stay green, change the returned `ok` to always `True` and keep the `detail` informative. Implement it as shown (ok reflects presence); the existing `test_run_doctor_returns_zero_on_all_pass` test must be updated in Step 4.

- [ ] **Step 4: Keep `test_run_doctor_returns_zero_on_all_pass` green**

That test patches `shutil.which` and `subprocess.run` and expects rc 0. With `check_face_model` added, it now also needs the model dir present. Update the test to create it:

```python
def test_run_doctor_returns_zero_on_all_pass(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("MISTRAL_API_KEY", "ms-x")
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"x")
    from photopipe.config import get_config
    model = get_config().vlm.model
    with patch(
        "photopipe.cli.doctor.shutil.which", side_effect=lambda c: f"/usr/bin/{c}"
    ), patch("photopipe.cli.doctor.subprocess.run") as run, patch(
        "urllib.request.urlopen", return_value=_fake_models_response([model])
    ):
        run.return_value = MagicMock(stdout="device 'x' is a scanner", stderr="")
        rc = run_doctor()
    assert rc == 0
```

- [ ] **Step 5: Add the README section**

In `README.md`, after the existing workflow section, add:

```markdown
### Faces (optional)

The **Faces** page groups people across a batch so you name each person
once instead of tagging every photo:

1. Pick a batch, click **Detect faces** — finds every face (the face
   model downloads automatically the first time, ~300 MB).
2. **Group faces** — clusters the faces by person.
3. Name each group; use **merge** if one person split into two groups.
4. **Apply names** — adds each name to the keywords of the photos that
   person appears in.

All face data — crops and embeddings — stays on your machine. Nothing
is sent to any API. (Cloud vision models refuse to identify people, so
this is local by necessity as well as by design.)
```

- [ ] **Step 6: Run tests, commit**

```bash
.venv/bin/pytest tests/test_doctor.py -v
.venv/bin/pytest tests/ -q
git add photopipe/cli/doctor.py tests/test_doctor.py README.md
git commit -m "Add face-model doctor check and Faces README section"
```

---

## Self-Review Checklist (run before handoff)

**Spec coverage:**

| Spec section | Implementing task(s) |
|---|---|
| §1 Library choice (InsightFace, HDBSCAN) | 2 (hdbscan), 3 (insightface) |
| §2 Data model (faces, face_clusters, models) | 1 |
| §3 `faces/` package — detector | 3 |
| §3 `faces/` package — clustering | 2 |
| §3 `faces/` package — service | 4 |
| §4 Workflow & UI (`pages/5_faces.py`) | 6 |
| §5 Error handling — no faces / few faces | 2 (few → all noise), 4 (no faces) |
| §5 Error handling — corrupt image | 4 (`detect_batch` try/except) |
| §5 Error handling — model not downloaded | 7 (doctor check) |
| §5 Error handling — embedding dim mismatch | 2 (`cluster_embeddings` ValueError) |
| §6 Testing | tests in tasks 1–5, 7 |
| §7 Migration plan | tasks 1–7 in matching order |
| Manual correction (merge / move) — spec §"Scope decisions" | 5 |
| Propagation idempotency — spec open question | 4 (`test_propagate_labels_is_idempotent`) |

**Gap noted:** the spec's open question on crop-thumbnail storage is resolved in Task 4 — crops live under `~/.photopipe/face_crops/<batch_id>/` (via `FaceService.crop_root`), cleared on every `detect_batch` re-run by `_clear_batch`.

**Type consistency:** `DetectedFace` (detector) vs `Face` (model) are intentionally distinct — `DetectedFace` is the transient detector output, `Face` is the persisted row; `FaceService.detect_batch` converts one to the other. `cluster_embeddings` returns `list[int]` with `-1` for noise — consumed consistently in `FaceService.cluster_batch`. `FaceBackend.detect` signature (`Path -> list[DetectedFace]`) matches the `MagicMock` backend used in `test_face_service.py`.

**Placeholder scan:** no "TBD"/"TODO"/"implement later"; every code step contains complete code.
