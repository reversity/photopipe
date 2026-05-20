# Face Clustering Design

**Date:** 2026-05-19
**Status:** Draft for review
**Author:** PhotoPipe owner + Claude (Opus 4.7, 1M context)
**Companion:** follows the May 2026 rebuild (`2026-05-17-photopipe-rebuild-design.md`, §10 deferred enhancements)

---

## Why this exists

PhotoPipe digitizes family photos and writes date/location/people metadata. People
are currently tagged only at the batch level — every photo in a batch gets the same
`people` list, which is coarse and often wrong (not everyone is in every photo).

Face clustering closes that gap: detect each face, group faces by person across a
batch, let the owner name a group once, and propagate that name to exactly the photos
that person appears in. The owner labels ~10 clusters instead of tagging hundreds of
photos by hand.

This must be done **locally**. The photos are family biometrics; sending face data to
a cloud API is the wrong default. It also can't be a Claude call — frontier vision
models refuse to identify specific real people. Local face-recognition libraries
(ArcFace embeddings) are the only workable path.

## Goals

- Detect every face in a batch's photos and compute a per-face embedding.
- Cluster faces by person without knowing the number of people in advance.
- Let the owner name a cluster once; propagate the name to the `final_keywords` of
  every photo containing a face in that cluster.
- Keep all face data on the machine — embeddings stored locally in SQLite, nothing
  sent to any API.
- Provide manual correction: merge two clusters, move a face to another cluster.

## Non-goals (this spec)

- Cross-batch global clustering. v1 clusters within one batch at a time.
- Incremental assignment of new faces to existing clusters via nearest-centroid.
  v1 re-clusters the whole batch each run.
- Face verification against a known gallery ("is this Grandma Rose?" as a yes/no).
- Age/expression/landmark analysis beyond what detection yields for free.
- Replacing the batch-level `people` field — face tags are additive to
  `final_keywords`.

---

## 1. Library choice

**InsightFace** (`buffalo_l` model pack) for detection + embedding:

- Detection model (SCRFD) finds face bounding boxes.
- Recognition model (ArcFace, ResNet-100) produces a normalized 512-d embedding per
  face. 99.8% LFW accuracy — well above what clustering needs.
- Runs via `onnxruntime`. On Apple Silicon, onnxruntime's CoreML execution provider
  offloads to the Neural Engine; falls back to CPU otherwise.
- One-time model download (~300 MB) to `~/.insightface/models/` on first use.

**HDBSCAN** for clustering:

- Density-based — does not require specifying the number of clusters.
- Naturally produces a `-1` "noise" label for faces that don't group with anything
  (a stranger in the background, a blurry profile). These become an "unsorted"
  bucket in the UI rather than forcing bad assignments.
- Operates on the 512-d embeddings with cosine distance.

The detector is fronted by a small interface (`FaceBackend`) so InspireFace
(CoreML/ANE-native, same research lineage) or Apple's Vision framework can be
swapped in later without touching clustering, persistence, or UI.

New dependencies: `insightface`, `onnxruntime`, `hdbscan`. (`numpy` already declared.)

## 2. Data model

Additive migration `002_faces.py` — no changes to existing tables.

### New table: `faces`
- `id` (uuid)
- `photo_id` (uuid, FK → photos)
- `batch_id` (uuid, FK → batches) — denormalized for fast per-batch queries
- `bbox` (json: `[x, y, w, h]` in pixels on the front image)
- `embedding` (blob: 512 float32 = 2048 bytes)
- `crop_path` (text: path to the saved face-crop thumbnail)
- `cluster_id` (uuid, nullable, FK → face_clusters) — set by the clustering step
- `detection_score` (float: detector confidence)
- `created_at` (timestamp)

### New table: `face_clusters`
- `id` (uuid)
- `batch_id` (uuid, FK → batches)
- `label` (text, nullable — the owner-entered name)
- `representative_face_id` (uuid, FK → faces — the crop shown as the cluster avatar)
- `is_noise` (bool — true for the single per-batch HDBSCAN noise bucket)
- `created_at` (timestamp)

### New models (`photopipe/models.py`)
- `Face` — mirrors the `faces` row; `embedding` carried as `list[float]` in memory.
- `FaceCluster` — mirrors the `face_clusters` row.

Embeddings stored as raw float32 blobs (`numpy.tobytes()` / `numpy.frombuffer`) —
compact and fast. They never leave SQLite.

## 3. `photopipe/faces/` package

### `detector.py`
```
class DetectedFace:        # value object
    bbox: tuple[int, int, int, int]
    embedding: list[float]  # 512-d, L2-normalized
    detection_score: float

class FaceBackend(Protocol):
    def detect(self, image_path: Path) -> list[DetectedFace]: ...

class InsightFaceBackend:   # default implementation
    # lazy-loads the buffalo_l model pack on first detect()
    def detect(self, image_path: Path) -> list[DetectedFace]: ...
```
One clear job: turn an image path into a list of `DetectedFace`. No DB, no clustering.
Lazy model load so importing the module is cheap and tests don't need the 300 MB pack.

### `clustering.py`
```
def cluster_embeddings(
    embeddings: list[list[float]],
    min_cluster_size: int = 3,
) -> list[int]:
    """Pure function. Returns a cluster label per embedding; -1 == noise."""
```
No I/O. Wraps HDBSCAN with cosine distance. Independently testable with synthetic
embedding clouds.

### `service.py`
```
class FaceService:
    def detect_batch(self, batch, *, db, progress=None) -> DetectResult
        # run the backend over every front image in the batch,
        # save crop thumbnails, persist `faces` rows (cluster_id NULL)

    def cluster_batch(self, batch, *, db) -> ClusterResult
        # load embeddings, call cluster_embeddings, create face_clusters
        # rows (one noise bucket + one per real cluster), set faces.cluster_id,
        # pick representative_face_id (highest detection_score per cluster)

    def name_cluster(self, cluster_id, label, *, db) -> None
        # set face_clusters.label

    def propagate_labels(self, batch, *, db) -> PropagateResult
        # for every named, non-noise cluster: add the label to
        # final_keywords of each photo that has a face in that cluster

    def merge_clusters(self, cluster_ids, *, db) -> None
    def move_face(self, face_id, target_cluster_id, *, db) -> None
```
Orchestration only — delegates detection to `detector`, math to `clustering`.

## 4. Workflow & UI (`pages/5_faces.py`)

Owner-only page, four steps shown as a vertical flow for one selected batch:

1. **Detect** — "Detect faces in this batch" button. Runs `detect_batch` with a
   progress bar. Reports "Found N faces across M photos."
2. **Cluster** — "Group faces" button. Runs `cluster_batch`. Reports
   "N people found, K unsorted faces."
3. **Name** — one row per cluster: the representative crop as an avatar, a strip of
   that cluster's face crops, and a text input for the name. The noise bucket renders
   last as "Unsorted faces" with per-face "move to cluster" controls. Cluster-level
   actions: merge selected clusters, set name.
4. **Apply** — "Apply names to photos" button. Runs `propagate_labels`. Reports
   "Tagged P photos with Q names." Names land in `final_keywords`; the existing
   finalize step writes them to EXIF/IPTC.

Re-running Detect or Cluster on a batch discards that batch's prior `faces` /
`face_clusters` rows and crop files first — clustering is cheap to redo and stale
partial state is worse than a recompute.

## 5. Error handling

- **Model not downloaded** — first `detect()` triggers the InsightFace download;
  surface a one-time "Downloading face model (~300 MB)…" status. A new
  `photopipe doctor` check reports whether the model pack is present.
- **Photo with no faces** — normal; contributes zero `faces` rows.
- **Fewer than `min_cluster_size` faces total** — skip HDBSCAN; every face goes to
  the noise bucket; the UI says "Not enough faces to group yet."
- **Corrupt / unreadable image** — detector skips it, logs, continues the batch.
- **onnxruntime CoreML provider unavailable** — onnxruntime falls back to CPU
  automatically; slower but correct. Logged once, not fatal.
- **Embedding dimension mismatch** (a backend returning non-512-d) — `clustering`
  rejects the batch with a clear error rather than producing garbage clusters.

## 6. Testing

- `clustering.py` — real HDBSCAN over synthetic embeddings: generate 3 tight
  Gaussian clouds + scattered outliers, assert it finds 3 clusters and marks the
  outliers `-1`. Pure, fast, no model.
- `detector.py` — `InsightFaceBackend` tested with the InsightFace model mocked;
  verify image-path-in → `DetectedFace` list-out plumbing, bbox/embedding shapes,
  the lazy-load happens once.
- `service.py` — against a real temp SQLite DB with the detector and clustering
  mocked: `detect_batch` persists rows, `cluster_batch` assigns ids + picks
  representatives, `propagate_labels` writes the right keyword to exactly the photos
  with a face in that cluster, `merge_clusters` / `move_face` mutate correctly.
- No Streamlit page tests (consistent with the rest of the project — pages are thin
  renderers over the service).
- Migration `002` — idempotency + additive-column tests, mirroring `test_migrations.py`.

## 7. Migration plan (rough sequence)

1. DB migration `002_faces.py` + `Face` / `FaceCluster` models.
2. `clustering.py` (pure, no deps beyond hdbscan) + tests.
3. `detector.py` with `FaceBackend` protocol + `InsightFaceBackend` + tests.
4. `service.py` detect/cluster/name/propagate + tests; bucket-style DB CRUD for
   `faces` / `face_clusters`.
5. `merge_clusters` / `move_face` correction methods + tests.
6. `pages/5_faces.py` thin renderer.
7. `photopipe doctor` check for the InsightFace model pack; `pyproject.toml` deps.
8. README section on the Faces workflow + the local-only privacy note.

Each step leaves the app runnable; the Faces page only appears once step 6 lands.

## Open questions for the implementation plan

- **Crop thumbnail storage** — under `~/.photopipe/face_crops/<batch_id>/`? Cleaned
  up when Detect re-runs. Resolve path convention during planning.
- **`min_cluster_size` default** — 3 is a reasonable start; a family photo set has
  many shots of the same few people. Expose as a slider on the page so the owner can
  retune without a code change.
- **Representative face** — highest detection score is the v1 rule; "largest face"
  or "most frontal" are alternatives. Score is simplest and fine.
- **Propagation idempotency** — `propagate_labels` must be safe to run twice without
  duplicating keywords; de-dupe `final_keywords` on write.
