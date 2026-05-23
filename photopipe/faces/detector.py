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
