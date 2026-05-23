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
