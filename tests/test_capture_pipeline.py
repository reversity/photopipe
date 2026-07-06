"""Tests for photopipe.capture_pipeline."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from photopipe.bucket_service import BucketService
from photopipe.capture_pipeline import CaptureProgress, capture_batch
from photopipe.database import Database


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "test.db")


@pytest.fixture
def bucket(db):
    svc = BucketService(db)
    return svc.open_bucket(label="Test bucket")


def test_capture_writes_photos_with_captured_phase(db, bucket, tmp_path):
    front = tmp_path / "f1.jpg"
    Image.new("RGB", (800, 600)).save(front)
    back = tmp_path / "f1_b.jpg"
    Image.new("RGB", (800, 600)).save(back)

    with patch("photopipe.capture_pipeline.scan_to_folder") as scan, patch(
        "photopipe.capture_pipeline.process_scanned_photo"
    ) as proc, patch("photopipe.capture_pipeline.HandwritingOCR") as ocr_cls:
        scan.return_value = [front, back]
        proc.return_value = {"cropped": True, "rotated": False, "error": None}
        ocr_instance = MagicMock()
        ocr_instance.ocr_back.return_value = MagicMock(
            text="June 1985",
            confidence=0.9,
            provider="mistral",
            extracted_date=None,
        )
        ocr_cls.return_value = ocr_instance
        result = capture_batch(bucket, db=db, scanner_device="fake", duplex=True)

    assert result.photos_added == 1
    photos = db.get_photos_by_bucket(bucket.id)
    assert len(photos) == 1
    # use_enum_values=True means p.phase is the string value, not the enum
    assert all(p.phase == "captured" for p in photos)
    assert photos[0].handwriting_ocr_text == "June 1985"
    assert photos[0].handwriting_ocr_provider == "mistral"
    assert photos[0].handwriting_ocr_confidence == 0.9


def test_capture_emits_progress_events(db, bucket, tmp_path):
    front = tmp_path / "p1.jpg"
    Image.new("RGB", (400, 400)).save(front)

    events: list[CaptureProgress] = []

    def collect(ev: CaptureProgress) -> None:
        events.append(ev)

    with patch("photopipe.capture_pipeline.scan_to_folder") as scan, patch(
        "photopipe.capture_pipeline.process_scanned_photo"
    ) as proc, patch("photopipe.capture_pipeline.HandwritingOCR") as ocr_cls:
        scan.return_value = [front]  # single front, no back
        proc.return_value = {"cropped": True, "error": None}
        # OCR class is constructed but ocr_back isn't called when there's no back
        ocr_cls.return_value = MagicMock()
        capture_batch(
            bucket,
            db=db,
            scanner_device="fake",
            duplex=False,
            progress=collect,
        )

    stages = {ev.stage for ev in events}
    assert "scanning" in stages
    assert "done" in stages


def test_capture_returns_empty_result_when_scanner_yields_nothing(db, bucket):
    with patch("photopipe.capture_pipeline.scan_to_folder") as scan, patch(
        "photopipe.capture_pipeline.HandwritingOCR"
    ) as ocr_cls:
        scan.return_value = []
        ocr_cls.return_value = MagicMock()
        result = capture_batch(bucket, db=db, scanner_device="fake", duplex=True)

    assert result.photos_added == 0
    assert result.errors  # at least one error message
    assert db.get_photos_by_bucket(bucket.id) == []


def test_capture_extracts_date_from_ocr_when_present(db, bucket, tmp_path):
    from datetime import date

    front = tmp_path / "f1.jpg"
    Image.new("RGB", (400, 400)).save(front)
    back = tmp_path / "f1_b.jpg"
    Image.new("RGB", (400, 400)).save(back)

    with patch("photopipe.capture_pipeline.scan_to_folder") as scan, patch(
        "photopipe.capture_pipeline.process_scanned_photo"
    ) as proc, patch("photopipe.capture_pipeline.HandwritingOCR") as ocr_cls:
        scan.return_value = [front, back]
        proc.return_value = {"cropped": True}
        ocr_instance = MagicMock()
        ocr_instance.ocr_back.return_value = MagicMock(
            text="06/15/1985",
            confidence=0.85,
            provider="mistral",
            extracted_date=date(1985, 6, 15),
        )
        ocr_cls.return_value = ocr_instance
        capture_batch(bucket, db=db, scanner_device="fake", duplex=True)

    photos = db.get_photos_by_bucket(bucket.id)
    assert photos[0].extracted_date == date(1985, 6, 15)
    assert photos[0].date_source == "ocr_back"


def test_scan_batch_without_device_raises_clear_error(monkeypatch, tmp_path):
    from photopipe import scanner as scanner_mod
    from photopipe.scanner import Scanner

    monkeypatch.setattr(scanner_mod, "find_fastfoto", lambda: None)
    s = Scanner(device_name=None)
    with pytest.raises(RuntimeError, match="No scanner found"):
        s.scan_batch(output_folder=tmp_path, name_prefix="photo")


def test_capture_surfaces_scanner_unreachable_as_error(db, bucket):
    with patch("photopipe.capture_pipeline.scan_to_folder") as scan:
        scan.side_effect = RuntimeError("No scanner found. Check that ...")
        result = capture_batch(bucket, db=db, scanner_device=None)
    assert result.photos_added == 0
    assert result.errors and "No scanner found" in result.errors[0]
