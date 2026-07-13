"""The process-wide scanner lock serializes captures across sessions."""
import threading
import time

import pytest
from PIL import Image

from photopipe.bucket_service import BucketService
from photopipe.database import Database
from photopipe import capture_pipeline as cp
from photopipe.scanner import scanner_session, scanner_in_use, ScannerBusy


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "test.db")


@pytest.fixture
def bucket(db):
    return BucketService(db).open_bucket(label="Lock test")


def test_scanner_session_is_exclusive():
    with scanner_session():
        assert scanner_in_use() is True
        with pytest.raises(ScannerBusy):
            with scanner_session():
                pass
    assert scanner_in_use() is False


def test_second_capture_refused_while_first_holds_scanner(db, bucket, monkeypatch, tmp_path):
    """A capture attempted while another holds the scanner is refused with a
    friendly message — it never launches a competing scan."""
    started = threading.Event()
    release = threading.Event()

    def slow_scan(*a, **k):
        started.set()
        release.wait(timeout=5)  # hold the scanner until the test lets go
        front = tmp_path / "photo_0001.jpg"
        Image.new("RGB", (400, 300)).save(front)
        return [front]

    monkeypatch.setattr(cp, "scan_to_folder", slow_scan)
    monkeypatch.setattr(cp, "process_scanned_photo", lambda *a, **k: None)
    monkeypatch.setattr(cp, "HandwritingOCR", lambda *a, **k: (_ for _ in ()).throw(Exception("no ocr")))

    results = {}

    def run_first():
        results["first"] = cp.capture_batch(bucket, db=db, scanner_device="dev")

    t = threading.Thread(target=run_first)
    t.start()
    assert started.wait(timeout=5), "first capture never started"

    # While the first still holds the scanner, a second attempt is refused fast
    second = cp.capture_batch(bucket, db=db, scanner_device="dev")
    assert second.photos_added == 0
    assert second.errors and "still finishing the previous stack" in second.errors[0]

    release.set()
    t.join(timeout=5)
    cp.wait_for_background()  # let the (patched) background work drain
    assert results["first"].photos_added == 1  # first completed normally
    assert scanner_in_use() is False


def test_lock_released_after_capture(db, bucket, monkeypatch, tmp_path):
    front = tmp_path / "photo_0001.jpg"
    Image.new("RGB", (400, 300)).save(front)
    monkeypatch.setattr(cp, "scan_to_folder", lambda *a, **k: [front])
    monkeypatch.setattr(cp, "process_scanned_photo", lambda *a, **k: None)
    monkeypatch.setattr(cp, "HandwritingOCR", lambda *a, **k: (_ for _ in ()).throw(Exception("no ocr")))
    cp.capture_batch(bucket, db=db, scanner_device="dev")
    assert scanner_in_use() is False  # a second stack can now start
    cp.wait_for_background()
