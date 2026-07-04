"""Tests for photopipe.bucket_triage."""
from datetime import date
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from photopipe.bucket_service import BucketService
from photopipe.bucket_triage import (
    _sample_across,
    ocr_date_rollup,
    suggest_bucket_context,
)
from photopipe.database import Database
from photopipe.models import PhotoPair, PhotoPhase


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "test.db")


def _bucket_with_photos(db, tmp_path, n, dates=()):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="Grandma's blue album")
    for i in range(n):
        front = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (200, 150)).save(front)
        photo = PhotoPair(
            bucket_id=bucket.id, batch_id="", sequence_num=i + 1,
            front_path=front, phase=PhotoPhase.CAPTURED,
            extracted_date=dates[i] if i < len(dates) else None,
        )
        db.create_photo(photo)
    return bucket


def _proposal(**overrides):
    base = {
        "suggested_batch_name": "Summer at the lake 1987",
        "container_text": "Post-it: '1987-89 lake trips'",
        "date_range": {"start": "1987-06-01", "end": "1989-08-31"},
        "era_guess": "late 1980s",
        "events": [
            {"description": "Lake vacation", "approx_date": "1987-07", "confidence": "high"},
            {"description": "Christmas at home", "approx_date": None, "confidence": "medium"},
        ],
        "location_guesses": ["Lake Erie"],
        "reasoning": "Cover Post-it plus clothing styles.",
        "confidence": "high",
    }
    base.update(overrides)
    return base


class TestRollup:
    def test_empty(self):
        assert ocr_date_rollup([]) == {"earliest": None, "latest": None, "count": 0}

    def test_min_max(self, db, tmp_path):
        bucket = _bucket_with_photos(
            db, tmp_path, 3, dates=(date(1987, 7, 4), None, date(1985, 12, 25))
        )
        photos = db.get_photos_by_bucket(bucket.id)
        rollup = ocr_date_rollup(photos)
        assert rollup == {"earliest": "1985-12-25", "latest": "1987-07-04", "count": 2}


class TestSampleAcross:
    def test_small_bucket_returned_whole(self):
        photos = list(range(5))
        assert _sample_across(photos, 12) == photos

    def test_spread_covers_whole_range(self):
        photos = list(range(100))
        sample = _sample_across(photos, 12)
        assert len(sample) == 12
        assert sample[0] == 0
        assert sample[-1] > 80  # reaches the back of the album


class TestSuggestBucketContext:
    def test_stores_proposal_on_bucket(self, db, tmp_path):
        bucket = _bucket_with_photos(db, tmp_path, 4)
        vlm = MagicMock()
        vlm.analyze.return_value = _proposal()
        result = suggest_bucket_context(bucket, db, vlm_client=vlm)
        assert result["suggested_batch_name"] == "Summer at the lake 1987"
        assert len(result["events"]) == 2
        stored = db.get_bucket(bucket.id)
        assert stored.suggested_context["era_guess"] == "late 1980s"
        assert stored.suggested_context["total_photos"] == 4

    def test_ocr_dates_override_visual_estimate(self, db, tmp_path):
        bucket = _bucket_with_photos(
            db, tmp_path, 3, dates=(date(1985, 12, 25), date(1987, 7, 4), None)
        )
        vlm = MagicMock()
        vlm.analyze.return_value = _proposal()
        result = suggest_bucket_context(bucket, db, vlm_client=vlm)
        assert result["date_range"] == {"start": "1985-12-25", "end": "1987-07-04"}
        assert result["ocr_date_rollup"]["count"] == 2

    def test_container_image_included_first(self, db, tmp_path):
        bucket = _bucket_with_photos(db, tmp_path, 2)
        cover = tmp_path / "cover.jpg"
        Image.new("RGB", (400, 300), color=(0, 0, 200)).save(cover)
        bucket.context_image_path = cover
        db.update_bucket(bucket)
        bucket = db.get_bucket(bucket.id)

        vlm = MagicMock()
        vlm.analyze.return_value = _proposal()
        suggest_bucket_context(bucket, db, vlm_client=vlm)
        kwargs = vlm.analyze.call_args.kwargs
        # container + 2 fronts
        assert len(kwargs["images"]) == 3
        assert "FIRST image is the container" in kwargs["per_call_prompt"]

    def test_empty_bucket_raises(self, db):
        svc = BucketService(db)
        bucket = svc.open_bucket(label="empty")
        with pytest.raises(ValueError):
            suggest_bucket_context(bucket, db, vlm_client=MagicMock())


class TestBucketRoundTrip:
    def test_context_fields_persist(self, db, tmp_path):
        svc = BucketService(db)
        bucket = svc.open_bucket(label="round trip")
        bucket.context_image_path = tmp_path / "cover.jpg"
        bucket.suggested_context = {"era_guess": "1970s"}
        db.update_bucket(bucket)
        loaded = db.get_bucket(bucket.id)
        assert loaded.context_image_path == tmp_path / "cover.jpg"
        assert loaded.suggested_context == {"era_guess": "1970s"}
