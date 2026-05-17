"""Tests for photopipe.curate_pipeline."""

from datetime import date
from unittest.mock import MagicMock

import pytest
from PIL import Image

from photopipe.curate_pipeline import (
    AIRunResult,
    apply_ai_results,
    run_ai_dating,
)
from photopipe.database import Database
from photopipe.models import Batch, PhotoPair, PhotoPhase


@pytest.fixture
def db(tmp_path):
    return Database(db_path=tmp_path / "test.db")


def _make_batch(db):
    batch = Batch(name="Test")
    db.create_batch(batch)
    return batch


def _make_photos(db, batch, count, tmp_path):
    photos = []
    for i in range(count):
        front = tmp_path / f"p{i}.jpg"
        Image.new("RGB", (200, 150)).save(front)
        photo = PhotoPair(
            batch_id=batch.id,
            sequence_num=i + 1,
            front_path=front,
            phase=PhotoPhase.CURATED,
        )
        db.create_photo(photo)
        photos.append(photo)
    return photos


def test_run_ai_dating_batches_images_into_groups(db, tmp_path):
    vlm = MagicMock()
    vlm.analyze.return_value = {
        "per_photo": [
            {"photo_index": i, "confidence": "medium", "evidence": ["x"], "year": 1985}
            for i in range(10)
        ],
        "coherence": {"same_event": True, "segment_breaks": [], "summary": "Same trip"},
    }
    batch = _make_batch(db)
    photos = _make_photos(db, batch, count=25, tmp_path=tmp_path)
    run_ai_dating(batch, photos, vlm_client=vlm, images_per_call=10)
    # 25 photos / 10 per call = 3 calls (10, 10, 5)
    assert vlm.analyze.call_count == 3


def test_apply_results_sets_extracted_date_only_on_undated_photos(db, tmp_path):
    batch = _make_batch(db)
    photos = _make_photos(db, batch, count=3, tmp_path=tmp_path)
    photos[0].extracted_date = date(1980, 1, 1)
    db.update_photo(photos[0])

    ai_result = AIRunResult(
        per_photo={
            photos[i].id: {
                "year": 1985, "month": 6, "confidence": "high", "evidence": ["x"]
            }
            for i in range(3)
        },
        coherence={"same_event": True, "segment_breaks": [], "summary": ""},
    )
    applied = apply_ai_results(batch, ai_result, photos, db=db)
    assert applied.updated == 2  # only the two undated ones
    assert applied.skipped == 1


def test_run_ai_dating_uses_cached_prompt_prefix(db, tmp_path):
    vlm = MagicMock()
    vlm.analyze.return_value = {
        "per_photo": [
            {"photo_index": 0, "confidence": "low", "evidence": [], "year": None}
        ],
        "coherence": {"same_event": True, "segment_breaks": [], "summary": ""},
    }
    batch = _make_batch(db)
    photos = _make_photos(db, batch, count=1, tmp_path=tmp_path)
    run_ai_dating(batch, photos, vlm_client=vlm, images_per_call=10)
    kwargs = vlm.analyze.call_args.kwargs
    assert "cached_prefix" in kwargs
    assert "expert at dating old photographs" in kwargs["cached_prefix"]
    assert kwargs.get("response_schema") is not None


def test_run_ai_dating_aggregates_coherence_segments(db, tmp_path):
    """When multiple calls report segment breaks, the aggregated result preserves them."""
    vlm = MagicMock()
    vlm.analyze.side_effect = [
        {
            "per_photo": [
                {"photo_index": i, "confidence": "medium", "evidence": [], "year": 1985}
                for i in range(10)
            ],
            "coherence": {
                "same_event": True,
                "segment_breaks": [
                    {"after_photo_index": 5, "reason": "Different clothing"}
                ],
                "summary": "Mostly summer 1985",
            },
        },
        {
            "per_photo": [
                {"photo_index": i, "confidence": "medium", "evidence": [], "year": 1990}
                for i in range(5)
            ],
            "coherence": {
                "same_event": False,
                "segment_breaks": [
                    {"after_photo_index": 2, "reason": "Different people"}
                ],
                "summary": "Some 1990s shots",
            },
        },
    ]
    batch = _make_batch(db)
    photos = _make_photos(db, batch, count=15, tmp_path=tmp_path)
    result = run_ai_dating(batch, photos, vlm_client=vlm, images_per_call=10)
    assert len(result.coherence["segment_breaks"]) == 2
    # Majority same_event: first call True, second False => 1 of 2 => False
    assert result.coherence["same_event"] is False
