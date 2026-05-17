"""Tests for photopipe.bucket_service."""

from pathlib import Path

import pytest

from photopipe.bucket_service import BucketService, BucketStats
from photopipe.database import Database
from photopipe.models import BucketStatus, PhotoPair, PhotoPhase


@pytest.fixture
def db(tmp_path):
    """Fresh Database instance with a per-test SQLite file.

    We point the cached config's database path at tmp_path before constructing
    Database. Database.__init__ reads get_config().paths.database when no
    explicit path is passed.
    """
    from photopipe.config import get_config

    cfg = get_config()
    cfg.paths.database = tmp_path / "test.db"
    return Database(db_path=tmp_path / "test.db")


def test_open_bucket_creates_record(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="Grandma's blue album", helper_name="Jo")
    # use_enum_values=True on Bucket means bucket.status is the string value
    assert bucket.status == BucketStatus.OPEN.value
    reloaded = db.get_bucket(bucket.id)
    assert reloaded is not None
    assert reloaded.label == "Grandma's blue album"
    assert reloaded.helper_name == "Jo"


def test_close_bucket_sets_closed_status(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="X")
    svc.close_bucket(bucket.id)
    reloaded = db.get_bucket(bucket.id)
    assert reloaded.status == BucketStatus.CLOSED.value
    assert reloaded.closed_at is not None


def test_bucket_stats_counts_photos(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="X")
    for i in range(3):
        photo = PhotoPair(
            bucket_id=bucket.id,
            batch_id="",  # captured-phase photos aren't in a batch yet
            sequence_num=i + 1,
            front_path=Path(f"/tmp/p{i}.jpg"),
            phase=PhotoPhase.CAPTURED,
        )
        db.create_photo(photo)
    stats = svc.get_stats(bucket.id)
    assert isinstance(stats, BucketStats)
    assert stats.photo_count == 3


def test_convert_to_batch_moves_photos_and_marks_bucket(db):
    svc = BucketService(db)
    bucket = svc.open_bucket(label="Trip", helper_name="Jo")
    photo = PhotoPair(
        bucket_id=bucket.id,
        batch_id="",
        sequence_num=1,
        front_path=Path("/tmp/p1.jpg"),
        phase=PhotoPhase.CAPTURED,
    )
    db.create_photo(photo)

    batch = svc.convert_to_batch(
        bucket.id,
        name="Trip 1985",
        location_description="Toledo, OH",
        people=["Mom", "Dad"],
    )
    reloaded_bucket = db.get_bucket(bucket.id)
    reloaded_photo = db.get_photo(photo.id)
    assert reloaded_bucket.status == BucketStatus.CONVERTED.value
    assert reloaded_bucket.batch_id == batch.id
    assert reloaded_photo.batch_id == batch.id
    assert reloaded_photo.phase == PhotoPhase.CURATED.value


def test_list_buckets_filters_by_status(db):
    svc = BucketService(db)
    open_bucket = svc.open_bucket(label="Open one")
    closed_bucket = svc.open_bucket(label="Closing")
    svc.close_bucket(closed_bucket.id)

    open_only = db.list_buckets(status=BucketStatus.OPEN)
    closed_only = db.list_buckets(status=BucketStatus.CLOSED)

    open_ids = {b.id for b in open_only}
    closed_ids = {b.id for b in closed_only}
    assert open_bucket.id in open_ids
    assert closed_bucket.id in closed_ids
    assert closed_bucket.id not in open_ids
