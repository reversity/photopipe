"""Bucket lifecycle: open, list, close, convert to batch.

A Bucket is a free-text label entered by the helper during capture.
No metadata, no AI; just a way to group raw scans for the owner to
curate later. When the owner is ready, ``convert_to_batch`` promotes
the bucket's photos into a real :class:`~photopipe.models.Batch`
with full owner-provided context, advancing each photo's phase from
``CAPTURED`` to ``CURATED``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional

from photopipe.database import Database
from photopipe.models import (
    Batch,
    Bucket,
    BucketStatus,
    PhotoPhase,
)


@dataclass
class BucketStats:
    """Aggregate stats for a single bucket."""

    photo_count: int
    photos_with_ocr: int
    photos_with_extracted_date: int
    helper_name: Optional[str]


class BucketService:
    """Manage bucket lifecycle and conversion to batches."""

    def __init__(self, db: Database):
        self.db = db

    def open_bucket(self, label: str, helper_name: Optional[str] = None) -> Bucket:
        """Create a new open bucket and persist it."""
        bucket = Bucket(label=label, helper_name=helper_name)
        self.db.create_bucket(bucket)
        return bucket

    def close_bucket(self, bucket_id: str) -> None:
        """Mark a bucket as closed (no longer accepting captures)."""
        bucket = self.db.get_bucket(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket {bucket_id} not found")
        bucket.status = BucketStatus.CLOSED
        bucket.closed_at = datetime.now()
        self.db.update_bucket(bucket)

    def get_stats(self, bucket_id: str) -> BucketStats:
        """Compute aggregate stats for the given bucket."""
        bucket = self.db.get_bucket(bucket_id)
        photos = self.db.get_photos_by_bucket(bucket_id)
        return BucketStats(
            photo_count=len(photos),
            photos_with_ocr=sum(1 for p in photos if p.handwriting_ocr_text),
            photos_with_extracted_date=sum(1 for p in photos if p.extracted_date),
            helper_name=bucket.helper_name if bucket else None,
        )

    def convert_to_batch(
        self,
        bucket_id: str,
        *,
        name: str,
        date_start: Optional[date] = None,
        date_end: Optional[date] = None,
        location_description: Optional[str] = None,
        people: Optional[list[str]] = None,
        event_description: Optional[str] = None,
    ) -> Batch:
        """Promote a bucket's photos into a new batch.

        Creates a :class:`Batch` with owner-provided context, re-parents every
        photo in the bucket to that batch, and advances each photo's phase
        from ``CAPTURED`` to ``CURATED``. The bucket itself moves to
        ``CONVERTED`` and gets a back-reference to the new batch.
        """
        bucket = self.db.get_bucket(bucket_id)
        if bucket is None:
            raise ValueError(f"Bucket {bucket_id} not found")

        batch = Batch(
            name=name,
            date_start=date_start,
            date_end=date_end,
            location_description=location_description,
            people=people or [],
            event_description=event_description,
        )
        self.db.create_batch(batch)

        # Re-parent the bucket's photos: set batch_id, advance phase to CURATED.
        for photo in self.db.get_photos_by_bucket(bucket_id):
            photo.batch_id = batch.id
            photo.phase = PhotoPhase.CURATED
            self.db.update_photo(photo)

        bucket.status = BucketStatus.CONVERTED
        bucket.batch_id = batch.id
        self.db.update_bucket(bucket)
        return batch
