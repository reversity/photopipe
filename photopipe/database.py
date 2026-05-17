"""
SQLite database operations for PhotoPipe.

Handles all database interactions including schema creation,
CRUD operations for batches and photos, and processing logs.
"""

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Optional, Generator

from photopipe.config import get_config
from photopipe.migrations import run_all_migrations
from photopipe.models import (
    Batch,
    BatchStatus,
    BatchTemplate,
    PhotoPair,
    PhotoStatus,
    ProcessingLogEntry,
    Location,
    DateSource,
    DateConfidence,
)


SCHEMA = """
-- Batches table
CREATE TABLE IF NOT EXISTS batches (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    date_start DATE,
    date_end DATE,
    location_description TEXT,
    location_lat REAL,
    location_lon REAL,
    event_description TEXT,
    people JSON,
    input_folder TEXT,
    status TEXT DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Photos table
CREATE TABLE IF NOT EXISTS photos (
    id TEXT PRIMARY KEY,
    batch_id TEXT REFERENCES batches(id),
    sequence_num INTEGER,
    front_path TEXT NOT NULL,
    back_path TEXT,

    -- OCR results
    ocr_text_back TEXT,
    ocr_confidence REAL,
    ocr_raw_results JSON,

    -- Extracted date
    extracted_date DATE,
    date_source TEXT,
    date_confidence TEXT,

    -- AI analysis
    ai_analysis JSON,

    -- Final reviewed metadata
    final_date DATE,
    final_location_lat REAL,
    final_location_lon REAL,
    final_location_description TEXT,
    final_description TEXT,
    final_keywords JSON,

    -- Processing status
    status TEXT DEFAULT 'ingested',
    needs_review BOOLEAN DEFAULT FALSE,
    review_notes TEXT,

    -- Output paths
    output_front_path TEXT,
    output_back_path TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Processing log
CREATE TABLE IF NOT EXISTS processing_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id TEXT REFERENCES photos(id),
    batch_id TEXT REFERENCES batches(id),
    action TEXT,
    details JSON,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Batch templates
CREATE TABLE IF NOT EXISTS batch_templates (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    location_description TEXT,
    location_lat REAL,
    location_lon REAL,
    people JSON,
    default_keywords JSON,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indices for performance
CREATE INDEX IF NOT EXISTS idx_photos_batch_id ON photos(batch_id);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
CREATE INDEX IF NOT EXISTS idx_photos_needs_review ON photos(needs_review);
CREATE INDEX IF NOT EXISTS idx_processing_log_photo_id ON processing_log(photo_id);
CREATE INDEX IF NOT EXISTS idx_processing_log_batch_id ON processing_log(batch_id);
"""


class Database:
    """SQLite database manager for PhotoPipe."""

    def __init__(self, db_path: Optional[Path] = None):
        """
        Initialize database connection.

        Args:
            db_path: Path to SQLite database file.
                     Defaults to path from config.
        """
        if db_path is None:
            db_path = get_config().paths.database

        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        """Initialize database schema.

        Runs the legacy ``SCHEMA`` script (so fresh DBs have all base tables)
        and then applies any pending migrations from
        :mod:`photopipe.migrations` (so existing DBs pick up new columns).
        """
        with self.connection() as conn:
            conn.executescript(SCHEMA)
            run_all_migrations(conn)

    @contextmanager
    def connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager for database connections."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    # ==================== Batch Operations ====================

    def create_batch(self, batch: Batch) -> Batch:
        """Create a new batch."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO batches (
                    id, name, date_start, date_end, location_description,
                    location_lat, location_lon, event_description, people,
                    input_folder, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    batch.id,
                    batch.name,
                    batch.date_start.isoformat() if batch.date_start else None,
                    batch.date_end.isoformat() if batch.date_end else None,
                    batch.location_description,
                    batch.location.latitude if batch.location else None,
                    batch.location.longitude if batch.location else None,
                    batch.event_description,
                    json.dumps(batch.people),
                    str(batch.input_folder) if batch.input_folder else None,
                    batch.status,
                    batch.created_at.isoformat(),
                    batch.updated_at.isoformat(),
                ),
            )
        self.log_action(batch_id=batch.id, action="batch_created", details={"name": batch.name})
        return batch

    def get_batch(self, batch_id: str) -> Optional[Batch]:
        """Get a batch by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM batches WHERE id = ?", (batch_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_batch(row)

    def get_batch_by_name(self, name: str) -> Optional[Batch]:
        """Get a batch by name."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM batches WHERE name = ?", (name,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_batch(row)

    def get_all_batches(self, status: Optional[BatchStatus] = None) -> list[Batch]:
        """Get all batches, optionally filtered by status."""
        with self.connection() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM batches WHERE status = ? ORDER BY created_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM batches ORDER BY created_at DESC"
                ).fetchall()

        return [self._row_to_batch(row) for row in rows]

    def update_batch(self, batch: Batch) -> Batch:
        """Update an existing batch."""
        batch.updated_at = datetime.now()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE batches SET
                    name = ?, date_start = ?, date_end = ?, location_description = ?,
                    location_lat = ?, location_lon = ?, event_description = ?,
                    people = ?, input_folder = ?, status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    batch.name,
                    batch.date_start.isoformat() if batch.date_start else None,
                    batch.date_end.isoformat() if batch.date_end else None,
                    batch.location_description,
                    batch.location.latitude if batch.location else None,
                    batch.location.longitude if batch.location else None,
                    batch.event_description,
                    json.dumps(batch.people),
                    str(batch.input_folder) if batch.input_folder else None,
                    batch.status,
                    batch.updated_at.isoformat(),
                    batch.id,
                ),
            )
        return batch

    def delete_batch(self, batch_id: str) -> bool:
        """Delete a batch and all associated photos."""
        with self.connection() as conn:
            # Delete associated photos first
            conn.execute("DELETE FROM photos WHERE batch_id = ?", (batch_id,))
            # Delete processing logs
            conn.execute("DELETE FROM processing_log WHERE batch_id = ?", (batch_id,))
            # Delete batch
            result = conn.execute("DELETE FROM batches WHERE id = ?", (batch_id,))
            return result.rowcount > 0

    def _row_to_batch(self, row: sqlite3.Row) -> Batch:
        """Convert database row to Batch object."""
        location = None
        if row["location_lat"] and row["location_lon"]:
            location = Location(
                description=row["location_description"] or "",
                latitude=row["location_lat"],
                longitude=row["location_lon"],
            )

        return Batch(
            id=row["id"],
            name=row["name"],
            date_start=date.fromisoformat(row["date_start"]) if row["date_start"] else None,
            date_end=date.fromisoformat(row["date_end"]) if row["date_end"] else None,
            location_description=row["location_description"],
            location=location,
            event_description=row["event_description"],
            people=json.loads(row["people"]) if row["people"] else [],
            input_folder=Path(row["input_folder"]) if row["input_folder"] else None,
            status=BatchStatus(row["status"]),
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ==================== Photo Operations ====================

    def create_photo(self, photo: PhotoPair) -> PhotoPair:
        """Create a new photo record."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO photos (
                    id, batch_id, sequence_num, front_path, back_path,
                    ocr_text_back, ocr_confidence, ocr_raw_results,
                    extracted_date, date_source, date_confidence,
                    ai_analysis, final_date, final_location_lat, final_location_lon,
                    final_location_description, final_description, final_keywords,
                    status, needs_review, review_notes,
                    output_front_path, output_back_path, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    photo.id,
                    photo.batch_id,
                    photo.sequence_num,
                    str(photo.front_path),
                    str(photo.back_path) if photo.back_path else None,
                    photo.ocr_text_back,
                    photo.ocr_raw_results.get("confidence") if photo.ocr_raw_results else None,
                    json.dumps(photo.ocr_raw_results) if photo.ocr_raw_results else None,
                    photo.extracted_date.isoformat() if photo.extracted_date else None,
                    photo.date_source,
                    photo.date_confidence,
                    json.dumps(photo.ai_analysis) if photo.ai_analysis else None,
                    photo.final_date.isoformat() if photo.final_date else None,
                    photo.final_location.latitude if photo.final_location else None,
                    photo.final_location.longitude if photo.final_location else None,
                    photo.final_location.description if photo.final_location else None,
                    photo.final_description,
                    json.dumps(photo.final_keywords),
                    photo.status,
                    photo.needs_review,
                    photo.review_notes,
                    str(photo.output_front_path) if photo.output_front_path else None,
                    str(photo.output_back_path) if photo.output_back_path else None,
                    photo.created_at.isoformat(),
                    photo.updated_at.isoformat(),
                ),
            )
        return photo

    def get_photo(self, photo_id: str) -> Optional[PhotoPair]:
        """Get a photo by ID."""
        with self.connection() as conn:
            row = conn.execute(
                "SELECT * FROM photos WHERE id = ?", (photo_id,)
            ).fetchone()

        if row is None:
            return None

        return self._row_to_photo(row)

    def get_photos_by_batch(
        self,
        batch_id: str,
        status: Optional[PhotoStatus] = None,
        needs_review: Optional[bool] = None,
    ) -> list[PhotoPair]:
        """Get all photos for a batch."""
        with self.connection() as conn:
            query = "SELECT * FROM photos WHERE batch_id = ?"
            params = [batch_id]

            if status:
                query += " AND status = ?"
                params.append(status)

            if needs_review is not None:
                query += " AND needs_review = ?"
                params.append(needs_review)

            query += " ORDER BY sequence_num"
            rows = conn.execute(query, params).fetchall()

        return [self._row_to_photo(row) for row in rows]

    def get_photos_needing_review(self, batch_id: Optional[str] = None) -> list[PhotoPair]:
        """Get all photos that need review."""
        with self.connection() as conn:
            if batch_id:
                rows = conn.execute(
                    """
                    SELECT * FROM photos
                    WHERE batch_id = ? AND needs_review = TRUE
                    ORDER BY sequence_num
                    """,
                    (batch_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM photos
                    WHERE needs_review = TRUE
                    ORDER BY batch_id, sequence_num
                    """
                ).fetchall()

        return [self._row_to_photo(row) for row in rows]

    def update_photo(self, photo: PhotoPair) -> PhotoPair:
        """Update an existing photo record."""
        photo.updated_at = datetime.now()
        with self.connection() as conn:
            conn.execute(
                """
                UPDATE photos SET
                    ocr_text_back = ?, ocr_confidence = ?, ocr_raw_results = ?,
                    extracted_date = ?, date_source = ?, date_confidence = ?,
                    ai_analysis = ?, final_date = ?, final_location_lat = ?,
                    final_location_lon = ?, final_location_description = ?,
                    final_description = ?, final_keywords = ?,
                    status = ?, needs_review = ?, review_notes = ?,
                    output_front_path = ?, output_back_path = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    photo.ocr_text_back,
                    photo.ocr_raw_results.get("confidence") if photo.ocr_raw_results else None,
                    json.dumps(photo.ocr_raw_results) if photo.ocr_raw_results else None,
                    photo.extracted_date.isoformat() if photo.extracted_date else None,
                    photo.date_source,
                    photo.date_confidence,
                    json.dumps(photo.ai_analysis) if photo.ai_analysis else None,
                    photo.final_date.isoformat() if photo.final_date else None,
                    photo.final_location.latitude if photo.final_location else None,
                    photo.final_location.longitude if photo.final_location else None,
                    photo.final_location.description if photo.final_location else None,
                    photo.final_description,
                    json.dumps(photo.final_keywords),
                    photo.status,
                    photo.needs_review,
                    photo.review_notes,
                    str(photo.output_front_path) if photo.output_front_path else None,
                    str(photo.output_back_path) if photo.output_back_path else None,
                    photo.updated_at.isoformat(),
                    photo.id,
                ),
            )
        return photo

    def delete_photo(self, photo_id: str) -> bool:
        """Delete a photo record."""
        with self.connection() as conn:
            conn.execute("DELETE FROM processing_log WHERE photo_id = ?", (photo_id,))
            result = conn.execute("DELETE FROM photos WHERE id = ?", (photo_id,))
            return result.rowcount > 0

    def get_batch_photo_count(self, batch_id: str) -> int:
        """Get the count of photos in a batch."""
        with self.connection() as conn:
            result = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            return result[0] if result else 0

    def get_batch_stats(self, batch_id: str) -> dict:
        """Get statistics for a batch."""
        with self.connection() as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE batch_id = ?", (batch_id,)
            ).fetchone()[0]

            by_status = {}
            for row in conn.execute(
                "SELECT status, COUNT(*) FROM photos WHERE batch_id = ? GROUP BY status",
                (batch_id,),
            ):
                by_status[row[0]] = row[1]

            by_date_source = {}
            for row in conn.execute(
                "SELECT date_source, COUNT(*) FROM photos WHERE batch_id = ? GROUP BY date_source",
                (batch_id,),
            ):
                by_date_source[row[0] or "none"] = row[1]

            needs_review = conn.execute(
                "SELECT COUNT(*) FROM photos WHERE batch_id = ? AND needs_review = TRUE",
                (batch_id,),
            ).fetchone()[0]

        return {
            "total": total,
            "by_status": by_status,
            "by_date_source": by_date_source,
            "needs_review": needs_review,
        }

    def _row_to_photo(self, row: sqlite3.Row) -> PhotoPair:
        """Convert database row to PhotoPair object."""
        final_location = None
        if row["final_location_lat"] and row["final_location_lon"]:
            final_location = Location(
                description=row["final_location_description"] or "",
                latitude=row["final_location_lat"],
                longitude=row["final_location_lon"],
            )

        return PhotoPair(
            id=row["id"],
            batch_id=row["batch_id"],
            sequence_num=row["sequence_num"],
            front_path=Path(row["front_path"]),
            back_path=Path(row["back_path"]) if row["back_path"] else None,
            ocr_text_back=row["ocr_text_back"],
            ocr_raw_results=json.loads(row["ocr_raw_results"]) if row["ocr_raw_results"] else None,
            extracted_date=date.fromisoformat(row["extracted_date"]) if row["extracted_date"] else None,
            date_source=DateSource(row["date_source"]) if row["date_source"] else None,
            date_confidence=DateConfidence(row["date_confidence"]) if row["date_confidence"] else None,
            ai_analysis=json.loads(row["ai_analysis"]) if row["ai_analysis"] else None,
            final_date=date.fromisoformat(row["final_date"]) if row["final_date"] else None,
            final_location=final_location,
            final_description=row["final_description"],
            final_keywords=json.loads(row["final_keywords"]) if row["final_keywords"] else [],
            status=PhotoStatus(row["status"]),
            needs_review=bool(row["needs_review"]),
            review_notes=row["review_notes"],
            output_front_path=Path(row["output_front_path"]) if row["output_front_path"] else None,
            output_back_path=Path(row["output_back_path"]) if row["output_back_path"] else None,
            created_at=datetime.fromisoformat(row["created_at"]),
            updated_at=datetime.fromisoformat(row["updated_at"]),
        )

    # ==================== Template Operations ====================

    def create_template(self, template: BatchTemplate) -> BatchTemplate:
        """Create a new batch template."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO batch_templates (
                    id, name, location_description, location_lat, location_lon,
                    people, default_keywords, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    template.id,
                    template.name,
                    template.location_description,
                    template.location.latitude if template.location else None,
                    template.location.longitude if template.location else None,
                    json.dumps(template.people),
                    json.dumps(template.default_keywords),
                    template.created_at.isoformat(),
                ),
            )
        return template

    def get_all_templates(self) -> list[BatchTemplate]:
        """Get all batch templates."""
        with self.connection() as conn:
            rows = conn.execute(
                "SELECT * FROM batch_templates ORDER BY name"
            ).fetchall()

        return [self._row_to_template(row) for row in rows]

    def delete_template(self, template_id: str) -> bool:
        """Delete a batch template."""
        with self.connection() as conn:
            result = conn.execute(
                "DELETE FROM batch_templates WHERE id = ?", (template_id,)
            )
            return result.rowcount > 0

    def _row_to_template(self, row: sqlite3.Row) -> BatchTemplate:
        """Convert database row to BatchTemplate object."""
        location = None
        if row["location_lat"] and row["location_lon"]:
            location = Location(
                description=row["location_description"] or "",
                latitude=row["location_lat"],
                longitude=row["location_lon"],
            )

        return BatchTemplate(
            id=row["id"],
            name=row["name"],
            location_description=row["location_description"],
            location=location,
            people=json.loads(row["people"]) if row["people"] else [],
            default_keywords=json.loads(row["default_keywords"]) if row["default_keywords"] else [],
            created_at=datetime.fromisoformat(row["created_at"]),
        )

    # ==================== Logging Operations ====================

    def log_action(
        self,
        action: str,
        photo_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        details: Optional[dict] = None,
    ) -> None:
        """Log a processing action."""
        with self.connection() as conn:
            conn.execute(
                """
                INSERT INTO processing_log (photo_id, batch_id, action, details, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    photo_id,
                    batch_id,
                    action,
                    json.dumps(details) if details else None,
                    datetime.now().isoformat(),
                ),
            )

    def get_logs(
        self,
        photo_id: Optional[str] = None,
        batch_id: Optional[str] = None,
        limit: int = 100,
    ) -> list[ProcessingLogEntry]:
        """Get processing logs."""
        with self.connection() as conn:
            query = "SELECT * FROM processing_log WHERE 1=1"
            params = []

            if photo_id:
                query += " AND photo_id = ?"
                params.append(photo_id)

            if batch_id:
                query += " AND batch_id = ?"
                params.append(batch_id)

            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)

            rows = conn.execute(query, params).fetchall()

        return [
            ProcessingLogEntry(
                id=row["id"],
                photo_id=row["photo_id"],
                batch_id=row["batch_id"],
                action=row["action"],
                details=json.loads(row["details"]) if row["details"] else {},
                timestamp=datetime.fromisoformat(row["timestamp"]),
            )
            for row in rows
        ]

    # ==================== Utility Methods ====================

    def check_photo_exists(self, front_path: Path) -> bool:
        """Check if a photo with the given front path already exists."""
        with self.connection() as conn:
            result = conn.execute(
                "SELECT 1 FROM photos WHERE front_path = ?", (str(front_path),)
            ).fetchone()
            return result is not None

    def get_next_sequence_num(self, batch_id: str) -> int:
        """Get the next sequence number for a batch."""
        with self.connection() as conn:
            result = conn.execute(
                "SELECT MAX(sequence_num) FROM photos WHERE batch_id = ?", (batch_id,)
            ).fetchone()
            max_seq = result[0] if result and result[0] else 0
            return max_seq + 1
