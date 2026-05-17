"""Migration 001: add buckets, ai_jobs, photo phase + handwriting OCR columns."""

import sqlite3
import warnings

MIGRATION_ID = "001_phase_and_buckets"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 001."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS buckets (
            id TEXT PRIMARY KEY,
            label TEXT NOT NULL,
            helper_name TEXT,
            status TEXT DEFAULT 'open',
            batch_id TEXT REFERENCES batches(id),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            closed_at TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS ai_jobs (
            id TEXT PRIMARY KEY,
            batch_id TEXT REFERENCES batches(id),
            provider TEXT NOT NULL,
            provider_job_id TEXT,
            status TEXT NOT NULL DEFAULT 'queued',
            submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            completed_at TIMESTAMP,
            photo_ids JSON,
            result_summary JSON
        );

        CREATE INDEX IF NOT EXISTS idx_buckets_status ON buckets(status);
        CREATE INDEX IF NOT EXISTS idx_ai_jobs_status ON ai_jobs(status);
        """
    )

    # Additive columns on existing tables. Each call is a no-op if the table
    # doesn't exist (fresh DB where legacy SCHEMA hasn't run) or the column
    # already exists (idempotency).
    _add_column(conn, "photos", "bucket_id", "TEXT REFERENCES buckets(id)")
    _add_column(conn, "photos", "phase", "TEXT DEFAULT 'finalized'")
    _add_column(conn, "photos", "handwriting_ocr_text", "TEXT")
    _add_column(conn, "photos", "handwriting_ocr_provider", "TEXT")
    _add_column(conn, "photos", "handwriting_ocr_confidence", "REAL")
    _add_column(conn, "batches", "source_bucket_ids", "JSON")


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _add_column(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    """Add ``col`` to ``table`` if the table exists and column is missing."""
    if not _table_exists(conn, table):
        warnings.warn(
            f"_add_column: skipped {table}.{col} because table {table!r} does not exist",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")
