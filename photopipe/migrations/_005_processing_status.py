"""Migration 005: track per-photo background-processing state.

Capture persists a photo row immediately, then autocrops/orients/OCRs it in a
background worker. If that worker is interrupted (server restart, crash), the
photo would silently stay un-cropped and its back unread. ``processing_status``
records where each photo is so unfinished work can be resumed on startup.

Existing rows default to ``done`` — they were already processed synchronously
before this feature, so a resume must not re-touch them.
"""

import sqlite3

MIGRATION_ID = "005_processing_status"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 005."""
    # The base SCHEMA creates `photos`; guard in case migrations run against a
    # bare DB (as some migration tests do) where it doesn't exist yet.
    exists = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='photos'"
    ).fetchone()
    if not exists:
        return
    columns = {row[1] for row in conn.execute("PRAGMA table_info(photos)")}
    if "processing_status" not in columns:
        conn.execute(
            "ALTER TABLE photos ADD COLUMN processing_status TEXT DEFAULT 'done'"
        )
