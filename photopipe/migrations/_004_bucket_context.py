"""Migration 004: bucket container photo + AI-suggested context.

A helper can photograph the physical container (album cover, envelope —
often carrying the owner's Post-it notes with approximate years) with the
Mac's camera; ``context_image_path`` stores that photo. ``suggested_context``
stores the AI triage proposal (dates, events, locations) that pre-fills the
convert-to-batch form.
"""

import sqlite3

MIGRATION_ID = "004_bucket_context"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 004."""
    columns = {row[1] for row in conn.execute("PRAGMA table_info(buckets)")}
    if "context_image_path" not in columns:
        conn.execute("ALTER TABLE buckets ADD COLUMN context_image_path TEXT")
    if "suggested_context" not in columns:
        conn.execute("ALTER TABLE buckets ADD COLUMN suggested_context JSON")
