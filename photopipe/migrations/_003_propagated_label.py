"""Migration 003: track the label each face cluster last propagated.

``propagate_labels`` adds cluster labels to photo keywords. Without a
record of what was applied, renaming a cluster after an apply left the
old name in the keywords forever. ``propagated_label`` stores the label
as last propagated so a re-apply can remove the stale one.
"""

import sqlite3

MIGRATION_ID = "003_propagated_label"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 003."""
    columns = {
        row[1] for row in conn.execute("PRAGMA table_info(face_clusters)")
    }
    if "propagated_label" not in columns:
        conn.execute("ALTER TABLE face_clusters ADD COLUMN propagated_label TEXT")
