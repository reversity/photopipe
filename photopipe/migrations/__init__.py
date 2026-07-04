"""Database migration runner for PhotoPipe.

Each migration module declares ``MIGRATION_ID`` and an ``up(conn)`` function.
``run_all_migrations`` is idempotent: it tracks applied migrations in the
``schema_migrations`` table and skips ones that have already run.
"""

import sqlite3

from photopipe.migrations import _001_phase_and_buckets, _002_faces, _003_propagated_label

MIGRATIONS = [_001_phase_and_buckets, _002_faces, _003_propagated_label]


def run_all_migrations(conn: sqlite3.Connection) -> None:
    """Apply any pending migrations to ``conn``.

    Safe to call multiple times. Each migration is recorded in
    ``schema_migrations`` after a successful ``up`` call.
    """
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            id TEXT PRIMARY KEY,
            applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    applied = {row[0] for row in conn.execute("SELECT id FROM schema_migrations")}
    for mod in MIGRATIONS:
        if mod.MIGRATION_ID not in applied:
            mod.up(conn)
            conn.execute(
                "INSERT INTO schema_migrations(id) VALUES (?)", (mod.MIGRATION_ID,)
            )
            conn.commit()
