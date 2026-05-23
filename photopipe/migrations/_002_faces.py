"""Migration 002: add faces and face_clusters tables.

These tables back the on-device face-clustering feature: every detected
face is persisted with its ArcFace embedding so clustering can re-run
without re-detecting, and clusters carry the owner-entered label that
later propagates to per-photo keywords.
"""

import sqlite3

MIGRATION_ID = "002_faces"


def up(conn: sqlite3.Connection) -> None:
    """Apply the schema changes for migration 002."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS face_clusters (
            id TEXT PRIMARY KEY,
            batch_id TEXT REFERENCES batches(id),
            label TEXT,
            representative_face_id TEXT,
            is_noise INTEGER NOT NULL DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS faces (
            id TEXT PRIMARY KEY,
            photo_id TEXT REFERENCES photos(id),
            batch_id TEXT REFERENCES batches(id),
            bbox JSON NOT NULL,
            embedding BLOB NOT NULL,
            crop_path TEXT,
            cluster_id TEXT REFERENCES face_clusters(id),
            detection_score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE INDEX IF NOT EXISTS idx_faces_batch_id ON faces(batch_id);
        CREATE INDEX IF NOT EXISTS idx_faces_cluster_id ON faces(cluster_id);
        CREATE INDEX IF NOT EXISTS idx_face_clusters_batch ON face_clusters(batch_id);
        """
    )
