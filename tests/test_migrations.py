"""Tests for photopipe.migrations."""

import sqlite3

import pytest

from photopipe.migrations import run_all_migrations


def test_migration_creates_buckets_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='buckets'"
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()


def test_migration_adds_phase_to_photos(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE photos (id TEXT PRIMARY KEY, front_path TEXT NOT NULL);
            INSERT INTO photos VALUES ('p1', '/tmp/x.jpg');
            """
        )
        run_all_migrations(conn)
        cur = conn.execute("SELECT phase FROM photos WHERE id='p1'")
        assert cur.fetchone()[0] == "finalized"  # existing rows default to finalized
    finally:
        conn.close()


def test_migration_is_idempotent(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        run_all_migrations(conn)  # second run should not raise
        cur = conn.execute("SELECT COUNT(*) FROM buckets")
        assert cur.fetchone()[0] == 0
    finally:
        conn.close()


def test_migration_adds_ai_jobs_table(tmp_path):
    db_path = tmp_path / "test.db"
    conn = sqlite3.connect(db_path)
    try:
        run_all_migrations(conn)
        cur = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='ai_jobs'"
        )
        assert cur.fetchone() is not None
    finally:
        conn.close()
