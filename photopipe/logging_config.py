"""Central logging setup for PhotoPipe.

One rotating log file at ``~/.photopipe/logs/photopipe.log`` (plus stderr,
which the launchd server captures to server.err.log). Call
:func:`setup_logging` once at each entry point (the Streamlit app and the
CLI); modules just call :func:`get_logger`.

The point is post-hoc debugging of intermittent field failures — chiefly the
network scanner being grabbed by another device — so scan attempts, exact
commands, timings, and error classifications all land here with timestamps.
"""
from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_CONFIGURED = False

LOG_FORMAT = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"


def log_dir() -> Path:
    """Directory holding PhotoPipe logs (honors the Docker data dir)."""
    data_dir = os.environ.get("PHOTOPIPE_DATA_DIR")
    base = Path(data_dir) if data_dir else Path.home() / ".photopipe"
    d = base / "logs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_file() -> Path:
    return log_dir() / "photopipe.log"


def setup_logging(level: int = logging.INFO) -> None:
    """Configure root logging once. Idempotent across Streamlit reruns."""
    global _CONFIGURED
    if _CONFIGURED:
        return

    root = logging.getLogger("photopipe")
    root.setLevel(level)

    # Env override for verbose field debugging: PHOTOPIPE_LOG_LEVEL=DEBUG
    env_level = os.environ.get("PHOTOPIPE_LOG_LEVEL")
    if env_level:
        root.setLevel(getattr(logging, env_level.upper(), level))

    fmt = logging.Formatter(LOG_FORMAT)

    try:
        fh = RotatingFileHandler(
            log_file(), maxBytes=2_000_000, backupCount=5, encoding="utf-8"
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)
    except OSError:
        # A read-only or missing home dir must never stop the app starting.
        pass

    sh = logging.StreamHandler()  # -> stderr -> launchd server.err.log
    sh.setFormatter(fmt)
    root.addHandler(sh)

    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Get a module logger under the ``photopipe`` namespace.

    Safe to call before :func:`setup_logging`; it just won't have handlers
    until setup runs (messages fall back to Python's default).
    """
    short = name.split(".")[-1] if name else "photopipe"
    return logging.getLogger(f"photopipe.{short}")
