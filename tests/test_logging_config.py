"""Logging setup writes to a rotating file under the data dir."""
import logging

from photopipe import logging_config
from photopipe.logging_config import get_logger, setup_logging


def test_setup_writes_to_log_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOPIPE_DATA_DIR", str(tmp_path))
    # Force a fresh configuration for this test
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    root = logging.getLogger("photopipe")
    root.handlers.clear()

    setup_logging()
    get_logger("scanner").info("hello from a test %d", 42)

    for h in root.handlers:
        h.flush()
    log_path = tmp_path / "logs" / "photopipe.log"
    assert log_path.exists()
    contents = log_path.read_text()
    assert "hello from a test 42" in contents
    assert "photopipe.scanner" in contents


def test_setup_is_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("PHOTOPIPE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(logging_config, "_CONFIGURED", False)
    root = logging.getLogger("photopipe")
    root.handlers.clear()
    setup_logging()
    n = len(root.handlers)
    setup_logging()
    assert len(root.handlers) == n  # no duplicate handlers
