"""Shared test fixtures.

Isolate the filesystem so tests never write into the user's real
~/Pictures / ~/.photopipe folders. Several modules read paths from
``get_config()`` and copy/scan files there (originals preservation, per-bucket
scan folders, finalize output); autouse-redirect those to a temp dir.
"""
import importlib

import pytest

import photopipe.config as cfgmod

# Modules that reference get_config() to resolve on-disk paths at runtime.
_PATH_USING_MODULES = [
    "photopipe.capture_pipeline",
    "photopipe.file_manager",
    "photopipe.autocrop",
    "photopipe.scanner",
    "photopipe.bucket_triage",
]


@pytest.fixture(autouse=True)
def isolate_filesystem(tmp_path, monkeypatch):
    base = tmp_path / "pp"
    cfg = cfgmod.Config(
        paths={
            "input_folder": str(base / "input"),
            "output_folder": str(base / "output"),
            "archive_folder": str(base / "archive"),
            "database": str(base / "photopipe.db"),
        }
    )
    for name in (
        cfg.paths.input_folder,
        cfg.paths.output_folder,
        cfg.paths.archive_folder,
    ):
        name.mkdir(parents=True, exist_ok=True)

    # Replace the bound get_config reference inside each path-using module so
    # its file operations land in the temp dir, not the real Pictures folder.
    for modname in _PATH_USING_MODULES:
        try:
            mod = importlib.import_module(modname)
        except Exception:
            continue
        if hasattr(mod, "get_config"):
            monkeypatch.setattr(mod, "get_config", lambda cfg=cfg: cfg, raising=False)
    yield
