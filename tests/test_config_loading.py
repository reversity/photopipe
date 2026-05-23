"""Config file discovery and legacy-format migration.

Two coupled bugs being fixed here:

1. ``find_config_file()`` previously walked ``DEFAULT_CONFIG_PATHS`` in
   declaration order, which put the repo's ``./config.yaml`` template
   **before** the user's ``~/.photopipe/config.yaml`` — so user
   customisations were silently ignored.

2. Once user dotfiles actually win, legacy files written by an older
   ``Config.to_yaml`` (which emitted ``!!python/object/apply:pathlib.PosixPath``
   tags) blow up ``safe_load`` on read. The loader has to fall back to
   ``yaml.FullLoader`` for those files and rewrite them in the new
   clean format so the next load goes through ``safe_load`` cleanly.
"""
import textwrap
from pathlib import Path

import pytest
import yaml

from photopipe import config as config_module
from photopipe.config import Config, find_config_file, get_config


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(content).lstrip())


def test_default_paths_put_user_dotfile_first():
    """Production order of DEFAULT_CONFIG_PATHS must list the user dotfile first.

    The other tests in this file monkeypatch the list — this one is the
    canary for the real production constant.
    """
    paths = config_module.DEFAULT_CONFIG_PATHS
    user_dotfile = Path.home() / ".photopipe" / "config.yaml"
    assert paths[0] == user_dotfile, (
        f"User dotfile must come first, got {paths[0]} — see follow-up #31"
    )


def test_user_dotfile_wins_over_cwd_template(tmp_path, monkeypatch):
    """When both ~/.photopipe/config.yaml and ./config.yaml exist, the user dotfile wins."""
    fake_home = tmp_path / "home"
    fake_cwd = tmp_path / "cwd"
    fake_home.mkdir()
    fake_cwd.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.chdir(fake_cwd)
    # Force the module-level constant to recompute against the patched Path.home / cwd.
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATHS",
        [
            fake_home / ".photopipe" / "config.yaml",
            fake_cwd / "config.yaml",
        ],
    )

    _write(
        fake_home / ".photopipe" / "config.yaml",
        """
        vlm:
          model: from-user-dotfile
        """,
    )
    _write(
        fake_cwd / "config.yaml",
        """
        vlm:
          model: from-repo-template
        """,
    )

    assert find_config_file() == fake_home / ".photopipe" / "config.yaml"


def test_cwd_template_used_when_no_user_dotfile(tmp_path, monkeypatch):
    fake_home = tmp_path / "home"
    fake_cwd = tmp_path / "cwd"
    fake_home.mkdir()
    fake_cwd.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    monkeypatch.chdir(fake_cwd)
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATHS",
        [
            fake_home / ".photopipe" / "config.yaml",
            fake_cwd / "config.yaml",
        ],
    )

    _write(fake_cwd / "config.yaml", "vlm: {model: cwd}")

    assert find_config_file() == fake_cwd / "config.yaml"


def test_legacy_python_tagged_yaml_migrates_on_load(tmp_path, monkeypatch):
    """A file with ``!!python/object/apply:pathlib.PosixPath`` tags must load AND get rewritten clean."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    user_yaml = fake_home / ".photopipe" / "config.yaml"
    user_yaml.parent.mkdir()
    user_yaml.write_text(
        "paths:\n"
        "  input_folder: !!python/object/apply:pathlib.PosixPath\n"
        "  - /tmp/legacy/input\n"
        "vlm:\n"
        "  model: legacy-model-marker\n"
    )
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATHS",
        [user_yaml],
    )
    get_config.cache_clear()

    cfg = get_config()
    assert cfg.vlm.model == "legacy-model-marker"

    # Post-migration, the file on disk no longer contains Python tags.
    text = user_yaml.read_text()
    assert "!!python/object" not in text
    # And it round-trips through safe_load cleanly.
    assert isinstance(yaml.safe_load(text)["vlm"]["model"], str)


def test_clean_yaml_left_untouched(tmp_path, monkeypatch):
    """A clean (already-safe) file must NOT be rewritten on load — that would churn mtimes."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    user_yaml = fake_home / ".photopipe" / "config.yaml"
    user_yaml.parent.mkdir()
    user_yaml.write_text("vlm:\n  model: clean-marker\n")
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATHS",
        [user_yaml],
    )
    get_config.cache_clear()

    before = user_yaml.stat().st_mtime_ns
    cfg = get_config()
    after = user_yaml.stat().st_mtime_ns

    assert cfg.vlm.model == "clean-marker"
    assert before == after, "clean YAML should not be rewritten on load"


def test_legacy_yaml_with_dropped_sections_still_loads(tmp_path, monkeypatch):
    """Legacy files often have removed sections (ocr:, ai_dating:). Pydantic must ignore them."""
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", staticmethod(lambda: fake_home))
    user_yaml = fake_home / ".photopipe" / "config.yaml"
    user_yaml.parent.mkdir()
    user_yaml.write_text(
        "paths:\n"
        "  database: !!python/object/apply:pathlib.PosixPath\n"
        "  - /tmp/legacy/db.sqlite\n"
        "ocr:\n"
        "  language: eng\n"
        "ai_dating:\n"
        "  model: claude-sonnet-4-20250514\n"
        "vlm:\n"
        "  model: kept-vlm-model\n"
    )
    monkeypatch.setattr(
        config_module,
        "DEFAULT_CONFIG_PATHS",
        [user_yaml],
    )
    get_config.cache_clear()

    cfg = get_config()
    assert cfg.vlm.model == "kept-vlm-model"
    # After migration the dropped sections are gone from disk.
    rewritten = yaml.safe_load(user_yaml.read_text())
    assert "ocr" not in rewritten
    assert "ai_dating" not in rewritten
