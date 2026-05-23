"""Round-trip tests for Config YAML serialization.

The save → load loop must use only ``yaml.safe_load``-compatible
constructs. ``yaml.dump`` on a Pydantic ``model_dump()`` output emits
``!!python/object/apply:pathlib.PosixPath`` tags for every Path field,
which ``safe_load`` refuses — silently breaking the Settings page's
Save button.
"""
from pathlib import Path

import yaml

from photopipe.config import Config


def test_to_yaml_writes_safe_loadable_output(tmp_path):
    out = tmp_path / "config.yaml"
    Config().to_yaml(out)
    # safe_load must not raise — the raw text must contain no Python tags
    text = out.read_text()
    assert "!!python/object" not in text
    yaml.safe_load(text)  # must not raise


def test_save_load_roundtrip_preserves_values(tmp_path):
    original = Config()
    # Mutate every category we expect users to edit via the Settings page.
    original.vlm.model = "claude-opus-4-7"
    original.vlm.cache_ttl = "1h"
    original.handwriting_ocr.provider = "mistral"
    original.handwriting_ocr.confidence_fallback_threshold = 0.42
    original.scanner.front_pattern = "custom_{num}.jpg"

    out = tmp_path / "config.yaml"
    original.to_yaml(out)
    loaded = Config.from_yaml(out)

    assert loaded.vlm.model == "claude-opus-4-7"
    assert loaded.vlm.cache_ttl == "1h"
    assert loaded.handwriting_ocr.provider == "mistral"
    assert loaded.handwriting_ocr.confidence_fallback_threshold == 0.42
    assert loaded.scanner.front_pattern == "custom_{num}.jpg"


def test_paths_serialize_as_plain_strings(tmp_path):
    """Path fields must serialize as plain strings, not Python objects.

    (We assert against the YAML directly rather than the reloaded
    ``Config.paths`` because ``PathsConfig.model_post_init`` runs
    ``expanduser().resolve()`` on every path — on macOS that turns
    ``/tmp/...`` into ``/private/tmp/...`` via symlink resolution.
    The post-resolve string is implementation detail; the property we
    care about is that the YAML contains plain strings, not Python
    object tags.)
    """
    original = Config()
    out = tmp_path / "config.yaml"
    original.to_yaml(out)
    text = out.read_text()
    assert "!!python/object" not in text
    raw = yaml.safe_load(text)
    assert isinstance(raw["paths"]["database"], str)
    assert isinstance(raw["paths"]["input_folder"], str)
    assert isinstance(raw["paths"]["output_folder"], str)
    assert isinstance(raw["paths"]["archive_folder"], str)
