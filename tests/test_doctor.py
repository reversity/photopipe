"""Tests for the photopipe doctor CLI."""
import json
import subprocess
import urllib.error
from unittest.mock import patch, MagicMock

from photopipe.cli.doctor import (
    check_exiftool,
    check_sane,
    check_scanner_discovery,
    check_anthropic_key,
    check_mistral_key,
    check_model_alias,
    check_face_model,
    run_doctor,
)


def _fake_models_response(model_ids):
    """Build a MagicMock context manager mimicking urlopen() for /v1/models."""
    payload = json.dumps({"data": [{"id": m} for m in model_ids]}).encode()
    cm = MagicMock()
    cm.__enter__.return_value.read.return_value = payload
    return cm


def test_check_exiftool_present():
    with patch("photopipe.cli.doctor.shutil.which", return_value="/usr/bin/exiftool"):
        c = check_exiftool()
    assert c.ok
    assert c.fix is None


def test_check_exiftool_missing():
    with patch("photopipe.cli.doctor.shutil.which", return_value=None):
        c = check_exiftool()
    assert not c.ok
    assert "brew install exiftool" in c.fix


def test_check_sane_present():
    with patch("photopipe.cli.doctor.shutil.which", return_value="/usr/bin/scanimage"):
        c = check_sane()
    assert c.ok
    assert c.fix is None


def test_check_sane_missing():
    with patch("photopipe.cli.doctor.shutil.which", return_value=None):
        c = check_sane()
    assert not c.ok
    assert "brew install sane-backends" in c.fix


def test_check_scanner_discovery_finds_scanner():
    with patch("photopipe.cli.doctor.shutil.which", return_value="/usr/bin/scanimage"), \
         patch("photopipe.cli.doctor.subprocess.run") as run:
        run.return_value = MagicMock(
            stdout="device `epsonds:net:...' is a Epson ...", stderr=""
        )
        c = check_scanner_discovery()
    assert c.ok


def test_check_scanner_discovery_no_scanner_includes_tahoe_hint():
    with patch("photopipe.cli.doctor.shutil.which", return_value="/usr/bin/scanimage"), \
         patch("photopipe.cli.doctor.subprocess.run") as run:
        run.return_value = MagicMock(
            stdout="No scanners were identified.", stderr=""
        )
        c = check_scanner_discovery()
    assert not c.ok
    assert "Local Network" in c.fix


def test_check_scanner_discovery_timeout():
    with patch("photopipe.cli.doctor.shutil.which", return_value="/usr/bin/scanimage"), \
         patch(
             "photopipe.cli.doctor.subprocess.run",
             side_effect=subprocess.TimeoutExpired(cmd="scanimage", timeout=5),
         ):
        c = check_scanner_discovery()
    assert not c.ok
    assert "timed out" in c.detail.lower()


def test_check_scanner_discovery_no_scanimage():
    with patch("photopipe.cli.doctor.shutil.which", return_value=None):
        c = check_scanner_discovery()
    assert not c.ok
    assert "scanimage not installed" in c.detail


def test_check_anthropic_key_present(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-abc123")
    c = check_anthropic_key()
    assert c.ok


def test_check_anthropic_key_missing(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = check_anthropic_key()
    assert not c.ok


def test_check_mistral_key_not_required_when_provider_claude(monkeypatch):
    from photopipe.config import get_config
    cfg = get_config()
    original = cfg.handwriting_ocr.provider
    cfg.handwriting_ocr.provider = "claude"
    try:
        monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
        c = check_mistral_key()
        assert c.ok
    finally:
        cfg.handwriting_ocr.provider = original


def test_check_mistral_key_present(monkeypatch):
    from photopipe.config import get_config
    cfg = get_config()
    original = cfg.handwriting_ocr.provider
    cfg.handwriting_ocr.provider = "auto"
    try:
        monkeypatch.setenv("MISTRAL_API_KEY", "ms-abc")
        c = check_mistral_key()
        assert c.ok
    finally:
        cfg.handwriting_ocr.provider = original


def test_check_model_alias_skips_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    c = check_model_alias()
    assert c.ok
    assert "not verified" in c.detail


def test_check_model_alias_valid_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    from photopipe.config import get_config
    model = get_config().vlm.model
    with patch(
        "urllib.request.urlopen",
        return_value=_fake_models_response([model, "claude-opus-4-7"]),
    ):
        c = check_model_alias()
    assert c.ok
    assert "valid" in c.detail


def test_check_model_alias_invalid_model(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    with patch(
        "urllib.request.urlopen",
        return_value=_fake_models_response(["claude-opus-4-7", "claude-sonnet-4-5"]),
    ):
        c = check_model_alias()
    assert not c.ok
    assert "NOT in /v1/models" in c.detail
    assert "claude-sonnet-4-5" in c.fix


def test_check_model_alias_tolerates_network_error(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    with patch(
        "urllib.request.urlopen",
        side_effect=urllib.error.URLError("connection refused"),
    ):
        c = check_model_alias()
    # Network failure must not fail the check — it's advisory only.
    assert c.ok
    assert "could not verify" in c.detail


def test_run_doctor_returns_zero_on_all_pass(monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("MISTRAL_API_KEY", "ms-x")
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"x")
    from photopipe.config import get_config
    model = get_config().vlm.model
    with patch(
        "photopipe.cli.doctor.shutil.which", side_effect=lambda c: f"/usr/bin/{c}"
    ), patch("photopipe.cli.doctor.subprocess.run") as run, patch(
        "urllib.request.urlopen", return_value=_fake_models_response([model])
    ):
        run.return_value = MagicMock(stdout="device 'x' is a scanner", stderr="")
        rc = run_doctor()
    assert rc == 0


def test_run_doctor_returns_nonzero_on_failure(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with patch("photopipe.cli.doctor.shutil.which", return_value=None):
        rc = run_doctor()
    assert rc != 0


def test_check_face_model_present(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    model_dir = tmp_path / ".insightface" / "models" / "buffalo_l"
    model_dir.mkdir(parents=True)
    (model_dir / "det_10g.onnx").write_bytes(b"x")
    c = check_face_model()
    assert c.ok


def test_check_face_model_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    c = check_face_model()
    assert not c.ok
    assert "downloaded" in c.detail.lower() or "first" in (c.fix or "").lower()
