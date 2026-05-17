"""Tests for the photopipe doctor CLI."""
import subprocess
from unittest.mock import patch, MagicMock

from photopipe.cli.doctor import (
    check_exiftool,
    check_sane,
    check_scanner_discovery,
    check_anthropic_key,
    check_mistral_key,
    run_doctor,
)


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


def test_run_doctor_returns_zero_on_all_pass(monkeypatch, capsys):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-x")
    monkeypatch.setenv("MISTRAL_API_KEY", "ms-x")
    with patch(
        "photopipe.cli.doctor.shutil.which", side_effect=lambda c: f"/usr/bin/{c}"
    ), patch("photopipe.cli.doctor.subprocess.run") as run:
        run.return_value = MagicMock(stdout="device 'x' is a scanner", stderr="")
        rc = run_doctor()
    assert rc == 0


def test_run_doctor_returns_nonzero_on_failure(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    with patch("photopipe.cli.doctor.shutil.which", return_value=None):
        rc = run_doctor()
    assert rc != 0
