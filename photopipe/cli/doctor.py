"""Diagnose common PhotoPipe setup issues."""
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from photopipe.config import get_config


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    fix: Optional[str] = None


def check_exiftool() -> Check:
    found = shutil.which("exiftool")
    return Check(
        "ExifTool installed", bool(found),
        detail=found or "not found in PATH",
        fix="brew install exiftool" if not found else None,
    )


def check_sane() -> Check:
    found = shutil.which("scanimage")
    return Check(
        "SANE / scanimage installed", bool(found),
        detail=found or "not found in PATH",
        fix="brew install sane-backends" if not found else None,
    )


def check_scanner_discovery() -> Check:
    if not shutil.which("scanimage"):
        return Check("Scanner discovery", False, "scanimage not installed")
    try:
        r = subprocess.run(
            ["scanimage", "-L"], capture_output=True, text=True, timeout=5
        )
        found_any = "No scanners were identified" not in (r.stdout + r.stderr)
        return Check(
            "Scanner discovery", found_any,
            detail=r.stdout.strip() or r.stderr.strip(),
            fix=(
                "On macOS Tahoe (26+), grant 'Local Network' permission to your "
                "terminal app in System Settings → Privacy & Security → Local Network. "
                "Tahoe periodically drops this permission silently after updates."
            ) if not found_any else None,
        )
    except subprocess.TimeoutExpired:
        return Check(
            "Scanner discovery", False, "scanimage -L timed out (5s)",
            fix="Network scanner may be unreachable, or Local Network permission missing.",
        )


def check_anthropic_key() -> Check:
    cfg = get_config()
    key = os.environ.get(cfg.vlm.api_key_env_var)
    return Check(
        "Anthropic API key", bool(key),
        detail=f"{cfg.vlm.api_key_env_var}: {'set' if key else 'unset'}",
        fix=f"export {cfg.vlm.api_key_env_var}=sk-..." if not key else None,
    )


def check_mistral_key() -> Check:
    cfg = get_config()
    key = os.environ.get(cfg.handwriting_ocr.mistral_api_key_env_var)
    if cfg.handwriting_ocr.provider == "claude":
        return Check("Mistral API key", True, "not required (provider=claude)")
    return Check(
        "Mistral API key", bool(key),
        detail=f"{cfg.handwriting_ocr.mistral_api_key_env_var}: {'set' if key else 'unset'}",
        fix=(
            f"export {cfg.handwriting_ocr.mistral_api_key_env_var}=...  "
            "(or set handwriting_ocr.provider=claude in config to skip Mistral)"
        ) if not key else None,
    )


def check_model_alias() -> Check:
    """Verify the configured Claude model exists in Anthropic's model list.

    Catches a stale/typo'd `vlm.model` config before the first real
    curate run fails with an opaque 404. Skips gracefully when no key
    is set (check_anthropic_key already reports that).
    """
    import json
    import urllib.error
    import urllib.request

    cfg = get_config()
    model = cfg.vlm.model
    key = os.environ.get(cfg.vlm.api_key_env_var)
    if not key:
        return Check(
            "Claude model alias", True,
            detail=f"{model} (not verified — no API key)",
        )
    try:
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/models?limit=100",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.load(resp)
        available = {m["id"] for m in data.get("data", [])}
    except (urllib.error.URLError, ValueError, KeyError, TimeoutError) as e:
        return Check(
            "Claude model alias", True,
            detail=f"{model} (could not verify: {e})",
        )
    return Check(
        "Claude model alias", model in available,
        detail=f"{model}: {'valid' if model in available else 'NOT in /v1/models'}",
        fix=(
            f"Set vlm.model in config to a valid alias. Available: "
            f"{', '.join(sorted(m for m in available if 'sonnet' in m or 'opus' in m))}"
        ) if model not in available else None,
    )


def check_face_model() -> Check:
    """Report whether the InsightFace buffalo_l model pack is downloaded.

    The pack (~300 MB) downloads automatically on the first face
    detection. This check just tells the owner whether that has
    happened yet — a missing pack is not an error.
    """
    from pathlib import Path

    model_dir = Path.home() / ".insightface" / "models" / "buffalo_l"
    present = model_dir.exists() and any(model_dir.glob("*.onnx"))
    return Check(
        "Face model (InsightFace buffalo_l)", present,
        detail="downloaded" if present else "not downloaded yet",
        fix=(
            "Downloads automatically (~300 MB) the first time you run "
            "Detect on the Faces page."
        ) if not present else None,
    )


CHECKS = [
    check_exiftool, check_sane, check_scanner_discovery,
    check_anthropic_key, check_mistral_key, check_model_alias,
    check_face_model,
]


def run_doctor() -> int:
    print("PhotoPipe Doctor")
    print("=" * 40)
    failed = 0
    for fn in CHECKS:
        c = fn()
        icon = "✓" if c.ok else "✗"
        print(f"{icon} {c.name}: {c.detail}")
        if not c.ok:
            failed += 1
            if c.fix:
                print(f"   → fix: {c.fix}")
    print("=" * 40)
    print(f"{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 0 if failed == 0 else 1
