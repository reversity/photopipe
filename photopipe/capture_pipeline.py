"""Headless capture pipeline.

Stages: scan -> pair fronts/backs -> autocrop -> handwriting OCR
        -> persist with phase=captured, attached to bucket.

No batch concept, no owner context, no period-dating AI runs here.
Everything is keyed on a :class:`~photopipe.models.Bucket` so the
helper can keep feeding pages without knowing what they belong to.

Adapter shims (``scan_to_folder``, ``pair_fronts_and_backs``) wrap the
existing ``photopipe.scanner`` / ``photopipe.pairing`` helpers, which
have different signatures than this pipeline assumes. Tests patch the
shims directly; future cleanup can fold the shims back into their
underlying modules.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.handwriting_ocr import HandwritingOCR
from photopipe.logging_config import get_logger
from photopipe.models import (
    Bucket,
    DateSource,
    PhotoPair,
    PhotoPhase,
    PhotoStatus,
)
from photopipe.scanner import Scanner

log = get_logger(__name__)


def process_scanned_photo(
    input_path: Path,
    output_path: Optional[Path] = None,
    use_ai_orientation: bool = True,
) -> dict:
    """Lazy shim around :func:`photopipe.autocrop.process_scanned_photo`.

    Imported at call time so the heavy ``cv2`` dependency doesn't have to
    be installed merely to import the capture pipeline (e.g. in test
    environments where ``process_scanned_photo`` is patched out anyway).
    """
    from photopipe.autocrop import process_scanned_photo as _impl  # noqa: WPS433

    return _impl(input_path, output_path=output_path, use_ai_orientation=use_ai_orientation)


# ---------------------------------------------------------------------------
# Adapter shims around legacy scanner/pairing APIs.
#
# The plan calls these by simpler names than the existing modules expose. We
# expose them at module level so tests can patch them directly, and so the
# pipeline body stays readable.
# ---------------------------------------------------------------------------


def _next_scan_sequence(output_folder: Path, name_prefix: str) -> int:
    """First sequence number that won't collide with files already on disk.

    Every capture run scans into the same input folder with the same prefix;
    numbering must continue past existing files or the scanner would silently
    overwrite earlier scans.
    """
    highest = 0
    if output_folder.exists():
        stem_re = re.compile(
            rf"{re.escape(name_prefix)}_(\d+)(?:_b)?$", re.IGNORECASE
        )
        for existing in output_folder.iterdir():
            m = stem_re.fullmatch(existing.stem)
            if m:
                highest = max(highest, int(m.group(1)))
    return highest + 1


def scan_to_folder(
    device: str,
    *,
    resolution: int = 600,
    duplex: bool = True,
    output_folder: Optional[Path] = None,
    name_prefix: Optional[str] = None,
    errors: Optional[list[str]] = None,
) -> list[Path]:
    """Scan a stack to ``output_folder`` and return the produced file paths.

    Thin wrapper around :class:`photopipe.scanner.Scanner`. Outputs are
    interleaved fronts and (when ``duplex``) backs in scan order — exactly
    the form :func:`pair_fronts_and_backs` expects. When ``errors`` is
    given, partial-scan failures (paper jam, timeout) append a message to
    it and the successfully scanned files are still returned.
    """
    cfg = get_config()
    if output_folder is None:
        output_folder = cfg.paths.input_folder
    if name_prefix is None:
        name_prefix = cfg.scanner.default_name_prefix

    scanner = Scanner(device_name=device)
    results = scanner.scan_batch(
        output_folder=output_folder,
        name_prefix=name_prefix,
        start_sequence=_next_scan_sequence(output_folder, name_prefix),
        resolution=resolution,
        duplex=duplex,
        error_sink=errors,
    )

    files: list[Path] = []
    for r in results:
        files.append(r.front_path)
        if r.back_path is not None:
            files.append(r.back_path)
    return files


def pair_fronts_and_backs(files: list[Path]) -> list[tuple[Path, Optional[Path]]]:
    """Pair an interleaved scanner output list into (front, back) tuples.

    Recognises the FastFoto ``_b`` suffix convention: any file ending in
    ``_b`` before its extension is treated as the back of the preceding
    front. Fronts without backs come back as ``(front, None)``.
    """
    pairs: list[tuple[Path, Optional[Path]]] = []
    i = 0
    while i < len(files):
        front = files[i]
        if front.stem.endswith("_b"):
            # Orphan back with no preceding front; skip.
            i += 1
            continue
        back: Optional[Path] = None
        if i + 1 < len(files) and files[i + 1].stem.endswith("_b"):
            back = files[i + 1]
            i += 2
        else:
            i += 1
        pairs.append((front, back))
    return pairs


# ---------------------------------------------------------------------------
# Public pipeline API
# ---------------------------------------------------------------------------


@dataclass
class CaptureProgress:
    """A single progress event emitted during capture."""

    stage: str
    current: int = 0
    total: int = 0
    message: str = ""


@dataclass
class CaptureResult:
    """Summary returned at the end of a capture run."""

    photos_added: int
    bucket_id: str
    errors: list[str] = field(default_factory=list)


def capture_batch(
    bucket: Bucket,
    *,
    db: Database,
    scanner_device: str,
    resolution: int = 600,
    duplex: bool = True,
    progress: Optional[Callable[[CaptureProgress], None]] = None,
) -> CaptureResult:
    """Run one capture pass: scan a stack into ``bucket``.

    Stages emitted via the ``progress`` callback: ``scanning``, ``pairing``,
    ``processing``, ``ocr``, ``ocr_error``, ``done``.
    """

    def emit(stage: str, **kw) -> None:
        if progress:
            progress(CaptureProgress(stage=stage, **kw))

    errors: list[str] = []

    log.info(
        "capture_batch start: bucket=%s label=%r device=%r res=%s duplex=%s",
        bucket.id, bucket.label, scanner_device, resolution, duplex,
    )
    emit("scanning", message="Scanning stack...")
    try:
        files = scan_to_folder(
            device=scanner_device, resolution=resolution, duplex=duplex, errors=errors
        )
    except RuntimeError as e:
        # Scanner unreachable / no device: a routine failure for the helper,
        # not a crash — surface it through the normal error channel.
        msg = str(e)
        log.warning("capture_batch scan failed for bucket=%s: %s", bucket.id, msg)
        emit("done", message=msg)
        return CaptureResult(photos_added=0, bucket_id=bucket.id, errors=[msg])
    if not files:
        msg = "Scanner returned no files"
        emit("done", message=msg)
        return CaptureResult(
            photos_added=0, bucket_id=bucket.id, errors=[msg]
        )

    emit("pairing", message=f"Pairing {len(files)} files...")
    pairs = pair_fronts_and_backs(files)

    # Continue numbering after any photos already attached to this bucket
    # (capture_batch can be called multiple times against the same bucket).
    existing_count = len(db.get_photos_by_bucket(bucket.id))
    next_seq = existing_count + 1

    # Snapshot pristine originals before autocrop mutates the scans in place.
    # Archive-at-finalize is too late: a bad crop would otherwise destroy the
    # only copy of the raw scan.
    originals_dir = get_config().paths.archive_folder / "_originals" / bucket.id
    originals_dir.mkdir(parents=True, exist_ok=True)
    for front, back in pairs:
        for src in (front, back):
            if src is None:
                continue
            dest = originals_dir / src.name
            if not dest.exists():
                try:
                    shutil.copy2(src, dest)
                except OSError as e:
                    errors.append(f"could not preserve original {src.name}: {e}")

    emit("processing", total=len(pairs))
    photos: list[PhotoPair] = []
    for i, (front, back) in enumerate(pairs):
        emit(
            "processing",
            current=i + 1,
            total=len(pairs),
            message=f"Crop + orient #{i + 1}",
        )
        try:
            process_scanned_photo(front, use_ai_orientation=True)
        except Exception as e:  # autocrop failures are non-fatal
            errors.append(f"autocrop failed for {front.name}: {e}")

        photo = PhotoPair(
            bucket_id=bucket.id,
            batch_id="",  # not in a batch until convert_to_batch
            sequence_num=next_seq + i,
            front_path=front,
            back_path=back,
            phase=PhotoPhase.CAPTURED,
            status=PhotoStatus.INGESTED,
        )
        photos.append(photo)
        db.create_photo(photo)

    # Handwriting OCR: synchronous per-photo for the MVP. Task 8 swaps in
    # the Batch API. OCR being unavailable (no API key) must not fail the
    # capture — the photos are already scanned and persisted.
    if any(photo.back_path for photo in photos):
        ocr = None
        try:
            ocr = HandwritingOCR()
        except Exception as e:
            msg = f"Handwriting OCR skipped: {e}"
            errors.append(msg)
            emit("ocr_error", message=msg)
        if ocr is not None:
            for i, photo in enumerate(photos):
                if not photo.back_path:
                    continue
                emit(
                    "ocr",
                    current=i + 1,
                    total=len(photos),
                    message=f"OCR back #{i + 1}",
                )
                try:
                    result = ocr.ocr_back(photo.back_path)
                    photo.handwriting_ocr_text = result.text
                    photo.handwriting_ocr_provider = result.provider
                    photo.handwriting_ocr_confidence = result.confidence
                    if result.extracted_date:
                        photo.extracted_date = result.extracted_date
                        photo.date_source = DateSource.OCR_BACK
                    db.update_photo(photo)
                except Exception as e:
                    msg = f"OCR failed for #{i + 1}: {e}"
                    log.warning("bucket=%s %s", bucket.id, msg)
                    errors.append(msg)
                    emit("ocr_error", message=msg)

    log.info(
        "capture_batch done: bucket=%s added=%d errors=%d",
        bucket.id, len(photos), len(errors),
    )
    emit("done", total=len(photos), current=len(photos))
    return CaptureResult(
        photos_added=len(photos), bucket_id=bucket.id, errors=errors
    )
