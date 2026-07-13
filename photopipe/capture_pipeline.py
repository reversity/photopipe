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
import threading
from concurrent.futures import ThreadPoolExecutor
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
from photopipe.scanner import Scanner, ScannerBusy, scanner_session

log = get_logger(__name__)


# Autocrop + AI orientation + handwriting OCR are slow (seconds per photo) and
# DON'T touch the scanner, so they run AFTER the scanner lock is released, in a
# single background worker. The helper can scan the next stack immediately;
# processing catches up in order. One worker keeps API load bounded and avoids
# any file/DB races between overlapping captures.
_bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pp-capture-bg")
_bg_lock = threading.Lock()
_bg_pending: dict[str, int] = {}  # bucket_id -> photos still being processed
_resumed = False  # process-level guard so resume runs only once per server


def background_pending(bucket_id: str) -> int:
    """How many just-captured photos are still processing in the background."""
    with _bg_lock:
        return _bg_pending.get(bucket_id, 0)


def wait_for_background(timeout: float = 15.0) -> bool:
    """Block until all queued background processing has finished.

    For tests and headless CLI runs (the Streamlit UI never blocks on this).
    Returns True if drained, False on timeout.
    """
    import time as _time

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        with _bg_lock:
            if not _bg_pending:
                return True
        _time.sleep(0.02)
    return False


def _bg_add(bucket_id: str, n: int) -> None:
    with _bg_lock:
        _bg_pending[bucket_id] = _bg_pending.get(bucket_id, 0) + n


def _bg_finish_one(bucket_id: str) -> None:
    with _bg_lock:
        remaining = _bg_pending.get(bucket_id, 0) - 1
        if remaining > 0:
            _bg_pending[bucket_id] = remaining
        else:
            _bg_pending.pop(bucket_id, None)


def _original_path_for(bucket_id: str, front_path: Path) -> Optional[Path]:
    """The pristine backup of a front, if it exists."""
    original = get_config().paths.archive_folder / "_originals" / bucket_id / front_path.name
    return original if original.exists() else None


def _process_one_photo(db: Database, bucket_id: str, photo, ocr) -> None:
    """Crop/orient the front (from its pristine original, so re-running is safe)
    and OCR the back, then mark the photo done. Best-effort; failures logged."""
    # Autocrop from the ORIGINAL into the working front path. Cropping from the
    # pristine copy (not the possibly-already-cropped working file) makes this
    # idempotent, so a resumed/retried run can't double-crop.
    cfg = get_config().autocrop
    original = _original_path_for(bucket_id, photo.front_path)
    if cfg.enabled and original is not None:
        try:
            process_scanned_photo(
                original, output_path=photo.front_path,
                use_ai_orientation=cfg.ai_orientation,
            )
        except Exception as e:
            log.warning("bg autocrop failed for %s: %s", photo.front_path, e)

    if photo.back_path and ocr is not None:
        try:
            result = ocr.ocr_back(photo.back_path)
            photo.handwriting_ocr_text = result.text
            photo.handwriting_ocr_provider = result.provider
            photo.handwriting_ocr_confidence = result.confidence
            if result.extracted_date:
                photo.extracted_date = result.extracted_date
                photo.date_source = DateSource.OCR_BACK
        except Exception as e:
            log.warning("bg OCR failed for %s: %s", photo.back_path, e)

    photo.processing_status = "done"
    try:
        db.update_photo(photo)
    except Exception as e:
        log.error("bg could not mark photo %s done: %s", photo.id, e)


def _process_captured_photos(bucket_id: str, db_path, photos: list) -> None:
    """Background: process each just-captured photo, updating the DB. Runs in the
    background worker thread with its own Database; never touches Streamlit.
    """
    db = Database(db_path)
    ocr = _make_ocr(photos)
    for photo in photos:
        try:
            _process_one_photo(db, bucket_id, photo, ocr)
        finally:
            _bg_finish_one(bucket_id)
    log.info("background processing complete for bucket=%s", bucket_id)


def _make_ocr(photos):
    """Construct one HandwritingOCR if any photo has a back; else None."""
    if not any(p.back_path for p in photos):
        return None
    try:
        return HandwritingOCR()
    except Exception as e:
        log.warning("bg handwriting OCR unavailable: %s", e)
        return None


def resume_pending_processing(db: Database) -> int:
    """Re-enqueue any photos whose background processing was interrupted.

    Called on app startup so a restart mid-backlog doesn't leave scans
    permanently un-cropped and their backs unread. Cropping from the pristine
    original makes re-processing idempotent, and OCR is naturally repeatable.
    Returns the number of photos re-queued.
    """
    pending = db.get_photos_pending_processing()
    if not pending:
        return 0
    # Group by bucket so the UI's per-bucket "finishing N" counter is right.
    by_bucket: dict[str, list] = {}
    for photo in pending:
        by_bucket.setdefault(photo.bucket_id or "", []).append(photo)
    for bucket_id, photos in by_bucket.items():
        _bg_add(bucket_id, len(photos))
        _bg_executor.submit(_process_captured_photos, bucket_id, db.db_path, photos)
    log.info("resumed background processing for %d interrupted photo(s)", len(pending))
    return len(pending)


def reprocess_bucket(db: Database, bucket_id: str) -> int:
    """Re-run crop/deskew/OCR for a bucket from its pristine originals.

    Lets the owner apply improved processing (e.g. the new deskew) to photos
    already captured. Only photos whose original was preserved can be re-cropped;
    returns how many were re-queued.
    """
    photos = db.get_photos_by_bucket(bucket_id)
    to_do = [p for p in photos if _original_path_for(bucket_id, p.front_path)]
    if not to_do:
        return 0
    _bg_add(bucket_id, len(to_do))
    _bg_executor.submit(_process_captured_photos, bucket_id, db.db_path, to_do)
    log.info("reprocessing %d photo(s) in bucket=%s from originals", len(to_do), bucket_id)
    return len(to_do)


def resume_pending_processing_once(db: Database) -> int:
    """Run :func:`resume_pending_processing` at most once per server process.

    Safe to call from every page load / session start — the guard means only
    the first call (after a server start) actually re-enqueues, so pending
    photos aren't double-submitted on later reruns.
    """
    global _resumed
    with _bg_lock:
        if _resumed:
            return 0
        _resumed = True
    try:
        return resume_pending_processing(db)
    except Exception as e:  # never let resume break app startup
        log.error("resume_pending_processing failed: %s", e)
        return 0


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

    # Refuse to start a second capture while one is already running — otherwise
    # two `scanimage` processes race the single-client network scanner and the
    # loser reports "Device busy". This makes it impossible to start a new stack
    # until the previous one has fully finished.
    try:
        with scanner_session():
            log.info(
                "capture_batch start: bucket=%s label=%r device=%r res=%s duplex=%s",
                bucket.id, bucket.label, scanner_device, resolution, duplex,
            )
            return _capture_locked(
                bucket, db=db, scanner_device=scanner_device, resolution=resolution,
                duplex=duplex, emit=emit, errors=errors,
            )
    except ScannerBusy:
        msg = (
            "The scanner is still finishing the previous stack. Please wait for "
            "it to finish (watch the progress above), then press Scan again."
        )
        log.warning("capture_batch refused (scanner busy) for bucket=%s", bucket.id)
        emit("done", message=msg)
        return CaptureResult(photos_added=0, bucket_id=bucket.id, errors=[msg])


def _capture_locked(
    bucket: Bucket,
    *,
    db: Database,
    scanner_device: str,
    resolution: int,
    duplex: bool,
    emit,
    errors: list,
) -> CaptureResult:
    """The body of :func:`capture_batch`, run while holding the scanner lock."""
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
    # only copy of the raw scan. Track which sources were actually preserved
    # so we NEVER autocrop a front whose backup failed (that would leave no
    # pristine copy of an irreplaceable scan).
    preserved: set[Path] = set()
    originals_dir = get_config().paths.archive_folder / "_originals" / bucket.id
    try:
        originals_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        log.error("could not create originals dir %s: %s", originals_dir, e)
        errors.append(f"could not create backup folder: {e}")
    for front, back in pairs:
        for src in (front, back):
            if src is None:
                continue
            dest = originals_dir / src.name
            if dest.exists():
                preserved.add(src)
                continue
            try:
                shutil.copy2(src, dest)
                preserved.add(src)
            except OSError as e:
                log.warning("could not preserve original %s: %s", src.name, e)
                errors.append(f"could not preserve original {src.name}: {e}")

    # Foreground: persist a raw row per photo FAST, then hand the slow work
    # (autocrop + AI orientation + OCR) to the background worker. This releases
    # the scanner as soon as the physical scan + DB writes are done, so the
    # helper can immediately scan the next stack.
    emit("saving", message=f"Saving {len(pairs)} photos...")
    photos: list[PhotoPair] = []
    for i, (front, back) in enumerate(pairs):
        if front not in preserved:
            # Backup failed — keep the raw scan untouched (no autocrop) so no
            # original is ever lost; it's still ingested and its back still OCR'd.
            # (The background derives "can crop" from the original's existence.)
            errors.append(
                f"kept {front.name} uncropped (its backup could not be saved)"
            )

        photo = PhotoPair(
            bucket_id=bucket.id,
            batch_id="",  # not in a batch until convert_to_batch
            sequence_num=next_seq + i,
            front_path=front,
            back_path=back,
            phase=PhotoPhase.CAPTURED,
            status=PhotoStatus.INGESTED,
            processing_status="pending",  # background worker flips to "done"
        )
        try:
            db.create_photo(photo)
            photos.append(photo)
        except Exception as e:
            log.error("failed to persist photo %s: %s", front.name, e)
            errors.append(f"could not save {front.name} to the library: {e}")

    if photos:
        _bg_add(bucket.id, len(photos))
        _bg_executor.submit(_process_captured_photos, bucket.id, db.db_path, photos)

    log.info(
        "capture_batch foreground done: bucket=%s added=%d errors=%d "
        "(queued for background processing)",
        bucket.id, len(photos), len(errors),
    )
    emit(
        "done", total=len(photos), current=len(photos),
        message=(
            f"Added {len(photos)} photos. Cropping and reading them in the "
            "background — you can scan the next stack now."
        ),
    )
    return CaptureResult(
        photos_added=len(photos), bucket_id=bucket.id, errors=errors
    )
