"""
File management operations for PhotoPipe.

Handles file copying, renaming, and output folder organization.
"""

import filecmp
import json
import shutil
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from PIL import Image

from photopipe.config import get_config
from photopipe.models import (
    Batch,
    PhotoPair,
    BatchReport,
    PhotoStatus,
    DateSource,
)
from photopipe.database import Database
from photopipe.metadata import write_metadata


def sanitize_filename(name: str) -> str:
    """
    Sanitize a string for use in filenames.

    Args:
        name: String to sanitize

    Returns:
        Safe filename string
    """
    # Replace spaces with underscores
    name = name.replace(" ", "_")

    # Remove or replace problematic characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        name = name.replace(char, "")

    # Remove leading/trailing dots and spaces
    name = name.strip(". ")

    return name


def generate_output_filename(
    photo: PhotoPair,
    batch: Batch,
    side: str = "front",  # "front" or "back"
) -> str:
    """
    Generate output filename based on template.

    Args:
        photo: PhotoPair to generate name for
        batch: Parent batch
        side: "front" or "back" (only used for back exports, front is default)

    Returns:
        Generated filename (without path)
    """
    config = get_config()

    # Get effective date
    photo_date = photo.final_date or photo.extracted_date or date.today()

    # Build filename from template
    template = config.output.filename_template
    batch_name = sanitize_filename(batch.name)

    # str.format silently ignores extra kwargs, so the {side} presence must
    # be checked explicitly — otherwise fronts and backs generate identical
    # names and the back overwrites the exported front.
    filename = template.format(
        date=photo_date.strftime("%Y-%m-%d"),
        batch_name=batch_name,
        sequence=photo.sequence_num,
        side=side,
    )
    if side == "back" and "{side}" not in template:
        filename = f"{filename}_back"

    # Add extension from original file
    if side == "front":
        ext = photo.front_path.suffix.lower()
    else:
        ext = photo.back_path.suffix.lower() if photo.back_path else ".jpg"

    return f"{filename}{ext}"


def generate_output_folder(batch: Batch) -> Path:
    """
    Generate output folder path based on batch date.

    All photos in a batch go to the same folder based on the batch start date.

    Args:
        batch: Batch to generate folder for

    Returns:
        Output folder path (with ~ expanded)
    """
    config = get_config()

    # Get date for folder structure - always use batch date to keep photos together
    if batch.date_start:
        year = str(batch.date_start.year)
        month = f"{batch.date_start.month:02d}"
    else:
        year = str(date.today().year)
        month = "00"

    batch_name = sanitize_filename(batch.name)

    # Build folder path from template
    template = config.output.folder_structure
    folder_path = template.format(
        year=year,
        month=month,
        batch_name=batch_name,
    )

    # Expand ~ and return absolute path
    output_folder = config.paths.output_folder / folder_path
    return Path(output_folder).expanduser().resolve()


def copy_to_archive(photo: PhotoPair, batch: Batch) -> tuple[Optional[Path], Optional[Path]]:
    """
    Copy original files to archive folder (untouched).

    Args:
        photo: PhotoPair to archive
        batch: Parent batch

    Returns:
        Tuple of (archived_front_path, archived_back_path)
    """
    config = get_config()
    batch_name = sanitize_filename(batch.name)

    archive_folder = (config.paths.archive_folder / batch_name).expanduser().resolve()
    archive_folder.mkdir(parents=True, exist_ok=True)

    front_dest = _archive_copy(photo.front_path, archive_folder)

    back_dest = None
    if photo.back_path and photo.back_path.exists():
        back_dest = _archive_copy(photo.back_path, archive_folder)

    return front_dest, back_dest


def _archive_copy(source: Path, archive_folder: Path) -> Path:
    """Copy ``source`` into the archive without ever losing a photo to a name
    collision: scanner filenames repeat across sessions and different batch
    names can sanitize to the same folder, so a same-named file that is NOT
    byte-identical gets a suffixed name instead of being skipped."""
    dest = archive_folder / source.name
    counter = 1
    while dest.exists():
        try:
            if filecmp.cmp(source, dest, shallow=False):
                return dest  # identical copy already archived
        except OSError:
            pass
        dest = archive_folder / f"{source.stem}_{counter}{source.suffix}"
        counter += 1
    shutil.copy2(source, dest)
    return dest


def finalize_photo(
    photo: PhotoPair,
    batch: Batch,
    db: Database,
    archive_originals: bool = True,
    export_backs: bool = False,
) -> PhotoPair:
    """
    Finalize a single photo: copy, rename, write metadata.

    Args:
        photo: PhotoPair to finalize
        batch: Parent batch
        db: Database instance
        archive_originals: Whether to copy originals to archive
        export_backs: Whether to include back images in output (default: False)

    Returns:
        Updated PhotoPair with output paths
    """
    config = get_config()

    # Archive originals first (backs are preserved in archive even if not exported)
    if archive_originals and config.output.preserve_originals:
        copy_to_archive(photo, batch)

    # Determine output folder - all photos in batch go to same folder
    output_folder = generate_output_folder(batch)
    output_folder.mkdir(parents=True, exist_ok=True)

    # Generate unique output filename for front
    # Format: PP_{date}_{batch}_{seq:04d}.jpg (PP prefix ensures uniqueness)
    front_filename = generate_output_filename(photo, batch, "front")
    front_output = output_folder / front_filename

    # Copy front image
    if not photo.front_path.exists():
        raise FileNotFoundError(f"Source file not found: {photo.front_path}")

    shutil.copy2(photo.front_path, front_output)

    # Verify copy succeeded
    if not front_output.exists():
        raise IOError(f"Failed to copy file to: {front_output}")

    # Write metadata to front
    write_metadata(photo, batch, front_output)
    photo.output_front_path = front_output

    # Handle back image (only if export_backs is True)
    if export_backs and photo.back_path and photo.back_path.exists():
        back_filename = generate_output_filename(photo, batch, "back")
        back_output = output_folder / back_filename

        shutil.copy2(photo.back_path, back_output)
        photo.output_back_path = back_output

    # Update status
    photo.status = PhotoStatus.FINALIZED
    db.update_photo(photo)
    db.log_action(
        photo_id=photo.id,
        batch_id=batch.id,
        action="finalized",
        details={
            "output_front": str(front_output),
            "output_back": str(photo.output_back_path) if photo.output_back_path else None,
        },
    )

    return photo


def finalize_batch(
    batch: Batch,
    db: Database,
    auto_approve_high_confidence: bool = False,
    finalize_all: bool = False,
    progress_callback: Optional[callable] = None,
) -> BatchReport:
    """
    Finalize all photos in a batch.

    Args:
        batch: Batch to finalize
        db: Database instance
        auto_approve_high_confidence: Auto-approve photos with high confidence dates
        finalize_all: Finalize all photos regardless of review status
        progress_callback: Optional callback(current, total) for progress

    Returns:
        BatchReport with summary
    """
    config = get_config()
    photos = db.get_photos_by_batch(batch.id)
    total = len(photos)

    # Track statistics
    date_sources: dict[str, int] = {}
    finalized_photos = []
    earliest_date: Optional[date] = None
    latest_date: Optional[date] = None

    for i, photo in enumerate(photos):
        # Auto-approve high confidence if enabled
        if auto_approve_high_confidence:
            # Handle both enum and string values for date_confidence
            confidence_val = photo.date_confidence.value if hasattr(photo.date_confidence, 'value') else photo.date_confidence
            source_val = photo.date_source.value if hasattr(photo.date_source, 'value') else photo.date_source

            if confidence_val == "high" and photo.extracted_date:
                photo.final_date = photo.extracted_date
                photo.needs_review = False
            # Also auto-approve AI-estimated dates
            elif source_val == "ai_estimated" and photo.extracted_date:
                photo.final_date = photo.extracted_date
                photo.needs_review = False

        # Skip photos that still need review (unless finalize_all or auto-approved)
        if photo.needs_review and not finalize_all and not auto_approve_high_confidence:
            continue

        # If no final date set, use extracted or batch default
        if not photo.final_date:
            if photo.extracted_date:
                photo.final_date = photo.extracted_date
            else:
                # Use batch default
                photo.final_date = batch.calculate_date_for_sequence(
                    photo.sequence_num, total
                )
                photo.date_source = DateSource.BATCH_DEFAULT

        # Finalize the photo
        photo = finalize_photo(photo, batch, db, export_backs=config.output.export_backs)
        finalized_photos.append(photo)

        # Track statistics
        if photo.date_source:
            source = photo.date_source.value if hasattr(photo.date_source, 'value') else photo.date_source
        else:
            source = "none"
        date_sources[source] = date_sources.get(source, 0) + 1

        if photo.final_date:
            if earliest_date is None or photo.final_date < earliest_date:
                earliest_date = photo.final_date
            if latest_date is None or photo.final_date > latest_date:
                latest_date = photo.final_date

        if progress_callback:
            progress_callback(i + 1, total)

    # Generate batch report
    output_folder = generate_output_folder(batch)
    output_folder.mkdir(parents=True, exist_ok=True)

    report = BatchReport(
        batch_name=batch.name,
        created=batch.created_at,
        finalized=datetime.now(),
        photo_count=len(finalized_photos),
        date_range={
            "earliest": earliest_date.isoformat() if earliest_date else None,
            "latest": latest_date.isoformat() if latest_date else None,
        },
        location={
            "description": batch.location_description,
            "coordinates": [batch.location.latitude, batch.location.longitude] if batch.location else None,
        } if batch.location_description or batch.location else None,
        date_source_breakdown=date_sources,
        people_tagged=batch.people,
        photos=[
            {
                "final_name": photo.output_front_path.name if photo.output_front_path else None,
                "original_name": photo.front_path.name,
                "date": photo.final_date.isoformat() if photo.final_date else None,
                "date_source": (photo.date_source.value if hasattr(photo.date_source, 'value') else photo.date_source) if photo.date_source else None,
                "date_confidence": (photo.date_confidence.value if hasattr(photo.date_confidence, 'value') else photo.date_confidence) if photo.date_confidence else None,
                "has_back": photo.back_path is not None,
            }
            for photo in finalized_photos
        ],
    )

    # Write report to output folder
    report_path = output_folder / "_batch_report.json"
    report_path.write_text(report.to_json())

    # Only mark complete once the work actually happened; a run that
    # finalized nothing (everything still needs review) is not complete.
    if finalized_photos:
        batch.status = "complete"
        db.update_batch(batch)

    db.log_action(
        batch_id=batch.id,
        action="batch_finalized",
        details={
            "photo_count": len(finalized_photos),
            "report_path": str(report_path),
        },
    )

    return report


def generate_web_copy(
    source_path: Path,
    dest_folder: Path,
    max_dimension: int = 2048,
) -> Path:
    """
    Generate a web-sized copy of an image.

    Args:
        source_path: Original image path
        dest_folder: Destination folder
        max_dimension: Maximum width or height

    Returns:
        Path to generated web copy
    """
    dest_folder.mkdir(parents=True, exist_ok=True)

    with Image.open(source_path) as img:
        # JPEG output can't hold alpha/palette modes
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Calculate new size maintaining aspect ratio
        width, height = img.size
        if width > max_dimension or height > max_dimension:
            if width > height:
                new_width = max_dimension
                new_height = int(height * (max_dimension / width))
            else:
                new_height = max_dimension
                new_width = int(width * (max_dimension / height))

            img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Save with reasonable quality
        dest_path = dest_folder / source_path.name
        img.save(dest_path, quality=85, optimize=True)

    return dest_path


def generate_thumbnail(
    source_path: Path,
    size: tuple[int, int] = (200, 200),
) -> Image.Image:
    """
    Generate a thumbnail for display in GUI.

    Args:
        source_path: Original image path
        size: Thumbnail size (width, height)

    Returns:
        PIL Image thumbnail
    """
    with Image.open(source_path) as img:
        img.thumbnail(size, Image.Resampling.LANCZOS)
        return img.copy()


def cleanup_input_folder(
    input_folder: Path,
    batch: Batch,
    db: Database,
    delete_processed: bool = False,
) -> int:
    """
    Clean up input folder after processing.

    Args:
        input_folder: Input folder to clean
        batch: Processed batch
        db: Database instance
        delete_processed: Whether to delete processed files

    Returns:
        Number of files cleaned up
    """
    if not delete_processed:
        return 0

    config = get_config()
    archive_folder = (
        config.paths.archive_folder / sanitize_filename(batch.name)
    ).expanduser()

    def safe_delete(src: Optional[Path], output_copy: Optional[Path]) -> bool:
        """Delete an input file only when a copy verifiably exists elsewhere
        (finalized output or archive) — never delete the only copy."""
        if not src or not src.exists():
            return False
        has_output = output_copy is not None and output_copy.exists()
        has_archive = (archive_folder / src.name).exists()
        if has_output or has_archive:
            src.unlink()
            return True
        return False

    photos = db.get_photos_by_batch(batch.id, status=PhotoStatus.FINALIZED)
    cleaned = 0

    for photo in photos:
        if safe_delete(photo.front_path, photo.output_front_path):
            cleaned += 1
        if safe_delete(photo.back_path, photo.output_back_path):
            cleaned += 1

    return cleaned
