"""
Metadata writing with ExifTool for PhotoPipe.

Handles writing EXIF, IPTC, and XMP metadata to image files.
"""

import json
import subprocess
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from photopipe.config import get_config
from photopipe.models import PhotoPair, Batch, Location
from photopipe.geocoding import parse_location_components


def get_exiftool_path() -> Optional[str]:
    """Find the exiftool executable."""
    # Try common locations
    paths_to_try = [
        "exiftool",  # In PATH
        "/opt/homebrew/bin/exiftool",  # Homebrew on Apple Silicon
        "/usr/local/bin/exiftool",  # Homebrew on Intel Mac
    ]
    for path in paths_to_try:
        try:
            result = subprocess.run(
                [path, "-ver"],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                return path
        except FileNotFoundError:
            continue
    return None


def check_exiftool_installed() -> bool:
    """Check if ExifTool is installed and accessible."""
    return get_exiftool_path() is not None


def read_metadata(image_path: Path) -> dict:
    """
    Read existing metadata from an image.

    Args:
        image_path: Path to image file

    Returns:
        Dictionary of metadata tags
    """
    exiftool = get_exiftool_path()
    if not exiftool:
        return {}

    try:
        result = subprocess.run(
            [exiftool, "-json", str(image_path)],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return data[0] if data else {}
    except Exception:
        pass
    return {}


def format_exif_date(d: date) -> str:
    """
    Format date for EXIF DateTimeOriginal field.

    Args:
        d: Date to format

    Returns:
        EXIF-formatted date string (YYYY:MM:DD HH:MM:SS)
    """
    return f"{d.year}:{d.month:02d}:{d.day:02d} 12:00:00"


def format_iptc_date(d: date) -> str:
    """
    Format date for IPTC DateCreated field.

    Args:
        d: Date to format

    Returns:
        IPTC-formatted date string (YYYY:MM:DD)
    """
    return f"{d.year}:{d.month:02d}:{d.day:02d}"


def format_gps_coordinate(value: float, is_latitude: bool) -> tuple[str, str]:
    """
    Format GPS coordinate for EXIF.

    Args:
        value: Coordinate value
        is_latitude: True for latitude, False for longitude

    Returns:
        Tuple of (coordinate_value, reference)
    """
    if is_latitude:
        ref = "N" if value >= 0 else "S"
    else:
        ref = "E" if value >= 0 else "W"

    abs_value = abs(value)
    return str(abs_value), ref


def build_exiftool_args(
    photo: PhotoPair,
    batch: Batch,
    output_path: Path,
) -> list[str]:
    """
    Build ExifTool argument list for writing metadata.

    Args:
        photo: PhotoPair with metadata to write
        batch: Parent batch with context
        output_path: Target file path

    Returns:
        List of ExifTool arguments
    """
    config = get_config()
    args = []

    # Get effective date (final > extracted)
    photo_date = photo.final_date or photo.extracted_date

    if photo_date:
        exif_date = format_exif_date(photo_date)
        iptc_date = format_iptc_date(photo_date)

        # EXIF dates
        args.extend([
            f"-DateTimeOriginal={exif_date}",
            f"-CreateDate={exif_date}",
            f"-ModifyDate={exif_date}",
        ])

        # IPTC date
        args.append(f"-IPTC:DateCreated={iptc_date}")

        # XMP date
        args.append(f"-XMP:DateCreated={photo_date.isoformat()}")

    # Location/GPS
    location = photo.final_location or batch.location
    if location:
        lat_val, lat_ref = format_gps_coordinate(location.latitude, True)
        lon_val, lon_ref = format_gps_coordinate(location.longitude, False)

        args.extend([
            f"-GPSLatitude={lat_val}",
            f"-GPSLatitudeRef={lat_ref}",
            f"-GPSLongitude={lon_val}",
            f"-GPSLongitudeRef={lon_ref}",
        ])

        # IPTC location components
        components = parse_location_components(location)
        if components.get("city"):
            args.append(f"-IPTC:City={components['city']}")
        if components.get("state"):
            args.append(f"-IPTC:Province-State={components['state']}")
        if components.get("country"):
            args.append(f"-IPTC:Country-PrimaryLocationName={components['country']}")

    # Description/Caption
    description = photo.final_description or batch.event_description
    if description:
        args.extend([
            f"-IPTC:Caption-Abstract={description}",
            f"-XMP:Description={description}",
        ])

    # Keywords (people + batch keywords)
    keywords = list(photo.final_keywords) if photo.final_keywords else []
    keywords.extend(batch.people)

    for keyword in set(keywords):
        args.append(f"-IPTC:Keywords={keyword}")
        args.append(f"-XMP:Subject={keyword}")

    # Copyright
    if photo_date:
        copyright_text = config.metadata.copyright_template.format(year=photo_date.year)
        args.extend([
            f"-IPTC:CopyrightNotice={copyright_text}",
            f"-XMP:Rights={copyright_text}",
        ])

    # Custom PhotoPipe metadata (XMP)
    if photo.date_confidence:
        args.append(f"-XMP-dc:Source=DateConfidence:{photo.date_confidence}")

    if photo.date_source:
        args.append(f"-XMP-photoshop:Instructions=DateSource:{photo.date_source}")

    # Batch name
    args.append(f"-XMP-photoshop:Headline={batch.name}")

    # OCR text (for searchability)
    if photo.ocr_text_back:
        # Truncate if too long
        ocr_text = photo.ocr_text_back[:500] if len(photo.ocr_text_back) > 500 else photo.ocr_text_back
        args.append(f"-XMP:UserComment={ocr_text}")

    # Don't create backup files
    args.append("-overwrite_original")

    # Target file
    args.append(str(output_path))

    return args


def write_metadata(
    photo: PhotoPair,
    batch: Batch,
    output_path: Path,
) -> bool:
    """
    Write all metadata fields using ExifTool.

    Args:
        photo: PhotoPair with metadata to write
        batch: Parent batch with context
        output_path: Target file path

    Returns:
        True if successful, False otherwise
    """
    exiftool = get_exiftool_path()
    if not exiftool:
        raise RuntimeError("ExifTool is not installed. Please install with: brew install exiftool")

    args = build_exiftool_args(photo, batch, output_path)

    try:
        result = subprocess.run(
            [exiftool] + args,
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception as e:
        raise RuntimeError(f"ExifTool execution failed: {e}")


def write_metadata_batch(
    photos: list[tuple[PhotoPair, Batch, Path]],
) -> dict[str, bool]:
    """
    Write metadata to multiple files efficiently.

    Uses ExifTool's batch processing capability.

    Args:
        photos: List of (PhotoPair, Batch, output_path) tuples

    Returns:
        Dictionary mapping photo IDs to success status
    """
    if not check_exiftool_installed():
        raise RuntimeError("ExifTool is not installed")

    results = {}

    for photo, batch, output_path in photos:
        try:
            success = write_metadata(photo, batch, output_path)
            results[photo.id] = success
        except Exception:
            results[photo.id] = False

    return results


def copy_metadata(source_path: Path, dest_path: Path) -> bool:
    """
    Copy all metadata from source to destination file.

    Args:
        source_path: Source image with metadata
        dest_path: Destination image to receive metadata

    Returns:
        True if successful
    """
    exiftool = get_exiftool_path()
    if not exiftool:
        return False

    try:
        result = subprocess.run(
            [
                exiftool,
                "-TagsFromFile", str(source_path),
                "-all:all",
                "-overwrite_original",
                str(dest_path),
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False


def strip_metadata(image_path: Path) -> bool:
    """
    Remove all metadata from an image.

    Args:
        image_path: Path to image

    Returns:
        True if successful
    """
    exiftool = get_exiftool_path()
    if not exiftool:
        return False

    try:
        result = subprocess.run(
            [
                exiftool,
                "-all=",
                "-overwrite_original",
                str(image_path),
            ],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except Exception:
        return False
