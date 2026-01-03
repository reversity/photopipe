"""
Front/back scan pairing logic for PhotoPipe.

Handles matching front images with their corresponding back scans
from the FastFoto FF-680W scanner output.
"""

import re
from pathlib import Path
from typing import Optional

from photopipe.config import get_config
from photopipe.database import Database
from photopipe.models import PhotoPair, Batch


# Common image extensions
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}


def extract_sequence_number(filename: str, pattern: str) -> Optional[int]:
    """
    Extract sequence number from filename based on pattern.

    Args:
        filename: The filename to parse (without extension)
        pattern: Pattern like "{name}_{num}" where {num} is the sequence placeholder

    Returns:
        The extracted sequence number, or None if no match
    """
    # Convert pattern to regex
    # Replace {name} with a non-capturing group for any characters
    regex_pattern = pattern.replace("{name}", r"(?:.+?)")
    # Replace {num} with a capturing group for digits
    regex_pattern = regex_pattern.replace("{num}", r"(\d+)")
    regex_pattern = regex_pattern.replace(".", r"\.")

    match = re.match(regex_pattern, filename, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


def is_back_image(filename: str, back_pattern: str) -> bool:
    """
    Check if a filename matches the back image pattern.

    Args:
        filename: The filename to check (without extension)
        back_pattern: Pattern like "{name}_{num}_b" (FastFoto uses _b suffix)

    Returns:
        True if this is a back image
    """
    # Convert pattern to regex
    regex_pattern = back_pattern.replace("{name}", r".+?")
    regex_pattern = regex_pattern.replace("{num}", r"\d+")
    regex_pattern = regex_pattern.replace(".", r"\.")
    return bool(re.match(regex_pattern, filename, re.IGNORECASE))


def find_back_for_front(
    front_path: Path,
    input_folder: Path,
    front_pattern: str,
    back_pattern: str,
) -> Optional[Path]:
    """
    Find the corresponding back image for a front image.

    Args:
        front_path: Path to the front image
        input_folder: Folder containing scans
        front_pattern: Pattern for front images (e.g., "{name}_{num}")
        back_pattern: Pattern for back images (e.g., "{name}_{num}_b")

    Returns:
        Path to back image if found, None otherwise
    """
    front_stem = front_path.stem
    front_ext = front_path.suffix

    # For FastFoto, the back filename is simply front + "_b"
    # Try this simple approach first
    back_stem = f"{front_stem}_b"

    # Try to find back with same extension first
    back_path = input_folder / f"{back_stem}{front_ext}"
    if back_path.exists():
        return back_path

    # Try other extensions
    for ext in IMAGE_EXTENSIONS:
        back_path = input_folder / f"{back_stem}{ext}"
        if back_path.exists():
            return back_path

        # Try uppercase extension
        back_path = input_folder / f"{back_stem}{ext.upper()}"
        if back_path.exists():
            return back_path

    return None


def scan_input_folder(
    input_folder: Path,
    front_pattern: Optional[str] = None,
    back_pattern: Optional[str] = None,
) -> dict[int, dict]:
    """
    Scan input folder and identify front/back image pairs.

    Args:
        input_folder: Folder containing scanner output
        front_pattern: Pattern for front images (from config if not provided)
        back_pattern: Pattern for back images (from config if not provided)

    Returns:
        Dictionary mapping sequence numbers to {"front": Path, "back": Optional[Path]}
    """
    config = get_config()
    front_pattern = front_pattern or config.scanner.front_pattern
    back_pattern = back_pattern or config.scanner.back_pattern

    # Strip extension from patterns for matching
    front_pattern_stem = front_pattern.replace(".jpg", "").replace(".JPG", "")
    back_pattern_stem = back_pattern.replace(".jpg", "").replace(".JPG", "")

    pairs: dict[int, dict] = {}

    # Scan all image files
    for file_path in input_folder.iterdir():
        if not file_path.is_file():
            continue

        if file_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        filename_stem = file_path.stem

        # Check if this is a back image (ends with _b)
        if is_back_image(filename_stem, back_pattern_stem):
            # Remove _b suffix to get the front pattern for sequence extraction
            front_equivalent = re.sub(r'_b$', '', filename_stem, flags=re.IGNORECASE)
            seq_num = extract_sequence_number(front_equivalent, front_pattern_stem)
            if seq_num is not None:
                if seq_num not in pairs:
                    pairs[seq_num] = {"front": None, "back": None}
                pairs[seq_num]["back"] = file_path
        else:
            # This is a front image
            seq_num = extract_sequence_number(filename_stem, front_pattern_stem)
            if seq_num is not None:
                if seq_num not in pairs:
                    pairs[seq_num] = {"front": None, "back": None}
                pairs[seq_num]["front"] = file_path

    return pairs


def pair_scans(
    input_folder: Path,
    batch: Batch,
    db: Database,
    skip_existing: bool = True,
) -> list[PhotoPair]:
    """
    Scan input folder and pair front/back images.

    Args:
        input_folder: Folder containing scanner output
        batch: The batch these photos belong to
        db: Database instance
        skip_existing: Skip files that are already in the database

    Returns:
        List of newly created PhotoPair objects
    """
    config = get_config()

    # Scan for all image pairs
    pairs_dict = scan_input_folder(
        input_folder,
        config.scanner.front_pattern,
        config.scanner.back_pattern,
    )

    photo_pairs: list[PhotoPair] = []
    orphaned_backs: list[Path] = []

    # Process pairs in sequence order
    for seq_num in sorted(pairs_dict.keys()):
        pair_data = pairs_dict[seq_num]
        front_path = pair_data.get("front")
        back_path = pair_data.get("back")

        # Handle orphaned back (no front)
        if front_path is None and back_path is not None:
            orphaned_backs.append(back_path)
            continue

        # Skip if no front image
        if front_path is None:
            continue

        # Check if already processed
        if skip_existing and db.check_photo_exists(front_path):
            continue

        # Create PhotoPair
        photo_pair = PhotoPair(
            batch_id=batch.id,
            sequence_num=db.get_next_sequence_num(batch.id),
            front_path=front_path,
            back_path=back_path,
        )

        # Save to database
        db.create_photo(photo_pair)
        photo_pairs.append(photo_pair)

        # Log the action
        db.log_action(
            photo_id=photo_pair.id,
            batch_id=batch.id,
            action="ingested",
            details={
                "front_path": str(front_path),
                "back_path": str(back_path) if back_path else None,
                "sequence_num": photo_pair.sequence_num,
            },
        )

    # Log orphaned backs (shouldn't happen normally)
    if orphaned_backs:
        db.log_action(
            batch_id=batch.id,
            action="orphaned_backs_found",
            details={"paths": [str(p) for p in orphaned_backs]},
        )

    return photo_pairs


def get_pairing_summary(input_folder: Path) -> dict:
    """
    Get a summary of what would be paired from an input folder.

    Useful for preview before actual ingestion.

    Args:
        input_folder: Folder to analyze

    Returns:
        Summary dictionary with counts and details
    """
    pairs_dict = scan_input_folder(input_folder)

    fronts_only = []
    backs_only = []
    complete_pairs = []

    for seq_num, pair_data in sorted(pairs_dict.items()):
        front = pair_data.get("front")
        back = pair_data.get("back")

        if front and back:
            complete_pairs.append({
                "sequence": seq_num,
                "front": str(front),
                "back": str(back),
            })
        elif front and not back:
            fronts_only.append({
                "sequence": seq_num,
                "front": str(front),
            })
        elif back and not front:
            backs_only.append({
                "sequence": seq_num,
                "back": str(back),
            })

    return {
        "total_images": len(pairs_dict),
        "complete_pairs": len(complete_pairs),
        "fronts_without_backs": len(fronts_only),
        "orphaned_backs": len(backs_only),
        "pairs": complete_pairs,
        "fronts_only": fronts_only,
        "backs_only": backs_only,
    }
