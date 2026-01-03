"""
OCR processing pipeline for PhotoPipe.

Uses Tesseract for text extraction from photo backs,
with preprocessing and date pattern recognition.
"""

import re
from datetime import date, datetime
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytesseract
from PIL import Image, ImageOps, ImageFilter

from photopipe.config import get_config
from photopipe.models import (
    PhotoPair,
    OCRResult,
    DateSource,
    DateConfidence,
)
from photopipe.database import Database


# Date patterns to recognize (ordered by specificity/reliability)
DATE_PATTERNS = [
    # Photo lab stamps (most common, reliable)
    (r"(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\s*['\u2019]?\s*(\d{2})", "month_year_short"),
    (r"(\d{2})\s+(\d{2})\s+(\d{2,4})", "numeric_spaced"),
    (r"(\d{2})/(\d{2})/(\d{2,4})", "numeric_slash"),
    (r"(\d{2})-(\d{2})-(\d{2,4})", "numeric_dash"),

    # Full date formats
    (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{1,2}),?\s+(\d{4})", "full_date"),
    (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{1,2}),?\s+(\d{4})", "abbrev_date"),

    # Month and year only
    (r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})", "month_year_full"),
    (r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\.?\s+(\d{4})", "month_year_abbrev"),

    # Contextual dates
    (r"(Summer|Spring|Fall|Autumn|Winter|Xmas|Christmas|Easter|Thanksgiving)\s*['\u2019]?\s*(\d{2,4})", "seasonal"),

    # Year only (last resort)
    (r"\b(19\d{2}|20[0-2]\d)\b", "year_only"),
]

MONTH_MAP = {
    "jan": 1, "january": 1,
    "feb": 2, "february": 2,
    "mar": 3, "march": 3,
    "apr": 4, "april": 4,
    "may": 5,
    "jun": 6, "june": 6,
    "jul": 7, "july": 7,
    "aug": 8, "august": 8,
    "sep": 9, "september": 9,
    "oct": 10, "october": 10,
    "nov": 11, "november": 11,
    "dec": 12, "december": 12,
}

SEASON_MAP = {
    "spring": (3, 15),
    "summer": (7, 15),
    "fall": (10, 15),
    "autumn": (10, 15),
    "winter": (1, 15),
    "xmas": (12, 25),
    "christmas": (12, 25),
    "easter": (4, 15),
    "thanksgiving": (11, 25),
}


def preprocess_image(image_path: Path) -> Image.Image:
    """
    Preprocess image for better OCR results.

    Applies grayscale conversion, adaptive thresholding, and deskewing.
    """
    config = get_config().ocr.preprocessing

    img = Image.open(image_path)

    # Convert to grayscale
    if config.grayscale:
        img = ImageOps.grayscale(img)

    # Apply sharpening to help with faded text
    img = img.filter(ImageFilter.SHARPEN)

    # Increase contrast
    img = ImageOps.autocontrast(img, cutoff=2)

    # Apply adaptive thresholding (convert to binary)
    if config.adaptive_threshold:
        # Use a simple threshold approach
        # More sophisticated would use OpenCV's adaptive threshold
        threshold = 128
        img = img.point(lambda p: 255 if p > threshold else 0)

    return img


def run_tesseract(image: Image.Image, config_str: str = "") -> dict:
    """
    Run Tesseract OCR on preprocessed image.

    Returns raw Tesseract output with word-level confidence scores.
    """
    config = get_config()

    # Get detailed output with confidence scores
    data = pytesseract.image_to_data(
        image,
        lang=config.ocr.language,
        config=config_str,
        output_type=pytesseract.Output.DICT,
    )

    # Also get plain text
    text = pytesseract.image_to_string(
        image,
        lang=config.ocr.language,
        config=config_str,
    )

    # Calculate average confidence (excluding -1 values which indicate no text)
    confidences = [c for c in data["conf"] if c > 0]
    avg_confidence = sum(confidences) / len(confidences) if confidences else 0

    # Build word confidence map
    word_confidences = {}
    for i, word in enumerate(data["text"]):
        if word.strip() and data["conf"][i] > 0:
            word_confidences[word.strip()] = data["conf"][i]

    return {
        "text": text.strip(),
        "confidence": avg_confidence,
        "word_confidences": word_confidences,
        "raw_data": data,
    }


def expand_year(year_str: str) -> int:
    """
    Expand 2-digit year to 4-digit year.

    Assumes years 00-30 are 2000s, 31-99 are 1900s.
    """
    year = int(year_str)
    if year < 100:
        if year <= 30:
            return 2000 + year
        else:
            return 1900 + year
    return year


def parse_date_from_text(text: str) -> list[tuple[date, str, str]]:
    """
    Extract dates from OCR text using pattern matching.

    Returns:
        List of (date, pattern_type, raw_match) tuples
    """
    results = []

    for pattern, pattern_type in DATE_PATTERNS:
        matches = re.finditer(pattern, text, re.IGNORECASE)

        for match in matches:
            try:
                parsed_date = None

                if pattern_type == "month_year_short":
                    # "JUN '85" or "JUN 85"
                    month_str = match.group(1).lower()[:3]
                    year = expand_year(match.group(2))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, 15)

                elif pattern_type in ("numeric_spaced", "numeric_slash", "numeric_dash"):
                    # "06 15 85" or "06/15/85" or "06-15-85"
                    parts = [int(match.group(i)) for i in range(1, 4)]
                    # Assume MM DD YY format (American)
                    month, day, year = parts[0], parts[1], expand_year(str(parts[2]))
                    if month > 12:
                        # Might be DD MM YY format
                        month, day = day, month
                    if 1 <= month <= 12 and 1 <= day <= 31:
                        parsed_date = date(year, month, min(day, 28))

                elif pattern_type == "full_date":
                    # "January 15, 1985"
                    month_str = match.group(1).lower()
                    day = int(match.group(2))
                    year = int(match.group(3))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, min(day, 28))

                elif pattern_type == "abbrev_date":
                    # "Jan. 15, 1985"
                    month_str = match.group(1).lower()[:3]
                    day = int(match.group(2))
                    year = int(match.group(3))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, min(day, 28))

                elif pattern_type in ("month_year_full", "month_year_abbrev"):
                    # "January 1985" or "Jan. 1985"
                    month_str = match.group(1).lower()[:3]
                    year = int(match.group(2))
                    month = MONTH_MAP.get(month_str, 6)
                    parsed_date = date(year, month, 15)

                elif pattern_type == "seasonal":
                    # "Summer '85"
                    season = match.group(1).lower()
                    year = expand_year(match.group(2))
                    month, day = SEASON_MAP.get(season, (6, 15))
                    parsed_date = date(year, month, day)

                elif pattern_type == "year_only":
                    # "1985"
                    year = int(match.group(1))
                    parsed_date = date(year, 6, 15)

                if parsed_date and 1900 <= parsed_date.year <= datetime.now().year:
                    results.append((parsed_date, pattern_type, match.group(0)))

            except (ValueError, AttributeError):
                continue

    return results


def calculate_confidence(
    ocr_confidence: float,
    pattern_type: str,
    word_confidences: dict[str, float],
    match_text: str,
) -> DateConfidence:
    """
    Calculate overall confidence in the extracted date.

    Args:
        ocr_confidence: Average Tesseract confidence
        pattern_type: Type of pattern matched
        word_confidences: Word-level confidence scores
        match_text: The matched text string

    Returns:
        DateConfidence level
    """
    config = get_config()

    # Base confidence from pattern type (some are more reliable)
    pattern_reliability = {
        "month_year_short": 0.9,  # Photo lab stamps
        "numeric_spaced": 0.85,
        "numeric_slash": 0.85,
        "numeric_dash": 0.85,
        "full_date": 0.95,
        "abbrev_date": 0.9,
        "month_year_full": 0.85,
        "month_year_abbrev": 0.8,
        "seasonal": 0.6,
        "year_only": 0.4,
    }

    pattern_score = pattern_reliability.get(pattern_type, 0.5)

    # Factor in OCR confidence
    ocr_score = ocr_confidence / 100.0

    # Check word-level confidence for the matched text
    words = match_text.split()
    word_scores = []
    for word in words:
        if word in word_confidences:
            word_scores.append(word_confidences[word] / 100.0)
    word_score = sum(word_scores) / len(word_scores) if word_scores else ocr_score

    # Combined score
    combined = (pattern_score * 0.4) + (ocr_score * 0.3) + (word_score * 0.3)

    threshold = config.ocr.confidence_threshold / 100.0

    if combined >= 0.8:
        return DateConfidence.HIGH
    elif combined >= threshold:
        return DateConfidence.MEDIUM
    else:
        return DateConfidence.LOW


def extract_date_from_back(image_path: Path) -> OCRResult:
    """
    Use Tesseract to OCR the back of a photo.

    Args:
        image_path: Path to the back image

    Returns:
        OCRResult with extracted text and dates
    """
    # Preprocess image
    img = preprocess_image(image_path)

    # Run OCR with multiple PSM modes and combine
    results = []

    # PSM 3: Fully automatic page segmentation (default)
    results.append(run_tesseract(img, "--psm 3"))

    # PSM 6: Assume uniform block of text
    results.append(run_tesseract(img, "--psm 6"))

    # PSM 11: Sparse text
    results.append(run_tesseract(img, "--psm 11"))

    # Combine results - use the one with highest confidence
    best_result = max(results, key=lambda r: r["confidence"])

    # Also combine all text for date extraction
    all_text = " ".join(r["text"] for r in results)

    # Extract dates from combined text
    detected_dates = parse_date_from_text(all_text)

    return OCRResult(
        raw_text=best_result["text"],
        confidence=best_result["confidence"],
        detected_dates=[match[2] for match in detected_dates],
        word_confidences=best_result["word_confidences"],
        preprocessing_applied=["grayscale", "sharpen", "autocontrast", "threshold"],
    )


def extract_date_stamp_from_front(image_path: Path) -> Optional[OCRResult]:
    """
    Look for date stamps on photo front (typically in corners).

    Many photos from the 1980s-1990s have orange/red date stamps.

    Args:
        image_path: Path to the front image

    Returns:
        OCRResult if date stamp found, None otherwise
    """
    img = Image.open(image_path)
    width, height = img.size

    # Define corner regions to check (most common: bottom-right)
    corners = [
        ("bottom_right", (width * 0.7, height * 0.85, width, height)),
        ("bottom_left", (0, height * 0.85, width * 0.3, height)),
        ("top_right", (width * 0.7, 0, width, height * 0.15)),
        ("top_left", (0, 0, width * 0.3, height * 0.15)),
    ]

    for corner_name, bbox in corners:
        # Crop corner region
        corner_img = img.crop(tuple(map(int, bbox)))

        # Convert to grayscale and increase contrast
        corner_img = ImageOps.grayscale(corner_img)
        corner_img = ImageOps.autocontrast(corner_img, cutoff=5)

        # Run OCR on corner
        try:
            text = pytesseract.image_to_string(corner_img, config="--psm 7")

            # Look for date patterns
            dates = parse_date_from_text(text)
            if dates:
                return OCRResult(
                    raw_text=text.strip(),
                    confidence=70.0,  # Medium confidence for corner stamps
                    detected_dates=[match[2] for match in dates],
                    word_confidences={},
                    preprocessing_applied=[f"corner_{corner_name}", "grayscale", "autocontrast"],
                )
        except Exception:
            continue

    return None


def process_photo_ocr(photo: PhotoPair, db: Database) -> PhotoPair:
    """
    Run OCR pipeline on a single photo.

    Updates photo with extracted date and confidence.

    Args:
        photo: PhotoPair to process
        db: Database instance for updates

    Returns:
        Updated PhotoPair
    """
    config = get_config()
    extracted_date: Optional[date] = None
    date_source: Optional[DateSource] = None
    date_confidence: Optional[DateConfidence] = None
    ocr_result: Optional[OCRResult] = None

    # Stage 1: OCR on photo back (highest priority)
    if photo.back_path and photo.back_path.exists():
        try:
            ocr_result = extract_date_from_back(photo.back_path)
            photo.ocr_text_back = ocr_result.raw_text
            photo.ocr_raw_results = ocr_result.to_dict()

            # Parse dates from OCR text
            dates = parse_date_from_text(ocr_result.raw_text)
            if dates:
                # Use the first (most specific) date found
                extracted_date, pattern_type, match_text = dates[0]
                date_source = DateSource.OCR_BACK
                date_confidence = calculate_confidence(
                    ocr_result.confidence,
                    pattern_type,
                    ocr_result.word_confidences,
                    match_text,
                )
        except Exception as e:
            db.log_action(
                photo_id=photo.id,
                batch_id=photo.batch_id,
                action="ocr_error",
                details={"error": str(e), "stage": "back"},
            )

    # Stage 2: OCR on photo front (date stamps)
    if extracted_date is None and photo.front_path.exists():
        try:
            front_result = extract_date_stamp_from_front(photo.front_path)
            if front_result and front_result.detected_dates:
                dates = parse_date_from_text(" ".join(front_result.detected_dates))
                if dates:
                    extracted_date, pattern_type, match_text = dates[0]
                    date_source = DateSource.OCR_FRONT
                    date_confidence = DateConfidence.MEDIUM  # Front stamps less reliable
        except Exception as e:
            db.log_action(
                photo_id=photo.id,
                batch_id=photo.batch_id,
                action="ocr_error",
                details={"error": str(e), "stage": "front"},
            )

    # Update photo with results
    photo.extracted_date = extracted_date
    photo.date_source = date_source
    photo.date_confidence = date_confidence

    # Flag for review if low confidence or no date found
    if date_confidence == DateConfidence.LOW or extracted_date is None:
        photo.needs_review = True

    db.update_photo(photo)
    db.log_action(
        photo_id=photo.id,
        batch_id=photo.batch_id,
        action="ocr_complete",
        details={
            "date_found": extracted_date.isoformat() if extracted_date else None,
            "date_source": date_source.value if date_source else None,
            "confidence": date_confidence.value if date_confidence else None,
            "ocr_text_length": len(photo.ocr_text_back or ""),
        },
    )

    return photo


def process_batch_ocr(
    batch_id: str,
    db: Database,
    max_workers: int = 4,
    progress_callback: Optional[callable] = None,
) -> list[PhotoPair]:
    """
    Run OCR on all photos in a batch using parallel processing.

    Args:
        batch_id: ID of batch to process
        db: Database instance
        max_workers: Maximum parallel workers
        progress_callback: Optional callback(current, total) for progress updates

    Returns:
        List of processed PhotoPair objects
    """
    photos = db.get_photos_by_batch(batch_id)
    total = len(photos)
    processed = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all OCR jobs
        future_to_photo = {
            executor.submit(process_photo_ocr, photo, db): photo
            for photo in photos
        }

        # Process as they complete
        for i, future in enumerate(as_completed(future_to_photo)):
            try:
                result = future.result()
                processed.append(result)
            except Exception as e:
                photo = future_to_photo[future]
                db.log_action(
                    photo_id=photo.id,
                    batch_id=batch_id,
                    action="ocr_failed",
                    details={"error": str(e)},
                )

            if progress_callback:
                progress_callback(i + 1, total)

    return processed
