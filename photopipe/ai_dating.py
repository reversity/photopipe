"""
AI-assisted date estimation using Claude Vision for PhotoPipe.

Uses Anthropic's Claude model to analyze photos and estimate
when they were taken based on visual clues.
"""

import base64
import json
from datetime import date
from pathlib import Path
from typing import Optional

from PIL import Image
import io

from photopipe.config import get_config
from photopipe.models import (
    PhotoPair,
    Batch,
    AIDateEstimate,
    DateSource,
    DateConfidence,
)
from photopipe.database import Database


AI_DATING_PROMPT = """Analyze this photograph to estimate when it was taken. Look for clues including:

1. **Clothing and fashion**: Styles, cuts, colors typical of specific eras
2. **Hairstyles**: Often strongly indicative of decade
3. **Technology visible**: TVs, phones, cars, cameras, computers
4. **Interior/exterior details**: Furniture styles, architecture, decor
5. **Photo characteristics**: Print style, color quality, aspect ratio, borders

Based on your analysis, provide:
1. **Estimated year or range**: Be as specific as confidence allows (e.g., "1985" or "mid-1980s" or "1983-1987")
2. **Confidence level**: high/medium/low
3. **Key evidence**: List the 2-3 most telling clues you observed
4. **Reasoning**: Brief explanation of how you arrived at the estimate

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "estimated_date": {
    "year": 1985,
    "year_range": [1983, 1987],
    "season": "summer",
    "confidence": "medium"
  },
  "evidence": [
    "Woman's permed hairstyle typical of mid-1980s",
    "Wood-paneled station wagon visible (common 1978-1988)",
    "Child wearing Members Only style jacket"
  ],
  "reasoning": "The combination of fashion elements and visible vehicle strongly suggest mid-1980s. The photo paper quality and slightly faded colors are consistent with prints from this era."
}

If you cannot determine a date, use:
{
  "estimated_date": {
    "year": null,
    "year_range": null,
    "season": null,
    "confidence": "low"
  },
  "evidence": [],
  "reasoning": "Unable to determine date from visible elements."
}"""


def resize_image_for_api(image_path: Path, max_dimension: int = 1024) -> bytes:
    """
    Resize image to reasonable dimensions for API to save tokens.

    Args:
        image_path: Path to original image
        max_dimension: Maximum width or height

    Returns:
        JPEG bytes of resized image
    """
    img = Image.open(image_path)

    # Convert to RGB if necessary (handles RGBA, etc.)
    if img.mode != "RGB":
        img = img.convert("RGB")

    # Resize if needed
    width, height = img.size
    if width > max_dimension or height > max_dimension:
        if width > height:
            new_width = max_dimension
            new_height = int(height * (max_dimension / width))
        else:
            new_height = max_dimension
            new_width = int(width * (max_dimension / height))
        img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Convert to JPEG bytes
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return buffer.getvalue()


def image_to_base64(image_bytes: bytes) -> str:
    """Convert image bytes to base64 string."""
    return base64.standard_b64encode(image_bytes).decode("utf-8")


def select_representative_photos(
    photos: list[PhotoPair],
    max_samples: int = 3,
) -> list[PhotoPair]:
    """
    Select representative photos for AI analysis.

    Selects photos from start, middle, and end of batch
    to get a representative sample.

    Args:
        photos: List of photos to select from
        max_samples: Maximum number to select

    Returns:
        Selected representative photos
    """
    if len(photos) <= max_samples:
        return photos

    selected = []
    indices = []

    # Always include first
    indices.append(0)

    # Include middle
    if max_samples >= 2:
        indices.append(len(photos) // 2)

    # Include last
    if max_samples >= 3:
        indices.append(len(photos) - 1)

    # Remove duplicates and sort
    indices = sorted(set(indices))

    return [photos[i] for i in indices[:max_samples]]


def parse_ai_response(response_text: str) -> Optional[AIDateEstimate]:
    """
    Parse Claude's JSON response into AIDateEstimate.

    Args:
        response_text: Raw response text from Claude

    Returns:
        AIDateEstimate object, or None if parsing failed
    """
    try:
        # Try to extract JSON from response
        text = response_text.strip()

        # Handle markdown code blocks
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        data = json.loads(text)
        est = data.get("estimated_date", {})

        # Parse year range
        year_range = None
        if est.get("year_range"):
            yr = est["year_range"]
            if isinstance(yr, list) and len(yr) == 2:
                year_range = (yr[0], yr[1])

        # Parse confidence
        confidence_str = est.get("confidence", "low")
        confidence = DateConfidence.LOW
        if confidence_str == "high":
            confidence = DateConfidence.HIGH
        elif confidence_str == "medium":
            confidence = DateConfidence.MEDIUM

        return AIDateEstimate(
            year=est.get("year"),
            year_range=year_range,
            season=est.get("season"),
            confidence=confidence,
            evidence=data.get("evidence", []),
            reasoning=data.get("reasoning", ""),
        )

    except (json.JSONDecodeError, KeyError, TypeError):
        return None


async def analyze_photo_with_ai(
    photo_path: Path,
    client,  # anthropic.Anthropic client
) -> Optional[AIDateEstimate]:
    """
    Analyze a single photo with Claude Vision.

    Args:
        photo_path: Path to photo to analyze
        client: Anthropic client instance

    Returns:
        AIDateEstimate, or None if analysis failed
    """
    config = get_config()

    # Prepare image
    image_bytes = resize_image_for_api(photo_path, config.ai_dating.max_image_dimension)
    image_base64 = image_to_base64(image_bytes)

    try:
        response = client.messages.create(
            model=config.ai_dating.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": AI_DATING_PROMPT,
                        },
                    ],
                }
            ],
        )

        response_text = response.content[0].text
        return parse_ai_response(response_text)

    except Exception as e:
        print(f"AI analysis error: {e}")
        return None


def analyze_photo_with_ai_sync(
    photo_path: Path,
    client,  # anthropic.Anthropic client
) -> Optional[AIDateEstimate]:
    """
    Synchronous version of analyze_photo_with_ai.

    Args:
        photo_path: Path to photo to analyze
        client: Anthropic client instance

    Returns:
        AIDateEstimate, or None if analysis failed
    """
    config = get_config()

    # Prepare image
    image_bytes = resize_image_for_api(photo_path, config.ai_dating.max_image_dimension)
    image_base64 = image_to_base64(image_bytes)

    try:
        response = client.messages.create(
            model=config.ai_dating.model,
            max_tokens=1024,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": image_base64,
                            },
                        },
                        {
                            "type": "text",
                            "text": AI_DATING_PROMPT,
                        },
                    ],
                }
            ],
        )

        response_text = response.content[0].text
        return parse_ai_response(response_text)

    except Exception as e:
        print(f"AI analysis error: {e}")
        return None


def estimate_batch_date_with_ai(
    batch: Batch,
    photos: list[PhotoPair],
    db: Database,
) -> Optional[AIDateEstimate]:
    """
    Use AI to estimate dates for a batch of photos.

    Analyzes representative samples and applies the result
    to all photos in the batch.

    Args:
        batch: Batch to analyze
        photos: Photos in the batch
        db: Database instance

    Returns:
        AIDateEstimate if successful, None otherwise
    """
    config = get_config()

    if not config.ai_dating.enabled:
        return None

    api_key = config.get_api_key()
    if not api_key:
        db.log_action(
            batch_id=batch.id,
            action="ai_dating_skipped",
            details={"reason": "No API key configured"},
        )
        return None

    # Import anthropic here to avoid import errors if not installed
    try:
        import anthropic
    except ImportError:
        db.log_action(
            batch_id=batch.id,
            action="ai_dating_skipped",
            details={"reason": "anthropic package not installed"},
        )
        return None

    # Select representative photos
    samples = select_representative_photos(photos, config.ai_dating.max_samples_per_batch)

    if not samples:
        return None

    # Create client
    client = anthropic.Anthropic(api_key=api_key)

    # Analyze each sample and collect results
    estimates: list[AIDateEstimate] = []

    for photo in samples:
        estimate = analyze_photo_with_ai_sync(photo.front_path, client)
        if estimate and estimate.year:
            estimates.append(estimate)

            db.log_action(
                photo_id=photo.id,
                batch_id=batch.id,
                action="ai_analyzed",
                details={
                    "year": estimate.year,
                    "confidence": estimate.confidence.value,
                    "evidence": estimate.evidence,
                },
            )

    if not estimates:
        return None

    # Combine estimates (use most common year or average)
    years = [e.year for e in estimates if e.year]
    if not years:
        return None

    # Use most confident estimate, or average if same confidence
    best_estimate = max(estimates, key=lambda e: (
        {"high": 3, "medium": 2, "low": 1}.get(e.confidence.value, 0),
        e.year or 0,
    ))

    # Combine evidence from all estimates
    all_evidence = []
    for e in estimates:
        all_evidence.extend(e.evidence)

    combined_estimate = AIDateEstimate(
        year=best_estimate.year,
        year_range=best_estimate.year_range,
        season=best_estimate.season,
        confidence=best_estimate.confidence,
        evidence=list(set(all_evidence))[:5],  # Top 5 unique evidence items
        reasoning=f"Based on analysis of {len(samples)} representative photos. " + best_estimate.reasoning,
    )

    db.log_action(
        batch_id=batch.id,
        action="ai_batch_estimate",
        details={
            "samples_analyzed": len(samples),
            "year": combined_estimate.year,
            "confidence": combined_estimate.confidence.value,
        },
    )

    return combined_estimate


def apply_ai_date_to_batch(
    batch: Batch,
    estimate: AIDateEstimate,
    photos: list[PhotoPair],
    db: Database,
) -> int:
    """
    Apply AI date estimate to all photos in batch that don't have dates.

    Args:
        batch: Batch being processed
        estimate: AI date estimate to apply
        photos: Photos in batch
        db: Database instance

    Returns:
        Number of photos updated
    """
    ai_date = estimate.get_best_date()
    if not ai_date:
        return 0

    updated = 0

    for photo in photos:
        # Only apply to photos without dates
        if photo.extracted_date is not None:
            continue

        photo.extracted_date = ai_date
        photo.date_source = DateSource.AI_ESTIMATED
        photo.date_confidence = estimate.confidence
        photo.ai_analysis = estimate.to_dict()
        photo.needs_review = True  # AI dates should always be reviewed

        db.update_photo(photo)
        updated += 1

    return updated


def is_ai_dating_available() -> bool:
    """
    Check if AI dating is available and configured.

    Returns:
        True if AI dating can be used
    """
    config = get_config()

    if not config.ai_dating.enabled:
        return False

    if not config.get_api_key():
        return False

    try:
        import anthropic
        return True
    except ImportError:
        return False
