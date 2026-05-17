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


AI_DATING_PROMPT_BASE = """Analyze this photograph to estimate when AND where it was taken. Look for clues including:

**For DATE:**
1. **Clothing and fashion**: Styles, cuts, colors typical of specific eras
2. **Hairstyles**: Often strongly indicative of decade
3. **Technology visible**: TVs, phones, cars, cameras, computers
4. **Interior/exterior details**: Furniture styles, architecture, decor
5. **Photo characteristics**: Print style, color quality, aspect ratio, borders
6. **Seasonal clues**: Clothing weight, vegetation, holiday decorations, weather

**For LOCATION:**
1. **Landmarks**: Buildings, signs, monuments
2. **Landscape**: Mountains, beaches, deserts, vegetation type
3. **Architecture style**: Regional building styles
4. **Signs/text**: Language, business names, license plates
5. **Cultural elements**: Clothing styles specific to regions, activities

Based on your analysis, provide:
1. **Estimated year**: Be specific if confident, or give a range
2. **Estimated month**: If seasonal clues allow (1-12), accounting for hemisphere
3. **Season**: spring/summer/fall/winter
4. **Location guess**: Your best guess at where this was taken
5. **Confidence levels**: high/medium/low for both date and location

Respond ONLY with valid JSON in this exact format (no markdown, no extra text):
{
  "estimated_date": {
    "year": 1985,
    "year_range": [1983, 1987],
    "month": 7,
    "season": "summer",
    "confidence": "medium"
  },
  "evidence": [
    "Woman's permed hairstyle typical of mid-1980s",
    "Wood-paneled station wagon visible (common 1978-1988)",
    "Child wearing Members Only style jacket"
  ],
  "reasoning": "The combination of fashion elements and visible vehicle strongly suggest mid-1980s.",
  "location": {
    "guess": "Southern California, USA",
    "confidence": "medium",
    "evidence": [
      "Palm trees visible in background",
      "Spanish-style architecture",
      "License plate appears to be California"
    ]
  }
}

If you cannot determine something, use null for that field. Always try to provide your best guess even with low confidence."""


def build_ai_prompt(batch: Optional[Batch] = None) -> str:
    """
    Build the AI prompt, incorporating batch context if available.

    Args:
        batch: Optional batch with user-provided hints

    Returns:
        Complete prompt string
    """
    prompt = AI_DATING_PROMPT_BASE

    if batch:
        hints = []

        if batch.date_start or batch.date_end:
            if batch.date_start and batch.date_end:
                hints.append(f"User suggests date range: {batch.date_start.year} to {batch.date_end.year}")
            elif batch.date_start:
                hints.append(f"User suggests approximate date: {batch.date_start.year}")

        if batch.location_description:
            hints.append(f"User suggests location: {batch.location_description}")

        if batch.people:
            hints.append(f"People in photos: {', '.join(batch.people)}")

        if batch.event_description:
            hints.append(f"Event/context: {batch.event_description}")

        if hints:
            prompt += "\n\n**CONTEXT FROM USER (use as hints, but verify with visual evidence):**\n"
            for hint in hints:
                prompt += f"- {hint}\n"
            prompt += "\nUse these hints to guide your analysis, but base your estimates primarily on visual evidence. Correct the user's suggestions if the visual evidence clearly contradicts them."

    return prompt


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

        # Parse location
        loc = data.get("location", {})
        location_guess = loc.get("guess") if loc else None
        location_confidence = loc.get("confidence") if loc else None
        location_evidence = loc.get("evidence", []) if loc else []

        return AIDateEstimate(
            year=est.get("year"),
            year_range=year_range,
            month=est.get("month"),
            season=est.get("season"),
            confidence=confidence,
            evidence=data.get("evidence", []),
            reasoning=data.get("reasoning", ""),
            location_guess=location_guess,
            location_confidence=location_confidence,
            location_evidence=location_evidence,
        )

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Parse error: {e}")
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
    batch: Optional[Batch] = None,
) -> Optional[AIDateEstimate]:
    """
    Synchronous version of analyze_photo_with_ai.

    Args:
        photo_path: Path to photo to analyze
        client: Anthropic client instance
        batch: Optional batch for context hints

    Returns:
        AIDateEstimate, or None if analysis failed
    """
    config = get_config()

    # Prepare image
    image_bytes = resize_image_for_api(photo_path, config.ai_dating.max_image_dimension)
    image_base64 = image_to_base64(image_bytes)

    # Build prompt with batch context
    prompt = build_ai_prompt(batch)

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
                            "text": prompt,
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
        estimate = analyze_photo_with_ai_sync(photo.front_path, client, batch)
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
    all_location_evidence = []
    for e in estimates:
        all_evidence.extend(e.evidence)
        all_location_evidence.extend(e.location_evidence)

    # Find best location guess (prefer higher confidence)
    location_estimates = [(e.location_guess, e.location_confidence, e.location_evidence)
                          for e in estimates if e.location_guess]
    best_location = None
    best_location_conf = None
    best_location_evidence = []
    if location_estimates:
        # Sort by confidence
        conf_order = {"high": 3, "medium": 2, "low": 1}
        location_estimates.sort(key=lambda x: conf_order.get(x[1], 0), reverse=True)
        best_location, best_location_conf, best_location_evidence = location_estimates[0]

    combined_estimate = AIDateEstimate(
        year=best_estimate.year,
        year_range=best_estimate.year_range,
        month=best_estimate.month,
        season=best_estimate.season,
        confidence=best_estimate.confidence,
        evidence=list(set(all_evidence))[:5],  # Top 5 unique evidence items
        reasoning=f"Based on analysis of {len(samples)} representative photos. " + best_estimate.reasoning,
        location_guess=best_location,
        location_confidence=best_location_conf,
        location_evidence=list(set(all_location_evidence))[:3],
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


class AIBatchAnalysis:
    """Results of AI analysis on a batch, potentially with segments."""
    def __init__(
        self,
        combined_estimate: Optional[AIDateEstimate],
        segments: list[dict] = None,  # List of {estimate, photo_indices, photos}
        per_photo_estimates: dict = None,  # photo_id -> AIDateEstimate
        has_multiple_segments: bool = False,
    ):
        self.combined_estimate = combined_estimate
        self.segments = segments or []
        self.per_photo_estimates = per_photo_estimates or {}
        self.has_multiple_segments = has_multiple_segments


def detect_segments(
    estimates: list[tuple[int, AIDateEstimate]],  # (photo_index, estimate)
    year_threshold: int = 5,
) -> list[list[int]]:
    """
    Detect distinct segments based on year differences.

    Args:
        estimates: List of (photo_index, estimate) tuples
        year_threshold: Years apart to consider different segment

    Returns:
        List of segment groups, each containing photo indices
    """
    if not estimates:
        return []

    # Sort by photo index
    sorted_estimates = sorted(estimates, key=lambda x: x[0])

    segments = []
    current_segment = [sorted_estimates[0][0]]
    current_year = sorted_estimates[0][1].year

    for idx, estimate in sorted_estimates[1:]:
        if estimate.year and current_year:
            if abs(estimate.year - current_year) > year_threshold:
                # New segment
                segments.append(current_segment)
                current_segment = [idx]
                current_year = estimate.year
            else:
                current_segment.append(idx)
                # Update running average
                current_year = (current_year + estimate.year) // 2
        else:
            current_segment.append(idx)

    segments.append(current_segment)
    return segments


def estimate_batch_with_segments(
    batch: Batch,
    photos: list[PhotoPair],
    db: Database,
    max_samples: int = 6,
) -> Optional[AIBatchAnalysis]:
    """
    Analyze a batch and detect if there are different segments.

    Args:
        batch: Batch to analyze
        photos: Photos in the batch
        db: Database instance
        max_samples: Maximum photos to analyze

    Returns:
        AIBatchAnalysis with combined and per-segment results
    """
    config = get_config()

    if not config.ai_dating.enabled:
        return None

    api_key = config.get_api_key()
    if not api_key:
        return None

    try:
        import anthropic
    except ImportError:
        return None

    # For segment detection, sample more photos spread across the batch
    if len(photos) <= max_samples:
        samples = list(enumerate(photos))
    else:
        # Sample evenly across batch
        step = len(photos) / max_samples
        samples = [(int(i * step), photos[int(i * step)]) for i in range(max_samples)]

    client = anthropic.Anthropic(api_key=api_key)

    # Analyze each sample
    indexed_estimates: list[tuple[int, AIDateEstimate]] = []
    per_photo_estimates = {}

    for idx, photo in samples:
        estimate = analyze_photo_with_ai_sync(photo.front_path, client, batch)
        if estimate:
            indexed_estimates.append((idx, estimate))
            per_photo_estimates[photo.id] = estimate

    if not indexed_estimates:
        return None

    # Detect segments
    segments_indices = detect_segments(indexed_estimates)
    has_multiple = len(segments_indices) > 1 and all(len(s) > 0 for s in segments_indices)

    # Build segment info
    segments = []
    for seg_indices in segments_indices:
        seg_estimates = [est for idx, est in indexed_estimates if idx in seg_indices]
        if seg_estimates:
            # Combine estimates for this segment
            best = max(seg_estimates, key=lambda e: (
                {"high": 3, "medium": 2, "low": 1}.get(e.confidence.value, 0),
                e.year or 0,
            ))

            # Find photo range
            photo_range = (min(seg_indices), max(seg_indices))

            segments.append({
                "estimate": best,
                "photo_indices": seg_indices,
                "photo_range": photo_range,
                "photo_count_approx": int((photo_range[1] - photo_range[0] + 1) * len(photos) / max(idx for idx, _ in samples)),
            })

    # Combined estimate (same as before)
    all_estimates = [est for _, est in indexed_estimates]
    best_estimate = max(all_estimates, key=lambda e: (
        {"high": 3, "medium": 2, "low": 1}.get(e.confidence.value, 0),
        e.year or 0,
    ))

    all_evidence = []
    all_location_evidence = []
    for e in all_estimates:
        all_evidence.extend(e.evidence)
        all_location_evidence.extend(e.location_evidence)

    location_estimates = [(e.location_guess, e.location_confidence, e.location_evidence)
                          for e in all_estimates if e.location_guess]
    best_location = None
    best_location_conf = None
    if location_estimates:
        conf_order = {"high": 3, "medium": 2, "low": 1}
        location_estimates.sort(key=lambda x: conf_order.get(x[1], 0), reverse=True)
        best_location, best_location_conf, _ = location_estimates[0]

    combined = AIDateEstimate(
        year=best_estimate.year,
        year_range=best_estimate.year_range,
        month=best_estimate.month,
        season=best_estimate.season,
        confidence=best_estimate.confidence,
        evidence=list(set(all_evidence))[:5],
        reasoning=f"Based on analysis of {len(samples)} photos. " + best_estimate.reasoning,
        location_guess=best_location,
        location_confidence=best_location_conf,
        location_evidence=list(set(all_location_evidence))[:3],
    )

    return AIBatchAnalysis(
        combined_estimate=combined,
        segments=segments,
        per_photo_estimates=per_photo_estimates,
        has_multiple_segments=has_multiple,
    )


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
