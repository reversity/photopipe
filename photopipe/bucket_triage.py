"""AI triage for capture buckets: propose batch context before the owner has
entered any.

The helper who scans an album has none of the owner's context; this module
recovers as much as possible from what the helper *did* capture:

- handwriting dates already OCR'd from photo backs during capture,
- the container photo (album cover / envelope, usually taken with the Mac's
  camera and often carrying the owner's Post-it notes with approximate years),
- a sample of the photos themselves, spread across the whole bucket because
  one album routinely spans multiple events and years.

The result is a *proposal* stored on the bucket (``suggested_context``) that
pre-fills the convert-to-batch form — the owner always confirms or edits.
"""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from photopipe.database import Database
from photopipe.models import Bucket
from photopipe.vlm_client import VLMClient, build_image_block

TRIAGE_PROMPT_PREFIX = """You are helping organize a family photo archive.
You will see, in order: possibly a photo of the physical container the photos
came from (an album cover or envelope), then a sample of scanned photos taken
from ACROSS the whole album (not consecutive pages).

Read everything the container shows — titles, dates, and especially any
handwritten notes or Post-it stickers: those were written by the archive's
owner and are the most trustworthy context available.

Then look at the sampled photos for era clues (clothing, hairstyles, cars,
photo finish/borders, color cast), location clues (signage, landmarks,
landscapes), and season clues.

IMPORTANT: one album very often contains MULTIPLE distinct events, sometimes
years apart. List every distinct event you can discern as a separate entry —
do not force everything into a single event."""

TRIAGE_SCHEMA = {
    "type": "object",
    "properties": {
        "suggested_batch_name": {"type": "string"},
        "container_text": {
            "type": "string",
            "description": "All text read off the container photo, verbatim, "
                           "including Post-it notes. Empty string if none.",
        },
        "date_range": {
            "type": "object",
            "properties": {
                "start": {"type": ["string", "null"], "description": "ISO date or null"},
                "end": {"type": ["string", "null"], "description": "ISO date or null"},
            },
            "required": ["start", "end"],
        },
        "era_guess": {"type": "string", "description": "e.g. 'late 1980s'"},
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "approx_date": {"type": ["string", "null"]},
                    "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                },
                "required": ["description", "confidence"],
            },
        },
        "location_guesses": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
    },
    "required": [
        "suggested_batch_name", "container_text", "date_range",
        "era_guess", "events", "location_guesses", "reasoning", "confidence",
    ],
    "additionalProperties": False,
}


def ocr_date_rollup(photos) -> dict:
    """Aggregate the handwriting dates already extracted during capture.

    These come off the photo backs, so they are stronger evidence than any
    visual estimate.
    """
    dates: list[date] = [p.extracted_date for p in photos if p.extracted_date]
    if not dates:
        return {"earliest": None, "latest": None, "count": 0}
    return {
        "earliest": min(dates).isoformat(),
        "latest": max(dates).isoformat(),
        "count": len(dates),
    }


def _sample_across(photos, sample_size: int) -> list:
    """Evenly-spaced sample over the whole bucket, in scan order.

    Sampling only the first pages would miss later events in a
    multi-event album.
    """
    if len(photos) <= sample_size:
        return list(photos)
    step = len(photos) / sample_size
    return [photos[int(i * step)] for i in range(sample_size)]


def suggest_bucket_context(
    bucket: Bucket,
    db: Database,
    *,
    vlm_client: Optional[VLMClient] = None,
    sample_size: int = 12,
) -> dict:
    """Build a context proposal for ``bucket`` and persist it on the bucket.

    Returns the proposal dict (also stored as ``bucket.suggested_context``).
    Raises if the bucket has no photos or the VLM is unavailable — callers
    gate on :func:`photopipe.vlm_client.is_vlm_available`.
    """
    photos = sorted(db.get_photos_by_bucket(bucket.id), key=lambda p: p.sequence_num)
    if not photos:
        raise ValueError(f"Bucket {bucket.id} has no photos to triage")

    rollup = ocr_date_rollup(photos)

    images = []
    has_container = bool(
        bucket.context_image_path and Path(bucket.context_image_path).exists()
    )
    if has_container:
        images.append(build_image_block(Path(bucket.context_image_path)))
    sampled = [
        p for p in _sample_across(photos, sample_size)
        if p.front_path and Path(p.front_path).exists()
    ]
    images.extend(build_image_block(Path(p.front_path)) for p in sampled)
    if not images:
        raise ValueError(f"Bucket {bucket.id} has no readable images to triage")

    ocr_hint = ""
    if rollup["count"]:
        ocr_hint = (
            f"\nHandwriting OCR from the photo backs already found {rollup['count']} "
            f"date(s) between {rollup['earliest']} and {rollup['latest']} — treat "
            "these as strong evidence for the date range."
        )
    container_hint = (
        "The FIRST image is the container (album cover / envelope)."
        if has_container
        else "No container photo was taken; work from the photos alone."
    )

    vlm = vlm_client or VLMClient()
    proposal = vlm.analyze(
        cached_prefix=TRIAGE_PROMPT_PREFIX,
        images=images,
        per_call_prompt=(
            f"{container_hint} The bucket's helper-entered label is: "
            f"\"{bucket.label}\". {len(sampled)} sampled photos follow, in scan "
            f"order, drawn evenly from {len(photos)} total.{ocr_hint}"
        ),
        response_schema=TRIAGE_SCHEMA,
        max_tokens=2048,
    )

    # OCR'd dates outrank visual estimates: when the backs gave us a range,
    # it wins over (or tightens) the model's guess.
    if rollup["count"]:
        proposal["date_range"] = {
            "start": rollup["earliest"],
            "end": rollup["latest"],
        }

    proposal["ocr_date_rollup"] = rollup
    proposal["sampled_photos"] = len(sampled)
    proposal["total_photos"] = len(photos)
    proposal["generated_at"] = datetime.now().isoformat()

    bucket.suggested_context = proposal
    db.update_bucket(bucket)
    return proposal
