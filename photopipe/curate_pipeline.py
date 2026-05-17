"""
Owner-driven curation pipeline.

Orchestrates: AI dating with multi-image batching, segment detection,
applying results to photos. Uses VLMClient for transport.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Optional

from photopipe.database import Database
from photopipe.models import (
    Batch,
    DateConfidence,
    DateSource,
    PhotoPair,
)
from photopipe.vlm_client import VLMClient, build_image_block


CURATE_PROMPT_PREFIX = """You are an expert at dating old photographs. You will be shown
a numbered set of family photos and asked for per-photo date/location estimates
PLUS a coherence assessment of the set.

For each photo, look at:
- Clothing styles, hairstyles, jewelry
- Vehicles, technology, signage
- Photo print characteristics (border style, color cast, paper)
- Visible season cues (clothing weight, foliage, decorations)
- Architecture and regional cues for location

For the SET, assess: do these photos appear to be from the same event, day,
year, or are there visible breaks in the timeline (different ages of the same
people, different locations, different fashion eras)?

You will be given the user's hints (date range, location, people). Treat them
as priors. Override them only when visual evidence strongly contradicts.

Respond using the provided tool with strict JSON. If you cannot determine a
field, use null."""


RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "per_photo": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "photo_index": {"type": "integer"},
                    "year": {"type": ["integer", "null"]},
                    "year_range": {
                        "type": ["array", "null"],
                        "items": {"type": "integer"},
                        "minItems": 2,
                        "maxItems": 2,
                    },
                    "month": {"type": ["integer", "null"]},
                    "season": {"type": ["string", "null"]},
                    "location_guess": {"type": ["string", "null"]},
                    "confidence": {
                        "type": "string",
                        "enum": ["high", "medium", "low"],
                    },
                    "evidence": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["photo_index", "confidence", "evidence"],
                "additionalProperties": False,
            },
        },
        "coherence": {
            "type": "object",
            "properties": {
                "same_event": {"type": "boolean"},
                "segment_breaks": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "after_photo_index": {"type": "integer"},
                            "reason": {"type": "string"},
                        },
                        "required": ["after_photo_index", "reason"],
                        "additionalProperties": False,
                    },
                },
                "summary": {"type": "string"},
            },
            "required": ["same_event", "segment_breaks", "summary"],
            "additionalProperties": False,
        },
    },
    "required": ["per_photo", "coherence"],
    "additionalProperties": False,
}


@dataclass
class AIRunResult:
    """Aggregated AI dating results across all batched calls."""

    per_photo: dict[str, dict]  # photo_id -> per-photo dict from schema
    coherence: dict             # aggregated coherence dict
    raw_responses: list[dict] = field(default_factory=list)


@dataclass
class ApplyResult:
    """Summary of applying AI results to photos in the database."""

    updated: int
    skipped: int
    errors: list[str] = field(default_factory=list)


def _batch_hint_prefix(batch: Batch) -> str:
    """Build the user-context preamble that goes AFTER the cached prefix."""
    parts: list[str] = []
    if batch.date_start or batch.date_end:
        parts.append(
            f"User suggests date range: {batch.date_start}–{batch.date_end}"
        )
    if batch.location_description:
        parts.append(f"User suggests location: {batch.location_description}")
    if batch.people:
        parts.append(f"People: {', '.join(batch.people)}")
    if batch.event_description:
        parts.append(f"Event: {batch.event_description}")
    if not parts:
        return ""
    return "User hints:\n- " + "\n- ".join(parts)


def run_ai_dating(
    batch: Batch,
    photos: list[PhotoPair],
    *,
    vlm_client: Optional[VLMClient] = None,
    images_per_call: int = 12,
) -> AIRunResult:
    """Run multi-image AI dating across the given photos.

    Photos are chunked into groups of ``images_per_call`` and each chunk is
    sent in a single VLM call with the cached prompt prefix and the strict
    response schema. Per-photo results are keyed by ``photo.id`` and
    coherence reports are aggregated across calls.
    """
    vlm = vlm_client or VLMClient()
    per_photo: dict[str, dict] = {}
    all_coherences: list[dict] = []
    raw: list[dict] = []

    hints = _batch_hint_prefix(batch)
    per_call_template = (
        f"{hints}\n\n"
        "Estimate per-photo date and location for these {n} photos. "
        "Photo indices are 0-based and match the image order."
    )

    for start in range(0, len(photos), images_per_call):
        chunk = photos[start:start + images_per_call]
        images = [build_image_block(p.front_path) for p in chunk]
        result = vlm.analyze(
            cached_prefix=CURATE_PROMPT_PREFIX,
            images=images,
            per_call_prompt=per_call_template.format(n=len(chunk)),
            response_schema=RESPONSE_SCHEMA,
            max_tokens=4096,
        )
        raw.append(result)
        for entry in result.get("per_photo", []):
            idx = entry.get("photo_index")
            if isinstance(idx, int) and 0 <= idx < len(chunk):
                per_photo[chunk[idx].id] = entry
        all_coherences.append(result.get("coherence", {}))

    # Aggregate coherence: union segment_breaks; majority vote on same_event.
    same_event = (
        sum(1 for c in all_coherences if c.get("same_event"))
        > len(all_coherences) / 2
    )
    coherence = {
        "same_event": same_event,
        "segment_breaks": [
            b for c in all_coherences for b in c.get("segment_breaks", [])
        ],
        "summary": " | ".join(
            c.get("summary", "") for c in all_coherences if c.get("summary")
        ),
    }
    return AIRunResult(per_photo=per_photo, coherence=coherence, raw_responses=raw)


def apply_ai_results(
    batch: Batch,
    ai_result: AIRunResult,
    photos: list[PhotoPair],
    *,
    db: Database,
    overwrite_existing: bool = False,
) -> ApplyResult:
    """Persist AI dating results to photo records.

    Skips photos that already have ``extracted_date`` set (unless
    ``overwrite_existing`` is True) and photos for which the AI did not
    return a year.
    """
    updated = 0
    skipped = 0
    errors: list[str] = []
    for photo in photos:
        entry = ai_result.per_photo.get(photo.id)
        if not entry or entry.get("year") is None:
            skipped += 1
            continue
        if photo.extracted_date is not None and not overwrite_existing:
            skipped += 1
            continue
        try:
            year = entry["year"]
            month = entry.get("month") or 6
            photo.extracted_date = date(year, month, 15)
            photo.date_source = DateSource.AI_ESTIMATED
            photo.date_confidence = DateConfidence(entry.get("confidence", "low"))
            photo.ai_analysis = entry
            photo.needs_review = True
            db.update_photo(photo)
            updated += 1
        except Exception as e:  # noqa: BLE001 - surface per-photo errors
            errors.append(f"photo {photo.id}: {e}")
    return ApplyResult(updated=updated, skipped=skipped, errors=errors)
