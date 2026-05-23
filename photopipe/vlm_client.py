"""
Thin transport-layer wrapper around Anthropic vision calls.

Owns: prompt caching, structured output, Batch API submission/polling.
Does NOT own: prompts, schemas, business logic — those live in the
caller (dating_pipeline.py, handwriting_ocr.py).
"""
from __future__ import annotations

import base64
import io
import os
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from photopipe.config import get_config


def is_vlm_available() -> bool:
    """True iff a Claude API key is configured and the SDK is importable."""
    import os
    cfg = get_config()
    if not os.environ.get(cfg.vlm.api_key_env_var):
        return False
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False


def resize_image_to_jpeg_bytes(image_path: Path, max_dim: int = 1568, quality: int = 85) -> bytes:
    """Resize image to fit max_dim and return JPEG-encoded bytes."""
    from PIL import Image
    import io
    with Image.open(image_path) as img:
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > max_dim:
            if w >= h:
                img = img.resize((max_dim, int(h * max_dim / w)), Image.Resampling.LANCZOS)
            else:
                img = img.resize((int(w * max_dim / h), max_dim), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()


def build_image_block(image_path: Path, max_dim: int = 1568) -> dict:
    """Encode a single image as a base64 JPEG content block.

    Resizes so the largest dimension is at most `max_dim` (Anthropic's
    1568px vision token grid). Always converts to RGB JPEG for consistency.
    """
    img_bytes = resize_image_to_jpeg_bytes(image_path, max_dim=max_dim)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(img_bytes).decode("ascii"),
        },
    }


class VLMClient:
    """Single entry point for Claude vision calls.

    Responsibilities:
      - Build messages with a cache-controlled prompt prefix.
      - Issue synchronous calls with optional structured-output (tool_use) mode.
      - Submit and poll Batch API jobs.

    The Anthropic SDK client is constructed lazily on first use so tests
    can inject a mock via `client._anthropic_client = <mock>`.

    Exceptions from the Anthropic SDK (rate limits, auth errors, network
    failures) propagate to the caller; this class does not retry or swallow.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):
        cfg = get_config()
        self.api_key = api_key or os.environ.get(cfg.vlm.api_key_env_var)
        self.model = model or cfg.vlm.model
        self.cache_ttl = cfg.vlm.cache_ttl
        self._anthropic_client: Any = None

    @property
    def client(self) -> Any:
        """Lazily instantiate the Anthropic SDK client."""
        if self._anthropic_client is None:
            import anthropic  # local import: SDK optional at import time
            headers: dict[str, str] = {}
            if self.cache_ttl == "1h":
                headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
            self._anthropic_client = anthropic.Anthropic(
                api_key=self.api_key,
                default_headers=headers,
            )
        return self._anthropic_client

    def _cache_control(self) -> dict:
        """Build the cache_control block honoring `self.cache_ttl`.

        Default ephemeral cache TTL is 5 minutes; only emit an explicit
        `ttl` key when a non-default value (e.g. "1h") is configured.
        """
        cc: dict = {"type": "ephemeral"}
        if self.cache_ttl == "1h":
            cc["ttl"] = "1h"
        return cc

    def _build_message(
        self,
        *,
        cached_prefix: str,
        images: list[dict],
        per_call_prompt: str,
    ) -> dict:
        """Compose a single user message with prefix + images + tail prompt.

        The prefix block is marked with `cache_control: ephemeral` so the
        Anthropic API can serve it from the prompt cache on repeat calls.
        """
        content: list[dict] = []
        if cached_prefix:
            content.append({
                "type": "text",
                "text": cached_prefix,
                "cache_control": self._cache_control(),
            })
        content.extend(images)
        if per_call_prompt:
            content.append({"type": "text", "text": per_call_prompt})
        return {"role": "user", "content": content}

    def analyze(
        self,
        *,
        cached_prefix: str,
        images: list[dict],
        per_call_prompt: str,
        response_schema: Optional[dict] = None,
        max_tokens: int = 2048,
    ) -> dict:
        """Synchronous vision call.

        When `response_schema` is provided, the call uses tool-use strict
        mode (forced `tool_choice`) so the model is guaranteed to emit a
        structured JSON object matching the schema. The parsed dict is
        returned directly.

        When no schema is provided, returns `{"text": <raw text>}`.

        Exceptions from the Anthropic SDK (rate limits, auth errors, network
        failures) propagate to the caller; this method does not retry or swallow.
        """
        message = self._build_message(
            cached_prefix=cached_prefix,
            images=images,
            per_call_prompt=per_call_prompt,
        )
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [message],
        }
        if response_schema is not None:
            kwargs["tools"] = [{
                "name": "respond",
                "description": "Respond with the requested structured data.",
                "input_schema": response_schema,
            }]
            kwargs["tool_choice"] = {"type": "tool", "name": "respond"}

        resp = self.client.messages.create(**kwargs)

        if response_schema is not None:
            for block in resp.content:
                if getattr(block, "type", None) == "tool_use":
                    return block.input
            # No tool_use block found — schema enforcement failed.
            raise RuntimeError(
                f"Expected tool_use block from structured-output call, got: "
                f"{[getattr(b, 'type', '?') for b in resp.content]}"
            )
        # When no schema, find first text block.
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                return {"text": block.text}
        return {"text": ""}

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a Batch API job.

        `requests` is a list of `{custom_id, params}` dicts following the
        Anthropic Batch API request shape. Returns the batch job id.

        Exceptions from the Anthropic SDK (rate limits, auth errors, network
        failures) propagate to the caller; this method does not retry or swallow.
        """
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def poll_batch(self, job_id: str) -> dict:
        """Poll a Batch API job.

        Returns `{"status": <processing_status>, "results": <list|None>}`.
        Results are materialized (the SDK returns a generator) only when
        the batch has reached the terminal `ended` state.

        Exceptions from the Anthropic SDK (rate limits, auth errors, network
        failures) propagate to the caller; this method does not retry or swallow.
        """
        batch = self.client.messages.batches.retrieve(job_id)
        result: dict[str, Any] = {
            "status": batch.processing_status,
            "results": None,
        }
        if batch.processing_status == "ended":
            result["results"] = list(
                self.client.messages.batches.results(job_id)
            )
        return result
