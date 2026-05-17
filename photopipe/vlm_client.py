"""
Thin transport-layer wrapper around Anthropic vision calls.

Owns: prompt caching, structured output, Batch API submission/polling.
Does NOT own: prompts, schemas, business logic — those live in the
caller (dating_pipeline.py, handwriting_ocr.py).
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from PIL import Image

from photopipe.config import get_config


@dataclass
class PromptSection:
    """A chunk of prompt text optionally marked for prompt caching."""
    text: str
    cached: bool = False  # True => mark with cache_control


def build_image_block(image_path: Path, max_dim: int = 1568) -> dict:
    """Encode a single image as a base64 JPEG content block.

    Resizes so the largest dimension is at most `max_dim` (Anthropic's
    1568px vision token grid). Always converts to RGB JPEG for consistency.
    """
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_dim:
        if w >= h:
            new_size = (max_dim, int(h * max_dim / w))
        else:
            new_size = (int(w * max_dim / h), max_dim)
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return {
        "type": "image",
        "source": {
            "type": "base64",
            "media_type": "image/jpeg",
            "data": base64.standard_b64encode(buf.getvalue()).decode("ascii"),
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
            self._anthropic_client = anthropic.Anthropic(api_key=self.api_key)
        return self._anthropic_client

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
                "cache_control": {"type": "ephemeral"},
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
            # Fallback: try to parse text as JSON.
            return json.loads(resp.content[0].text)
        return {"text": resp.content[0].text}

    def submit_batch(self, requests: list[dict]) -> str:
        """Submit a Batch API job.

        `requests` is a list of `{custom_id, params}` dicts following the
        Anthropic Batch API request shape. Returns the batch job id.
        """
        batch = self.client.messages.batches.create(requests=requests)
        return batch.id

    def poll_batch(self, job_id: str) -> dict:
        """Poll a Batch API job.

        Returns `{"status": <processing_status>, "results": <list|None>}`.
        Results are materialized (the SDK returns a generator) only when
        the batch has reached the terminal `ended` state.
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
