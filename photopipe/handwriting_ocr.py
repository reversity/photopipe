"""
Handwriting OCR on photo backs.

Primary: Mistral OCR 3 (cheap, purpose-built).
Fallback: Claude vision (when Mistral confidence < threshold).

Both paths feed into date_parser to extract dates.
"""
from __future__ import annotations

import base64
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Optional

from photopipe.config import get_config
from photopipe.date_parser import parse_date_from_text
from photopipe.vlm_client import VLMClient, build_image_block


HANDWRITING_VLM_PROMPT = """Read any handwritten or stamped text on this photo back.
Return the literal text exactly as written (preserve line breaks).
If you see a date, even partial, include it verbatim.
If there is no readable text, respond with an empty string."""


@dataclass
class OCRResult:
    text: str
    confidence: float
    provider: str  # "mistral" | "claude_vlm" | "none"
    extracted_date: Optional[date] = None


class HandwritingOCR:
    """Run handwriting OCR on photo backs.

    Mistral OCR 3 is the primary provider; Claude VLM is the fallback when
    Mistral confidence falls below the configured threshold. The fallback
    is suppressed when ``provider == "mistral"`` (caller has explicitly
    pinned a single provider).

    The Mistral SDK client is constructed lazily on first use so tests can
    mock ``_call_mistral`` directly without needing the SDK installed.
    Exceptions from either underlying SDK propagate to the caller.
    """

    def __init__(
        self,
        mistral_api_key: Optional[str] = None,
        vlm_client: Optional[VLMClient] = None,
    ):
        cfg = get_config().handwriting_ocr
        self.cfg = cfg
        self.mistral_api_key = mistral_api_key or os.environ.get(cfg.mistral_api_key_env_var)
        self.vlm_client = vlm_client if vlm_client is not None else VLMClient()
        self._mistral_client: Any = None

    @property
    def mistral_client(self) -> Any:
        """Lazily instantiate the Mistral SDK client.

        The mistralai >= 2.0 package exposes ``Mistral`` from ``mistralai.client``;
        we fall back to a top-level ``mistralai`` import for older 1.x layouts.
        Local import keeps the SDK optional at module import time.
        """
        if self._mistral_client is None and self.mistral_api_key:
            try:
                from mistralai.client import Mistral
            except ImportError:  # pragma: no cover — older SDK layout
                from mistralai import Mistral
            self._mistral_client = Mistral(api_key=self.mistral_api_key)
        return self._mistral_client

    def ocr_back(self, image_path: Path) -> OCRResult:
        """Run handwriting OCR on a single photo back.

        Returns an :class:`OCRResult` whose ``provider`` indicates which
        path produced the text. When the parsed text yields any recognizable
        date, ``extracted_date`` is populated with the first match.
        """
        result: Optional[OCRResult] = None

        # Try Mistral first when configured and a client is available.
        if self.cfg.provider in ("mistral", "auto") and self.mistral_client:
            result = self._call_mistral(image_path)

        # Fall back to VLM unless the user has pinned provider="mistral".
        if (
            result is None
            or result.confidence < self.cfg.confidence_fallback_threshold
        ) and self.cfg.provider != "mistral":
            result = self._call_vlm(image_path)

        if result is None:
            result = OCRResult(text="", confidence=0.0, provider="none")

        dates = parse_date_from_text(result.text)
        if dates:
            result.extracted_date = dates[0][0]
        return result

    def _call_mistral(self, image_path: Path) -> OCRResult:
        """Run Mistral OCR 3 on a single image.

        Assumed SDK shape (mistralai >= 1.0):
          - ``client.ocr.process(model, document)`` returns an object with
            ``pages``; each page has ``markdown`` and optionally ``regions``
            with per-region ``confidence`` floats.
        If the actual SDK surface differs, this method is the only place
        that needs to change — tests mock it directly.
        """
        with open(image_path, "rb") as f:
            img_b64 = base64.standard_b64encode(f.read()).decode("ascii")
        resp = self.mistral_client.ocr.process(
            model=self.cfg.mistral_model,
            document={
                "type": "image_url",
                "image_url": f"data:image/jpeg;base64,{img_b64}",
            },
        )
        text = "\n".join(page.markdown for page in resp.pages)
        confs = [
            region.confidence
            for page in resp.pages
            for region in getattr(page, "regions", [])
            if region.confidence is not None
        ]
        avg_conf = sum(confs) / len(confs) if confs else 0.5
        return OCRResult(text=text.strip(), confidence=avg_conf, provider="mistral")

    def _call_vlm(self, image_path: Path) -> OCRResult:
        """Fall back to the Claude VLM for handwriting OCR.

        Uses tool-use structured output to coerce the model into emitting
        a JSON object with ``text`` (required) and ``confidence`` (optional;
        defaults to 0.7 when omitted).
        """
        block = build_image_block(image_path)
        out = self.vlm_client.analyze(
            cached_prefix=HANDWRITING_VLM_PROMPT,
            images=[block],
            per_call_prompt="Read the text on this back.",
            response_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["text"],
                "additionalProperties": False,
            },
        )
        return OCRResult(
            text=out.get("text", ""),
            confidence=out.get("confidence", 0.7),
            provider="claude_vlm",
        )
