"""Tests for photopipe.handwriting_ocr."""
from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest
from PIL import Image

from photopipe.config import get_config
from photopipe.handwriting_ocr import HandwritingOCR, HandwritingResult


@pytest.fixture
def fake_back(tmp_path):
    p = tmp_path / "back.jpg"
    Image.new("RGB", (800, 600), color="white").save(p)
    return p


def test_mistral_high_confidence_returns_result(fake_back):
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=MagicMock())
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = HandwritingResult(
            text="Mom and Dad, Summer 1985", confidence=0.9, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.provider == "mistral"
    assert result.confidence == 0.9
    assert "1985" in result.text


def test_low_confidence_falls_back_to_vlm(fake_back):
    vlm = MagicMock()
    vlm.analyze.return_value = {"text": "Aug 1992"}
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=vlm)
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = HandwritingResult(
            text="???", confidence=0.3, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.provider == "claude"
    assert "1992" in result.text
    vlm.analyze.assert_called_once()


def test_extracts_date_from_ocr_text(fake_back):
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=MagicMock())
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = HandwritingResult(
            text="June 1985", confidence=0.85, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.extracted_date is not None
    assert result.extracted_date.year == 1985
    assert result.extracted_date.month == 6


def test_provider_mistral_blocks_vlm_fallback(fake_back):
    """When provider is explicitly 'mistral', the VLM must not be called
    even if Mistral returns low confidence."""
    vlm = MagicMock()
    ocr = HandwritingOCR(mistral_api_key="fake", vlm_client=vlm)
    ocr.cfg = ocr.cfg.model_copy(update={"provider": "mistral"})
    with patch.object(ocr, "_call_mistral") as mock_mistral:
        mock_mistral.return_value = HandwritingResult(
            text="garbled", confidence=0.1, provider="mistral"
        )
        result = ocr.ocr_back(fake_back)
    assert result.provider == "mistral"
    assert result.confidence == 0.1
    vlm.analyze.assert_not_called()


def test_vlm_only_when_no_mistral_key(fake_back):
    """With no mistral key and provider='auto', VLM is the only path."""
    vlm = MagicMock()
    vlm.analyze.return_value = {"text": "1990"}
    ocr = HandwritingOCR(mistral_api_key=None, vlm_client=vlm)
    # No env var either
    with patch.dict("os.environ", {}, clear=False):
        result = ocr.ocr_back(fake_back)
    assert result.provider == "claude"
    assert "1990" in result.text


def test_vlm_default_confidence_when_schema_omits_it(fake_back):
    """When VLM returns {'text': ...} without 'confidence', defaults to 0.7."""
    vlm = MagicMock()
    vlm.analyze.return_value = {"text": "some text"}
    ocr = HandwritingOCR(mistral_api_key=None, vlm_client=vlm)
    result = ocr.ocr_back(fake_back)
    assert result.provider == "claude"
    assert result.confidence == 0.7


def test_provider_mistral_requires_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("MISTRAL_API_KEY", raising=False)
    cfg = get_config()
    cfg.handwriting_ocr.provider = "mistral"
    try:
        with pytest.raises(ValueError, match="no Mistral API key"):
            HandwritingOCR(mistral_api_key=None, vlm_client=MagicMock())
    finally:
        cfg.handwriting_ocr.provider = "auto"  # restore default for other tests
