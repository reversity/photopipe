"""Tests for photopipe.vlm_client."""
from unittest.mock import MagicMock

import pytest
from PIL import Image

from photopipe.vlm_client import VLMClient, build_image_block


def test_build_image_block_resizes_large_image(tmp_path):
    img_path = tmp_path / "big.jpg"
    Image.new("RGB", (4000, 3000), color="red").save(img_path)
    block = build_image_block(img_path, max_dim=1568)
    assert block["type"] == "image"
    assert block["source"]["media_type"] == "image/jpeg"
    assert block["source"]["type"] == "base64"
    assert isinstance(block["source"]["data"], str)
    assert len(block["source"]["data"]) > 0


def test_build_image_block_preserves_small_image(tmp_path):
    img_path = tmp_path / "small.jpg"
    Image.new("RGB", (800, 600), color="blue").save(img_path)
    block = build_image_block(img_path, max_dim=1568)
    assert block["type"] == "image"
    # Should still encode without crashing for sub-threshold images.
    assert len(block["source"]["data"]) > 0


def test_prompt_prefix_marked_for_cache():
    client = VLMClient(api_key="fake")
    msg = client._build_message(
        cached_prefix="LONG INSTRUCTION PROMPT" * 100,
        images=[],
        per_call_prompt="describe",
    )
    # First content block (text prefix) must have cache_control set.
    assert msg["content"][0].get("cache_control") == {"type": "ephemeral"}
    assert msg["content"][0]["type"] == "text"
    # Trailing per-call prompt is appended without cache_control.
    assert msg["content"][-1] == {"type": "text", "text": "describe"}
    assert msg["role"] == "user"


def test_build_message_omits_empty_prefix_and_prompt():
    client = VLMClient(api_key="fake")
    msg = client._build_message(
        cached_prefix="",
        images=[{"type": "image", "source": {}}],
        per_call_prompt="",
    )
    assert msg["content"] == [{"type": "image", "source": {}}]


def test_realtime_call_uses_structured_output():
    """When response_schema is provided, the client must configure tool_use
    with strict mode and extract the structured `input` from the tool_use block.

    Note (test deviation): the implementation lazily constructs the Anthropic
    client through a `.client` property. Setting `_anthropic_client` directly
    short-circuits the lazy-init and is the cleanest way to inject a mock.
    """
    client = VLMClient(api_key="fake")
    mock_client = MagicMock()
    client._anthropic_client = mock_client  # bypass lazy init

    tool_use_block = MagicMock()
    tool_use_block.type = "tool_use"
    tool_use_block.input = {"year": 1985, "confidence": "high"}

    mock_resp = MagicMock()
    mock_resp.content = [tool_use_block]
    mock_client.messages.create.return_value = mock_resp

    result = client.analyze(
        cached_prefix="instructions",
        images=[],
        per_call_prompt="describe",
        response_schema={
            "type": "object",
            "properties": {"year": {"type": "integer"}},
        },
    )

    assert result == {"year": 1985, "confidence": "high"}
    kwargs = mock_client.messages.create.call_args.kwargs
    # Structured output must be configured via tools + tool_choice.
    assert "tools" in kwargs
    assert kwargs["tools"][0]["name"] == "respond"
    assert kwargs["tool_choice"] == {"type": "tool", "name": "respond"}
    assert kwargs["model"] == client.model


def test_realtime_call_without_schema_returns_text():
    client = VLMClient(api_key="fake")
    mock_client = MagicMock()
    client._anthropic_client = mock_client

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = "hello world"
    mock_resp = MagicMock()
    mock_resp.content = [text_block]
    mock_client.messages.create.return_value = mock_resp

    result = client.analyze(
        cached_prefix="instructions",
        images=[],
        per_call_prompt="describe",
    )
    assert result == {"text": "hello world"}
    kwargs = mock_client.messages.create.call_args.kwargs
    assert "tools" not in kwargs


def test_batch_api_submission_returns_job_id():
    client = VLMClient(api_key="fake")
    mock_client = MagicMock()
    client._anthropic_client = mock_client

    mock_batch = MagicMock()
    mock_batch.id = "msgbatch_abc123"
    mock_client.messages.batches.create.return_value = mock_batch

    job_id = client.submit_batch([
        {"custom_id": "p1", "params": {"messages": []}},
    ])
    assert job_id == "msgbatch_abc123"
    kwargs = mock_client.messages.batches.create.call_args.kwargs
    assert kwargs["requests"] == [{"custom_id": "p1", "params": {"messages": []}}]


def test_poll_batch_returns_status_and_no_results_when_in_progress():
    client = VLMClient(api_key="fake")
    mock_client = MagicMock()
    client._anthropic_client = mock_client

    mock_batch = MagicMock()
    mock_batch.processing_status = "in_progress"
    mock_client.messages.batches.retrieve.return_value = mock_batch

    result = client.poll_batch("msgbatch_xyz")
    assert result["status"] == "in_progress"
    assert result["results"] is None
    mock_client.messages.batches.results.assert_not_called()


def test_poll_batch_returns_results_when_ended():
    client = VLMClient(api_key="fake")
    mock_client = MagicMock()
    client._anthropic_client = mock_client

    mock_batch = MagicMock()
    mock_batch.processing_status = "ended"
    mock_client.messages.batches.retrieve.return_value = mock_batch
    mock_client.messages.batches.results.return_value = iter(["r1", "r2"])

    result = client.poll_batch("msgbatch_xyz")
    assert result["status"] == "ended"
    assert result["results"] == ["r1", "r2"]


def test_cache_ttl_1h_propagates_to_cache_control():
    client = VLMClient(api_key="fake")
    client.cache_ttl = "1h"
    msg = client._build_message(
        cached_prefix="LONG PREFIX",
        images=[],
        per_call_prompt="x",
    )
    assert msg["content"][0]["cache_control"] == {"type": "ephemeral", "ttl": "1h"}


def test_cache_ttl_5m_omits_ttl_key():
    client = VLMClient(api_key="fake")
    client.cache_ttl = "5m"
    msg = client._build_message(
        cached_prefix="LONG PREFIX",
        images=[],
        per_call_prompt="x",
    )
    # 5m is the default ephemeral TTL — no `ttl` key needed
    assert msg["content"][0]["cache_control"] == {"type": "ephemeral"}


def test_analyze_propagates_sdk_exceptions():
    client = VLMClient(api_key="fake")
    client._anthropic_client = MagicMock()
    client._anthropic_client.messages.create.side_effect = RuntimeError("rate limit")
    with pytest.raises(RuntimeError, match="rate limit"):
        client.analyze(cached_prefix="x", images=[], per_call_prompt="y")
