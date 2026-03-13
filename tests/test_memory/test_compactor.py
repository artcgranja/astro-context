"""Tests for anchor.memory.compactor (TierCompactor)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.compactor import TierCompactor
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_compactor(response_content: str = "Summary text") -> tuple[TierCompactor, MagicMock]:
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = _make_llm_response(response_content)
    mock_llm.model_id = "test/model"
    mock_llm.provider_name = "test"
    compactor = TierCompactor(llm=mock_llm, tokenizer=FakeTokenizer())
    return compactor, mock_llm


class TestTierCompactorSummarize:
    def test_summarize_tier1(self) -> None:
        compactor, mock_llm = _make_compactor("Detailed summary of conversation")
        result = compactor.summarize("user: hello\nassistant: hi", target_tier=1, target_tokens=500)
        assert result == "Detailed summary of conversation"
        mock_llm.invoke.assert_called_once()
        # Check prompt contains target tokens
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        assert any("500" in str(m.content) for m in messages)

    def test_summarize_tier2(self) -> None:
        compactor, mock_llm = _make_compactor("Compact summary")
        result = compactor.summarize("Detailed summary text", target_tier=2, target_tokens=100)
        assert result == "Compact summary"

    def test_summarize_tier3(self) -> None:
        compactor, mock_llm = _make_compactor("Headline")
        result = compactor.summarize("Compact text", target_tier=3, target_tokens=20)
        assert result == "Headline"

    def test_progressive_merge(self) -> None:
        compactor, mock_llm = _make_compactor("Merged summary")
        result = compactor.summarize(
            "New content",
            target_tier=1,
            target_tokens=500,
            existing_summary="Old summary",
        )
        assert result == "Merged summary"
        call_args = mock_llm.invoke.call_args
        messages = call_args[0][0]
        prompt_text = str(messages[0].content)
        assert "Old summary" in prompt_text
        assert "New content" in prompt_text

    def test_summarize_fallback_on_error(self) -> None:
        compactor, mock_llm = _make_compactor()
        mock_llm.invoke.side_effect = Exception("LLM error")
        result = compactor.summarize("Some content here", target_tier=1, target_tokens=500)
        # Falls back to raw content (possibly truncated)
        assert "Some content here" in result
