"""Tests for progressive summarization data models and memory."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from anchor.models.memory import FactType, KeyFact, SummaryTier, TierConfig


class TestFactType:
    def test_all_values(self) -> None:
        assert FactType.DECISION == "decision"
        assert FactType.ENTITY == "entity"
        assert FactType.NUMBER == "number"
        assert FactType.DATE == "date"
        assert FactType.PREFERENCE == "preference"
        assert FactType.CONSTRAINT == "constraint"


class TestKeyFact:
    def test_create_minimal(self) -> None:
        fact = KeyFact(fact_type=FactType.DECISION, content="Use FastAPI", source_tier=0)
        assert fact.fact_type == FactType.DECISION
        assert fact.content == "Use FastAPI"
        assert fact.source_tier == 0
        assert fact.id  # auto-generated UUID
        assert fact.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            KeyFact(fact_type=FactType.NUMBER, content="x", source_tier=0, token_count=-1)


class TestSummaryTier:
    def test_create(self) -> None:
        tier = SummaryTier(level=1, content="Summary text", source_turn_count=5)
        assert tier.level == 1
        assert tier.content == "Summary text"
        assert tier.source_turn_count == 5
        assert tier.token_count == 0

    def test_token_count_non_negative(self) -> None:
        with pytest.raises(ValidationError):
            SummaryTier(level=1, content="x", source_turn_count=1, token_count=-1)


class TestTierConfig:
    def test_create_default(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        assert config.level == 0
        assert config.max_tokens == 4096
        assert config.target_tokens == 0
        assert config.priority == 7

    def test_frozen(self) -> None:
        config = TierConfig(level=0, max_tokens=4096)
        with pytest.raises(AttributeError):
            config.level = 1  # type: ignore[misc]


from anchor.memory.callbacks import ProgressiveSummarizationCallback


class TestProgressiveSummarizationCallback:
    def test_protocol_exists(self) -> None:
        assert hasattr(ProgressiveSummarizationCallback, 'on_tier_cascade')
        assert hasattr(ProgressiveSummarizationCallback, 'on_facts_extracted')
        assert hasattr(ProgressiveSummarizationCallback, 'on_compaction_error')

    def test_satisfies_protocol(self) -> None:
        class MyCallback:
            def on_tier_cascade(self, from_tier, to_tier, tokens_in, tokens_out):
                pass
            def on_facts_extracted(self, facts, source_tier):
                pass
            def on_compaction_error(self, tier, error):
                pass

        assert isinstance(MyCallback(), ProgressiveSummarizationCallback)


from unittest.mock import MagicMock

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.models.memory import TierConfig
from anchor.protocols.memory import ConversationMemory
from tests.conftest import FakeTokenizer


def _make_llm_response(content: str) -> LLMResponse:
    return LLMResponse(
        content=content,
        usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
        model="test",
        provider="test",
        stop_reason=StopReason.STOP,
    )


def _make_mock_llm(summary: str = "Summary", facts: str = "[]") -> MagicMock:
    mock = MagicMock()
    mock.invoke.side_effect = lambda msgs, **kw: _make_llm_response(
        facts if "Extract key facts" in str(msgs[0].content) else summary
    )
    mock.model_id = "test/model"
    mock.provider_name = "test"
    return mock


class TestProgressiveConstruction:
    def test_default_config(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert mem.turns == []
        assert mem.total_tokens == 0
        assert mem.summary is None
        assert mem.facts == []

    def test_custom_tier_config(self) -> None:
        mock_llm = _make_mock_llm()
        config = [
            TierConfig(level=0, max_tokens=100),
            TierConfig(level=1, max_tokens=50, target_tokens=25),
            TierConfig(level=2, max_tokens=20, target_tokens=10),
            TierConfig(level=3, max_tokens=10, target_tokens=5),
        ]
        mem = ProgressiveSummarizationMemory(
            max_tokens=200, llm=mock_llm, tier_config=config, tokenizer=FakeTokenizer()
        )
        assert mem.tier_tokens == {0: 0, 1: 0, 2: 0, 3: 0}

    def test_negative_max_tokens_raises(self) -> None:
        with pytest.raises(ValueError):
            ProgressiveSummarizationMemory(max_tokens=-1, llm=_make_mock_llm())

    def test_satisfies_conversation_memory_protocol(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=8192, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        assert isinstance(mem, ConversationMemory)

    def test_clear_resets_everything(self) -> None:
        mock_llm = _make_mock_llm()
        mem = ProgressiveSummarizationMemory(
            max_tokens=100, llm=mock_llm, tokenizer=FakeTokenizer()
        )
        mem.add_message("user", "hello world")
        assert len(mem.turns) == 1
        mem.clear()
        assert mem.turns == []
        assert mem.facts == []
        assert mem.summary is None
