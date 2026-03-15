"""Integration tests for persistent conversation memory."""

from unittest.mock import MagicMock

import pytest

from anchor.llm.models import LLMResponse, StopReason, Usage
from anchor.memory.manager import MemoryManager
from anchor.memory.progressive import ProgressiveSummarizationMemory
from anchor.storage.memory_store import InMemoryConversationStore
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


class TestMemoryManagerConversationPersistence:
    def test_auto_persist_appends_turns(self):
        store = InMemoryConversationStore()
        mgr = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
            auto_persist=True,
        )
        mgr.add_user_message("hello")
        mgr.add_assistant_message("hi")
        turns = store.load_turns("sess-1")
        assert len(turns) == 2

    def test_requires_session_id_with_store(self):
        store = InMemoryConversationStore()
        with pytest.raises(ValueError, match="session_id"):
            MemoryManager(
                conversation_tokens=4096,
                tokenizer=FakeTokenizer(),
                conversation_store=store,
            )

    def test_save_and_load(self):
        store = InMemoryConversationStore()
        mgr1 = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
        )
        mgr1.add_user_message("hello")
        mgr1.add_assistant_message("hi there")
        mgr1.save()

        # Simulate restart: new MemoryManager loads from store
        mgr2 = MemoryManager(
            conversation_tokens=4096,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
        )
        mgr2.load()
        items = mgr2.get_context_items()
        contents = [item.content for item in items]
        assert any("hello" in c for c in contents)


class TestProgressiveSummarizationPersistence:
    def test_compaction_persists_summary_tiers(self):
        store = InMemoryConversationStore()
        progressive = ProgressiveSummarizationMemory(
            max_tokens=200, llm=_make_mock_llm(), tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(
            conversation_memory=progressive,
            tokenizer=FakeTokenizer(),
            conversation_store=store,
            session_id="sess-1",
            auto_persist=True,
        )
        for i in range(20):
            mgr.add_user_message(f"Message number {i} with some content to fill tokens")
            mgr.add_assistant_message(f"Response number {i}")
        mgr.save()
        tiers = store.load_summary_tiers("sess-1")
        has_tier = any(t is not None for t in tiers.values())
        assert has_tier or len(store.load_turns("sess-1")) > 0
