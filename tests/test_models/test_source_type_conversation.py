"""Tests for SourceType.CONVERSATION and the MEMORY/CONVERSATION split.

Verifies that:
- SourceType.CONVERSATION exists with the correct value
- SlidingWindowMemory produces CONVERSATION-typed items
- SummaryBufferMemory produces CONVERSATION-typed items
- MemoryManager persistent facts keep MEMORY source
- MemoryManager conversation items have CONVERSATION source
- classify_window_items correctly classifies CONVERSATION items
- Budget defaults include CONVERSATION allocations
"""

from __future__ import annotations

from astro_context.formatters.utils import classify_window_items
from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.memory.summary_buffer import SummaryBufferMemory
from astro_context.models.budget_defaults import (
    default_agent_budget,
    default_chat_budget,
    default_rag_budget,
)
from astro_context.models.context import ContextItem, ContextWindow, SourceType
from astro_context.models.memory import ConversationTurn
from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer

# ---------------------------------------------------------------------------
# SourceType enum
# ---------------------------------------------------------------------------


class TestSourceTypeConversation:
    """SourceType.CONVERSATION exists and has the correct value."""

    def test_conversation_value(self) -> None:
        assert SourceType.CONVERSATION == "conversation"

    def test_conversation_is_distinct_from_memory(self) -> None:
        assert SourceType.CONVERSATION != SourceType.MEMORY

    def test_all_source_types_present(self) -> None:
        """All six source types should be present."""
        expected = {"retrieval", "memory", "system", "user", "tool", "conversation"}
        actual = {member.value for member in SourceType}
        assert expected == actual


# ---------------------------------------------------------------------------
# SlidingWindowMemory
# ---------------------------------------------------------------------------


class TestSlidingWindowConversationSource:
    """SlidingWindowMemory.to_context_items() returns CONVERSATION source."""

    def test_single_turn_has_conversation_source(self) -> None:
        mem = SlidingWindowMemory(max_tokens=1000, tokenizer=FakeTokenizer())
        mem.add_turn("user", "Hello")
        items = mem.to_context_items()
        assert len(items) == 1
        assert items[0].source == SourceType.CONVERSATION

    def test_multiple_turns_all_conversation_source(self) -> None:
        mem = SlidingWindowMemory(max_tokens=1000, tokenizer=FakeTokenizer())
        mem.add_turn("user", "Hello")
        mem.add_turn("assistant", "Hi there!")
        mem.add_turn("user", "How are you?")
        items = mem.to_context_items()
        assert all(item.source == SourceType.CONVERSATION for item in items)


# ---------------------------------------------------------------------------
# SummaryBufferMemory
# ---------------------------------------------------------------------------


def _simple_compact(turns: list[ConversationTurn]) -> str:
    return "Summary: " + "; ".join(t.content for t in turns)


class TestSummaryBufferConversationSource:
    """SummaryBufferMemory.to_context_items() returns CONVERSATION source."""

    def test_window_items_have_conversation_source(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100,
            compact_fn=_simple_compact,
            tokenizer=FakeTokenizer(),
        )
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        items = buf.to_context_items()
        assert len(items) == 2
        assert all(item.source == SourceType.CONVERSATION for item in items)

    def test_summary_item_has_conversation_source(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=3,
            compact_fn=_simple_compact,
            tokenizer=FakeTokenizer(),
        )
        buf.add_message("user", "Hello")
        buf.add_message("assistant", "World")
        buf.add_message("user", "three four five")

        items = buf.to_context_items()
        summary_items = [i for i in items if i.metadata.get("summary") is True]
        assert len(summary_items) == 1
        assert summary_items[0].source == SourceType.CONVERSATION


# ---------------------------------------------------------------------------
# MemoryManager
# ---------------------------------------------------------------------------


class TestMemoryManagerSourceTypes:
    """MemoryManager produces correct source types for persistent vs conversation."""

    def test_persistent_items_have_memory_source(self) -> None:
        from astro_context.memory.manager import MemoryManager

        store = InMemoryEntryStore()
        mgr = MemoryManager(
            conversation_tokens=2000,
            tokenizer=FakeTokenizer(),
            persistent_store=store,
        )
        mgr.add_fact("User prefers dark mode", tags=["preference"])
        items = mgr.get_context_items()

        persistent_items = [i for i in items if i.metadata.get("memory_entry_id")]
        assert len(persistent_items) == 1
        assert persistent_items[0].source == SourceType.MEMORY

    def test_conversation_items_have_conversation_source(self) -> None:
        from astro_context.memory.manager import MemoryManager

        mgr = MemoryManager(conversation_tokens=2000, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi there!")
        items = mgr.get_context_items()

        assert len(items) == 2
        assert all(item.source == SourceType.CONVERSATION for item in items)

    def test_mixed_persistent_and_conversation(self) -> None:
        from astro_context.memory.manager import MemoryManager

        store = InMemoryEntryStore()
        mgr = MemoryManager(
            conversation_tokens=2000,
            tokenizer=FakeTokenizer(),
            persistent_store=store,
        )
        mgr.add_fact("User likes Python")
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")

        items = mgr.get_context_items()

        memory_items = [i for i in items if i.source == SourceType.MEMORY]
        conversation_items = [i for i in items if i.source == SourceType.CONVERSATION]

        assert len(memory_items) == 1  # persistent fact
        assert len(conversation_items) == 2  # conversation turns
        assert memory_items[0].content == "User likes Python"


# ---------------------------------------------------------------------------
# classify_window_items
# ---------------------------------------------------------------------------


class TestClassifyWindowItemsConversation:
    """classify_window_items() correctly classifies CONVERSATION items."""

    def test_conversation_items_in_memory_bucket(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.CONVERSATION,
                token_count=1,
                metadata={"role": "user"},
            )
        )
        result = classify_window_items(window)

        assert len(result.memory_items) == 1
        assert result.memory_items[0].source == SourceType.CONVERSATION
        assert result.system_parts == []
        assert result.context_parts == []

    def test_memory_items_still_in_memory_bucket(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="User prefers dark mode",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "system"},
            )
        )
        result = classify_window_items(window)

        assert len(result.memory_items) == 1
        assert result.memory_items[0].source == SourceType.MEMORY

    def test_mixed_memory_and_conversation_in_same_bucket(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="persistent fact",
                source=SourceType.MEMORY,
                token_count=2,
                metadata={"role": "system"},
            )
        )
        window.add_item(
            ContextItem(
                content="user message",
                source=SourceType.CONVERSATION,
                token_count=2,
                metadata={"role": "user"},
            )
        )
        result = classify_window_items(window)

        assert len(result.memory_items) == 2
        sources = {item.source for item in result.memory_items}
        assert SourceType.MEMORY in sources
        assert SourceType.CONVERSATION in sources


# ---------------------------------------------------------------------------
# Budget defaults
# ---------------------------------------------------------------------------


class TestBudgetDefaultsConversation:
    """Budget defaults include CONVERSATION allocations."""

    def test_chat_budget_has_conversation(self) -> None:
        budget = default_chat_budget(10000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources
        assert SourceType.MEMORY in sources

    def test_rag_budget_has_conversation(self) -> None:
        budget = default_rag_budget(10000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources
        assert SourceType.MEMORY in sources

    def test_agent_budget_has_conversation(self) -> None:
        budget = default_agent_budget(10000)
        sources = {a.source for a in budget.allocations}
        assert SourceType.CONVERSATION in sources
        assert SourceType.MEMORY in sources

    def test_conversation_and_memory_have_different_priorities(self) -> None:
        budget = default_chat_budget(10000)
        mem = next(a for a in budget.allocations if a.source == SourceType.MEMORY)
        conv = next(a for a in budget.allocations if a.source == SourceType.CONVERSATION)
        assert mem.priority > conv.priority  # MEMORY=8 > CONVERSATION=7
