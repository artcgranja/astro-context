"""Tests for MemoryManager unified memory stack (SlidingWindowMemory + SummaryBufferMemory)."""

from __future__ import annotations

from astro_context.memory.manager import MemoryManager
from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.memory.summary_buffer import SummaryBufferMemory
from astro_context.models.context import SourceType
from astro_context.models.memory import ConversationTurn
from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer


def _simple_compact(turns: list[ConversationTurn]) -> str:
    """Simple compaction: join turn contents."""
    return "Summary: " + "; ".join(t.content for t in turns)


def _progressive_compact(turns: list[ConversationTurn], prev: str | None) -> str:
    """Progressive compaction: append to previous summary."""
    new_part = "; ".join(t.content for t in turns)
    if prev is not None:
        return f"{prev} | {new_part}"
    return f"Summary: {new_part}"


# ---- Default behavior (creates SlidingWindowMemory) ----


class TestMemoryManagerDefault:
    """MemoryManager with default parameters creates a SlidingWindowMemory."""

    def test_default_creates_sliding_window(self) -> None:
        mgr = MemoryManager(conversation_tokens=1000, tokenizer=FakeTokenizer())
        assert isinstance(mgr.conversation, SlidingWindowMemory)

    def test_default_conversation_type(self) -> None:
        mgr = MemoryManager(conversation_tokens=1000, tokenizer=FakeTokenizer())
        assert mgr.conversation_type == "sliding_window"

    def test_default_messages_work(self) -> None:
        mgr = MemoryManager(conversation_tokens=1000, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi there!")
        assert len(mgr.conversation.turns) == 2

    def test_default_get_context_items(self) -> None:
        mgr = MemoryManager(conversation_tokens=1000, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert len(items) == 1
        assert items[0].source == SourceType.CONVERSATION


# ---- Explicit SlidingWindowMemory ----


class TestMemoryManagerExplicitSlidingWindow:
    """MemoryManager with explicit SlidingWindowMemory passed as conversation_memory."""

    def test_explicit_sliding_window(self) -> None:
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        mgr = MemoryManager(conversation_memory=window, tokenizer=FakeTokenizer())
        assert mgr.conversation is window

    def test_explicit_sliding_window_type(self) -> None:
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        mgr = MemoryManager(conversation_memory=window, tokenizer=FakeTokenizer())
        assert mgr.conversation_type == "sliding_window"

    def test_explicit_sliding_window_messages(self) -> None:
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        mgr = MemoryManager(conversation_memory=window, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        assert len(window.turns) == 2

    def test_explicit_sliding_window_context_items(self) -> None:
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        mgr = MemoryManager(conversation_memory=window, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert len(items) == 1


# ---- SummaryBufferMemory ----


class TestMemoryManagerSummaryBuffer:
    """MemoryManager with SummaryBufferMemory as conversation_memory."""

    def test_summary_buffer_conversation_type(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        assert mgr.conversation_type == "summary_buffer"

    def test_summary_buffer_conversation_property(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        assert mgr.conversation is buf

    def test_add_user_message_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        assert len(buf.turns) == 1
        assert buf.turns[0].role == "user"
        assert buf.turns[0].content == "Hello"

    def test_add_assistant_message_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_assistant_message("Hi there!")
        assert len(buf.turns) == 1
        assert buf.turns[0].role == "assistant"

    def test_add_system_message_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_system_message("You are helpful.")
        assert len(buf.turns) == 1
        assert buf.turns[0].role == "system"

    def test_add_tool_message_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_tool_message("Tool output.")
        assert len(buf.turns) == 1
        assert buf.turns[0].role == "tool"

    def test_mixed_messages_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.add_user_message("How are you?")
        mgr.add_assistant_message("Good!")
        assert len(buf.turns) == 4

    def test_get_context_items_with_summary_buffer_no_eviction(self) -> None:
        """Without eviction, context items come from the live window only."""
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("World")
        items = mgr.get_context_items()
        assert len(items) == 2
        assert all(
            i.source in (SourceType.MEMORY, SourceType.CONVERSATION) for i in items
        )
        # No summary item
        assert all(i.metadata.get("summary") is not True for i in items)

    def test_get_context_items_with_summary_buffer_with_eviction(self) -> None:
        """After eviction, context items include the summary item."""
        buf = SummaryBufferMemory(
            max_tokens=3, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("World")
        mgr.add_user_message("three four five")  # triggers eviction

        items = mgr.get_context_items()
        summary_items = [i for i in items if i.metadata.get("summary") is True]
        assert len(summary_items) == 1
        assert summary_items[0].source in (SourceType.MEMORY, SourceType.CONVERSATION)

    def test_clear_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=3, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("World")
        mgr.add_user_message("three four five")  # triggers eviction
        assert buf.summary is not None

        mgr.clear()
        assert len(buf.turns) == 0
        assert buf.summary is None
        assert mgr.get_context_items() == []


# ---- Persistent store alongside SummaryBufferMemory ----


class TestMemoryManagerPersistentWithSummaryBuffer:
    """Persistent store works alongside SummaryBufferMemory."""

    def test_persistent_plus_summary_buffer(self) -> None:
        store = InMemoryEntryStore()
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(
            conversation_memory=buf,
            persistent_store=store,
            tokenizer=FakeTokenizer(),
        )
        mgr.add_fact("User prefers dark mode", tags=["preference"])
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")

        items = mgr.get_context_items()
        # 1 persistent + 2 conversation
        assert len(items) == 3

    def test_persistent_items_before_conversation_with_summary_buffer(self) -> None:
        store = InMemoryEntryStore()
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(
            conversation_memory=buf,
            persistent_store=store,
            tokenizer=FakeTokenizer(),
        )
        mgr.add_user_message("Hello")
        mgr.add_fact("persistent fact")

        items = mgr.get_context_items()
        # Persistent items come first
        assert items[0].metadata.get("memory_entry_id") is not None
        assert items[1].metadata.get("role") == "user"

    def test_persistent_items_have_priority_8_with_summary_buffer(self) -> None:
        store = InMemoryEntryStore()
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(
            conversation_memory=buf,
            persistent_store=store,
            tokenizer=FakeTokenizer(),
        )
        mgr.add_fact("fact")
        items = mgr.get_context_items()
        persistent = [i for i in items if i.metadata.get("memory_entry_id")]
        assert persistent[0].priority == 8

    def test_clear_clears_both_persistent_and_summary(self) -> None:
        store = InMemoryEntryStore()
        buf = SummaryBufferMemory(
            max_tokens=3, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(
            conversation_memory=buf,
            persistent_store=store,
            tokenizer=FakeTokenizer(),
        )
        mgr.add_fact("fact")
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("World")
        mgr.add_user_message("three four five")  # triggers summary

        mgr.clear()
        assert store.list_all() == []
        assert buf.summary is None
        assert len(buf.turns) == 0
        assert mgr.get_context_items() == []


# ---- conversation_memory overrides other parameters ----


class TestConversationMemoryOverrides:
    """When conversation_memory is provided, conversation_tokens/tokenizer/on_evict are ignored."""

    def test_conversation_memory_overrides_conversation_tokens(self) -> None:
        """conversation_tokens is ignored when conversation_memory is provided."""
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        # conversation_tokens=100 should be ignored since we pass conversation_memory
        mgr = MemoryManager(
            conversation_tokens=100,
            conversation_memory=window,
            tokenizer=FakeTokenizer(),
        )
        assert mgr.conversation is window
        assert mgr.conversation.max_tokens == 500  # type: ignore[union-attr]

    def test_conversation_memory_overrides_on_evict(self) -> None:
        """on_evict is ignored when conversation_memory is provided."""
        called: list[bool] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            called.append(True)

        # Create a window without on_evict
        window = SlidingWindowMemory(max_tokens=3, tokenizer=FakeTokenizer())
        mgr = MemoryManager(
            conversation_memory=window,
            on_evict=on_evict,  # This should be ignored
            tokenizer=FakeTokenizer(),
        )
        mgr.add_user_message("first")
        mgr.add_assistant_message("second")
        mgr.add_user_message("three four five")
        # on_evict was not wired to the window, so it should not be called
        assert len(called) == 0

    def test_conversation_memory_overrides_tokenizer(self) -> None:
        """tokenizer used for conversation is the one in the provided memory object."""
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        mgr = MemoryManager(
            conversation_memory=window,
            tokenizer=FakeTokenizer(),  # Different instance, but same type
        )
        mgr.add_user_message("Hello world")
        turns = mgr.conversation.turns
        # Token count should come from the window's tokenizer
        assert turns[0].token_count == 2  # FakeTokenizer: "Hello world" = 2 tokens

    def test_conversation_tokens_validation_skipped_with_conversation_memory(self) -> None:
        """conversation_tokens <= 0 does not raise when conversation_memory is provided."""
        window = SlidingWindowMemory(max_tokens=500, tokenizer=FakeTokenizer())
        # This should NOT raise even though conversation_tokens is invalid
        mgr = MemoryManager(
            conversation_tokens=-1,
            conversation_memory=window,
            tokenizer=FakeTokenizer(),
        )
        assert mgr.conversation is window


# ---- Progressive SummaryBufferMemory ----


class TestMemoryManagerProgressiveSummary:
    """MemoryManager with progressive SummaryBufferMemory."""

    def test_progressive_summary_accumulates(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=2,
            progressive_compact_fn=_progressive_compact,
            tokenizer=FakeTokenizer(),
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())

        mgr.add_user_message("first")
        mgr.add_assistant_message("second")
        # At 2 tokens. Adding third triggers eviction.
        mgr.add_user_message("third")

        assert buf.summary is not None
        assert "first" in buf.summary

    def test_progressive_summary_in_context_items(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=2,
            progressive_compact_fn=_progressive_compact,
            tokenizer=FakeTokenizer(),
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())

        mgr.add_user_message("first")
        mgr.add_assistant_message("second")
        mgr.add_user_message("third")

        items = mgr.get_context_items()
        summary_items = [i for i in items if i.metadata.get("summary") is True]
        assert len(summary_items) == 1


# ---- Repr ----


class TestMemoryManagerReprUnified:
    """__repr__ reflects the conversation memory type."""

    def test_repr_with_sliding_window(self) -> None:
        mgr = MemoryManager(conversation_tokens=1000, tokenizer=FakeTokenizer())
        r = repr(mgr)
        assert "MemoryManager" in r
        assert "SlidingWindowMemory" in r

    def test_repr_with_summary_buffer(self) -> None:
        buf = SummaryBufferMemory(
            max_tokens=100, compact_fn=_simple_compact, tokenizer=FakeTokenizer()
        )
        mgr = MemoryManager(conversation_memory=buf, tokenizer=FakeTokenizer())
        r = repr(mgr)
        assert "MemoryManager" in r
        assert "SummaryBufferMemory" in r
