"""Tests for astro_context.memory.manager."""

from __future__ import annotations

from astro_context.memory.manager import MemoryManager
from astro_context.models.context import SourceType
from tests.conftest import FakeTokenizer


def _make_manager(conversation_tokens: int = 1000) -> MemoryManager:
    """Create a MemoryManager with FakeTokenizer."""
    return MemoryManager(conversation_tokens=conversation_tokens, tokenizer=FakeTokenizer())


class TestMemoryManagerMessages:
    """add_user_message and add_assistant_message."""

    def test_add_user_message(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello!")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello!"

    def test_add_assistant_message(self) -> None:
        mgr = _make_manager()
        mgr.add_assistant_message("Hi there!")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "assistant"
        assert turns[0].content == "Hi there!"

    def test_add_system_message(self) -> None:
        mgr = _make_manager()
        mgr.add_system_message("You are helpful.")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "system"

    def test_mixed_messages(self) -> None:
        mgr = _make_manager(conversation_tokens=2000)
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.add_user_message("How are you?")
        mgr.add_assistant_message("I'm good!")
        assert len(mgr.conversation.turns) == 4


class TestMemoryManagerGetContextItems:
    """get_context_items returns items with correct priority."""

    def test_returns_context_items(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi there")
        items = mgr.get_context_items()
        assert len(items) == 2

    def test_items_have_memory_source(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert all(item.source == SourceType.MEMORY for item in items)

    def test_items_have_specified_priority(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items(priority=9)
        assert all(item.priority == 9 for item in items)

    def test_default_priority_is_seven(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert all(item.priority == 7 for item in items)

    def test_empty_manager_returns_empty_list(self) -> None:
        mgr = _make_manager()
        assert mgr.get_context_items() == []


class TestMemoryManagerClear:
    """clear resets all state."""

    def test_clear(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.clear()
        assert len(mgr.conversation.turns) == 0
        assert mgr.get_context_items() == []


class TestMemoryManagerConversationProperty:
    """conversation property gives access to sliding window."""

    def test_conversation_property(self) -> None:
        mgr = _make_manager(conversation_tokens=2048)
        assert mgr.conversation.max_tokens == 2048
