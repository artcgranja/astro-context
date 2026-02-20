"""Tests for astro_context.memory.manager."""

from __future__ import annotations

import pytest

from astro_context.models.context import SourceType
from tests.conftest import make_memory_manager


class TestMemoryManagerMessages:
    """add_user_message and add_assistant_message."""

    def test_add_user_message(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello!")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "user"
        assert turns[0].content == "Hello!"

    def test_add_assistant_message(self) -> None:
        mgr = make_memory_manager()
        mgr.add_assistant_message("Hi there!")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "assistant"
        assert turns[0].content == "Hi there!"

    def test_add_system_message(self) -> None:
        mgr = make_memory_manager()
        mgr.add_system_message("You are helpful.")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "system"

    def test_add_tool_message(self) -> None:
        mgr = make_memory_manager()
        mgr.add_tool_message("Tool output here.")
        turns = mgr.conversation.turns
        assert len(turns) == 1
        assert turns[0].role == "tool"
        assert turns[0].content == "Tool output here."

    def test_mixed_messages(self) -> None:
        mgr = make_memory_manager(conversation_tokens=2000)
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.add_user_message("How are you?")
        mgr.add_assistant_message("I'm good!")
        assert len(mgr.conversation.turns) == 4

    def test_mixed_messages_with_tool(self) -> None:
        mgr = make_memory_manager(conversation_tokens=2000)
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Let me check")
        mgr.add_tool_message("Result: 42")
        mgr.add_assistant_message("The answer is 42")
        turns = mgr.conversation.turns
        assert len(turns) == 4
        assert turns[2].role == "tool"


class TestMemoryManagerGetContextItems:
    """get_context_items returns items with correct priority."""

    def test_returns_context_items(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi there")
        items = mgr.get_context_items()
        assert len(items) == 2

    def test_items_have_memory_source(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert all(item.source == SourceType.MEMORY for item in items)

    def test_items_have_specified_priority(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items(priority=9)
        assert all(item.priority == 9 for item in items)

    def test_default_priority_is_seven(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello")
        items = mgr.get_context_items()
        assert all(item.priority == 7 for item in items)

    def test_empty_manager_returns_empty_list(self) -> None:
        mgr = make_memory_manager()
        assert mgr.get_context_items() == []


class TestMemoryManagerClear:
    """clear resets all state."""

    def test_clear(self) -> None:
        mgr = make_memory_manager()
        mgr.add_user_message("Hello")
        mgr.add_assistant_message("Hi")
        mgr.clear()
        assert len(mgr.conversation.turns) == 0
        assert mgr.get_context_items() == []


class TestMemoryManagerConversationProperty:
    """conversation property gives access to sliding window."""

    def test_conversation_property(self) -> None:
        mgr = make_memory_manager(conversation_tokens=2048)
        assert mgr.conversation.max_tokens == 2048


class TestMemoryManagerRepr:
    """__repr__ returns a useful string representation."""

    def test_repr(self) -> None:
        mgr = make_memory_manager(conversation_tokens=1000)
        r = repr(mgr)
        assert "MemoryManager" in r
        assert "SlidingWindowMemory" in r

    def test_repr_after_messages(self) -> None:
        mgr = make_memory_manager(conversation_tokens=1000)
        mgr.add_user_message("Hello")
        r = repr(mgr)
        assert "turns=1" in r


class TestMemoryManagerSlots:
    """__slots__ prevents arbitrary attribute assignment."""

    def test_cannot_set_arbitrary_attribute(self) -> None:
        mgr = make_memory_manager()
        with pytest.raises(AttributeError):
            mgr.some_random_attr = "oops"  # type: ignore[attr-defined]
