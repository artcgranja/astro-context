"""Tests for astro_context.formatters.utils."""

from __future__ import annotations

from astro_context.formatters.utils import (
    ClassifiedItems,
    classify_window_items,
    ensure_alternating_roles,
    get_message_role,
)
from astro_context.models.context import ContextItem, ContextWindow, SourceType


class TestGetMessageRole:
    """get_message_role() returns a validated chat role."""

    def test_role_user(self) -> None:
        item = ContextItem(
            content="Hello",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": "user"},
        )
        assert get_message_role(item) == "user"

    def test_role_assistant(self) -> None:
        item = ContextItem(
            content="Hi there!",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": "assistant"},
        )
        assert get_message_role(item) == "assistant"

    def test_role_system(self) -> None:
        item = ContextItem(
            content="You are helpful.",
            source=SourceType.SYSTEM,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "system"

    def test_role_tool_is_allowed(self) -> None:
        """'tool' is in _ALLOWED_ROLES and should be returned as-is."""
        item = ContextItem(
            content="Tool output",
            source=SourceType.TOOL,
            token_count=1,
            metadata={"role": "tool"},
        )
        assert get_message_role(item) == "tool"

    def test_bogus_role_falls_back_to_user(self) -> None:
        item = ContextItem(
            content="Something",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": "bogus"},
        )
        assert get_message_role(item) == "user"

    def test_missing_role_defaults_to_user(self) -> None:
        item = ContextItem(
            content="No role key",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={},
        )
        assert get_message_role(item) == "user"

    def test_non_string_role_falls_back_to_user(self) -> None:
        """An integer role is str()-converted; '123' is not in _ALLOWED_ROLES."""
        item = ContextItem(
            content="Numeric role",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": 123},
        )
        assert get_message_role(item) == "user"


class TestClassifyWindowItems:
    """classify_window_items() partitions items by source type."""

    def test_system_items_go_to_system_parts(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Be concise.", source=SourceType.SYSTEM, token_count=5)
        )
        result = classify_window_items(window)

        assert result.system_parts == ["Be helpful.", "Be concise."]
        assert result.memory_items == []
        assert result.context_parts == []

    def test_memory_items_go_to_memory_items(self) -> None:
        window = ContextWindow(max_tokens=10000)
        item_user = ContextItem(
            content="Hello",
            source=SourceType.MEMORY,
            token_count=5,
            metadata={"role": "user"},
        )
        item_assistant = ContextItem(
            content="Hi there!",
            source=SourceType.MEMORY,
            token_count=5,
            metadata={"role": "assistant"},
        )
        window.add_item(item_user)
        window.add_item(item_assistant)
        result = classify_window_items(window)

        assert result.system_parts == []
        assert len(result.memory_items) == 2
        assert result.memory_items[0].content == "Hello"
        assert result.memory_items[1].content == "Hi there!"
        assert result.context_parts == []

    def test_retrieval_items_go_to_context_parts(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="doc about Python",
                source=SourceType.RETRIEVAL,
                token_count=5,
            )
        )
        result = classify_window_items(window)

        assert result.system_parts == []
        assert result.memory_items == []
        assert result.context_parts == ["doc about Python"]

    def test_user_and_tool_items_go_to_context_parts(self) -> None:
        """USER and TOOL source types are 'everything else' and land in context_parts."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="user question", source=SourceType.USER, token_count=5)
        )
        window.add_item(
            ContextItem(content="tool output", source=SourceType.TOOL, token_count=5)
        )
        result = classify_window_items(window)

        assert result.system_parts == []
        assert result.memory_items == []
        assert result.context_parts == ["user question", "tool output"]

    def test_mixed_items_correctly_classified(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System prompt", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="User says hi",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Assistant replies",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        window.add_item(
            ContextItem(
                content="A relevant document",
                source=SourceType.RETRIEVAL,
                token_count=5,
            )
        )
        window.add_item(
            ContextItem(content="tool result", source=SourceType.TOOL, token_count=5)
        )
        result = classify_window_items(window)

        assert result.system_parts == ["System prompt"]
        assert len(result.memory_items) == 2
        assert result.memory_items[0].content == "User says hi"
        assert result.memory_items[1].content == "Assistant replies"
        assert result.context_parts == ["A relevant document", "tool result"]

    def test_empty_window_returns_empty_results(self) -> None:
        window = ContextWindow(max_tokens=10000)
        result = classify_window_items(window)

        assert result.system_parts == []
        assert result.memory_items == []
        assert result.context_parts == []

    def test_returns_classified_items_named_tuple(self) -> None:
        window = ContextWindow(max_tokens=10000)
        result = classify_window_items(window)
        assert isinstance(result, ClassifiedItems)

    def test_memory_items_preserve_full_context_item(self) -> None:
        """Memory items retain the full ContextItem (not just content strings)."""
        window = ContextWindow(max_tokens=10000)
        item = ContextItem(
            id="mem-1",
            content="Remember this",
            source=SourceType.MEMORY,
            token_count=5,
            score=0.8,
            priority=7,
            metadata={"role": "user", "turn": 1},
        )
        window.add_item(item)
        result = classify_window_items(window)

        assert len(result.memory_items) == 1
        classified = result.memory_items[0]
        assert classified.id == "mem-1"
        assert classified.score == 0.8
        assert classified.priority == 7
        assert classified.metadata == {"role": "user", "turn": 1}


class TestEnsureAlternatingRoles:
    """ensure_alternating_roles() merges consecutive same-role messages."""

    def test_empty_list_returns_empty(self) -> None:
        assert ensure_alternating_roles([]) == []

    def test_single_message_unchanged(self) -> None:
        messages = [{"role": "user", "content": "Hello"}]
        result = ensure_alternating_roles(messages)
        assert result == [{"role": "user", "content": "Hello"}]

    def test_normal_alternation_unchanged(self) -> None:
        """user -> assistant -> user should pass through without changes."""
        messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "user", "content": "How are you?"},
        ]
        result = ensure_alternating_roles(messages)
        assert result == messages

    def test_consecutive_user_messages_merged(self) -> None:
        """user -> user should merge into a single user message."""
        messages = [
            {"role": "user", "content": "Context info"},
            {"role": "user", "content": "My question"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Context info\n\nMy question"

    def test_user_user_assistant_becomes_user_assistant(self) -> None:
        """user -> user -> assistant should become user -> assistant."""
        messages = [
            {"role": "user", "content": "Context"},
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "Answer"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Context\n\nQuestion"
        assert result[1]["role"] == "assistant"
        assert result[1]["content"] == "Answer"

    def test_three_consecutive_same_role_merged(self) -> None:
        """Three consecutive user messages merge into one."""
        messages = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
            {"role": "user", "content": "C"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 1
        assert result[0]["content"] == "A\n\nB\n\nC"

    def test_system_messages_never_merged(self) -> None:
        """System messages are passed through without merging."""
        messages = [
            {"role": "system", "content": "Instruction A"},
            {"role": "system", "content": "Instruction B"},
            {"role": "user", "content": "Hello"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 3
        assert result[0] == {"role": "system", "content": "Instruction A"}
        assert result[1] == {"role": "system", "content": "Instruction B"}
        assert result[2] == {"role": "user", "content": "Hello"}

    def test_system_between_users_prevents_merge(self) -> None:
        """A system message between two user messages keeps them separate."""
        messages = [
            {"role": "user", "content": "First user"},
            {"role": "system", "content": "System note"},
            {"role": "user", "content": "Second user"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 3

    def test_extra_keys_preserved_from_first_message(self) -> None:
        """Extra keys (e.g. cache_control) from the first message in a
        merged group are preserved."""
        messages = [
            {"role": "user", "content": "Context", "cache_control": {"type": "ephemeral"}},
            {"role": "user", "content": "Question"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 1
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[0]["content"] == "Context\n\nQuestion"

    def test_all_same_role_merged(self) -> None:
        """All messages with the same non-system role collapse to one."""
        messages = [
            {"role": "assistant", "content": "Part 1"},
            {"role": "assistant", "content": "Part 2"},
            {"role": "assistant", "content": "Part 3"},
        ]
        result = ensure_alternating_roles(messages)
        assert len(result) == 1
        assert result[0]["content"] == "Part 1\n\nPart 2\n\nPart 3"

    def test_does_not_mutate_input(self) -> None:
        """The original list and dicts are not modified."""
        original = [
            {"role": "user", "content": "A"},
            {"role": "user", "content": "B"},
        ]
        original_copy = [dict(m) for m in original]
        ensure_alternating_roles(original)
        assert original == original_copy
