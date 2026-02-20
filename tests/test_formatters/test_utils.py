"""Tests for astro_context.formatters.utils."""

from __future__ import annotations

from astro_context.formatters.utils import ClassifiedItems, classify_window_items, get_message_role
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

    def test_role_tool_falls_back_to_user(self) -> None:
        """'tool' is not in _ALLOWED_ROLES so it should fall back to 'user'."""
        item = ContextItem(
            content="Tool output",
            source=SourceType.TOOL,
            token_count=1,
            metadata={"role": "tool"},
        )
        assert get_message_role(item) == "user"

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
