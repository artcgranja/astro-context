"""Tests for astro_context.formatters.anthropic."""

from __future__ import annotations

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.models.context import ContextItem, ContextWindow, SourceType


class TestAnthropicFormatter:
    """AnthropicFormatter produces correct dict structure."""

    def test_format_type(self) -> None:
        formatter = AnthropicFormatter()
        assert formatter.format_type == "anthropic"

    def test_system_prompt(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert isinstance(output, dict)
        assert output["system"] == "You are helpful."
        assert output["messages"] == []

    def test_multiple_system_prompts_joined(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Be concise.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["system"] == "Be helpful.\n\nBe concise."

    def test_memory_items_become_messages(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Hi there!",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][0]["content"] == "Hello"
        assert output["messages"][1]["role"] == "assistant"

    def test_retrieval_items_become_context_message(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert "doc about Python" in output["messages"][0]["content"]

    def test_context_message_inserted_before_memory(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="user msg",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        # Context message should be inserted at index 0
        assert output["messages"][0]["content"].startswith("Here is relevant context:")
        assert output["messages"][1]["content"] == "user msg"

    def test_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output == {"system": "", "messages": []}

    def test_memory_default_role_is_user(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="no role specified",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={},  # No role key
            )
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["messages"][0]["role"] == "user"

    def test_full_structure(self) -> None:
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
            ContextItem(content="A relevant document", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)

        assert output["system"] == "System prompt"
        assert len(output["messages"]) == 2
        # Context block first, then conversation
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert output["messages"][1]["content"] == "User says hi"
