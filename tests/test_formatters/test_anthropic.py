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
        assert output["system"] == [{"type": "text", "text": "You are helpful."}]
        assert output["messages"] == []

    def test_multiple_system_prompts_separate_blocks(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Be concise.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter()
        output = formatter.format(window)
        assert output["system"] == [
            {"type": "text", "text": "Be helpful."},
            {"type": "text", "text": "Be concise."},
        ]

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
        assert output == {"system": [], "messages": []}

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

        assert output["system"] == [{"type": "text", "text": "System prompt"}]
        assert len(output["messages"]) == 2
        # Context block first, then conversation
        assert "Here is relevant context:" in output["messages"][0]["content"]
        assert output["messages"][1]["content"] == "User says hi"


class TestAnthropicFormatterCaching:
    """AnthropicFormatter with enable_caching=True adds cache_control."""

    def test_caching_adds_cache_control_to_system(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        assert output["system"] == [
            {
                "type": "text",
                "text": "You are helpful.",
                "cache_control": {"type": "ephemeral"},
            }
        ]

    def test_caching_adds_cache_control_to_context_message(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        assert output["messages"][0]["cache_control"] == {"type": "ephemeral"}

    def test_caching_disabled_no_cache_control(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = AnthropicFormatter(enable_caching=False)
        output = formatter.format(window)
        assert "cache_control" not in output["system"][0]
        assert "cache_control" not in output["messages"][0]

    def test_caching_full_structure(self) -> None:
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
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)

        # System block has cache_control
        assert output["system"][0]["cache_control"] == {"type": "ephemeral"}
        # Context message has cache_control
        assert output["messages"][0]["cache_control"] == {"type": "ephemeral"}
        # Regular user message does NOT have cache_control
        assert "cache_control" not in output["messages"][1]

    def test_caching_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        assert output == {"system": [], "messages": []}

    def test_caching_no_context_only_system(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="Hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        formatter = AnthropicFormatter(enable_caching=True)
        output = formatter.format(window)
        # System has cache_control
        assert output["system"][0]["cache_control"] == {"type": "ephemeral"}
        # User message does NOT have cache_control (it's not a context message)
        assert "cache_control" not in output["messages"][0]
