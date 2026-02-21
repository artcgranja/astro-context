"""Tests for astro_context.formatters.openai."""

from __future__ import annotations

from astro_context.formatters.openai import OpenAIFormatter
from astro_context.models.context import ContextItem, ContextWindow, SourceType


class TestOpenAIFormatter:
    """OpenAIFormatter produces correct dict structure."""

    def test_format_type(self) -> None:
        formatter = OpenAIFormatter()
        assert formatter.format_type == "openai"

    def test_system_prompt_as_system_message(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        assert isinstance(output, dict)
        assert "messages" in output
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "system"
        assert output["messages"][0]["content"] == "You are helpful."

    def test_multiple_system_prompts_joined(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Be helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Be concise.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        # System parts joined into one system message
        assert output["messages"][0]["role"] == "system"
        assert output["messages"][0]["content"] == "Be helpful.\n\nBe concise."

    def test_retrieval_items_as_user_context(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc about Python", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        # Retrieval content uses 'user' role (not 'system') to prevent
        # privilege escalation from untrusted retrieved documents.
        assert len(output["messages"]) == 1
        msg = output["messages"][0]
        assert msg["role"] == "user"
        assert "Relevant context:" in msg["content"]
        assert "doc about Python" in msg["content"]

    def test_memory_items_as_conversation_messages(self) -> None:
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
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][1]["role"] == "assistant"

    def test_full_structure_ordering(self) -> None:
        """System messages come first, then retrieval context merged with user conversation."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="System prompt", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(
                content="User hello",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(content="A document", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        messages = output["messages"]

        # Should be: system prompt, then merged user message (context + conversation)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "System prompt"
        assert messages[1]["role"] == "user"
        assert "Relevant context:" in messages[1]["content"]
        assert "User hello" in messages[1]["content"]

    def test_empty_window(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        assert output == {"messages": []}

    def test_memory_default_role_is_user(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="no role",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={},
            )
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        assert output["messages"][0]["role"] == "user"

    def test_multiple_context_docs_joined(self) -> None:
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc one", source=SourceType.RETRIEVAL, token_count=5)
        )
        window.add_item(
            ContextItem(content="doc two", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        # Both docs should appear in one user context message
        context_msg = output["messages"][0]
        assert context_msg["role"] == "user"
        assert "doc one" in context_msg["content"]
        assert "doc two" in context_msg["content"]
        assert "---" in context_msg["content"]  # Separator


class TestOpenAIFormatterRoleAlternation:
    """OpenAIFormatter enforces strict user/assistant alternation."""

    def test_context_block_and_user_memory_merged(self) -> None:
        """Context block (user) + first user memory message are merged."""
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
            ContextItem(content="doc", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        # System-less: context(user) + memory(user) merged into one
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "user"
        assert "Relevant context:" in output["messages"][0]["content"]
        assert "Hello" in output["messages"][0]["content"]

    def test_consecutive_user_memory_items_merged(self) -> None:
        """user -> user -> assistant becomes user -> assistant."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="First",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Second",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        window.add_item(
            ContextItem(
                content="Reply",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 2
        assert output["messages"][0]["role"] == "user"
        assert "First" in output["messages"][0]["content"]
        assert "Second" in output["messages"][0]["content"]
        assert output["messages"][1]["role"] == "assistant"

    def test_proper_alternation_unchanged(self) -> None:
        """user -> assistant -> user passes through without merging."""
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
                content="Hi!",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "assistant"},
            )
        )
        window.add_item(
            ContextItem(
                content="How are you?",
                source=SourceType.MEMORY,
                token_count=5,
                metadata={"role": "user"},
            )
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 3
        assert output["messages"][0]["role"] == "user"
        assert output["messages"][1]["role"] == "assistant"
        assert output["messages"][2]["role"] == "user"

    def test_system_messages_not_merged(self) -> None:
        """System messages are never merged with each other or with user."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="Instruction A", source=SourceType.SYSTEM, token_count=5)
        )
        window.add_item(
            ContextItem(content="Instruction B", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        # System parts are joined into one message by the formatter itself
        # (not by ensure_alternating_roles), so we still get 1 system message
        assert len(output["messages"]) == 1
        assert output["messages"][0]["role"] == "system"

    def test_empty_window_returns_empty_messages(self) -> None:
        window = ContextWindow(max_tokens=10000)
        formatter = OpenAIFormatter()
        output = formatter.format(window)
        assert output["messages"] == []
