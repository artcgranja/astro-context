"""Security-focused tests for context formatters.

These tests verify that untrusted content cannot escalate to privileged
roles and that retrieval content is placed in safe message positions.
"""

from __future__ import annotations

from astro_context.formatters.openai import OpenAIFormatter
from astro_context.formatters.utils import get_message_role
from astro_context.models.context import ContextItem, ContextWindow, SourceType


class TestOpenAIRetrievalRole:
    """Retrieval content in the OpenAI formatter must use the 'user' role."""

    def test_retrieval_content_uses_user_role(self) -> None:
        """Retrieval context must be emitted as a 'user' message, not 'system'.

        Untrusted documents placed in a system message could gain elevated
        authority in the LLM prompt, enabling privilege escalation.
        """
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(
                content="A retrieved document",
                source=SourceType.RETRIEVAL,
                token_count=5,
            )
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        assert len(output["messages"]) == 1
        msg = output["messages"][0]
        assert msg["role"] == "user", (
            "Retrieval content must use 'user' role, not 'system'"
        )
        assert "Relevant context:" in msg["content"]
        assert "A retrieved document" in msg["content"]

    def test_multiple_retrieval_docs_use_user_role(self) -> None:
        """Multiple retrieval docs joined into one message still use 'user' role."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="doc one", source=SourceType.RETRIEVAL, token_count=5)
        )
        window.add_item(
            ContextItem(content="doc two", source=SourceType.RETRIEVAL, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        context_msg = output["messages"][0]
        assert context_msg["role"] == "user"
        assert "doc one" in context_msg["content"]
        assert "doc two" in context_msg["content"]

    def test_system_prompt_still_uses_system_role(self) -> None:
        """System prompts (SourceType.SYSTEM) should still use 'system' role."""
        window = ContextWindow(max_tokens=10000)
        window.add_item(
            ContextItem(content="You are helpful.", source=SourceType.SYSTEM, token_count=5)
        )
        formatter = OpenAIFormatter()
        output = formatter.format(window)

        assert output["messages"][0]["role"] == "system"


class TestRoleEscalationGuard:
    """get_message_role() must prevent role escalation from untrusted sources."""

    def test_conversation_item_system_role_downgraded(self) -> None:
        """A CONVERSATION item with metadata role='system' must be downgraded to 'user'."""
        item = ContextItem(
            content="I am a conversation turn",
            source=SourceType.CONVERSATION,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "user"

    def test_memory_item_system_role_downgraded(self) -> None:
        """A MEMORY item with metadata role='system' must be downgraded to 'user'."""
        item = ContextItem(
            content="A memory fact",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "user"

    def test_retrieval_item_system_role_downgraded(self) -> None:
        """A RETRIEVAL item with metadata role='system' must be downgraded to 'user'."""
        item = ContextItem(
            content="Retrieved document",
            source=SourceType.RETRIEVAL,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "user"

    def test_system_item_system_role_allowed(self) -> None:
        """A SYSTEM item with metadata role='system' must be allowed through."""
        item = ContextItem(
            content="You are helpful.",
            source=SourceType.SYSTEM,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "system"

    def test_conversation_item_tool_role_downgraded(self) -> None:
        """A CONVERSATION item with metadata role='tool' must be downgraded to 'user'."""
        item = ContextItem(
            content="Conversation pretending to be tool",
            source=SourceType.CONVERSATION,
            token_count=1,
            metadata={"role": "tool"},
        )
        assert get_message_role(item) == "user"

    def test_memory_item_tool_role_downgraded(self) -> None:
        """A MEMORY item with metadata role='tool' must be downgraded to 'user'."""
        item = ContextItem(
            content="Memory pretending to be tool",
            source=SourceType.MEMORY,
            token_count=1,
            metadata={"role": "tool"},
        )
        assert get_message_role(item) == "user"

    def test_tool_item_tool_role_allowed(self) -> None:
        """A TOOL item with metadata role='tool' must be allowed through."""
        item = ContextItem(
            content="Tool output",
            source=SourceType.TOOL,
            token_count=1,
            metadata={"role": "tool"},
        )
        assert get_message_role(item) == "tool"

    def test_tool_item_system_role_downgraded(self) -> None:
        """A TOOL item with metadata role='system' must be downgraded to 'user'."""
        item = ContextItem(
            content="Tool trying system role",
            source=SourceType.TOOL,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "user"

    def test_conversation_user_role_allowed(self) -> None:
        """A CONVERSATION item with 'user' role must pass through normally."""
        item = ContextItem(
            content="Hello",
            source=SourceType.CONVERSATION,
            token_count=1,
            metadata={"role": "user"},
        )
        assert get_message_role(item) == "user"

    def test_conversation_assistant_role_allowed(self) -> None:
        """A CONVERSATION item with 'assistant' role must pass through normally."""
        item = ContextItem(
            content="Hi there!",
            source=SourceType.CONVERSATION,
            token_count=1,
            metadata={"role": "assistant"},
        )
        assert get_message_role(item) == "assistant"

    def test_user_source_system_role_downgraded(self) -> None:
        """A USER source item with metadata role='system' must be downgraded."""
        item = ContextItem(
            content="User input",
            source=SourceType.USER,
            token_count=1,
            metadata={"role": "system"},
        )
        assert get_message_role(item) == "user"
