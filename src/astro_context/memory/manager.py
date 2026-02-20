"""Memory manager that coordinates conversation memory."""

from __future__ import annotations

from astro_context.models.context import ContextItem
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

from .sliding_window import SlidingWindowMemory


class MemoryManager:
    """Coordinates different memory strategies and produces context items.

    For MVP, this wraps SlidingWindowMemory. Future versions will add:
    - Progressive summarization (summarize old turns instead of evicting)
    - Persistent memory (long-term facts)
    - Semantic memory search
    """

    __slots__ = ("_conversation", "_tokenizer")

    def __init__(
        self,
        conversation_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if conversation_tokens <= 0:
            msg = "conversation_tokens must be a positive integer"
            raise ValueError(msg)
        self._tokenizer = tokenizer or get_default_counter()
        self._conversation = SlidingWindowMemory(
            max_tokens=conversation_tokens,
            tokenizer=self._tokenizer,
        )

    def __repr__(self) -> str:
        return (
            f"MemoryManager(conversation={self._conversation!r})"
        )

    @property
    def conversation(self) -> SlidingWindowMemory:
        """Access the underlying conversation memory."""
        return self._conversation

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self._conversation.add_turn("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self._conversation.add_turn("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation history."""
        self._conversation.add_turn("system", content)

    def add_tool_message(self, content: str) -> None:
        """Add a tool message to the conversation history."""
        self._conversation.add_turn("tool", content)

    def get_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Get all memory as context items for pipeline assembly."""
        return self._conversation.to_context_items(priority=priority)

    def clear(self) -> None:
        self._conversation.clear()
