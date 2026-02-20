"""Memory manager that coordinates conversation memory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.memory import ConversationTurn, MemoryEntry, MemoryType
from astro_context.protocols.storage import MemoryEntryStore
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

from .sliding_window import SlidingWindowMemory
from .summary_buffer import SummaryBufferMemory


class MemoryManager:
    """Coordinates different memory strategies and produces context items.

    Wraps ``SlidingWindowMemory`` or ``SummaryBufferMemory`` for conversation
    history and optionally integrates a persistent ``MemoryEntryStore`` for
    long-term facts.

    When *conversation_memory* is provided it is used directly as the
    conversation backend, and *conversation_tokens*, *tokenizer*, and
    *on_evict* are ignored.  Otherwise a new ``SlidingWindowMemory`` is
    created from those parameters (backwards-compatible default).
    """

    __slots__ = ("_conversation", "_persistent_store", "_tokenizer")

    def __init__(
        self,
        conversation_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
        on_evict: Callable[[list[ConversationTurn]], None] | None = None,
        persistent_store: MemoryEntryStore | None = None,
        conversation_memory: SlidingWindowMemory | SummaryBufferMemory | None = None,
    ) -> None:
        self._tokenizer = tokenizer or get_default_counter()
        if conversation_memory is not None:
            self._conversation: SlidingWindowMemory | SummaryBufferMemory = (
                conversation_memory
            )
        else:
            if conversation_tokens <= 0:
                msg = "conversation_tokens must be a positive integer"
                raise ValueError(msg)
            self._conversation = SlidingWindowMemory(
                max_tokens=conversation_tokens,
                tokenizer=self._tokenizer,
                on_evict=on_evict,
            )
        self._persistent_store = persistent_store

    def __repr__(self) -> str:
        has_store = self._persistent_store is not None
        return (
            f"MemoryManager(conversation={self._conversation!r}, "
            f"persistent_store={'yes' if has_store else 'none'})"
        )

    @property
    def conversation(self) -> SlidingWindowMemory | SummaryBufferMemory:
        """Access the underlying conversation memory."""
        return self._conversation

    @property
    def conversation_type(self) -> str:
        """Return the type of the underlying conversation memory.

        Returns ``"sliding_window"`` or ``"summary_buffer"``.
        """
        if isinstance(self._conversation, SummaryBufferMemory):
            return "summary_buffer"
        return "sliding_window"

    @property
    def persistent_store(self) -> MemoryEntryStore | None:
        """Access the underlying persistent memory store, if any."""
        return self._persistent_store

    # ---- Conversation helpers ----

    def _add_message(self, role: str, content: str) -> None:
        """Add a message to the conversation backend (works with both types)."""
        if isinstance(self._conversation, SummaryBufferMemory):
            self._conversation.add_message(role, content)  # type: ignore[arg-type]
        else:
            self._conversation.add_turn(role, content)  # type: ignore[arg-type]

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation history."""
        self._add_message("user", content)

    def add_assistant_message(self, content: str) -> None:
        """Add an assistant message to the conversation history."""
        self._add_message("assistant", content)

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation history."""
        self._add_message("system", content)

    def add_tool_message(self, content: str) -> None:
        """Add a tool message to the conversation history."""
        self._add_message("tool", content)

    # ---- Persistent fact management ----

    def add_fact(
        self,
        content: str,
        tags: list[str] | None = None,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Create and store a persistent memory entry.

        Parameters:
            content: The textual content of the memory.
            tags: Optional classification tags.
            memory_type: The cognitive type of the memory.
            metadata: Arbitrary key-value metadata.

        Returns:
            The newly created ``MemoryEntry``.

        Raises:
            RuntimeError: If no persistent store has been configured.
        """
        if self._persistent_store is None:
            msg = "No persistent_store configured on this MemoryManager"
            raise RuntimeError(msg)
        entry = MemoryEntry(
            content=content,
            tags=tags or [],
            memory_type=memory_type,
            metadata=metadata or {},
        )
        self._persistent_store.add(entry)
        return entry

    def get_relevant_facts(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search the persistent store for entries relevant to *query*.

        Returns an empty list when no persistent store is configured.
        """
        if self._persistent_store is None:
            return []
        return self._persistent_store.search(query, top_k=top_k)

    def get_all_facts(self) -> list[MemoryEntry]:
        """Return every entry in the persistent store.

        Returns an empty list when no persistent store is configured.
        """
        if self._persistent_store is None:
            return []
        return self._persistent_store.list_all()

    def delete_fact(self, entry_id: str) -> bool:
        """Delete a persistent memory entry by id.

        Returns ``False`` when no persistent store is configured or the
        entry does not exist.
        """
        if self._persistent_store is None:
            return False
        return self._persistent_store.delete(entry_id)

    # ---- Context assembly ----

    def get_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Get all memory as context items for pipeline assembly.

        Persistent memory facts are included at priority 8 (between
        system=10 and conversation=7).  Conversation turns use the
        caller-supplied *priority* (default 7).
        """
        items: list[ContextItem] = []

        # Persistent facts first (higher priority)
        if self._persistent_store is not None:
            for entry in self._persistent_store.list_all():
                if entry.is_expired:
                    continue
                token_count = self._tokenizer.count_tokens(entry.content)
                item = ContextItem(
                    content=entry.content,
                    source=SourceType.MEMORY,
                    score=entry.relevance_score,
                    priority=8,
                    token_count=token_count,
                    metadata={
                        "memory_entry_id": entry.id,
                        "memory_type": str(entry.memory_type),
                        "tags": entry.tags,
                    },
                    created_at=entry.created_at,
                )
                items.append(item)

        # Conversation turns
        items.extend(self._conversation.to_context_items(priority=priority))
        return items

    def clear(self) -> None:
        """Clear conversation history and persistent store (if present)."""
        self._conversation.clear()
        if self._persistent_store is not None:
            self._persistent_store.clear()
