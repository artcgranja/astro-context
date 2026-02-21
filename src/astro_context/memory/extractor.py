"""Memory extraction from conversation turns.

Provides a callback-based extractor that delegates to user-provided
functions. The library never calls an LLM -- users supply their own
extraction logic (which may or may not involve an LLM).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from astro_context.models.memory import MemoryEntry, MemoryType

if TYPE_CHECKING:
    from astro_context.models.memory import ConversationTurn


class CallbackExtractor:
    """Delegates memory extraction to a user-provided function.

    The user function receives a list of conversation turns and returns
    a list of plain dictionaries. Each dictionary must contain at least
    a ``"content"`` key; optional keys include ``"tags"``,
    ``"memory_type"``, ``"metadata"``, ``"relevance_score"``,
    ``"user_id"``, ``"session_id"``, and any other ``MemoryEntry`` field.

    Example user function::

        def my_extractor(turns):
            return [
                {"content": "User prefers dark mode", "tags": ["preference"]},
                {"content": "User's name is Alice", "memory_type": "semantic"},
            ]
    """

    __slots__ = ("_default_type", "_extract_fn")

    def __init__(
        self,
        extract_fn: Callable[[list[ConversationTurn]], list[dict[str, Any]]],
        default_type: MemoryType = MemoryType.SEMANTIC,
    ) -> None:
        self._extract_fn = extract_fn
        self._default_type = default_type

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]:
        """Extract memory entries from conversation turns.

        Parameters:
            turns: Recent conversation turns to extract memories from.

        Returns:
            A list of ``MemoryEntry`` objects built from the user
            function's output dictionaries.

        Raises:
            ValueError: If a returned dictionary is missing the required
                ``"content"`` key.
        """
        raw_results = self._extract_fn(turns)
        entries: list[MemoryEntry] = []

        for raw_original in raw_results:
            raw = dict(raw_original)  # defensive copy
            if "content" not in raw:
                msg = "extraction result must contain a 'content' key"
                raise ValueError(msg)

            # Resolve memory_type: use provided string/enum or fall back to default
            memory_type_raw = raw.pop("memory_type", None)
            if memory_type_raw is not None:
                memory_type = MemoryType(memory_type_raw)
            else:
                memory_type = self._default_type

            # Build the source_turns list from turn timestamps if not provided
            source_turns: list[str] = raw.pop("source_turns", [])
            if not source_turns:
                source_turns = [t.timestamp.isoformat() for t in turns]

            entry = MemoryEntry(
                content=raw.pop("content"),
                memory_type=memory_type,
                source_turns=source_turns,
                **raw,
            )
            entries.append(entry)

        return entries
