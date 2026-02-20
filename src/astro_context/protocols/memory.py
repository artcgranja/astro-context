"""Protocols for memory extensibility.

All memory extension points are defined as ``@runtime_checkable`` Protocols
(PEP 544).  Any object with the matching method signatures satisfies the
protocol -- no inheritance required.
"""

from __future__ import annotations

from enum import StrEnum
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from astro_context.models.memory import ConversationTurn, MemoryEntry


class MemoryOperation(StrEnum):
    """Actions that a MemoryConsolidator can prescribe."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


@runtime_checkable
class CompactionStrategy(Protocol):
    """Compacts evicted conversation turns into a summary."""

    def compact(self, turns: list[ConversationTurn]) -> str: ...


@runtime_checkable
class AsyncCompactionStrategy(Protocol):
    """Async variant of CompactionStrategy."""

    async def compact(self, turns: list[ConversationTurn]) -> str: ...


@runtime_checkable
class MemoryExtractor(Protocol):
    """Extracts structured memories from conversation turns."""

    def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...


@runtime_checkable
class AsyncMemoryExtractor(Protocol):
    """Async variant of MemoryExtractor."""

    async def extract(self, turns: list[ConversationTurn]) -> list[MemoryEntry]: ...


@runtime_checkable
class MemoryConsolidator(Protocol):
    """Decides how to merge new memories with existing ones.

    Returns a list of ``(MemoryOperation, entry | None)`` tuples describing
    the actions to take.
    """

    def consolidate(
        self, new_entries: list[MemoryEntry], existing: list[MemoryEntry]
    ) -> list[tuple[MemoryOperation, MemoryEntry | None]]: ...


@runtime_checkable
class EvictionPolicy(Protocol):
    """Decides which turns to evict when memory is full.

    Returns indices of turns to evict from the provided list.
    """

    def select_for_eviction(
        self, turns: list[ConversationTurn], tokens_to_free: int
    ) -> list[int]: ...


@runtime_checkable
class MemoryDecay(Protocol):
    """Computes a retention score for a memory entry.

    Returns a float from 0.0 (forget) to 1.0 (perfect retention).
    """

    def compute_retention(self, entry: MemoryEntry) -> float: ...


@runtime_checkable
class QueryEnricher(Protocol):
    """Enriches a query with memory context before retrieval."""

    def enrich(self, query: str, memory_items: list[MemoryEntry]) -> str: ...


@runtime_checkable
class RecencyScorer(Protocol):
    """Computes recency scores for memory items.

    ``index=0`` is the oldest item, ``index=total-1`` is the newest.
    Returns a float from 0.0 to 1.0.
    """

    def score(self, index: int, total: int) -> float: ...
