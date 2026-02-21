"""In-memory implementation of MemoryEntryStore with filtering support."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryType

from astro_context.models.memory import MemoryEntry
from astro_context.storage._filters import (
    list_all_entries,
    matches_filters,
    search_entries,
    search_filtered_entries,
)


class InMemoryEntryStore:
    """In-memory store for MemoryEntry objects with advanced filtering.

    Implements the ``MemoryEntryStore`` protocol and adds extra methods for
    filtered search, per-user deletion, and single-entry retrieval.

    Useful for testing and single-session applications.
    For persistence across sessions, use ``JsonFileMemoryStore``.
    """

    __slots__ = ("_entries",)

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol methods
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Add or overwrite a memory entry."""
        self._entries[entry.id] = entry

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search entries by substring match, excluding expired entries.

        Results are sorted by relevance_score (descending) as a tiebreaker.
        """
        return search_entries(self._entries, query, top_k)

    def list_all(self) -> list[MemoryEntry]:
        """Return all non-expired entries."""
        return list_all_entries(self._entries)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True if found and deleted."""
        return self._entries.pop(entry_id, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        self._entries.clear()

    # ------------------------------------------------------------------
    # Extra methods (beyond MemoryEntryStore protocol)
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by id, or None if not found."""
        return self._entries.get(entry_id)

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        """Return all entries including expired ones."""
        return list(self._entries.values())

    def delete_by_user(self, user_id: str) -> int:
        """Delete all entries belonging to a user. Returns count deleted."""
        to_delete = [
            eid for eid, entry in self._entries.items()
            if entry.user_id == user_id
        ]
        for eid in to_delete:
            del self._entries[eid]
        return len(to_delete)

    def search_filtered(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        tags: list[str] | None = None,
        created_after: datetime | None = None,
        created_before: datetime | None = None,
    ) -> list[MemoryEntry]:
        """Search entries with optional filters, excluding expired entries.

        Parameters:
            query: Substring to match against entry content.
            top_k: Maximum number of results to return.
            user_id: Filter to entries with this user_id.
            session_id: Filter to entries with this session_id.
            memory_type: Filter to entries with this memory type (string or MemoryType).
            tags: Entries must contain ALL specified tags.
            created_after: Only entries created after this datetime.
            created_before: Only entries created before this datetime.

        Returns:
            Matching entries sorted by relevance_score descending.
        """
        return search_filtered_entries(
            self._entries,
            query,
            top_k,
            user_id=user_id,
            session_id=session_id,
            memory_type=memory_type,
            tags=tags,
            created_after=created_after,
            created_before=created_before,
        )

    @staticmethod
    def _matches_filters(
        entry: MemoryEntry,
        *,
        user_id: str | None,
        session_id: str | None,
        memory_type_str: str | None,
        tags: list[str] | None,
        created_after: datetime | None,
        created_before: datetime | None,
    ) -> bool:
        """Check whether an entry passes all provided filters.

        Delegates to :func:`astro_context.storage._filters.matches_filters`.
        Kept for backwards compatibility.
        """
        return matches_filters(
            entry,
            user_id=user_id,
            session_id=session_id,
            memory_type_str=memory_type_str,
            tags=tags,
            created_after=created_after,
            created_before=created_before,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(entries={len(self._entries)})"
