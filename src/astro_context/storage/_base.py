"""Base mixin for MemoryEntry stores.

Provides shared query, filtering, and lookup methods so that
:class:`InMemoryEntryStore` and :class:`JsonFileMemoryStore` need only
implement mutation + persistence logic.

Subclasses must define ``self._entries: dict[str, MemoryEntry]`` before
calling any mixin method.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from astro_context.storage._filters import (
    list_all_entries,
    matches_filters,
    search_entries,
    search_filtered_entries,
)

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryEntry, MemoryType


class BaseEntryStoreMixin:
    """Mixin providing shared read/query/delete-by-user methods.

    Concrete stores inherit from this and implement ``add()``, ``delete()``,
    ``clear()``, ``__init__()`` (to create ``self._entries``), and optionally
    override ``_after_mutation()`` to persist changes.

    The ``_after_mutation()`` hook is called after any mixin method that
    modifies ``self._entries`` (currently ``delete_by_user``).  The default
    implementation is a no-op; file-backed stores override it to flush to disk.
    """

    # Declared here for type-checkers; concrete classes actually create it.
    _entries: dict[str, MemoryEntry]

    # ------------------------------------------------------------------
    # Mutation hook
    # ------------------------------------------------------------------

    def _after_mutation(self) -> None:
        """Hook called after the mixin mutates ``_entries``.

        Override in subclasses to trigger persistence (e.g. auto-save).
        The default implementation does nothing.
        """

    # ------------------------------------------------------------------
    # Read / query methods
    # ------------------------------------------------------------------

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search entries by substring match, excluding expired entries.

        Results are sorted by relevance_score (descending) as a tiebreaker.
        """
        return search_entries(self._entries, query, top_k)

    def list_all(self) -> list[MemoryEntry]:
        """Return all non-expired entries."""
        return list_all_entries(self._entries)

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by id, or ``None`` if not found."""
        return self._entries.get(entry_id)

    def list_all_unfiltered(self) -> list[MemoryEntry]:
        """Return all entries including expired ones."""
        return list(self._entries.values())

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

    # ------------------------------------------------------------------
    # Mutation methods delegated from concrete stores
    # ------------------------------------------------------------------

    def delete_by_user(self, user_id: str) -> int:
        """Delete all entries belonging to a user. Returns count deleted."""
        to_delete = [
            eid for eid, entry in self._entries.items()
            if entry.user_id == user_id
        ]
        for eid in to_delete:
            del self._entries[eid]
        if to_delete:
            self._after_mutation()
        return len(to_delete)

    # ------------------------------------------------------------------
    # Static helpers (kept for backwards compatibility)
    # ------------------------------------------------------------------

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
