"""Shared filtering utilities for MemoryEntry stores.

Both :class:`InMemoryEntryStore` and :class:`JsonFileMemoryStore` delegate
their ``search``, ``list_all``, ``search_filtered``, and filter-matching
logic to the functions in this module.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryEntry, MemoryType


def matches_filters(
    entry: MemoryEntry,
    *,
    user_id: str | None,
    session_id: str | None,
    memory_type_str: str | None,
    tags: list[str] | None,
    created_after: datetime | None,
    created_before: datetime | None,
) -> bool:
    """Check whether an entry passes all provided filters."""
    if user_id is not None and entry.user_id != user_id:
        return False
    if session_id is not None and entry.session_id != session_id:
        return False
    if memory_type_str is not None and str(entry.memory_type) != memory_type_str:
        return False
    if tags is not None and not all(t in entry.tags for t in tags):
        return False
    if created_after is not None and entry.created_at <= created_after:
        return False
    return not (created_before is not None and entry.created_at >= created_before)


def search_entries(
    entries: dict[str, MemoryEntry],
    query: str,
    top_k: int = 5,
) -> list[MemoryEntry]:
    """Search entries by substring match, excluding expired entries.

    Results are sorted by relevance_score (descending).
    """
    query_lower = query.lower()
    results: list[MemoryEntry] = []
    for entry in entries.values():
        if entry.is_expired:
            continue
        if query_lower in entry.content.lower():
            results.append(entry)
    results.sort(key=lambda e: e.relevance_score, reverse=True)
    return results[:top_k]


def list_all_entries(entries: dict[str, MemoryEntry]) -> list[MemoryEntry]:
    """Return all non-expired entries."""
    return [e for e in entries.values() if not e.is_expired]


def search_filtered_entries(
    entries: dict[str, MemoryEntry],
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

    Parameters
    ----------
    entries:
        The dict of id -> MemoryEntry to search over.
    query:
        Substring to match against entry content.
    top_k:
        Maximum number of results to return.
    user_id:
        Filter to entries with this user_id.
    session_id:
        Filter to entries with this session_id.
    memory_type:
        Filter to entries with this memory type (string or MemoryType).
    tags:
        Entries must contain ALL specified tags.
    created_after:
        Only entries created after this datetime.
    created_before:
        Only entries created before this datetime.

    Returns
    -------
    list[MemoryEntry]
        Matching entries sorted by relevance_score descending.
    """
    query_lower = query.lower()
    memory_type_str = str(memory_type) if memory_type is not None else None

    results: list[MemoryEntry] = []
    for entry in entries.values():
        if entry.is_expired:
            continue
        if query_lower and query_lower not in entry.content.lower():
            continue
        if not matches_filters(
            entry,
            user_id=user_id,
            session_id=session_id,
            memory_type_str=memory_type_str,
            tags=tags,
            created_after=created_after,
            created_before=created_before,
        ):
            continue
        results.append(entry)

    results.sort(key=lambda e: e.relevance_score, reverse=True)
    return results[:top_k]
