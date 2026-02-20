"""In-memory implementation of MemoryEntryStore."""

from __future__ import annotations

from astro_context.models.memory import MemoryEntry


class InMemoryEntryStore:
    """Simple in-memory store for MemoryEntry objects.

    Useful for testing and single-session applications.
    For persistence across sessions, use JsonFileEntryStore.
    """

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}

    def add(self, entry: MemoryEntry) -> None:
        self._entries[entry.id] = entry

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        query_lower = query.lower()
        scored = []
        for entry in self._entries.values():
            if query_lower in entry.content.lower():
                scored.append(entry)
        scored.sort(key=lambda e: e.relevance_score, reverse=True)
        return scored[:top_k]

    def list_all(self) -> list[MemoryEntry]:
        return list(self._entries.values())

    def delete(self, entry_id: str) -> bool:
        return self._entries.pop(entry_id, None) is not None

    def clear(self) -> None:
        self._entries.clear()
