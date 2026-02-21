"""In-memory implementation of MemoryEntryStore with filtering support."""

from __future__ import annotations

import threading

from astro_context.models.memory import MemoryEntry
from astro_context.storage._base import BaseEntryStoreMixin


class InMemoryEntryStore(BaseEntryStoreMixin):
    """In-memory store for MemoryEntry objects with advanced filtering.

    Implements the ``MemoryEntryStore`` protocol and adds extra methods for
    filtered search, per-user deletion, and single-entry retrieval.

    Useful for testing and single-session applications.
    For persistence across sessions, use ``JsonFileMemoryStore``.
    """

    __slots__ = ("_entries", "_lock")

    def __init__(self) -> None:
        self._entries: dict[str, MemoryEntry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol methods
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Add or overwrite a memory entry."""
        with self._lock:
            self._entries[entry.id] = entry

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id. Returns True if found and deleted."""
        with self._lock:
            return self._entries.pop(entry_id, None) is not None

    def clear(self) -> None:
        """Remove all entries."""
        with self._lock:
            self._entries.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(entries={len(self._entries)})"
