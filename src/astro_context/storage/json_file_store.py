"""JSON-file-backed persistent store for MemoryEntry."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryType

from astro_context.models.memory import MemoryEntry


class JsonFileMemoryStore:
    """Persistent MemoryEntry store backed by a JSON file.

    Implements the ``MemoryEntryStore`` protocol and adds extra methods for
    filtered search, per-user deletion, data export, and explicit save/load.

    Suitable for development and single-process applications.
    Not suitable for concurrent multi-process access.

    Auto-saves to disk on every mutation (``add``, ``delete``, ``clear``).
    Call ``save()`` explicitly after bulk operations or direct ``_entries``
    manipulation if needed.

    Example::

        store = JsonFileMemoryStore("memories.json")
        store.add(MemoryEntry(content="hello"))
        entries = store.search("hello")
    """

    __slots__ = ("_dirty", "_entries", "_file_path")

    def __init__(self, file_path: str | Path) -> None:
        self._file_path = Path(file_path)
        self._entries: dict[str, MemoryEntry] = {}
        self._dirty: bool = False
        if self._file_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol methods
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Add or overwrite a memory entry and persist to disk."""
        self._entries[entry.id] = entry
        self._dirty = True
        self.save()

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search entries by substring match, excluding expired entries.

        Results are sorted by relevance_score (descending).
        """
        query_lower = query.lower()
        now = datetime.now(UTC)
        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if entry.expires_at is not None and entry.expires_at <= now:
                continue
            if query_lower in entry.content.lower():
                results.append(entry)
        results.sort(key=lambda e: e.relevance_score, reverse=True)
        return results[:top_k]

    def list_all(self) -> list[MemoryEntry]:
        """Return all non-expired entries."""
        now = datetime.now(UTC)
        return [
            e for e in self._entries.values()
            if e.expires_at is None or e.expires_at > now
        ]

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id, persist to disk. Returns True if found."""
        removed = self._entries.pop(entry_id, None) is not None
        if removed:
            self._dirty = True
            self.save()
        return removed

    def clear(self) -> None:
        """Remove all entries and persist the empty state to disk."""
        self._entries.clear()
        self._dirty = True
        self.save()

    # ------------------------------------------------------------------
    # Persistence methods
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush all entries to the JSON file on disk."""
        data: list[dict[str, Any]] = [
            entry.model_dump(mode="json") for entry in self._entries.values()
        ]
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
        self._dirty = False

    def load(self) -> None:
        """Reload entries from the JSON file on disk."""
        if not self._file_path.exists():
            self._entries.clear()
            self._dirty = False
            return

        text = self._file_path.read_text(encoding="utf-8")
        if not text.strip():
            self._entries.clear()
            self._dirty = False
            return

        raw_list: list[dict[str, Any]] = json.loads(text)
        self._entries.clear()
        for raw in raw_list:
            entry = MemoryEntry.model_validate(raw)
            self._entries[entry.id] = entry
        self._dirty = False

    # ------------------------------------------------------------------
    # Extra methods (beyond MemoryEntryStore protocol)
    # ------------------------------------------------------------------

    def get(self, entry_id: str) -> MemoryEntry | None:
        """Retrieve a single entry by id, or None if not found."""
        return self._entries.get(entry_id)

    def delete_by_user(self, user_id: str) -> int:
        """Delete all entries belonging to a user. Returns count deleted."""
        to_delete = [
            eid for eid, entry in self._entries.items()
            if entry.user_id == user_id
        ]
        for eid in to_delete:
            del self._entries[eid]
        if to_delete:
            self._dirty = True
            self.save()
        return len(to_delete)

    def export_user_entries(self, user_id: str) -> list[MemoryEntry]:
        """Export all entries belonging to a user (GDPR data portability).

        Returns a list of all entries with the given user_id, including
        expired entries (for completeness of the data export).
        """
        return [
            entry for entry in self._entries.values()
            if entry.user_id == user_id
        ]

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
        query_lower = query.lower()
        now = datetime.now(UTC)
        memory_type_str = str(memory_type) if memory_type is not None else None

        results: list[MemoryEntry] = []
        for entry in self._entries.values():
            if entry.expires_at is not None and entry.expires_at <= now:
                continue
            if query_lower and query_lower not in entry.content.lower():
                continue
            if not self._matches_filters(
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

    def __repr__(self) -> str:
        return f"{type(self).__name__}(file={self._file_path!s}, entries={len(self._entries)})"
