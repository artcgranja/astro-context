"""JSON-file-backed persistent store for MemoryEntry."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryType

from astro_context.exceptions import StorageError
from astro_context.models.memory import MemoryEntry
from astro_context.storage._filters import (
    list_all_entries,
    matches_filters,
    search_entries,
    search_filtered_entries,
)

logger = logging.getLogger(__name__)


class JsonFileMemoryStore:
    """Persistent MemoryEntry store backed by a JSON file.

    Implements the ``MemoryEntryStore`` protocol and adds extra methods for
    filtered search, per-user deletion, data export, and explicit save/load.

    Suitable for development and single-process applications.
    Not suitable for concurrent multi-process access.

    By default (``auto_save=True``), mutations (``add``, ``delete``, ``clear``)
    are persisted to disk immediately.  Set ``auto_save=False`` for batch
    operations and call ``save()`` explicitly when ready.

    Example::

        store = JsonFileMemoryStore("memories.json")
        store.add(MemoryEntry(content="hello"))
        entries = store.search("hello")
    """

    __slots__ = ("_auto_save", "_dirty", "_entries", "_file_path")

    def __init__(self, file_path: str | Path, *, auto_save: bool = True) -> None:
        self._file_path = Path(file_path).resolve()
        self._entries: dict[str, MemoryEntry] = {}
        self._dirty: bool = False
        self._auto_save: bool = auto_save
        if self._file_path.exists():
            self.load()

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol methods
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Add or overwrite a memory entry and optionally persist to disk."""
        self._entries[entry.id] = entry
        self._dirty = True
        if self._auto_save:
            self._maybe_save()

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search entries by substring match, excluding expired entries.

        Results are sorted by relevance_score (descending).
        """
        return search_entries(self._entries, query, top_k)

    def list_all(self) -> list[MemoryEntry]:
        """Return all non-expired entries."""
        return list_all_entries(self._entries)

    def delete(self, entry_id: str) -> bool:
        """Delete an entry by id, persist to disk. Returns True if found."""
        removed = self._entries.pop(entry_id, None) is not None
        if removed:
            self._dirty = True
            if self._auto_save:
                self._maybe_save()
        return removed

    def clear(self) -> None:
        """Remove all entries and persist the empty state to disk."""
        self._entries.clear()
        self._dirty = True
        if self._auto_save:
            self._maybe_save()

    # ------------------------------------------------------------------
    # Persistence methods
    # ------------------------------------------------------------------

    def save(self) -> None:
        """Flush all entries to the JSON file on disk.

        Uses atomic write (temp file + rename) to prevent corruption.
        Always writes when called explicitly; use internal ``_maybe_save()``
        for dirty-flag-aware auto-saving.
        """
        data: list[dict[str, Any]] = [
            entry.model_dump(mode="json") for entry in self._entries.values()
        ]
        self._file_path.parent.mkdir(parents=True, exist_ok=True)

        content = json.dumps(data, indent=2, default=str)

        fd, tmp_path = tempfile.mkstemp(dir=self._file_path.parent, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            Path(tmp_path).replace(self._file_path)
        except BaseException:
            Path(tmp_path).unlink(missing_ok=True)
            raise

        self._dirty = False

    def _maybe_save(self) -> None:
        """Write to disk only if in-memory state has changed since the last save."""
        if self._dirty:
            self.save()

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

        try:
            raw_list: list[dict[str, Any]] = json.loads(text)
        except json.JSONDecodeError as e:
            msg = f"Failed to load memory store from {self._file_path}: invalid JSON"
            raise StorageError(msg) from e

        self._entries.clear()
        for raw in raw_list:
            try:
                entry = MemoryEntry.model_validate(raw)
            except Exception:
                logger.warning(
                    "Skipping malformed entry in %s: %s",
                    self._file_path,
                    raw.get("id", "<unknown>"),
                )
                continue
            self._entries[entry.id] = entry
        self._dirty = False

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
        if to_delete:
            self._dirty = True
            if self._auto_save:
                self._maybe_save()
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
        return f"{type(self).__name__}(file={self._file_path!s}, entries={len(self._entries)})"
