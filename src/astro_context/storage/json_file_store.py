"""JSON-file-backed persistent store for MemoryEntry."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

from astro_context.exceptions import StorageError
from astro_context.models.memory import MemoryEntry
from astro_context.storage._base import BaseEntryStoreMixin

logger = logging.getLogger(__name__)


class JsonFileMemoryStore(BaseEntryStoreMixin):
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
    # Mutation hook (called by BaseEntryStoreMixin.delete_by_user)
    # ------------------------------------------------------------------

    def _after_mutation(self) -> None:
        """Persist changes to disk when auto-save is enabled."""
        self._dirty = True
        if self._auto_save:
            self._maybe_save()

    # ------------------------------------------------------------------
    # MemoryEntryStore protocol methods
    # ------------------------------------------------------------------

    def add(self, entry: MemoryEntry) -> None:
        """Add or overwrite a memory entry and optionally persist to disk."""
        self._entries[entry.id] = entry
        self._dirty = True
        if self._auto_save:
            self._maybe_save()

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

    def export_user_entries(self, user_id: str) -> list[MemoryEntry]:
        """Export all entries belonging to a user (GDPR data portability).

        Returns a list of all entries with the given user_id, including
        expired entries (for completeness of the data export).
        """
        return [
            entry for entry in self._entries.values()
            if entry.user_id == user_id
        ]

    def __repr__(self) -> str:
        return f"{type(self).__name__}(file={self._file_path!s}, entries={len(self._entries)})"
