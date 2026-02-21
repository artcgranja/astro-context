"""Tests specific to astro_context.storage.json_file_store.JsonFileMemoryStore.

Shared behavioral tests (search, filtering, CRUD) have been moved to
``test_entry_store_shared.py`` which runs against both store implementations.
This file retains only JsonFileMemoryStore-specific tests: persistence,
file operations, export, and repr.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from astro_context.storage.json_file_store import JsonFileMemoryStore
from tests.conftest import make_memory_entry as _make_entry


@pytest.fixture
def store_path(tmp_path: Path) -> Path:
    """Return a path for a temporary JSON file."""
    return tmp_path / "memories.json"


@pytest.fixture
def store(store_path: Path) -> JsonFileMemoryStore:
    """Return a fresh JsonFileMemoryStore backed by a temp file."""
    return JsonFileMemoryStore(store_path)


# ---------------------------------------------------------------------------
# File creation and persistence basics
# ---------------------------------------------------------------------------


class TestJsonFileStoreFileOperations:
    """File creation, loading, and persistence on mutations."""

    def test_creates_file_on_first_save(self, store_path: Path) -> None:
        assert not store_path.exists()
        jstore = JsonFileMemoryStore(store_path)
        jstore.add(_make_entry(entry_id="f1"))
        assert store_path.exists()

    def test_loads_entries_from_existing_file(self, store_path: Path) -> None:
        # Create store, add entry, let auto-save write file
        jstore1 = JsonFileMemoryStore(store_path)
        jstore1.add(_make_entry(entry_id="persist1", content="persistent data"))

        # Create a second store from the same file
        jstore2 = JsonFileMemoryStore(store_path)
        result = jstore2.get("persist1")
        assert result is not None
        assert result.content == "persistent data"

    def test_add_persists_to_disk(self, store: JsonFileMemoryStore, store_path: Path) -> None:
        store.add(_make_entry(entry_id="d1", content="disk entry"))
        assert store_path.exists()

        # Read back via a new store instance
        new_store = JsonFileMemoryStore(store_path)
        assert new_store.get("d1") is not None

    def test_delete_persists_to_disk(self, store: JsonFileMemoryStore, store_path: Path) -> None:
        store.add(_make_entry(entry_id="d2", content="to be deleted"))
        store.delete("d2")

        new_store = JsonFileMemoryStore(store_path)
        assert new_store.get("d2") is None

    def test_clear_persists_to_disk(self, store: JsonFileMemoryStore, store_path: Path) -> None:
        store.add(_make_entry(entry_id="c1"))
        store.add(_make_entry(entry_id="c2"))
        store.clear()

        new_store = JsonFileMemoryStore(store_path)
        assert new_store.list_all() == []

    def test_roundtrip_add_reload(self, store_path: Path) -> None:
        """Add entries, create a new store from same file, entries survive."""
        store1 = JsonFileMemoryStore(store_path)
        store1.add(_make_entry(entry_id="rt1", content="round trip 1"))
        store1.add(_make_entry(entry_id="rt2", content="round trip 2"))

        store2 = JsonFileMemoryStore(store_path)
        ids = {e.id for e in store2.list_all()}
        assert ids == {"rt1", "rt2"}

    def test_explicit_save_and_load(self, store: JsonFileMemoryStore) -> None:
        entry = _make_entry(entry_id="sl1", content="save-load")
        store._entries[entry.id] = entry
        store.save()

        store._entries.clear()
        assert store.get("sl1") is None  # cleared in-memory

        store.load()
        assert store.get("sl1") is not None
        assert store.get("sl1").content == "save-load"  # type: ignore[union-attr]

    def test_handles_missing_file_gracefully(self, tmp_path: Path) -> None:
        """Opening a store with a non-existent file yields an empty store."""
        missing = tmp_path / "no_such_file.json"
        jstore = JsonFileMemoryStore(missing)
        assert jstore.list_all() == []

    def test_creates_parent_directories_on_save(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c" / "memories.json"
        assert not nested.parent.exists()

        jstore = JsonFileMemoryStore(nested)
        jstore.add(_make_entry(entry_id="nested1"))
        assert nested.exists()


# ---------------------------------------------------------------------------
# delete_by_user persistence
# ---------------------------------------------------------------------------


class TestJsonFileStoreDeleteByUserPersistence:
    """delete_by_user() should persist changes to disk via _after_mutation."""

    def test_delete_by_user_works_and_persists(
        self, store: JsonFileMemoryStore, store_path: Path
    ) -> None:
        store.add(_make_entry(entry_id="u1", user_id="alice"))
        store.add(_make_entry(entry_id="u2", user_id="alice"))
        store.add(_make_entry(entry_id="u3", user_id="bob"))

        count = store.delete_by_user("alice")
        assert count == 2

        # Verify persistence
        new_store = JsonFileMemoryStore(store_path)
        assert new_store.get("u1") is None
        assert new_store.get("u2") is None
        assert new_store.get("u3") is not None


# ---------------------------------------------------------------------------
# export_user_entries (JsonFileMemoryStore-specific)
# ---------------------------------------------------------------------------


class TestJsonFileStoreExportUserEntries:
    """export_user_entries() is specific to JsonFileMemoryStore."""

    def test_export_user_entries_returns_all_for_user(
        self, store: JsonFileMemoryStore
    ) -> None:
        store.add(_make_entry(entry_id="eu1", user_id="alice", content="memory A"))
        store.add(_make_entry(entry_id="eu2", user_id="alice", content="memory B"))
        store.add(_make_entry(entry_id="eu3", user_id="bob", content="memory C"))

        exported = store.export_user_entries("alice")
        assert len(exported) == 2
        ids = {e.id for e in exported}
        assert ids == {"eu1", "eu2"}

    def test_export_user_entries_includes_expired(self, store: JsonFileMemoryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        store.add(
            _make_entry(entry_id="ee1", user_id="alice", content="expired one", expires_at=past)
        )
        store.add(_make_entry(entry_id="ee2", user_id="alice", content="active one"))

        exported = store.export_user_entries("alice")
        assert len(exported) == 2
        ids = {e.id for e in exported}
        assert "ee1" in ids


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestJsonFileStoreRepr:
    """Test __repr__ for debugging."""

    def test_repr_includes_file_path(self, store: JsonFileMemoryStore) -> None:
        r = repr(store)
        assert "JsonFileMemoryStore" in r
        assert "memories.json" in r
