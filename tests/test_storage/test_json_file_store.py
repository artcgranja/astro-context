"""Tests for astro_context.storage.json_file_store.JsonFileMemoryStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from astro_context.models.memory import MemoryEntry, MemoryType
from astro_context.storage.json_file_store import JsonFileMemoryStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    entry_id: str = "e1",
    content: str = "some memory content",
    relevance_score: float = 0.5,
    user_id: str | None = None,
    session_id: str | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    tags: list[str] | None = None,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryEntry:
    """Build a MemoryEntry with sensible test defaults."""
    kwargs: dict = {
        "id": entry_id,
        "content": content,
        "relevance_score": relevance_score,
        "memory_type": memory_type,
    }
    if user_id is not None:
        kwargs["user_id"] = user_id
    if session_id is not None:
        kwargs["session_id"] = session_id
    if tags is not None:
        kwargs["tags"] = tags
    if created_at is not None:
        kwargs["created_at"] = created_at
    if expires_at is not None:
        kwargs["expires_at"] = expires_at
    return MemoryEntry(**kwargs)


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
# search (delegates to same logic as InMemoryEntryStore)
# ---------------------------------------------------------------------------


class TestJsonFileStoreSearch:
    """Search works the same as the in-memory variant."""

    def test_search_finds_substring(self, store: JsonFileMemoryStore) -> None:
        store.add(_make_entry(entry_id="s1", content="Python rocks"))
        store.add(_make_entry(entry_id="s2", content="Java is fine"))

        results = store.search("python")
        assert len(results) == 1
        assert results[0].id == "s1"

    def test_search_respects_top_k(self, store: JsonFileMemoryStore) -> None:
        for i in range(10):
            store.add(_make_entry(entry_id=f"k{i}", content=f"memory item {i}"))
        results = store.search("memory", top_k=3)
        assert len(results) == 3

    def test_search_excludes_expired(self, store: JsonFileMemoryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        store.add(_make_entry(entry_id="exp", content="memory expired", expires_at=past))
        store.add(_make_entry(entry_id="ok", content="memory valid"))

        results = store.search("memory")
        ids = {e.id for e in results}
        assert "exp" not in ids
        assert "ok" in ids


# ---------------------------------------------------------------------------
# get, delete_by_user
# ---------------------------------------------------------------------------


class TestJsonFileStoreExtraMethods:
    """get(), delete_by_user(), and export_user_entries()."""

    def test_get_returns_entry_by_id(self, store: JsonFileMemoryStore) -> None:
        store.add(_make_entry(entry_id="g1", content="get me"))
        result = store.get("g1")
        assert result is not None
        assert result.content == "get me"

    def test_get_returns_none_for_missing(self, store: JsonFileMemoryStore) -> None:
        assert store.get("no-id") is None

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

    def test_delete_by_user_returns_zero_when_no_match(
        self, store: JsonFileMemoryStore
    ) -> None:
        store.add(_make_entry(entry_id="x1", user_id="alice"))
        assert store.delete_by_user("unknown") == 0

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
# search_filtered
# ---------------------------------------------------------------------------


class TestJsonFileStoreSearchFiltered:
    """Filtered search with all filter types."""

    def test_filters_by_user_id(self, store: JsonFileMemoryStore) -> None:
        store.add(_make_entry(entry_id="f1", content="data point", user_id="alice"))
        store.add(_make_entry(entry_id="f2", content="data point", user_id="bob"))

        results = store.search_filtered("data", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_filters_by_session_id(self, store: JsonFileMemoryStore) -> None:
        store.add(_make_entry(entry_id="s1", content="info x", session_id="sess-a"))
        store.add(_make_entry(entry_id="s2", content="info x", session_id="sess-b"))

        results = store.search_filtered("info", session_id="sess-a")
        assert len(results) == 1
        assert results[0].id == "s1"

    def test_filters_by_memory_type(self, store: JsonFileMemoryStore) -> None:
        store.add(
            _make_entry(
                entry_id="mt1", content="procedure steps", memory_type=MemoryType.PROCEDURAL
            )
        )
        store.add(
            _make_entry(entry_id="mt2", content="procedure info", memory_type=MemoryType.SEMANTIC)
        )

        results = store.search_filtered("procedure", memory_type=MemoryType.PROCEDURAL)
        assert len(results) == 1
        assert results[0].id == "mt1"

    def test_filters_by_tags_must_match_all(self, store: JsonFileMemoryStore) -> None:
        store.add(_make_entry(entry_id="tg1", content="tagged item", tags=["python", "ml"]))
        store.add(_make_entry(entry_id="tg2", content="tagged item", tags=["python"]))

        results = store.search_filtered("tagged", tags=["python", "ml"])
        assert len(results) == 1
        assert results[0].id == "tg1"

    def test_filters_by_created_after(self, store: JsonFileMemoryStore) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        store.add(_make_entry(entry_id="old", content="note here", created_at=old))
        store.add(_make_entry(entry_id="new", content="note here", created_at=recent))

        results = store.search_filtered("note", created_after=cutoff)
        assert len(results) == 1
        assert results[0].id == "new"

    def test_filters_by_created_before(self, store: JsonFileMemoryStore) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        store.add(_make_entry(entry_id="old", content="archival data", created_at=old))
        store.add(_make_entry(entry_id="new", content="archival data", created_at=recent))

        results = store.search_filtered("archival", created_before=cutoff)
        assert len(results) == 1
        assert results[0].id == "old"

    def test_combines_multiple_filters(self, store: JsonFileMemoryStore) -> None:
        ts = datetime(2025, 3, 15, tzinfo=UTC)
        store.add(
            _make_entry(
                entry_id="hit",
                content="combined filter test",
                user_id="alice",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                tags=["important"],
                created_at=ts,
            )
        )
        store.add(
            _make_entry(
                entry_id="miss",
                content="combined filter test",
                user_id="bob",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                tags=["important"],
                created_at=ts,
            )
        )

        results = store.search_filtered(
            "combined",
            user_id="alice",
            session_id="s1",
            memory_type=MemoryType.EPISODIC,
            tags=["important"],
            created_after=datetime(2025, 1, 1, tzinfo=UTC),
            created_before=datetime(2025, 12, 31, tzinfo=UTC),
        )
        assert len(results) == 1
        assert results[0].id == "hit"

    def test_search_filtered_excludes_expired(self, store: JsonFileMemoryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        store.add(
            _make_entry(
                entry_id="expired",
                content="filtered content",
                user_id="alice",
                expires_at=past,
            )
        )
        store.add(
            _make_entry(
                entry_id="valid",
                content="filtered content",
                user_id="alice",
            )
        )

        results = store.search_filtered("filtered", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "valid"


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestJsonFileStoreRepr:
    """Test __repr__ for debugging."""

    def test_repr_includes_file_path(self, store: JsonFileMemoryStore) -> None:
        r = repr(store)
        assert "JsonFileMemoryStore" in r
        assert "memories.json" in r
