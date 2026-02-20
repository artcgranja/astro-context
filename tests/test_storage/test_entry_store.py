"""Tests for astro_context.storage.json_memory_store.InMemoryEntryStore."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from astro_context.models.memory import MemoryEntry, MemoryType
from astro_context.storage.json_memory_store import InMemoryEntryStore

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
def store() -> InMemoryEntryStore:
    """Return a fresh InMemoryEntryStore."""
    return InMemoryEntryStore()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreBasicCRUD:
    """Core add / search / list_all / delete / clear operations."""

    def test_add_stores_an_entry(self, store: InMemoryEntryStore) -> None:
        entry = _make_entry(entry_id="a1", content="hello world")
        store.add(entry)
        assert store.get("a1") is not None
        assert store.get("a1") == entry

    def test_search_finds_entries_by_substring(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="s1", content="Python is great"))
        store.add(_make_entry(entry_id="s2", content="Java is verbose"))
        store.add(_make_entry(entry_id="s3", content="Python and Java"))

        results = store.search("python")
        assert len(results) == 2
        ids = {e.id for e in results}
        assert ids == {"s1", "s3"}

    def test_search_returns_top_k_results(self, store: InMemoryEntryStore) -> None:
        for i in range(10):
            store.add(_make_entry(entry_id=f"t{i}", content=f"topic {i} about memory"))
        results = store.search("memory", top_k=3)
        assert len(results) == 3

    def test_search_sorts_by_relevance_score(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="lo", content="memory low", relevance_score=0.2))
        store.add(_make_entry(entry_id="hi", content="memory high", relevance_score=0.9))
        store.add(_make_entry(entry_id="mid", content="memory mid", relevance_score=0.5))

        results = store.search("memory")
        assert [e.id for e in results] == ["hi", "mid", "lo"]

    def test_search_filters_out_expired_entries(self, store: InMemoryEntryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

        store.add(_make_entry(entry_id="expired", content="memory old", expires_at=past))
        store.add(_make_entry(entry_id="valid", content="memory new", expires_at=future))
        store.add(_make_entry(entry_id="no_expiry", content="memory forever"))

        results = store.search("memory")
        ids = {e.id for e in results}
        assert "expired" not in ids
        assert "valid" in ids
        assert "no_expiry" in ids

    def test_list_all_returns_all_non_expired_entries(self, store: InMemoryEntryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        store.add(_make_entry(entry_id="a"))
        store.add(_make_entry(entry_id="b"))
        store.add(_make_entry(entry_id="c", expires_at=past))

        all_entries = store.list_all()
        ids = {e.id for e in all_entries}
        assert ids == {"a", "b"}

    def test_delete_removes_an_entry(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="del1"))
        assert store.delete("del1") is True
        assert store.get("del1") is None

    def test_delete_returns_false_for_missing(self, store: InMemoryEntryStore) -> None:
        assert store.delete("nonexistent") is False

    def test_clear_removes_all_entries(self, store: InMemoryEntryStore) -> None:
        for i in range(5):
            store.add(_make_entry(entry_id=f"c{i}"))
        store.clear()
        assert store.list_all() == []


# ---------------------------------------------------------------------------
# Extra methods: get, delete_by_user
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreExtraMethods:
    """get() and delete_by_user() beyond the basic protocol."""

    def test_get_returns_entry_by_id(self, store: InMemoryEntryStore) -> None:
        entry = _make_entry(entry_id="g1", content="retrievable")
        store.add(entry)
        result = store.get("g1")
        assert result is not None
        assert result.content == "retrievable"

    def test_get_returns_none_for_missing_id(self, store: InMemoryEntryStore) -> None:
        assert store.get("ghost") is None

    def test_delete_by_user_deletes_all_for_user(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="u1", user_id="alice"))
        store.add(_make_entry(entry_id="u2", user_id="alice"))
        store.add(_make_entry(entry_id="u3", user_id="bob"))

        count = store.delete_by_user("alice")
        assert count == 2
        assert store.get("u1") is None
        assert store.get("u2") is None
        assert store.get("u3") is not None

    def test_delete_by_user_returns_count_of_deleted(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="x1", user_id="carol"))
        store.add(_make_entry(entry_id="x2", user_id="carol"))
        store.add(_make_entry(entry_id="x3", user_id="carol"))

        assert store.delete_by_user("carol") == 3

    def test_delete_by_user_returns_zero_for_unknown_user(
        self, store: InMemoryEntryStore
    ) -> None:
        store.add(_make_entry(entry_id="y1", user_id="dave"))
        assert store.delete_by_user("nobody") == 0


# ---------------------------------------------------------------------------
# search_filtered
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreSearchFiltered:
    """Filtered search with user_id, session_id, memory_type, tags, date range."""

    def test_filters_by_user_id(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="f1", content="data point", user_id="alice"))
        store.add(_make_entry(entry_id="f2", content="data point", user_id="bob"))

        results = store.search_filtered("data", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "f1"

    def test_filters_by_session_id(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="s1", content="info here", session_id="sess-a"))
        store.add(_make_entry(entry_id="s2", content="info here", session_id="sess-b"))

        results = store.search_filtered("info", session_id="sess-a")
        assert len(results) == 1
        assert results[0].id == "s1"

    def test_filters_by_memory_type(self, store: InMemoryEntryStore) -> None:
        store.add(
            _make_entry(entry_id="m1", content="procedure steps", memory_type=MemoryType.PROCEDURAL)
        )
        store.add(
            _make_entry(entry_id="m2", content="procedure memory", memory_type=MemoryType.SEMANTIC)
        )

        results = store.search_filtered("procedure", memory_type=MemoryType.PROCEDURAL)
        assert len(results) == 1
        assert results[0].id == "m1"

    def test_filters_by_memory_type_as_string(self, store: InMemoryEntryStore) -> None:
        store.add(
            _make_entry(entry_id="m3", content="episodic event", memory_type=MemoryType.EPISODIC)
        )
        store.add(
            _make_entry(entry_id="m4", content="episodic story", memory_type=MemoryType.SEMANTIC)
        )

        results = store.search_filtered("episodic", memory_type="episodic")
        assert len(results) == 1
        assert results[0].id == "m3"

    def test_filters_by_tags_must_match_all(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="t1", content="tagged entry", tags=["python", "ml"]))
        store.add(_make_entry(entry_id="t2", content="tagged entry", tags=["python"]))
        store.add(_make_entry(entry_id="t3", content="tagged entry", tags=["ml", "deep"]))

        results = store.search_filtered("tagged", tags=["python", "ml"])
        assert len(results) == 1
        assert results[0].id == "t1"

    def test_filters_by_created_after(self, store: InMemoryEntryStore) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        store.add(_make_entry(entry_id="old", content="historical note", created_at=old))
        store.add(_make_entry(entry_id="new", content="historical note", created_at=recent))

        results = store.search_filtered("historical", created_after=cutoff)
        assert len(results) == 1
        assert results[0].id == "new"

    def test_filters_by_created_before(self, store: InMemoryEntryStore) -> None:
        old = datetime(2024, 1, 1, tzinfo=UTC)
        recent = datetime(2025, 6, 1, tzinfo=UTC)
        cutoff = datetime(2025, 1, 1, tzinfo=UTC)

        store.add(_make_entry(entry_id="old", content="archival data", created_at=old))
        store.add(_make_entry(entry_id="new", content="archival data", created_at=recent))

        results = store.search_filtered("archival", created_before=cutoff)
        assert len(results) == 1
        assert results[0].id == "old"

    def test_combines_multiple_filters(self, store: InMemoryEntryStore) -> None:
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
                entry_id="miss_user",
                content="combined filter test",
                user_id="bob",
                session_id="s1",
                memory_type=MemoryType.EPISODIC,
                tags=["important"],
                created_at=ts,
            )
        )
        store.add(
            _make_entry(
                entry_id="miss_type",
                content="combined filter test",
                user_id="alice",
                session_id="s1",
                memory_type=MemoryType.SEMANTIC,
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

    def test_search_filtered_excludes_expired_entries(
        self, store: InMemoryEntryStore
    ) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

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
                expires_at=future,
            )
        )

        results = store.search_filtered("filtered", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "valid"

    def test_search_filtered_empty_query_matches_all(
        self, store: InMemoryEntryStore
    ) -> None:
        """An empty query string should match all entries (no substring filtering)."""
        store.add(_make_entry(entry_id="a", content="alpha", user_id="alice"))
        store.add(_make_entry(entry_id="b", content="beta", user_id="alice"))

        results = store.search_filtered("", user_id="alice")
        assert len(results) == 2


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreRepr:
    """Test __repr__ for debugging."""

    def test_repr_includes_entry_count(self, store: InMemoryEntryStore) -> None:
        store.add(_make_entry(entry_id="r1"))
        assert "1" in repr(store)
        assert "InMemoryEntryStore" in repr(store)
