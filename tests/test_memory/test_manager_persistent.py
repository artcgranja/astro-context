"""Tests for MemoryManager persistent store integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from astro_context.memory.manager import MemoryManager
from astro_context.models.context import SourceType
from astro_context.models.memory import ConversationTurn, MemoryEntry, MemoryType
from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer


def _make_manager(
    conversation_tokens: int = 2000,
    on_evict: object = None,
    persistent_store: object = None,
) -> MemoryManager:
    """Create a MemoryManager with FakeTokenizer and optional store/callback."""
    return MemoryManager(
        conversation_tokens=conversation_tokens,
        tokenizer=FakeTokenizer(),
        on_evict=on_evict,  # type: ignore[arg-type]
        persistent_store=persistent_store,  # type: ignore[arg-type]
    )


class TestMemoryManagerOnEvict:
    """MemoryManager with on_evict parameter."""

    def test_on_evict_callback_is_called(self) -> None:
        evicted: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted.append(turns)

        mgr = _make_manager(conversation_tokens=3, on_evict=on_evict)
        mgr.add_user_message("first")
        mgr.add_assistant_message("second")
        mgr.add_user_message("three four five")

        assert len(evicted) >= 1

    def test_on_evict_receives_evicted_turns(self) -> None:
        evicted: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted.append(turns)

        mgr = _make_manager(conversation_tokens=3, on_evict=on_evict)
        mgr.add_user_message("first")
        mgr.add_assistant_message("second")
        mgr.add_user_message("three four five")

        all_evicted = [t.content for batch in evicted for t in batch]
        assert "first" in all_evicted


class TestMemoryManagerPersistentStore:
    """MemoryManager with persistent_store parameter."""

    def test_manager_with_persistent_store(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        assert mgr.persistent_store is store

    def test_manager_without_persistent_store(self) -> None:
        mgr = _make_manager()
        assert mgr.persistent_store is None


class TestMemoryManagerAddFact:
    """add_fact creates and stores MemoryEntry."""

    def test_add_fact_creates_entry(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("User prefers dark mode", tags=["preference"])
        assert isinstance(entry, MemoryEntry)
        assert entry.content == "User prefers dark mode"
        assert entry.tags == ["preference"]

    def test_add_fact_stores_in_persistent_store(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("fact content")
        stored = store.get(entry.id)
        assert stored is not None
        assert stored.content == "fact content"

    def test_add_fact_default_memory_type(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("fact")
        assert entry.memory_type == MemoryType.SEMANTIC

    def test_add_fact_custom_memory_type(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("procedure", memory_type=MemoryType.PROCEDURAL)
        assert entry.memory_type == MemoryType.PROCEDURAL

    def test_add_fact_with_metadata(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("fact", metadata={"source": "chat"})
        assert entry.metadata == {"source": "chat"}

    def test_add_fact_raises_without_persistent_store(self) -> None:
        mgr = _make_manager()
        with pytest.raises(RuntimeError, match="No persistent_store"):
            mgr.add_fact("some fact")


class TestMemoryManagerGetRelevantFacts:
    """get_relevant_facts searches the persistent store."""

    def test_search_returns_matching_entries(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("User prefers dark mode")
        mgr.add_fact("User likes Python programming")
        mgr.add_fact("The weather is sunny")

        results = mgr.get_relevant_facts("Python")
        assert len(results) >= 1
        assert any("Python" in r.content for r in results)

    def test_search_respects_top_k(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        for i in range(10):
            mgr.add_fact(f"fact number {i}")

        results = mgr.get_relevant_facts("fact", top_k=3)
        assert len(results) <= 3

    def test_returns_empty_without_store(self) -> None:
        mgr = _make_manager()
        results = mgr.get_relevant_facts("anything")
        assert results == []


class TestMemoryManagerGetAllFacts:
    """get_all_facts returns all entries from the store."""

    def test_returns_all_entries(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("fact 1")
        mgr.add_fact("fact 2")
        mgr.add_fact("fact 3")

        all_facts = mgr.get_all_facts()
        assert len(all_facts) == 3

    def test_returns_empty_without_store(self) -> None:
        mgr = _make_manager()
        assert mgr.get_all_facts() == []


class TestMemoryManagerDeleteFact:
    """delete_fact removes an entry by ID."""

    def test_delete_existing_fact(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("to be deleted")
        assert mgr.delete_fact(entry.id) is True
        assert store.get(entry.id) is None

    def test_delete_nonexistent_fact(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        assert mgr.delete_fact("nonexistent-id") is False

    def test_delete_without_store(self) -> None:
        mgr = _make_manager()
        assert mgr.delete_fact("any-id") is False


class TestMemoryManagerGetContextItemsWithPersistent:
    """get_context_items includes persistent memory items."""

    def test_includes_persistent_items(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("persistent fact")
        mgr.add_user_message("Hello")

        items = mgr.get_context_items()
        # Should have 1 persistent + 1 conversation
        assert len(items) == 2

    def test_persistent_items_have_priority_8(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("persistent fact")

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        assert len(persistent_items) == 1
        assert persistent_items[0].priority == 8

    def test_persistent_items_have_memory_source(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("fact")

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        assert persistent_items[0].source == SourceType.MEMORY

    def test_persistent_items_have_correct_metadata(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        entry = mgr.add_fact("fact", tags=["tag1"])

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        meta = persistent_items[0].metadata
        assert meta["memory_entry_id"] == entry.id
        assert meta["memory_type"] == str(MemoryType.SEMANTIC)
        assert meta["tags"] == ["tag1"]

    def test_persistent_items_have_token_count(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("one two three")

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        # FakeTokenizer: "one two three" = 3 tokens
        assert persistent_items[0].token_count == 3

    def test_persistent_items_come_before_conversation(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_user_message("Hello")
        mgr.add_fact("persistent fact")

        items = mgr.get_context_items()
        # First item should be persistent (priority 8), then conversation (priority 7)
        assert items[0].metadata.get("memory_entry_id") is not None
        assert items[1].metadata.get("role") == "user"


class TestMemoryManagerExpiredEntries:
    """Expired entries are filtered from get_context_items."""

    def test_expired_entries_excluded(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)

        # Add a non-expired entry
        mgr.add_fact("active fact")

        # Add an expired entry directly to the store
        expired_entry = MemoryEntry(
            content="expired fact",
            expires_at=datetime.now(UTC) - timedelta(hours=1),
        )
        store.add(expired_entry)

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        # Only the non-expired entry should appear
        assert len(persistent_items) == 1
        assert persistent_items[0].content == "active fact"

    def test_non_expired_entries_included(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)

        # Add an entry that expires in the future
        future_entry = MemoryEntry(
            content="future fact",
            expires_at=datetime.now(UTC) + timedelta(hours=24),
        )
        store.add(future_entry)

        items = mgr.get_context_items()
        persistent_items = [i for i in items if i.metadata.get("memory_entry_id") is not None]
        assert len(persistent_items) == 1
        assert persistent_items[0].content == "future fact"


class TestMemoryManagerClearWithPersistent:
    """clear() also clears the persistent store."""

    def test_clear_clears_persistent_store(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        mgr.add_fact("fact 1")
        mgr.add_fact("fact 2")
        mgr.add_user_message("Hello")

        mgr.clear()

        assert len(mgr.conversation.turns) == 0
        assert store.list_all() == []
        assert mgr.get_context_items() == []

    def test_clear_without_persistent_store(self) -> None:
        mgr = _make_manager()
        mgr.add_user_message("Hello")
        mgr.clear()
        assert len(mgr.conversation.turns) == 0


class TestMemoryManagerReprWithPersistent:
    """__repr__ reflects persistent store presence."""

    def test_repr_with_store(self) -> None:
        store = InMemoryEntryStore()
        mgr = _make_manager(persistent_store=store)
        r = repr(mgr)
        assert "MemoryManager" in r
        assert "yes" in r  # persistent_store='yes'

    def test_repr_without_store(self) -> None:
        mgr = _make_manager()
        r = repr(mgr)
        assert "none" in r  # persistent_store='none'
