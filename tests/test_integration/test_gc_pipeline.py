"""Integration tests for GC + pipeline lifecycle.

Tests the full lifecycle:
1. Create MemoryManager with InMemoryEntryStore as persistent_store
2. Add several facts via manager.add_fact()
3. Set some facts to be expired (expires_at in the past)
4. Create MemoryGarbageCollector with the store
5. Run gc.collect() -- verify expired facts are pruned
6. Create a ContextPipeline with the memory manager
7. Run pipeline.build() -- verify only surviving facts appear in context
8. Verify expired facts are NOT in the result
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from astro_context.memory.gc import MemoryGarbageCollector
from astro_context.memory.manager import MemoryManager
from astro_context.models.context import ContextResult, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager_with_store() -> tuple[MemoryManager, InMemoryEntryStore]:
    """Create a MemoryManager backed by an InMemoryEntryStore."""
    store = InMemoryEntryStore()
    tokenizer = FakeTokenizer()
    manager = MemoryManager(
        conversation_tokens=2000,
        tokenizer=tokenizer,
        persistent_store=store,
    )
    return manager, store


def _make_pipeline(manager: MemoryManager) -> ContextPipeline:
    """Create a ContextPipeline with FakeTokenizer and the given memory manager."""
    pipeline = ContextPipeline(max_tokens=8192, tokenizer=FakeTokenizer())
    pipeline.with_memory(manager)
    return pipeline


# ---------------------------------------------------------------------------
# GC collects expired facts
# ---------------------------------------------------------------------------


class TestGCCollectsExpiredFacts:
    """MemoryGarbageCollector removes expired entries."""

    def test_gc_removes_expired_facts(self) -> None:
        manager, store = _make_manager_with_store()

        # Add non-expired facts
        manager.add_fact("Python is a programming language")
        manager.add_fact("Context engineering is important")

        # Add expired facts (manually set expires_at in the past)
        past = datetime.now(UTC) - timedelta(hours=1)
        expired1 = manager.add_fact("Old fact one")
        expired2 = manager.add_fact("Old fact two")

        # Manually update expires_at on the stored entries
        updated1 = expired1.model_copy(update={"expires_at": past})
        updated2 = expired2.model_copy(update={"expires_at": past})
        store.add(updated1)
        store.add(updated2)

        # Before GC: 4 total entries, 2 expired
        all_entries = store.list_all_unfiltered()
        assert len(all_entries) == 4

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        assert stats.expired_pruned == 2
        assert stats.decayed_pruned == 0
        assert stats.total_remaining == 2
        assert not stats.dry_run

    def test_gc_dry_run_does_not_delete(self) -> None:
        manager, store = _make_manager_with_store()

        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("Expired fact")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        gc = MemoryGarbageCollector(store)
        stats = gc.collect(dry_run=True)

        assert stats.expired_pruned == 1
        assert stats.dry_run
        # Entry should still be in the store
        assert len(store.list_all_unfiltered()) == 1


# ---------------------------------------------------------------------------
# Pipeline shows only surviving facts after GC
# ---------------------------------------------------------------------------


class TestGCPipelineIntegration:
    """After GC, only surviving facts appear in pipeline output."""

    def test_pipeline_excludes_gc_pruned_facts(self) -> None:
        manager, store = _make_manager_with_store()

        # Add surviving facts
        manager.add_fact("Python is great")
        manager.add_fact("Rust is fast")

        # Add expired facts
        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("Outdated info")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        # Run GC to prune expired entries
        gc = MemoryGarbageCollector(store)
        stats = gc.collect()
        assert stats.expired_pruned == 1

        # Build pipeline
        pipeline = _make_pipeline(manager)
        result = pipeline.build(QueryBundle(query_str="test"))

        assert isinstance(result, ContextResult)

        # Verify only surviving facts are in the context
        contents = [item.content for item in result.window.items]
        assert "Python is great" in contents
        assert "Rust is fast" in contents
        assert "Outdated info" not in contents

    def test_pipeline_with_system_prompt_and_gc(self) -> None:
        manager, store = _make_manager_with_store()

        # Add facts
        manager.add_fact("Surviving fact")

        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("Dead fact")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        # GC
        gc = MemoryGarbageCollector(store)
        gc.collect()

        # Build pipeline with system prompt
        pipeline = _make_pipeline(manager)
        pipeline.add_system_prompt("You are a helpful assistant.")

        result = pipeline.build(QueryBundle(query_str="test"))

        contents = [item.content for item in result.window.items]
        assert "You are a helpful assistant." in contents
        assert "Surviving fact" in contents
        assert "Dead fact" not in contents

    def test_pipeline_with_conversation_and_gc(self) -> None:
        manager, store = _make_manager_with_store()

        # Add facts and conversation
        manager.add_fact("Important fact")

        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("Expired fact")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        manager.add_user_message("Hello")
        manager.add_assistant_message("Hi there!")

        # GC
        gc = MemoryGarbageCollector(store)
        gc.collect()

        # Build pipeline
        pipeline = _make_pipeline(manager)
        result = pipeline.build(QueryBundle(query_str="test"))

        contents = [item.content for item in result.window.items]
        assert "Important fact" in contents
        assert "Expired fact" not in contents
        # Conversation turns should still be present
        assert "Hello" in contents
        assert "Hi there!" in contents

    def test_expired_facts_not_in_result_without_gc(self) -> None:
        """MemoryManager.get_context_items already skips expired entries."""
        manager, store = _make_manager_with_store()

        manager.add_fact("Active fact")

        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("Expired fact")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        # Build pipeline WITHOUT running GC
        pipeline = _make_pipeline(manager)
        result = pipeline.build(QueryBundle(query_str="test"))

        contents = [item.content for item in result.window.items]
        assert "Active fact" in contents
        # Even without GC, the MemoryManager skips expired entries
        assert "Expired fact" not in contents


# ---------------------------------------------------------------------------
# Memory source types in pipeline
# ---------------------------------------------------------------------------


class TestGCPipelineSourceTypes:
    """Verify source types are correct after GC + pipeline."""

    def test_persistent_facts_have_memory_source(self) -> None:
        manager, _store = _make_manager_with_store()
        manager.add_fact("A persistent fact")

        pipeline = _make_pipeline(manager)
        result = pipeline.build(QueryBundle(query_str="test"))

        memory_items = [
            item for item in result.window.items if item.source == SourceType.MEMORY
        ]
        assert len(memory_items) == 1
        assert memory_items[0].content == "A persistent fact"

    def test_gc_stats_repr(self) -> None:
        manager, store = _make_manager_with_store()

        past = datetime.now(UTC) - timedelta(hours=1)
        expired = manager.add_fact("To be pruned")
        updated = expired.model_copy(update={"expires_at": past})
        store.add(updated)

        gc = MemoryGarbageCollector(store)
        stats = gc.collect()

        repr_str = repr(stats)
        assert "applied" in repr_str
        assert "expired_pruned=1" in repr_str
