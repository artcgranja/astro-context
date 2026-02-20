"""Tests for eviction-based auto-promotion of memories.

Validates that the ``create_eviction_promoter`` callback correctly
extracts, consolidates, and stores memories when conversation turns
are evicted from ``SlidingWindowMemory``.
"""

from __future__ import annotations

from typing import Any

from astro_context.memory.extractor import CallbackExtractor
from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.models.memory import ConversationTurn, MemoryEntry
from astro_context.pipeline.memory_steps import create_eviction_promoter
from astro_context.storage.json_memory_store import InMemoryEntryStore
from tests.conftest import FakeTokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _simple_extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Extract one memory per turn with a 'Promoted:' prefix."""
    return [{"content": f"Promoted: {t.content}"} for t in turns]


def _empty_extract_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
    """Extract nothing."""
    return []


def _make_extractor(
    fn: Any = None,
) -> CallbackExtractor:
    return CallbackExtractor(extract_fn=fn or _simple_extract_fn)


# ===========================================================================
# TestEvictionTriggersExtraction
# ===========================================================================


class TestEvictionTriggersExtraction:
    """Eviction of turns triggers the extraction function."""

    def test_eviction_calls_extractor(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")        # 1 token
        mem.add_turn("assistant", "second")   # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens -> evicts first & second

        stored = store.list_all()
        assert len(stored) == 2
        contents = [e.content for e in stored]
        assert "Promoted: first" in contents
        assert "Promoted: second" in contents

    def test_no_eviction_means_no_extraction(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=100, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "hello")
        mem.add_turn("assistant", "world")

        assert len(store.list_all()) == 0


# ===========================================================================
# TestExtractedEntriesAreStored
# ===========================================================================


class TestExtractedEntriesAreStored:
    """Extracted MemoryEntry objects are properly persisted in the store."""

    def test_entries_are_valid_memory_entries(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        for entry in store.list_all():
            assert isinstance(entry, MemoryEntry)
            assert entry.content.startswith("Promoted:")
            assert entry.id  # Has a UUID

    def test_entries_have_default_semantic_type(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        for entry in store.list_all():
            assert str(entry.memory_type) == "semantic"

    def test_entries_searchable_in_store(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        results = store.search("Promoted")
        assert len(results) > 0

    def test_empty_extraction_stores_nothing(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor(fn=_empty_extract_fn)
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        assert len(store.list_all()) == 0


# ===========================================================================
# TestConsolidationPreventsDuplicates
# ===========================================================================


class TestConsolidationPreventsDuplicates:
    """Consolidator prevents duplicate entries from being stored."""

    def test_duplicate_content_skipped(self) -> None:
        store = InMemoryEntryStore()
        existing = MemoryEntry(id="old-1", content="Promoted: first")
        store.add(existing)

        extractor = _make_extractor()

        class SkipDupConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                existing_contents = {e.content for e in existing}
                results: list[tuple[str, MemoryEntry | None]] = []
                for entry in new_entries:
                    if entry.content in existing_contents:
                        results.append(("none", None))
                    else:
                        results.append(("add", entry))
                return results

        promoter = create_eviction_promoter(
            extractor=extractor,
            store=store,
            consolidator=SkipDupConsolidator(),  # type: ignore[arg-type]
        )

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")         # 1 token
        mem.add_turn("assistant", "second")    # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens -> evicts first & second

        stored = store.list_all()
        # "Promoted: first" was duplicate (already in store), so only
        # "Promoted: second" should be added
        assert len(stored) == 2
        ids = [e.id for e in stored]
        assert "old-1" in ids

    def test_update_action_overwrites(self) -> None:
        store = InMemoryEntryStore()
        existing = MemoryEntry(id="old-1", content="Promoted: first")
        store.add(existing)

        extractor = _make_extractor()

        class UpdateConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                return [("update", entry) for entry in new_entries]

        promoter = create_eviction_promoter(
            extractor=extractor,
            store=store,
            consolidator=UpdateConsolidator(),  # type: ignore[arg-type]
        )

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        stored = store.list_all()
        # original + 2 updated entries
        assert len(stored) == 3


# ===========================================================================
# TestExtractorErrorHandling
# ===========================================================================


class TestExtractorErrorHandling:
    """Extractor errors do not crash eviction or the memory pipeline."""

    def test_extractor_error_silently_handled(self) -> None:
        store = InMemoryEntryStore()

        def failing_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            msg = "extractor crashed"
            raise RuntimeError(msg)

        extractor = _make_extractor(fn=failing_fn)
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Should not raise -- error swallowed by promoter + SlidingWindowMemory
        mem.add_turn("user", "three four five")

        # Eviction still happened
        remaining = [t.content for t in mem.turns]
        assert "three four five" in remaining
        assert "first" not in remaining

        # Nothing stored due to error
        assert len(store.list_all()) == 0

    def test_store_error_silently_handled(self) -> None:
        class FailingStore:
            def add(self, entry: MemoryEntry) -> None:
                msg = "store crashed"
                raise RuntimeError(msg)

            def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
                return []

            def list_all(self) -> list[MemoryEntry]:
                return []

            def delete(self, entry_id: str) -> bool:
                return False

            def clear(self) -> None:
                pass

        extractor = _make_extractor()
        promoter = create_eviction_promoter(
            extractor=extractor,
            store=FailingStore(),  # type: ignore[arg-type]
        )

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Should not raise
        mem.add_turn("user", "three four five")

        # Eviction still happened
        assert mem.turns[-1].content == "three four five"

    def test_consolidator_error_silently_handled(self) -> None:
        store = InMemoryEntryStore()
        extractor = _make_extractor()

        class FailingConsolidator:
            def consolidate(
                self,
                new_entries: list[MemoryEntry],
                existing: list[MemoryEntry],
            ) -> list[tuple[str, MemoryEntry | None]]:
                msg = "consolidator crashed"
                raise RuntimeError(msg)

        promoter = create_eviction_promoter(
            extractor=extractor,
            store=store,
            consolidator=FailingConsolidator(),  # type: ignore[arg-type]
        )

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Should not raise
        mem.add_turn("user", "three four five")

        assert mem.turns[-1].content == "three four five"
        assert len(store.list_all()) == 0


# ===========================================================================
# TestWorksWithSlidingWindowMemory
# ===========================================================================


class TestWorksWithSlidingWindowMemory:
    """End-to-end: promoter works correctly as SlidingWindowMemory.on_evict."""

    def test_multiple_eviction_rounds(self) -> None:
        """Multiple rounds of eviction each trigger extraction."""
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )

        # Round 1: fill and evict
        mem.add_turn("user", "a")           # 1 token
        mem.add_turn("assistant", "b")      # 1 token
        mem.add_turn("user", "c d e")       # 3 tokens -> evicts a, b

        assert len(store.list_all()) == 2

        # Round 2: fill and evict again
        mem.add_turn("assistant", "f g h")  # 3 tokens -> evicts "c d e"

        assert len(store.list_all()) == 3
        contents = [e.content for e in store.list_all()]
        assert "Promoted: a" in contents
        assert "Promoted: b" in contents
        assert "Promoted: c d e" in contents

    def test_promoter_preserves_sliding_window_behavior(self) -> None:
        """Adding the promoter does not change SlidingWindowMemory's core behavior."""
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=5, tokenizer=FakeTokenizer(), on_evict=promoter
        )

        mem.add_turn("user", "one")
        mem.add_turn("assistant", "two")
        mem.add_turn("user", "three")
        mem.add_turn("assistant", "four")
        # At 4 tokens. Adding 3-token message -> needs to free 2 tokens
        mem.add_turn("user", "five six seven")

        # Sliding window should still work correctly
        assert mem.total_tokens <= 5
        assert mem.turns[-1].content == "five six seven"

    def test_context_items_still_work_after_eviction(self) -> None:
        """to_context_items works correctly after eviction with promoter."""
        store = InMemoryEntryStore()
        extractor = _make_extractor()
        promoter = create_eviction_promoter(extractor=extractor, store=store)

        mem = SlidingWindowMemory(
            max_tokens=3, tokenizer=FakeTokenizer(), on_evict=promoter
        )
        mem.add_turn("user", "old")
        mem.add_turn("assistant", "message")
        mem.add_turn("user", "new big turn")  # 3 tokens -> evicts old, message

        context_items = mem.to_context_items()
        assert len(context_items) == 1
        assert context_items[0].content == "new big turn"
