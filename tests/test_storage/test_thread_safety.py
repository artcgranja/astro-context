"""Thread safety tests for in-memory storage implementations.

Verifies that concurrent access from multiple threads does not cause data
loss or corruption in InMemoryContextStore, InMemoryVectorStore,
InMemoryDocumentStore, InMemoryEntryStore, and JsonFileMemoryStore.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from pathlib import Path

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.memory import MemoryEntry
from astro_context.storage.json_file_store import JsonFileMemoryStore
from astro_context.storage.json_memory_store import InMemoryEntryStore
from astro_context.storage.memory_store import (
    InMemoryContextStore,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)
from tests.conftest import make_embedding

NUM_THREADS = 10
ITEMS_PER_THREAD = 50


def _run_threads(
    targets: list[tuple[Callable[[int], None], int]],
    barrier: threading.Barrier,
) -> list[Exception]:
    """Launch threads from (callable, tid) pairs and collect errors."""
    errors: list[Exception] = []

    def _wrap(fn: Callable[[int], None], tid: int) -> None:
        try:
            barrier.wait()
            fn(tid)
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=_wrap, args=(fn, tid)) for fn, tid in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


# ---------------------------------------------------------------------------
# InMemoryContextStore
# ---------------------------------------------------------------------------


class TestInMemoryContextStoreThreadSafety:
    """Concurrent add/get on InMemoryContextStore from multiple threads."""

    def test_concurrent_add_get(self) -> None:
        store = InMemoryContextStore()
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                item_id = f"t{tid}-{i}"
                item = ContextItem(
                    id=item_id,
                    content=f"content-{tid}-{i}",
                    source=SourceType.USER,
                    token_count=1,
                )
                store.add(item)
                retrieved = store.get(item_id)
                assert retrieved is not None
                assert retrieved.id == item_id

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store.get_all()) == NUM_THREADS * ITEMS_PER_THREAD

    def test_concurrent_add_delete(self) -> None:
        store = InMemoryContextStore()
        barrier = threading.Barrier(NUM_THREADS)

        def writer(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(
                    ContextItem(
                        id=f"t{tid}-{i}",
                        content=f"content-{tid}-{i}",
                        source=SourceType.USER,
                    )
                )

        def deleter(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.delete(f"t{tid}-{i}")

        half = NUM_THREADS // 2
        targets: list[tuple[Callable[[int], None], int]] = [
            *[(writer, t) for t in range(half)],
            *[(deleter, t) for t in range(half)],
        ]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        # No corruption -- get_all returns a valid list
        store.get_all()

    def test_concurrent_clear(self) -> None:
        store = InMemoryContextStore()
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(
                    ContextItem(
                        id=f"t{tid}-{i}",
                        content=f"c-{tid}-{i}",
                        source=SourceType.USER,
                    )
                )
                if i % 10 == 0:
                    store.clear()

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)
        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreThreadSafety:
    """Concurrent add_embedding/search on InMemoryVectorStore."""

    def test_concurrent_add_search(self) -> None:
        store = InMemoryVectorStore()
        barrier = threading.Barrier(NUM_THREADS)

        def writer(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                emb = make_embedding(tid * ITEMS_PER_THREAD + i)
                store.add_embedding(f"t{tid}-{i}", emb)

        def searcher(tid: int) -> None:
            query = make_embedding(tid)
            for _ in range(ITEMS_PER_THREAD):
                store.search(query, top_k=5)

        half = NUM_THREADS // 2
        targets: list[tuple[Callable[[int], None], int]] = [
            *[(writer, t) for t in range(half)],
            *[(searcher, t) for t in range(half)],
        ]
        errors = _run_threads(targets, barrier)
        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_add_no_data_loss(self) -> None:
        store = InMemoryVectorStore()
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                emb = make_embedding(tid * ITEMS_PER_THREAD + i)
                store.add_embedding(f"t{tid}-{i}", emb)

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store._embeddings) == NUM_THREADS * ITEMS_PER_THREAD

    def test_concurrent_add_delete(self) -> None:
        store = InMemoryVectorStore()
        barrier = threading.Barrier(NUM_THREADS)

        def writer(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                emb = make_embedding(tid * ITEMS_PER_THREAD + i)
                store.add_embedding(f"t{tid}-{i}", emb)

        def deleter(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.delete(f"t{tid}-{i}")

        half = NUM_THREADS // 2
        targets: list[tuple[Callable[[int], None], int]] = [
            *[(writer, t) for t in range(half)],
            *[(deleter, t) for t in range(half)],
        ]
        errors = _run_threads(targets, barrier)
        assert not errors, f"Thread errors: {errors}"


# ---------------------------------------------------------------------------
# InMemoryDocumentStore
# ---------------------------------------------------------------------------


class TestInMemoryDocumentStoreThreadSafety:
    """Concurrent operations on InMemoryDocumentStore."""

    def test_concurrent_add_get(self) -> None:
        store = InMemoryDocumentStore()
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                doc_id = f"t{tid}-{i}"
                store.add_document(doc_id, f"content-{tid}-{i}")
                assert store.get_document(doc_id) is not None

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store.list_documents()) == NUM_THREADS * ITEMS_PER_THREAD


# ---------------------------------------------------------------------------
# InMemoryEntryStore (uses BaseEntryStoreMixin)
# ---------------------------------------------------------------------------


class TestInMemoryEntryStoreThreadSafety:
    """Concurrent operations on InMemoryEntryStore."""

    def test_concurrent_add_search(self) -> None:
        store = InMemoryEntryStore()
        barrier = threading.Barrier(NUM_THREADS)

        def writer(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(MemoryEntry(content=f"memory-{tid}-{i}", user_id=f"user-{tid}"))

        def reader(tid: int) -> None:
            for _ in range(ITEMS_PER_THREAD):
                store.search(f"memory-{tid}", top_k=5)

        half = NUM_THREADS // 2
        targets: list[tuple[Callable[[int], None], int]] = [
            *[(writer, t) for t in range(half)],
            *[(reader, t) for t in range(half)],
        ]
        errors = _run_threads(targets, barrier)
        assert not errors, f"Thread errors: {errors}"

    def test_concurrent_add_no_data_loss(self) -> None:
        store = InMemoryEntryStore()
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(MemoryEntry(id=f"entry-{tid}-{i}", content=f"memory-{tid}-{i}"))

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store.list_all_unfiltered()) == NUM_THREADS * ITEMS_PER_THREAD


# ---------------------------------------------------------------------------
# JsonFileMemoryStore
# ---------------------------------------------------------------------------


class TestJsonFileMemoryStoreThreadSafety:
    """Concurrent operations on JsonFileMemoryStore."""

    def test_concurrent_add(self, tmp_path: Path) -> None:
        store = JsonFileMemoryStore(tmp_path / "test.json", auto_save=False)
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(MemoryEntry(id=f"entry-{tid}-{i}", content=f"memory-{tid}-{i}"))

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store.list_all_unfiltered()) == NUM_THREADS * ITEMS_PER_THREAD

    def test_concurrent_add_with_auto_save(self, tmp_path: Path) -> None:
        store = JsonFileMemoryStore(tmp_path / "test_auto.json", auto_save=True)
        barrier = threading.Barrier(NUM_THREADS)

        def worker(tid: int) -> None:
            for i in range(ITEMS_PER_THREAD):
                store.add(MemoryEntry(id=f"entry-{tid}-{i}", content=f"memory-{tid}-{i}"))

        targets = [(worker, t) for t in range(NUM_THREADS)]
        errors = _run_threads(targets, barrier)

        assert not errors, f"Thread errors: {errors}"
        assert len(store.list_all_unfiltered()) == NUM_THREADS * ITEMS_PER_THREAD
