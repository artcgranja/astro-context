"""Tests for astro_context.storage.memory_store."""

from __future__ import annotations

import math

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.protocols.storage import ContextStore, DocumentStore, VectorStore
from astro_context.storage.memory_store import (
    InMemoryContextStore,
    InMemoryDocumentStore,
    InMemoryVectorStore,
)

# ---------------------------------------------------------------------------
# InMemoryContextStore
# ---------------------------------------------------------------------------


class TestInMemoryContextStoreCRUD:
    """CRUD operations on InMemoryContextStore."""

    def test_add_and_get(self, context_store: InMemoryContextStore) -> None:
        item = ContextItem(id="test-1", content="hello", source=SourceType.USER, token_count=1)
        context_store.add(item)
        retrieved = context_store.get("test-1")
        assert retrieved is not None
        assert retrieved.content == "hello"

    def test_get_nonexistent_returns_none(self, context_store: InMemoryContextStore) -> None:
        assert context_store.get("does-not-exist") is None

    def test_get_all_empty(self, context_store: InMemoryContextStore) -> None:
        assert context_store.get_all() == []

    def test_get_all_returns_all_items(self, context_store: InMemoryContextStore) -> None:
        for i in range(3):
            context_store.add(
                ContextItem(id=f"item-{i}", content=f"content-{i}", source=SourceType.USER)
            )
        all_items = context_store.get_all()
        assert len(all_items) == 3

    def test_delete_existing(self, context_store: InMemoryContextStore) -> None:
        item = ContextItem(id="del-me", content="bye", source=SourceType.USER)
        context_store.add(item)
        assert context_store.delete("del-me") is True
        assert context_store.get("del-me") is None

    def test_delete_nonexistent(self, context_store: InMemoryContextStore) -> None:
        assert context_store.delete("nope") is False

    def test_clear(self, context_store: InMemoryContextStore) -> None:
        for i in range(3):
            context_store.add(
                ContextItem(id=f"item-{i}", content=f"content-{i}", source=SourceType.USER)
            )
        context_store.clear()
        assert context_store.get_all() == []

    def test_add_overwrites_same_id(self, context_store: InMemoryContextStore) -> None:
        context_store.add(
            ContextItem(id="dup", content="original", source=SourceType.USER)
        )
        context_store.add(
            ContextItem(id="dup", content="replacement", source=SourceType.USER)
        )
        item = context_store.get("dup")
        assert item is not None
        assert item.content == "replacement"


class TestInMemoryContextStoreProtocol:
    """InMemoryContextStore satisfies the ContextStore protocol."""

    def test_isinstance_check(self) -> None:
        assert isinstance(InMemoryContextStore(), ContextStore)


# ---------------------------------------------------------------------------
# InMemoryVectorStore
# ---------------------------------------------------------------------------


class TestInMemoryVectorStoreOperations:
    """add_embedding, search, and delete on InMemoryVectorStore."""

    def test_add_and_search(self, vector_store: InMemoryVectorStore) -> None:
        from tests.conftest import make_embedding

        emb1 = make_embedding(1)
        emb2 = make_embedding(2)
        vector_store.add_embedding("v1", emb1)
        vector_store.add_embedding("v2", emb2)

        results = vector_store.search(emb1, top_k=2)
        assert len(results) == 2
        # The first result should be v1 since query matches v1 exactly
        assert results[0][0] == "v1"
        assert results[0][1] > results[1][1]

    def test_search_empty_store(self, vector_store: InMemoryVectorStore) -> None:
        from tests.conftest import make_embedding

        results = vector_store.search(make_embedding(1), top_k=5)
        assert results == []

    def test_search_top_k_limits_results(self, vector_store: InMemoryVectorStore) -> None:
        from tests.conftest import make_embedding

        for i in range(10):
            vector_store.add_embedding(f"v{i}", make_embedding(i))
        results = vector_store.search(make_embedding(0), top_k=3)
        assert len(results) == 3

    def test_delete_existing(self, vector_store: InMemoryVectorStore) -> None:
        from tests.conftest import make_embedding

        vector_store.add_embedding("v1", make_embedding(1))
        assert vector_store.delete("v1") is True
        results = vector_store.search(make_embedding(1), top_k=5)
        assert len(results) == 0

    def test_delete_nonexistent(self, vector_store: InMemoryVectorStore) -> None:
        assert vector_store.delete("nope") is False

    def test_add_embedding_with_metadata(self, vector_store: InMemoryVectorStore) -> None:
        from tests.conftest import make_embedding

        vector_store.add_embedding("v1", make_embedding(1), metadata={"source": "test"})
        assert vector_store._metadata["v1"] == {"source": "test"}


class TestInMemoryVectorStoreCosineSimilarity:
    """Cosine similarity correctness with known vectors."""

    def test_identical_vectors_similarity_is_one(self) -> None:
        vec = [1.0, 0.0, 0.0]
        result = InMemoryVectorStore._cosine_similarity(vec, vec)
        assert result == pytest.approx(1.0)

    def test_orthogonal_vectors_similarity_is_zero(self) -> None:
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        result = InMemoryVectorStore._cosine_similarity(a, b)
        assert result == pytest.approx(0.0)

    def test_opposite_vectors_similarity_is_negative_one(self) -> None:
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        result = InMemoryVectorStore._cosine_similarity(a, b)
        assert result == pytest.approx(-1.0)

    def test_known_similarity(self) -> None:
        a = [1.0, 2.0, 3.0]
        b = [4.0, 5.0, 6.0]
        # Manual: dot = 32, |a| = sqrt(14), |b| = sqrt(77)
        expected = 32 / (math.sqrt(14) * math.sqrt(77))
        result = InMemoryVectorStore._cosine_similarity(a, b)
        assert result == pytest.approx(expected, abs=1e-9)

    def test_zero_vector_returns_zero(self) -> None:
        a = [0.0, 0.0, 0.0]
        b = [1.0, 2.0, 3.0]
        assert InMemoryVectorStore._cosine_similarity(a, b) == 0.0


class TestInMemoryVectorStoreProtocol:
    """InMemoryVectorStore satisfies the VectorStore protocol."""

    def test_isinstance_check(self) -> None:
        assert isinstance(InMemoryVectorStore(), VectorStore)


# ---------------------------------------------------------------------------
# InMemoryDocumentStore
# ---------------------------------------------------------------------------


class TestInMemoryDocumentStoreCRUD:
    """CRUD operations on InMemoryDocumentStore."""

    def test_add_and_get_document(self) -> None:
        store = InMemoryDocumentStore()
        store.add_document("doc1", "Hello world", metadata={"author": "test"})
        assert store.get_document("doc1") == "Hello world"

    def test_get_nonexistent_returns_none(self) -> None:
        store = InMemoryDocumentStore()
        assert store.get_document("nope") is None

    def test_list_documents_empty(self) -> None:
        store = InMemoryDocumentStore()
        assert store.list_documents() == []

    def test_list_documents(self) -> None:
        store = InMemoryDocumentStore()
        store.add_document("doc1", "a")
        store.add_document("doc2", "b")
        doc_ids = store.list_documents()
        assert set(doc_ids) == {"doc1", "doc2"}

    def test_delete_existing(self) -> None:
        store = InMemoryDocumentStore()
        store.add_document("doc1", "content")
        assert store.delete_document("doc1") is True
        assert store.get_document("doc1") is None

    def test_delete_nonexistent(self) -> None:
        store = InMemoryDocumentStore()
        assert store.delete_document("nope") is False

    def test_add_without_metadata(self) -> None:
        store = InMemoryDocumentStore()
        store.add_document("doc1", "no meta")
        assert store.get_document("doc1") == "no meta"
        assert "doc1" not in store._metadata

    def test_overwrite_document(self) -> None:
        store = InMemoryDocumentStore()
        store.add_document("doc1", "original")
        store.add_document("doc1", "updated")
        assert store.get_document("doc1") == "updated"


class TestInMemoryDocumentStoreProtocol:
    """InMemoryDocumentStore satisfies the DocumentStore protocol."""

    def test_isinstance_check(self) -> None:
        assert isinstance(InMemoryDocumentStore(), DocumentStore)
