"""Tests for astro_context.retrieval.dense."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.retrieval.dense import DenseRetriever
from astro_context.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import FakeTokenizer, make_embedding


def _make_items() -> list[ContextItem]:
    """Create items for dense retrieval tests."""
    return [
        ContextItem(
            id=f"dense-{i}",
            content=text,
            source=SourceType.RETRIEVAL,
            token_count=10,
            priority=5,
        )
        for i, text in enumerate(
            [
                "Python programming language",
                "Java programming language",
                "Machine learning with neural networks",
                "Context engineering for LLMs",
                "Cooking pasta recipes",
            ]
        )
    ]


def _embed_fn(text: str) -> list[float]:
    """Deterministic embedding function: hashes text to a seed."""
    seed = sum(ord(c) for c in text) % 10000
    return make_embedding(seed)


def _make_retriever(
    vs: InMemoryVectorStore | None = None,
    cs: InMemoryContextStore | None = None,
    embed_fn=_embed_fn,
) -> DenseRetriever:
    """Create a DenseRetriever with a patched default counter."""
    vs = vs or InMemoryVectorStore()
    cs = cs or InMemoryContextStore()
    with patch("astro_context.retrieval.dense.get_default_counter", return_value=FakeTokenizer()):
        return DenseRetriever(vs, cs, embed_fn=embed_fn)


class TestDenseRetrieverIndex:
    """DenseRetriever.index() stores items in both vector and context stores."""

    def test_index_returns_count(self) -> None:
        retriever = _make_retriever()
        items = _make_items()
        count = retriever.index(items)
        assert count == 5

    def test_index_stores_in_context_store(self) -> None:
        cs = InMemoryContextStore()
        retriever = _make_retriever(cs=cs)
        items = _make_items()
        retriever.index(items)
        assert cs.get("dense-0") is not None
        assert cs.get("dense-4") is not None

    def test_index_stores_in_vector_store(self) -> None:
        vs = InMemoryVectorStore()
        retriever = _make_retriever(vs=vs)
        items = _make_items()
        retriever.index(items)
        # Vector store should have embeddings for all items
        results = vs.search(_embed_fn("anything"), top_k=10)
        assert len(results) == 5

    def test_index_without_embed_fn_raises(self) -> None:
        retriever = _make_retriever(embed_fn=None)
        with pytest.raises(ValueError, match="embed_fn must be provided"):
            retriever.index(_make_items())


class TestDenseRetrieverRetrieve:
    """DenseRetriever.retrieve() returns scored ContextItems."""

    def test_retrieve_returns_scored_items(self) -> None:
        retriever = _make_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="Python programming language")
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3
        # All results should have retrieval source
        for item in results:
            assert item.source == SourceType.RETRIEVAL
            assert 0.0 <= item.score <= 1.0
            assert item.metadata.get("retrieval_method") == "dense"

    def test_retrieve_without_embed_fn_and_no_query_embedding_raises(self) -> None:
        vs = InMemoryVectorStore()
        cs = InMemoryContextStore()
        # First index with an embed_fn
        retriever_for_index = _make_retriever(vs=vs, cs=cs)
        retriever_for_index.index(_make_items())

        # Then create a retriever without embed_fn
        retriever = _make_retriever(vs=vs, cs=cs, embed_fn=None)
        query = QueryBundle(query_str="test")
        with pytest.raises(ValueError, match=r"Either provide query\.embedding"):
            retriever.retrieve(query)

    def test_retrieve_with_query_embedding_works(self) -> None:
        vs = InMemoryVectorStore()
        cs = InMemoryContextStore()
        retriever_for_index = _make_retriever(vs=vs, cs=cs)
        retriever_for_index.index(_make_items())

        # Create retriever without embed_fn but provide embedding in query
        retriever = _make_retriever(vs=vs, cs=cs, embed_fn=None)
        query_embedding = _embed_fn("Python programming")
        query = QueryBundle(query_str="Python programming", embedding=query_embedding)
        results = retriever.retrieve(query, top_k=3)
        assert len(results) == 3

    def test_retrieve_top_k_limits_output(self) -> None:
        retriever = _make_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming")
        results = retriever.retrieve(query, top_k=2)
        assert len(results) == 2

    def test_retrieve_scores_are_normalized(self) -> None:
        retriever = _make_retriever()
        retriever.index(_make_items())

        query = QueryBundle(query_str="programming")
        results = retriever.retrieve(query, top_k=5)
        for item in results:
            assert 0.0 <= item.score <= 1.0
