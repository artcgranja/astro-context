"""Shared helpers for retrieval tests."""

from __future__ import annotations

from unittest.mock import patch

from anchor.retrieval.dense import DenseRetriever
from anchor.retrieval.sparse import SparseRetriever
from anchor.storage.memory_store import InMemoryContextStore, InMemoryVectorStore
from tests.conftest import FakeTokenizer


def make_dense_retriever(
    vs: InMemoryVectorStore | None = None,
    cs: InMemoryContextStore | None = None,
    embed_fn=None,
) -> DenseRetriever:
    """Create a DenseRetriever with a patched default counter."""
    vs = vs or InMemoryVectorStore()
    cs = cs or InMemoryContextStore()
    with patch("anchor.retrieval.dense.get_default_counter", return_value=FakeTokenizer()):
        return DenseRetriever(vs, cs, embed_fn=embed_fn)


def make_sparse_retriever(tokenize_fn=None) -> SparseRetriever:
    """Create a SparseRetriever with a patched default counter."""
    with patch("anchor.retrieval.sparse.get_default_counter", return_value=FakeTokenizer()):
        return SparseRetriever(tokenize_fn=tokenize_fn)
