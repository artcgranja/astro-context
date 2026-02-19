"""Shared fixtures for astro-context tests."""

from __future__ import annotations

import math

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.storage.memory_store import InMemoryContextStore, InMemoryVectorStore


def make_embedding(seed: int, dim: int = 128) -> list[float]:
    """Create a deterministic fake embedding using math.sin.

    Each seed produces a unique but reproducible unit-ish vector.
    The same seed always returns the same embedding, and different
    seeds produce different (approximately orthogonal) embeddings
    when dim is large enough relative to the number of seeds.
    """
    raw = [math.sin(seed * 1000 + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        return raw
    return [x / norm for x in raw]


class FakeTokenizer:
    """A simple tokenizer that splits on whitespace for testing.

    Satisfies the Tokenizer protocol without requiring tiktoken's
    network-downloaded encoding data.
    """

    def count_tokens(self, text: str) -> int:
        """Count tokens by splitting on whitespace."""
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token limit."""
        if max_tokens <= 0:
            return ""
        words = text.split()
        if len(words) <= max_tokens:
            return text
        return " ".join(words[:max_tokens])


@pytest.fixture()
def counter() -> FakeTokenizer:
    """Return a FakeTokenizer instance for testing."""
    return FakeTokenizer()


@pytest.fixture()
def context_store() -> InMemoryContextStore:
    """Return a fresh InMemoryContextStore."""
    return InMemoryContextStore()


@pytest.fixture()
def vector_store() -> InMemoryVectorStore:
    """Return a fresh InMemoryVectorStore."""
    return InMemoryVectorStore()


@pytest.fixture()
def sample_items(counter: FakeTokenizer) -> list[ContextItem]:
    """Return 5 ContextItems with token counts from FakeTokenizer."""
    texts = [
        "Python is a high-level programming language known for readability.",
        "Machine learning models require large amounts of training data to perform well.",
        "Context engineering is the art of assembling the right information for LLMs.",
        "Vector databases enable efficient similarity search over embeddings.",
        "Retrieval-augmented generation combines search with language model generation.",
    ]
    items: list[ContextItem] = []
    for i, text in enumerate(texts):
        token_count = counter.count_tokens(text)
        items.append(
            ContextItem(
                id=f"item-{i}",
                content=text,
                source=SourceType.RETRIEVAL,
                score=0.5 + i * 0.1,
                priority=5 + i,
                token_count=token_count,
            )
        )
    return items


@pytest.fixture()
def sample_query() -> QueryBundle:
    """Return a sample QueryBundle for testing."""
    return QueryBundle(
        query_str="What is context engineering?",
        embedding=make_embedding(seed=99),
    )
