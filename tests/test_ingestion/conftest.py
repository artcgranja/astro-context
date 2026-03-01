"""Shared fixtures for ingestion tests."""

from __future__ import annotations

import math

import pytest

from astro_context.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SemanticChunker,
    SentenceChunker,
)
from astro_context.ingestion.hierarchical import ParentChildChunker, ParentExpander
from tests.conftest import FakeTokenizer


def _fake_embed_fn(texts: list[str]) -> list[list[float]]:
    """Generate deterministic embeddings based on text content.

    Similar texts get similar embeddings, different texts get different ones.
    Uses a hash-based approach for determinism.
    """
    embeddings: list[list[float]] = []
    for text in texts:
        seed = hash(text) % 10000
        raw = [math.sin(seed * 1000 + i) for i in range(64)]
        norm = math.sqrt(sum(x * x for x in raw))
        embeddings.append([x / norm for x in raw] if norm > 0 else raw)
    return embeddings


@pytest.fixture
def fake_tokenizer() -> FakeTokenizer:
    return FakeTokenizer()


@pytest.fixture
def fixed_chunker(fake_tokenizer: FakeTokenizer) -> FixedSizeChunker:
    return FixedSizeChunker(chunk_size=10, overlap=2, tokenizer=fake_tokenizer)


@pytest.fixture
def recursive_chunker(fake_tokenizer: FakeTokenizer) -> RecursiveCharacterChunker:
    return RecursiveCharacterChunker(chunk_size=10, overlap=2, tokenizer=fake_tokenizer)


@pytest.fixture
def sentence_chunker(fake_tokenizer: FakeTokenizer) -> SentenceChunker:
    return SentenceChunker(chunk_size=10, overlap=1, tokenizer=fake_tokenizer)


@pytest.fixture
def semantic_chunker(fake_tokenizer: FakeTokenizer) -> SemanticChunker:
    return SemanticChunker(
        embed_fn=_fake_embed_fn,
        tokenizer=fake_tokenizer,
        chunk_size=20,
        min_chunk_size=3,
        threshold=0.5,
    )


@pytest.fixture
def parent_child_chunker(fake_tokenizer: FakeTokenizer) -> ParentChildChunker:
    return ParentChildChunker(
        parent_chunk_size=20,
        child_chunk_size=5,
        parent_overlap=2,
        child_overlap=1,
        tokenizer=fake_tokenizer,
    )


@pytest.fixture
def parent_expander() -> ParentExpander:
    return ParentExpander()
