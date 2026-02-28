"""Shared fixtures for ingestion tests."""

from __future__ import annotations

import pytest

from tests.conftest import FakeTokenizer

from astro_context.ingestion.chunkers import (
    FixedSizeChunker,
    RecursiveCharacterChunker,
    SentenceChunker,
)


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
