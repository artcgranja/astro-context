"""Shared helpers for pipeline tests."""

from __future__ import annotations

from astro_context.models.context import ContextItem, SourceType
from astro_context.pipeline.pipeline import ContextPipeline
from tests.conftest import FakeTokenizer


def make_pipeline(max_tokens: int = 8192) -> ContextPipeline:
    """Create a ContextPipeline with a FakeTokenizer."""
    return ContextPipeline(max_tokens=max_tokens, tokenizer=FakeTokenizer())


def make_items(count: int = 3) -> list[ContextItem]:
    """Create a list of ContextItems for pipeline testing."""
    tokenizer = FakeTokenizer()
    return [
        ContextItem(
            id=f"item-{i}",
            content=f"Test document number {i} with some content.",
            source=SourceType.RETRIEVAL,
            score=0.5 + i * 0.1,
            priority=5,
            token_count=tokenizer.count_tokens(f"Test document number {i} with some content."),
        )
        for i in range(count)
    ]
