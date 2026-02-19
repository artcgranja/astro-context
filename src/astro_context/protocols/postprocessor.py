"""PostProcessor protocol definitions.

PostProcessors transform context items after retrieval.
Examples: reranking, filtering, deduplication, PII removal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for synchronous post-processing of retrieved context items."""

    def process(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]: ...


@runtime_checkable
class AsyncPostProcessor(Protocol):
    """Protocol for asynchronous post-processing (e.g., LLM-based reranking)."""

    async def aprocess(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]: ...
