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
    ) -> list[ContextItem]:
        """Transform a list of context items after retrieval.

        Common transformations include reranking, deduplication,
        filtering by relevance score, and PII removal.

        Parameters:
            items: The context items to post-process, typically the
                output of a retriever.
            query: The original query bundle.  Implementations may use
                this for query-aware transformations (e.g., reranking).
                ``None`` indicates query-agnostic processing.

        Returns:
            A new (or modified) list of ``ContextItem`` objects.  The
            list may be shorter, longer, or reordered compared to the
            input depending on the transformation applied.
        """
        ...


@runtime_checkable
class AsyncPostProcessor(Protocol):
    """Protocol for asynchronous post-processing (e.g., LLM-based reranking)."""

    async def aprocess(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]:
        """Asynchronously transform a list of context items after retrieval.

        This is the async counterpart of ``PostProcessor.process``,
        intended for transformations that involve I/O-bound work such as
        LLM-based reranking or external API calls.

        Parameters:
            items: The context items to post-process.
            query: The original query bundle for query-aware
                transformations, or ``None`` for query-agnostic
                processing.

        Returns:
            A new (or modified) list of ``ContextItem`` objects.
        """
        ...
