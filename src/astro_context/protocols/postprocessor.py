"""PostProcessor protocol definition.

PostProcessors transform context items after retrieval.
Examples: reranking, filtering, deduplication, PII removal.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


@runtime_checkable
class PostProcessor(Protocol):
    """Protocol for post-processing retrieved context items."""

    def process(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]: ...
