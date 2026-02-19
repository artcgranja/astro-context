"""Retriever protocol definitions.

Any object with a `retrieve` (or `aretrieve`) method matching these
signatures can be used as a retriever in the pipeline -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


@runtime_checkable
class Retriever(Protocol):
    """Protocol for synchronous retrieval strategies."""

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...


@runtime_checkable
class AsyncRetriever(Protocol):
    """Protocol for asynchronous retrieval strategies.

    Async retrievers are used with ``ContextPipeline.abuild()`` for
    non-blocking I/O during embedding lookups, database queries, etc.
    """

    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...
