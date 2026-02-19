"""Retriever protocol definition.

Any object with a `retrieve` method matching this signature
can be used as a retriever in the pipeline -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


@runtime_checkable
class Retriever(Protocol):
    """Protocol for all retrieval strategies."""

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]: ...
