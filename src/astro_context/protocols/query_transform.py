"""Query transformation protocol definitions.

Any object with a ``transform`` (or ``atransform``) method matching these
signatures can be used as a query transformer -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.query import QueryBundle


@runtime_checkable
class QueryTransformer(Protocol):
    """Protocol for synchronous query transformation strategies.

    A query transformer takes a single query and produces one or more
    derived queries that may improve retrieval quality (e.g. via query
    expansion, decomposition, or hypothetical document generation).
    """

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Transform a query into one or more derived queries.

        Parameters:
            query: The original query bundle to transform.

        Returns:
            A list of ``QueryBundle`` objects derived from the input
            query.  Always returns at least one query.
        """
        ...


@runtime_checkable
class AsyncQueryTransformer(Protocol):
    """Protocol for asynchronous query transformation strategies.

    Async transformers are used when the transformation requires I/O
    (e.g. calling an LLM to generate hypothetical documents or query
    variations).
    """

    async def atransform(self, query: QueryBundle) -> list[QueryBundle]:
        """Asynchronously transform a query into one or more derived queries.

        This is the async counterpart of ``QueryTransformer.transform``,
        intended for use when the transformation involves network calls
        or other async I/O.

        Parameters:
            query: The original query bundle to transform.

        Returns:
            A list of ``QueryBundle`` objects derived from the input
            query.  Always returns at least one query.
        """
        ...
