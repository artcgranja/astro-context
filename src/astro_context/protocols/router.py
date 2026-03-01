"""Protocol definition for query routing."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.query import QueryBundle


@runtime_checkable
class QueryRouter(Protocol):
    """Routes queries to named retriever backends.

    A router inspects the query and returns the name of the retriever
    that should handle it. The name must match a key in the retriever
    map provided to ``RoutedRetriever``.
    """

    def route(self, query: QueryBundle) -> str:
        """Determine which retriever should handle this query.

        Parameters:
            query: The query to route.

        Returns:
            The name/key of the target retriever.
        """
        ...
