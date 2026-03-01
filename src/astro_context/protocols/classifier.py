"""Query classification protocol definitions.

Any object with a ``classify`` method matching this signature can be used
as a query classifier -- no inheritance required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.query import QueryBundle


@runtime_checkable
class QueryClassifier(Protocol):
    """Protocol for query classification strategies.

    A query classifier inspects a ``QueryBundle`` and returns a string
    label indicating the category or intent of the query.  This label
    can be used by downstream components (e.g. routers or classified
    retriever steps) to select an appropriate retrieval strategy.
    """

    def classify(self, query: QueryBundle) -> str:
        """Classify a query and return a string label.

        Parameters:
            query: The query bundle to classify.

        Returns:
            A string label representing the query category.
        """
        ...
