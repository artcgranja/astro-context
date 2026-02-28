"""Query transformation pipeline for chaining multiple transformers."""

from __future__ import annotations

import logging

from astro_context.models.query import QueryBundle
from astro_context.protocols.query_transform import AsyncQueryTransformer, QueryTransformer

logger = logging.getLogger(__name__)


class QueryTransformPipeline:
    """Chains multiple query transformers and deduplicates results.

    Each transformer is applied to every query produced by the previous
    stage, producing a flat list of unique queries at the end.

    Parameters:
        transformers: An ordered sequence of ``QueryTransformer``
            implementations to apply.
    """

    __slots__ = ("_transformers",)

    def __init__(self, transformers: list[QueryTransformer]) -> None:
        if not transformers:
            msg = "At least one transformer is required"
            raise ValueError(msg)
        self._transformers = list(transformers)

    def __repr__(self) -> str:
        names = ", ".join(repr(t) for t in self._transformers)
        return f"QueryTransformPipeline([{names}])"

    @staticmethod
    def _deduplicate(queries: list[QueryBundle]) -> list[QueryBundle]:
        """Remove duplicate queries based on query_str, preserving order."""
        seen: set[str] = set()
        unique: list[QueryBundle] = []
        for q in queries:
            if q.query_str not in seen:
                seen.add(q.query_str)
                unique.append(q)
        return unique

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        """Apply all transformers in sequence and deduplicate.

        Parameters:
            query: The original query to transform.

        Returns:
            A deduplicated list of ``QueryBundle`` objects produced by
            chaining all transformers.
        """
        current: list[QueryBundle] = [query]
        for transformer in self._transformers:
            next_queries: list[QueryBundle] = []
            for q in current:
                next_queries.extend(transformer.transform(q))
            current = next_queries
        result = self._deduplicate(current)
        logger.debug(
            "QueryTransformPipeline produced %d unique queries from %d transformers",
            len(result),
            len(self._transformers),
        )
        return result

    async def atransform(self, query: QueryBundle) -> list[QueryBundle]:
        """Async version: apply all transformers in sequence and deduplicate.

        Transformers that implement ``AsyncQueryTransformer`` are called
        via ``atransform``; others fall back to the synchronous
        ``transform`` method.

        Parameters:
            query: The original query to transform.

        Returns:
            A deduplicated list of ``QueryBundle`` objects produced by
            chaining all transformers.
        """
        current: list[QueryBundle] = [query]
        for transformer in self._transformers:
            next_queries: list[QueryBundle] = []
            for q in current:
                if isinstance(transformer, AsyncQueryTransformer):
                    next_queries.extend(await transformer.atransform(q))
                else:
                    next_queries.extend(transformer.transform(q))
            current = next_queries
        result = self._deduplicate(current)
        logger.debug(
            "QueryTransformPipeline (async) produced %d unique queries from %d transformers",
            len(result),
            len(self._transformers),
        )
        return result
