"""Native async reranker implementations."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle

logger = logging.getLogger(__name__)


class AsyncCrossEncoderReranker:
    """Async reranker using a user-provided cross-encoder scoring function.

    Scores all items concurrently via ``asyncio.gather`` and returns
    the top-k by score descending.

    Implements the ``AsyncReranker`` protocol.

    Parameters:
        score_fn: Async callable that takes ``(query_str, doc_content)``
            and returns a relevance score (higher = more relevant).
    """

    __slots__ = ("_score_fn",)

    def __init__(
        self,
        score_fn: Callable[[str, str], Awaitable[float]],
    ) -> None:
        self._score_fn = score_fn

    def __repr__(self) -> str:
        return f"AsyncCrossEncoderReranker(score_fn={'set'})"

    async def arerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int = 10,
    ) -> list[ContextItem]:
        """Asynchronously rerank items using cross-encoder scoring.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            Reranked list of context items, truncated to ``top_k``.
        """
        if not items:
            return []

        scores = await asyncio.gather(
            *(self._score_fn(query.query_str, item.content) for item in items)
        )

        scored: list[tuple[float, ContextItem]] = []
        for score, item in zip(scores, items, strict=True):
            clamped = max(0.0, min(1.0, score))
            updated = item.model_copy(update={"score": clamped})
            scored.append((score, updated))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:top_k]]


class AsyncCohereReranker:
    """Async reranker using a batch reranking callback.

    The callback receives ``(query_str, documents, top_k)`` and returns
    a list of ranked indices. This class maps those indices back to
    ``ContextItem`` objects.

    Implements the ``AsyncReranker`` protocol.

    Parameters:
        rerank_fn: Async callable that takes ``(query, documents, top_k)``
            and returns a list of original indices in ranked order.
    """

    __slots__ = ("_rerank_fn",)

    def __init__(
        self,
        rerank_fn: Callable[[str, list[str], int], Awaitable[list[int]]],
    ) -> None:
        self._rerank_fn = rerank_fn

    def __repr__(self) -> str:
        return f"AsyncCohereReranker(rerank_fn={'set'})"

    async def arerank(
        self,
        query: QueryBundle,
        items: list[ContextItem],
        top_k: int = 10,
    ) -> list[ContextItem]:
        """Asynchronously rerank items using the batch reranking callback.

        Parameters:
            query: The query bundle containing the user's query text.
            items: Candidate context items to rerank.
            top_k: Maximum number of items to return.

        Returns:
            Reranked list of context items in ranked order.
        """
        if not items:
            return []

        documents = [item.content for item in items]
        indices = await self._rerank_fn(query.query_str, documents, top_k)

        result: list[ContextItem] = []
        for idx in indices:
            if 0 <= idx < len(items):
                result.append(items[idx])

        return result[:top_k]
