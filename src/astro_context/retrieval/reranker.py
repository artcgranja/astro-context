"""Reranker postprocessor for two-stage retrieval."""

from __future__ import annotations

from collections.abc import Callable

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


class ScoreReranker:
    """Reranks context items using a user-provided scoring function.

    The scoring function receives (query_str, document_content) and returns
    a relevance score (higher = more relevant). Items are re-sorted by
    this score and optionally truncated to top_k.

    Usage::

        def my_scorer(query: str, doc: str) -> float:
            # Call a cross-encoder, Cohere rerank API, etc.
            return model.predict(query, doc)

        reranker = ScoreReranker(score_fn=my_scorer, top_k=10)
        reranked_items = reranker.process(items, query)
    """

    def __init__(
        self,
        score_fn: Callable[[str, str], float],
        top_k: int | None = None,
    ) -> None:
        self._score_fn = score_fn
        self._top_k = top_k

    @property
    def format_type(self) -> str:
        return "reranker"

    def process(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]:
        if not items or query is None:
            return items

        scored_items: list[tuple[float, ContextItem]] = []
        for item in items:
            score = self._score_fn(query.query_str, item.content)
            updated = item.model_copy(update={"score": max(0.0, min(1.0, score))})
            scored_items.append((score, updated))

        scored_items.sort(key=lambda x: x[0], reverse=True)
        result = [item for _, item in scored_items]

        if self._top_k is not None:
            result = result[: self._top_k]

        return result
