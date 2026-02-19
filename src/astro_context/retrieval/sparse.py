"""Sparse (BM25) retrieval.

Requires the 'bm25' extra: pip install astro-context[bm25]
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.tokens.counter import get_default_counter

if TYPE_CHECKING:
    from rank_bm25 import BM25Okapi


class SparseRetriever:
    """BM25-based retrieval over tokenized documents.

    Implements the Retriever protocol.
    """

    def __init__(
        self,
        tokenize_fn: Callable[[str], list[str]] | None = None,
    ) -> None:
        self._tokenize_fn = tokenize_fn or self._default_tokenize
        self._bm25: BM25Okapi | None = None
        self._items: list[ContextItem] = []
        self._counter = get_default_counter()

    @staticmethod
    def _default_tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenization."""
        return text.lower().split()

    def index(self, items: list[ContextItem]) -> int:
        """Build the BM25 index from context items."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as e:
            msg = (
                "rank-bm25 is required for SparseRetriever. "
                "Install it with: pip install astro-context[bm25]"
            )
            raise ImportError(msg) from e

        self._items = list(items)
        tokenized_corpus = [self._tokenize_fn(item.content) for item in self._items]
        self._bm25 = BM25Okapi(tokenized_corpus)
        return len(self._items)

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items using BM25 scoring."""
        if self._bm25 is None:
            msg = "Must call index() before retrieve()"
            raise RuntimeError(msg)

        tokenized_query = self._tokenize_fn(query.query_str)
        scores = self._bm25.get_scores(tokenized_query)

        max_score = max(scores) if max(scores) > 0 else 1.0
        scored_indices = [(i, float(s / max_score)) for i, s in enumerate(scores)]
        scored_indices.sort(key=lambda x: x[1], reverse=True)
        scored_indices = scored_indices[:top_k]

        items: list[ContextItem] = []
        for idx, score in scored_indices:
            if score <= 0:
                continue
            item = self._items[idx]
            scored_item = ContextItem(
                id=item.id,
                content=item.content,
                source=SourceType.RETRIEVAL,
                score=score,
                priority=item.priority,
                token_count=item.token_count or self._counter.count_tokens(item.content),
                metadata={**item.metadata, "retrieval_method": "sparse_bm25"},
                created_at=item.created_at,
            )
            items.append(scored_item)
        return items
