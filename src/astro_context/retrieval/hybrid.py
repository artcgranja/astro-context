"""Hybrid retrieval combining multiple retrievers with Reciprocal Rank Fusion."""

from __future__ import annotations

import logging

from astro_context.exceptions import RetrieverError
from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle
from astro_context.protocols.retriever import Retriever

logger = logging.getLogger(__name__)


class HybridRetriever:
    """Combines multiple retrievers using Reciprocal Rank Fusion (RRF).

    RRF score for document d across N ranking lists:
        RRF(d) = sum(weight_i / (rrf_k + rank_i(d))) for each list i

    The smoothing constant rrf_k (default 60) prevents top-ranked items
    from dominating excessively. This is the standard value from
    the original RRF paper.

    Implements the Retriever protocol.
    """

    __slots__ = ("_retrievers", "_rrf_k", "_weights")

    def __init__(
        self,
        retrievers: list[Retriever],
        rrf_k: int = 60,
        weights: list[float] | None = None,
    ) -> None:
        if not retrievers:
            msg = "At least one retriever is required"
            raise ValueError(msg)
        self._retrievers = retrievers
        self._rrf_k = rrf_k
        if weights is not None:
            if len(weights) != len(retrievers):
                msg = "weights must have same length as retrievers"
                raise ValueError(msg)
            self._weights = weights
        else:
            self._weights = [1.0] * len(retrievers)

    def __repr__(self) -> str:
        return (
            f"HybridRetriever(retrievers={len(self._retrievers)}, "
            f"rrf_k={self._rrf_k}, weights={self._weights})"
        )

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve from all sub-retrievers and fuse results with RRF."""
        all_rankings: list[list[ContextItem]] = []
        successful_weights: list[float] = []
        for retriever, weight in zip(self._retrievers, self._weights, strict=True):
            try:
                results = retriever.retrieve(query, top_k=top_k)
                all_rankings.append(results)
                successful_weights.append(weight)
            except Exception as exc:
                logger.warning(
                    "Sub-retriever %r failed, skipping: %s",
                    retriever, exc,
                )
                continue

        if not all_rankings:
            msg = "All sub-retrievers failed during hybrid retrieval"
            raise RetrieverError(msg)

        rrf_scores: dict[str, float] = {}
        item_map: dict[str, ContextItem] = {}

        for ranking, weight in zip(all_rankings, successful_weights, strict=True):
            for rank, item in enumerate(ranking, start=1):
                rrf_scores[item.id] = rrf_scores.get(item.id, 0.0) + weight / (self._rrf_k + rank)
                if item.id not in item_map or item.score > item_map[item.id].score:
                    item_map[item.id] = item

        sorted_ids = sorted(rrf_scores, key=lambda x: rrf_scores[x], reverse=True)[:top_k]

        if not sorted_ids:
            return []

        max_rrf = rrf_scores[sorted_ids[0]]
        min_rrf = rrf_scores[sorted_ids[-1]] if len(sorted_ids) > 1 else 0.0
        score_range = max_rrf - min_rrf if max_rrf > min_rrf else 1.0

        fused_results: list[ContextItem] = []
        for item_id in sorted_ids:
            original = item_map[item_id]
            normalized_score = (
                (rrf_scores[item_id] - min_rrf) / score_range if score_range > 0 else 1.0
            )
            fused_item = original.model_copy(update={
                "score": min(1.0, max(0.0, normalized_score)),
                "metadata": {
                    **original.metadata,
                    "retrieval_method": "hybrid_rrf",
                    "rrf_raw_score": rrf_scores[item_id],
                },
            })
            fused_results.append(fused_item)
        return fused_results
