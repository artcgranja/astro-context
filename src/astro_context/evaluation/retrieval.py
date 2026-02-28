"""Pure-computation retrieval metrics calculator.

No LLM dependencies -- all metrics are computed from ranked results and a
known set of relevant document IDs.
"""

from __future__ import annotations

import logging
import math

from astro_context.evaluation.models import RetrievalMetrics
from astro_context.models.context import ContextItem

logger = logging.getLogger(__name__)


class RetrievalMetricsCalculator:
    """Compute standard retrieval metrics against a ground-truth relevance set.

    Parameters:
        k: Default cutoff for top-k evaluation.  Can be overridden per call.
    """

    __slots__ = ("_k",)

    def __init__(self, k: int = 10) -> None:
        if k < 1:
            msg = f"k must be >= 1, got {k}"
            raise ValueError(msg)
        self._k = k

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        retrieved: list[ContextItem],
        relevant: list[str],
        k: int | None = None,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality.

        Parameters:
            retrieved: Items returned by the retriever in ranked order.
            relevant: IDs of the ground-truth relevant documents.
            k: Cutoff to use.  Falls back to the instance default.

        Returns:
            A ``RetrievalMetrics`` instance with precision@k, recall@k,
            F1@k, MRR, NDCG, and hit_rate.
        """
        effective_k = k if k is not None else self._k
        if effective_k < 1:
            msg = f"k must be >= 1, got {effective_k}"
            raise ValueError(msg)

        top_k = retrieved[:effective_k]
        relevant_set = set(relevant)

        precision = self._precision_at_k(top_k, relevant_set)
        recall = self._recall_at_k(top_k, relevant_set)
        f1 = self._f1(precision, recall)
        mrr = self._mrr(top_k, relevant_set)
        ndcg = self._ndcg(top_k, relevant_set)
        hit = self._hit_rate(top_k, relevant_set)

        return RetrievalMetrics(
            precision_at_k=precision,
            recall_at_k=recall,
            f1_at_k=f1,
            mrr=mrr,
            ndcg=ndcg,
            hit_rate=hit,
        )

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _precision_at_k(top_k: list[ContextItem], relevant: set[str]) -> float:
        """Fraction of retrieved items that are relevant."""
        if not top_k:
            return 0.0
        hits = sum(1 for item in top_k if item.id in relevant)
        return hits / len(top_k)

    @staticmethod
    def _recall_at_k(top_k: list[ContextItem], relevant: set[str]) -> float:
        """Fraction of relevant items that were retrieved."""
        if not relevant:
            return 0.0
        hits = sum(1 for item in top_k if item.id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _f1(precision: float, recall: float) -> float:
        """Harmonic mean of precision and recall."""
        if precision + recall == 0.0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _mrr(top_k: list[ContextItem], relevant: set[str]) -> float:
        """Reciprocal rank of the first relevant item."""
        for rank, item in enumerate(top_k, start=1):
            if item.id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg(top_k: list[ContextItem], relevant: set[str]) -> float:
        """Normalized Discounted Cumulative Gain (binary relevance)."""
        if not relevant or not top_k:
            return 0.0

        # DCG: sum of 1/log2(rank+1) for each relevant hit
        dcg = 0.0
        for rank, item in enumerate(top_k, start=1):
            if item.id in relevant:
                dcg += 1.0 / math.log2(rank + 1)

        # Ideal DCG: best possible ranking (all relevant items first)
        ideal_hits = min(len(relevant), len(top_k))
        idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))

        if idcg == 0.0:
            return 0.0
        return dcg / idcg

    @staticmethod
    def _hit_rate(top_k: list[ContextItem], relevant: set[str]) -> float:
        """1.0 if at least one relevant item appears in top_k, else 0.0."""
        return 1.0 if any(item.id in relevant for item in top_k) else 0.0
