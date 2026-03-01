"""Pipeline evaluator that orchestrates retrieval and RAG evaluation."""

from __future__ import annotations

import logging
from typing import Any

from astro_context.evaluation.models import EvaluationResult, RAGMetrics, RetrievalMetrics
from astro_context.evaluation.rag import LLMRAGEvaluator
from astro_context.evaluation.retrieval import RetrievalMetricsCalculator
from astro_context.models.context import ContextItem

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """Orchestrates retrieval and RAG evaluation into a single result.

    Combines a ``RetrievalMetricsCalculator`` for offline retrieval metrics
    with an optional ``LLMRAGEvaluator`` for LLM-judged RAG quality.

    Parameters:
        retrieval_calculator: Calculator for precision/recall/MRR/NDCG.
            Defaults to a new ``RetrievalMetricsCalculator()`` if not provided.
        rag_evaluator: Optional LLM-based evaluator for faithfulness, relevancy,
            context precision, and context recall.
    """

    __slots__ = ("_rag_evaluator", "_retrieval_calculator")

    def __init__(
        self,
        *,
        retrieval_calculator: RetrievalMetricsCalculator | None = None,
        rag_evaluator: LLMRAGEvaluator | None = None,
    ) -> None:
        self._retrieval_calculator = retrieval_calculator or RetrievalMetricsCalculator()
        self._rag_evaluator = rag_evaluator

    def evaluate_retrieval(
        self,
        retrieved: list[ContextItem],
        relevant: list[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality only.

        Parameters:
            retrieved: Items returned by the retriever in ranked order.
            relevant: IDs of the ground-truth relevant documents.
            k: Cutoff for top-k evaluation.

        Returns:
            A ``RetrievalMetrics`` instance.
        """
        return self._retrieval_calculator.evaluate(retrieved, relevant, k=k)

    def evaluate_rag(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGMetrics:
        """Evaluate RAG output quality only.

        Parameters:
            query: The original user query.
            answer: The generated answer.
            contexts: Context strings fed to the generator.
            ground_truth: Optional reference answer for recall.

        Returns:
            A ``RAGMetrics`` instance.

        Raises:
            ValueError: If no RAG evaluator was configured.
        """
        if self._rag_evaluator is None:
            msg = "No RAG evaluator configured; pass rag_evaluator to PipelineEvaluator"
            raise ValueError(msg)
        return self._rag_evaluator.evaluate(query, answer, contexts, ground_truth)

    def evaluate(
        self,
        query: str,
        answer: str,
        retrieved: list[ContextItem],
        relevant: list[str],
        contexts: list[str],
        ground_truth: str | None = None,
        k: int = 10,
        metadata: dict[str, Any] | None = None,
    ) -> EvaluationResult:
        """Run both retrieval and RAG evaluation.

        Parameters:
            query: The original user query.
            answer: The generated answer.
            retrieved: Items returned by the retriever in ranked order.
            relevant: IDs of the ground-truth relevant documents.
            contexts: Context strings fed to the generator.
            ground_truth: Optional reference answer for recall.
            k: Cutoff for top-k evaluation.
            metadata: Arbitrary metadata for the result.

        Returns:
            An ``EvaluationResult`` combining retrieval and RAG metrics.
        """
        retrieval_metrics = self._retrieval_calculator.evaluate(retrieved, relevant, k=k)

        rag_metrics: RAGMetrics | None = None
        if self._rag_evaluator is not None:
            rag_metrics = self._rag_evaluator.evaluate(query, answer, contexts, ground_truth)

        return EvaluationResult(
            retrieval_metrics=retrieval_metrics,
            rag_metrics=rag_metrics,
            metadata=metadata or {},
        )
