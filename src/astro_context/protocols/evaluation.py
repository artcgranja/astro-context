"""Evaluation protocol definitions.

Any object implementing ``evaluate`` with the matching signature can be used
as an evaluator -- no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from astro_context.evaluation.models import RAGMetrics, RetrievalMetrics
from astro_context.models.context import ContextItem


@runtime_checkable
class HumanEvaluator(Protocol):
    """Protocol for human-in-the-loop evaluators.

    Implementations collect human judgments and compute inter-annotator
    agreement metrics.
    """

    def add_judgment(self, judgment: Any) -> None:
        """Record a single human judgment.

        Parameters:
            judgment: The judgment object to record.
        """
        ...

    def compute_agreement(self) -> float:
        """Compute inter-annotator agreement.

        Returns:
            A float representing the agreement score (e.g. Cohen's kappa).
        """
        ...


@runtime_checkable
class RAGEvaluator(Protocol):
    """Protocol for RAG-quality evaluators.

    Implementations assess faithfulness, relevancy, context precision and
    context recall -- typically by delegating to an LLM judge.
    """

    def evaluate(
        self,
        query: str,
        answer: str,
        contexts: list[str],
        ground_truth: str | None = None,
    ) -> RAGMetrics:
        """Evaluate RAG output quality.

        Parameters:
            query: The original user query.
            answer: The generated answer.
            contexts: The context strings that were fed to the generator.
            ground_truth: Optional reference answer for recall computation.

        Returns:
            A ``RAGMetrics`` instance with all computed scores.
        """
        ...


@runtime_checkable
class RetrievalEvaluator(Protocol):
    """Protocol for retrieval-quality evaluators.

    Implementations compute precision, recall, MRR, NDCG, etc. by comparing
    retrieved items against a set of known-relevant document IDs.
    """

    def evaluate(
        self,
        retrieved: list[ContextItem],
        relevant: list[str],
        k: int = 10,
    ) -> RetrievalMetrics:
        """Evaluate retrieval quality.

        Parameters:
            retrieved: The items returned by the retriever, in ranked order.
            relevant: IDs of documents that are relevant to the query.
            k: Number of top results to consider.

        Returns:
            A ``RetrievalMetrics`` instance with all computed scores.
        """
        ...
