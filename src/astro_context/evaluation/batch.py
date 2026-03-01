"""Batch evaluation for running evaluations over entire datasets."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astro_context.evaluation.evaluator import PipelineEvaluator
from astro_context.evaluation.models import EvaluationResult, RetrievalMetrics
from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle
from astro_context.protocols.retriever import Retriever

logger = logging.getLogger(__name__)

__all__ = [
    "AggregatedMetrics",
    "BatchEvaluator",
    "EvaluationDataset",
    "EvaluationSample",
]


class EvaluationSample(BaseModel):
    """A single evaluation sample.

    Parameters:
        query: The user query.
        expected_ids: IDs of expected/relevant items.
        ground_truth_answer: Optional reference answer for RAG evaluation.
        contexts: Optional pre-retrieved contexts for RAG evaluation.
        metadata: Arbitrary metadata for this sample.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    expected_ids: list[str] = Field(default_factory=list)
    ground_truth_answer: str | None = None
    contexts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationDataset(BaseModel):
    """A collection of evaluation samples.

    Parameters:
        name: Name of this dataset.
        samples: The evaluation samples.
        metadata: Dataset-level metadata.
    """

    model_config = ConfigDict(frozen=True)

    name: str = "default"
    samples: list[EvaluationSample] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def __len__(self) -> int:
        """Return the number of samples in this dataset."""
        return len(self.samples)

    def __iter__(self) -> Iterator[EvaluationSample]:  # type: ignore[override]
        """Iterate over the evaluation samples."""
        return iter(self.samples)


class AggregatedMetrics(BaseModel):
    """Aggregated metrics across a batch of evaluations.

    Parameters:
        count: Number of samples evaluated.
        mean_precision: Mean precision@k across all samples.
        mean_recall: Mean recall@k.
        mean_f1: Mean F1@k.
        mean_mrr: Mean MRR.
        mean_ndcg: Mean NDCG.
        mean_hit_rate: Mean hit rate.
        p95_precision: 95th percentile precision.
        p95_recall: 95th percentile recall.
        min_precision: Minimum precision.
        min_recall: Minimum recall.
        per_sample_results: Individual evaluation results.
        metadata: Arbitrary metadata.
    """

    model_config = ConfigDict(frozen=True)

    count: int
    mean_precision: float = 0.0
    mean_recall: float = 0.0
    mean_f1: float = 0.0
    mean_mrr: float = 0.0
    mean_ndcg: float = 0.0
    mean_hit_rate: float = 0.0
    p95_precision: float = 0.0
    p95_recall: float = 0.0
    min_precision: float = 0.0
    min_recall: float = 0.0
    per_sample_results: list[EvaluationResult] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


def _percentile(values: list[float], p: float) -> float:
    """Compute the p-th percentile of a list of values.

    Parameters:
        values: The values to compute the percentile over.
        p: The percentile to compute (0-100).

    Returns:
        The p-th percentile value, or 0.0 if the list is empty.
    """
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = int(len(sorted_v) * p / 100.0)
    return sorted_v[min(idx, len(sorted_v) - 1)]


class BatchEvaluator:
    """Runs evaluation over an entire dataset and aggregates results.

    Parameters:
        evaluator: A PipelineEvaluator to run per sample.
        retriever: A Retriever to fetch items for each sample query.
        top_k: Number of items to retrieve per query.
    """

    __slots__ = ("_evaluator", "_retriever", "_top_k")

    def __init__(
        self,
        *,
        evaluator: PipelineEvaluator,
        retriever: Retriever,
        top_k: int = 10,
    ) -> None:
        self._evaluator = evaluator
        self._retriever = retriever
        self._top_k = top_k

    def evaluate(
        self,
        dataset: EvaluationDataset,
        k: int = 10,
    ) -> AggregatedMetrics:
        """Evaluate all samples in the dataset.

        For each sample:
        1. Retrieve items using self._retriever.
        2. Evaluate retrieval quality against sample.expected_ids.
        3. Collect per-sample results.
        4. Aggregate into mean/p95/min metrics.

        Parameters:
            dataset: The dataset of evaluation samples.
            k: Cutoff for top-k evaluation metrics.

        Returns:
            An ``AggregatedMetrics`` instance with aggregated statistics.
        """
        results: list[EvaluationResult] = []
        metrics_list: list[RetrievalMetrics] = []

        for sample in dataset:
            query = QueryBundle(query_str=sample.query)
            retrieved: list[ContextItem] = self._retriever.retrieve(
                query, top_k=self._top_k
            )
            retrieval_metrics = self._evaluator.evaluate_retrieval(
                retrieved, sample.expected_ids, k=k
            )
            result = EvaluationResult(
                retrieval_metrics=retrieval_metrics,
                metadata=dict(sample.metadata),
            )
            results.append(result)
            metrics_list.append(retrieval_metrics)

        if not metrics_list:
            return AggregatedMetrics(count=0, per_sample_results=results)

        count = len(metrics_list)
        precisions = [m.precision_at_k for m in metrics_list]
        recalls = [m.recall_at_k for m in metrics_list]
        f1s = [m.f1_at_k for m in metrics_list]
        mrrs = [m.mrr for m in metrics_list]
        ndcgs = [m.ndcg for m in metrics_list]
        hit_rates = [m.hit_rate for m in metrics_list]

        return AggregatedMetrics(
            count=count,
            mean_precision=sum(precisions) / count,
            mean_recall=sum(recalls) / count,
            mean_f1=sum(f1s) / count,
            mean_mrr=sum(mrrs) / count,
            mean_ndcg=sum(ndcgs) / count,
            mean_hit_rate=sum(hit_rates) / count,
            p95_precision=_percentile(precisions, 95),
            p95_recall=_percentile(recalls, 95),
            min_precision=min(precisions),
            min_recall=min(recalls),
            per_sample_results=results,
        )
