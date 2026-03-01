"""Human-in-the-loop evaluation for retrieval quality assessment.

Provides a collector for human relevance judgments and utilities to
compute inter-annotator agreement and convert judgments into evaluation
datasets.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astro_context.evaluation.ab_testing import EvaluationDataset, EvaluationSample

logger = logging.getLogger(__name__)


class HumanJudgment(BaseModel):
    """A single human relevance judgment for a query-document pair.

    Parameters:
        query: The query that was evaluated.
        item_id: The ID of the document or item being judged.
        relevance: Relevance score from 0 (not relevant) to 3 (highly relevant).
        annotator: Identifier for the person who made the judgment.
        metadata: Arbitrary metadata for the judgment.
    """

    model_config = ConfigDict(frozen=True)

    query: str
    item_id: str
    relevance: int = Field(ge=0, le=3)
    annotator: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class HumanEvaluationCollector:
    """Collect human relevance judgments and compute agreement metrics.

    Supports multiple annotators judging the same query-document pairs,
    with inter-annotator agreement measured via Cohen's kappa.
    """

    __slots__ = ("_judgments",)

    def __init__(self) -> None:
        self._judgments: list[HumanJudgment] = []

    def add_judgment(self, judgment: HumanJudgment) -> None:
        """Add a single judgment.

        Parameters:
            judgment: The human judgment to record.
        """
        self._judgments.append(judgment)

    def add_judgments(self, judgments: list[HumanJudgment]) -> None:
        """Add multiple judgments at once.

        Parameters:
            judgments: The list of human judgments to record.
        """
        self._judgments.extend(judgments)

    @property
    def judgments(self) -> list[HumanJudgment]:
        """Return a copy of all collected judgments."""
        return list(self._judgments)

    def compute_agreement(self) -> float:
        """Compute Cohen's kappa for inter-annotator agreement.

        Finds all (query, item_id) pairs judged by exactly two annotators
        and computes kappa over their ratings.

        Returns:
            Cohen's kappa coefficient.  Returns 0.0 if there are fewer than
            two annotators or no overlapping judgments.
        """
        # Group judgments by (query, item_id)
        pair_judgments: dict[tuple[str, str], list[HumanJudgment]] = defaultdict(list)
        for j in self._judgments:
            pair_judgments[(j.query, j.item_id)].append(j)

        # Collect paired ratings from distinct annotators
        ratings_1: list[int] = []
        ratings_2: list[int] = []
        for pair_list in pair_judgments.values():
            # Find all unique annotators for this pair
            annotators: dict[str, int] = {}
            for j in pair_list:
                if j.annotator not in annotators:
                    annotators[j.annotator] = j.relevance
            # Only consider pairs with exactly 2+ annotators
            annotator_items = list(annotators.items())
            if len(annotator_items) >= 2:
                ratings_1.append(annotator_items[0][1])
                ratings_2.append(annotator_items[1][1])

        if not ratings_1:
            return 0.0

        # Compute observed agreement
        n = len(ratings_1)
        observed = sum(1 for a, b in zip(ratings_1, ratings_2, strict=True) if a == b) / n

        # Compute expected agreement by chance
        categories = set(ratings_1) | set(ratings_2)
        expected = 0.0
        for cat in categories:
            p1 = sum(1 for r in ratings_1 if r == cat) / n
            p2 = sum(1 for r in ratings_2 if r == cat) / n
            expected += p1 * p2

        if expected == 1.0:
            return 1.0 if observed == 1.0 else 0.0

        kappa = (observed - expected) / (1 - expected)
        return kappa

    def to_dataset(self, threshold: int = 2) -> EvaluationDataset:
        """Convert collected judgments into an EvaluationDataset.

        Groups judgments by query.  For each query, items whose mean
        relevance rating meets or exceeds the threshold are considered
        relevant.

        Parameters:
            threshold: Minimum mean relevance for an item to be considered
                relevant (default 2).

        Returns:
            An ``EvaluationDataset`` with one sample per query.
        """
        # Group judgments by query, then by item_id
        query_items: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for j in self._judgments:
            query_items[j.query][j.item_id].append(j.relevance)

        samples: list[EvaluationSample] = []
        for query in sorted(query_items):
            relevant_ids: list[str] = []
            for item_id, ratings in sorted(query_items[query].items()):
                mean_rating = sum(ratings) / len(ratings)
                if mean_rating >= threshold:
                    relevant_ids.append(item_id)
            samples.append(
                EvaluationSample(query=query, relevant_ids=relevant_ids),
            )

        return EvaluationDataset(samples=samples)

    def compute_metrics(self) -> dict[str, float]:
        """Compute summary metrics across all collected judgments.

        Returns:
            A dict with keys ``mean_relevance``, ``agreement``,
            ``num_judgments``, ``num_annotators``, and ``num_queries``.
        """
        if not self._judgments:
            return {
                "mean_relevance": 0.0,
                "agreement": 0.0,
                "num_judgments": 0.0,
                "num_annotators": 0.0,
                "num_queries": 0.0,
            }

        mean_rel = sum(j.relevance for j in self._judgments) / len(self._judgments)
        agreement = self.compute_agreement()
        annotators = {j.annotator for j in self._judgments}
        queries = {j.query for j in self._judgments}

        return {
            "mean_relevance": mean_rel,
            "agreement": agreement,
            "num_judgments": float(len(self._judgments)),
            "num_annotators": float(len(annotators)),
            "num_queries": float(len(queries)),
        }

    def __repr__(self) -> str:
        return f"HumanEvaluationCollector(judgments={len(self._judgments)})"
