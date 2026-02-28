"""Tests for RetrievalMetricsCalculator."""

from __future__ import annotations

import math

import pytest

from astro_context.evaluation.models import RetrievalMetrics
from astro_context.evaluation.retrieval import RetrievalMetricsCalculator
from astro_context.models.context import ContextItem, SourceType
from astro_context.protocols.evaluation import RetrievalEvaluator


def _item(doc_id: str) -> ContextItem:
    """Create a minimal ContextItem with a given ID."""
    return ContextItem(id=doc_id, content=f"doc {doc_id}", source=SourceType.RETRIEVAL)


class TestRetrievalMetricsCalculator:
    """Test the pure-computation retrieval metrics."""

    def test_protocol_compliance(self) -> None:
        calc = RetrievalMetricsCalculator()
        assert isinstance(calc, RetrievalEvaluator)

    def test_perfect_retrieval(self) -> None:
        """All retrieved items are relevant and all relevant items are retrieved."""
        retrieved = [_item("a"), _item("b"), _item("c")]
        relevant = ["a", "b", "c"]
        calc = RetrievalMetricsCalculator(k=3)

        m = calc.evaluate(retrieved, relevant)

        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0
        assert m.f1_at_k == 1.0
        assert m.mrr == 1.0
        assert m.ndcg == 1.0
        assert m.hit_rate == 1.0

    def test_no_relevant_retrieved(self) -> None:
        """None of the retrieved items are relevant."""
        retrieved = [_item("x"), _item("y"), _item("z")]
        relevant = ["a", "b"]
        calc = RetrievalMetricsCalculator(k=3)

        m = calc.evaluate(retrieved, relevant)

        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0
        assert m.f1_at_k == 0.0
        assert m.mrr == 0.0
        assert m.ndcg == 0.0
        assert m.hit_rate == 0.0

    def test_partial_retrieval(self) -> None:
        """Some retrieved items are relevant."""
        retrieved = [_item("a"), _item("x"), _item("b"), _item("y")]
        relevant = ["a", "b", "c"]
        calc = RetrievalMetricsCalculator(k=4)

        m = calc.evaluate(retrieved, relevant)

        # precision = 2/4 = 0.5
        assert m.precision_at_k == pytest.approx(0.5)
        # recall = 2/3
        assert m.recall_at_k == pytest.approx(2.0 / 3.0)
        # f1 = 2 * 0.5 * (2/3) / (0.5 + 2/3)
        expected_f1 = 2.0 * 0.5 * (2.0 / 3.0) / (0.5 + 2.0 / 3.0)
        assert m.f1_at_k == pytest.approx(expected_f1)
        # MRR: first relevant at rank 1 -> 1.0
        assert m.mrr == 1.0
        assert m.hit_rate == 1.0

    def test_mrr_first_relevant_at_rank_3(self) -> None:
        """MRR should be 1/rank of first relevant item."""
        retrieved = [_item("x"), _item("y"), _item("a")]
        relevant = ["a"]
        calc = RetrievalMetricsCalculator(k=3)

        m = calc.evaluate(retrieved, relevant)
        assert m.mrr == pytest.approx(1.0 / 3.0)

    def test_ndcg_known_value(self) -> None:
        """NDCG with known positions."""
        # relevant: a, b, c.  Retrieved order: x, a, b
        # DCG = 0 + 1/log2(3) + 1/log2(4)
        # IDCG (3 relevant, 3 slots) = 1/log2(2) + 1/log2(3) + 1/log2(4)
        retrieved = [_item("x"), _item("a"), _item("b")]
        relevant = ["a", "b", "c"]
        calc = RetrievalMetricsCalculator(k=3)

        m = calc.evaluate(retrieved, relevant)

        dcg = 1.0 / math.log2(3) + 1.0 / math.log2(4)
        idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3) + 1.0 / math.log2(4)
        assert m.ndcg == pytest.approx(dcg / idcg)

    def test_empty_retrieved(self) -> None:
        """Empty retrieval results should yield all-zero metrics."""
        calc = RetrievalMetricsCalculator(k=5)
        m = calc.evaluate([], ["a", "b"])

        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0
        assert m.f1_at_k == 0.0
        assert m.mrr == 0.0
        assert m.ndcg == 0.0
        assert m.hit_rate == 0.0

    def test_empty_relevant(self) -> None:
        """No relevant documents should yield zero recall and zero NDCG."""
        calc = RetrievalMetricsCalculator(k=3)
        m = calc.evaluate([_item("a"), _item("b")], [])

        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0
        assert m.f1_at_k == 0.0
        assert m.mrr == 0.0
        assert m.ndcg == 0.0
        assert m.hit_rate == 0.0

    def test_k_override(self) -> None:
        """Per-call k should override the constructor default."""
        retrieved = [_item("a"), _item("b"), _item("c"), _item("d")]
        relevant = ["a", "b"]
        calc = RetrievalMetricsCalculator(k=10)

        m = calc.evaluate(retrieved, relevant, k=2)

        # Only top 2: a (relevant), b (relevant) -> precision 1.0
        assert m.precision_at_k == 1.0
        assert m.recall_at_k == 1.0

    def test_k_less_than_retrieved(self) -> None:
        """Cutoff should limit which items are considered."""
        retrieved = [_item("x"), _item("y"), _item("a")]
        relevant = ["a"]
        calc = RetrievalMetricsCalculator(k=2)

        m = calc.evaluate(retrieved, relevant)

        # k=2 means only [x, y] are considered -- 'a' at position 3 is excluded
        assert m.precision_at_k == 0.0
        assert m.recall_at_k == 0.0
        assert m.hit_rate == 0.0

    def test_invalid_k_constructor(self) -> None:
        with pytest.raises(ValueError, match="k must be >= 1"):
            RetrievalMetricsCalculator(k=0)

    def test_invalid_k_evaluate(self) -> None:
        calc = RetrievalMetricsCalculator()
        with pytest.raises(ValueError, match="k must be >= 1"):
            calc.evaluate([], [], k=0)

    def test_returns_frozen_model(self) -> None:
        calc = RetrievalMetricsCalculator(k=3)
        m = calc.evaluate([_item("a")], ["a"])
        assert isinstance(m, RetrievalMetrics)
        with pytest.raises(Exception):  # noqa: B017
            m.precision_at_k = 0.5  # type: ignore[misc]
