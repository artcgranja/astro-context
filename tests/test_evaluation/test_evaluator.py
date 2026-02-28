"""Tests for PipelineEvaluator orchestrator."""

from __future__ import annotations

import pytest

from astro_context.evaluation.evaluator import PipelineEvaluator
from astro_context.evaluation.models import EvaluationResult, RAGMetrics, RetrievalMetrics
from astro_context.evaluation.rag import LLMRAGEvaluator
from astro_context.evaluation.retrieval import RetrievalMetricsCalculator
from astro_context.models.context import ContextItem, SourceType


def _item(doc_id: str) -> ContextItem:
    """Create a minimal ContextItem with a given ID."""
    return ContextItem(id=doc_id, content=f"doc {doc_id}", source=SourceType.RETRIEVAL)


class TestPipelineEvaluator:
    """Test the orchestrator combining retrieval + RAG evaluation."""

    def test_evaluate_retrieval_only(self) -> None:
        pe = PipelineEvaluator()
        m = pe.evaluate_retrieval(
            retrieved=[_item("a"), _item("b")],
            relevant=["a"],
            k=2,
        )
        assert isinstance(m, RetrievalMetrics)
        assert m.precision_at_k == pytest.approx(0.5)
        assert m.recall_at_k == 1.0

    def test_evaluate_rag_only(self) -> None:
        rag = LLMRAGEvaluator(
            faithfulness_fn=lambda _a, _c: 0.9,
            relevancy_fn=lambda _q, _a: 0.8,
            precision_fn=lambda _q, _c: 0.7,
            recall_fn=lambda _q, _c, _gt: 0.6,
        )
        pe = PipelineEvaluator(rag_evaluator=rag)
        m = pe.evaluate_rag("Q", "A", ["c"], "gt")
        assert isinstance(m, RAGMetrics)
        assert m.faithfulness == pytest.approx(0.9)

    def test_evaluate_rag_raises_without_evaluator(self) -> None:
        pe = PipelineEvaluator()
        with pytest.raises(ValueError, match="No RAG evaluator configured"):
            pe.evaluate_rag("Q", "A", ["c"])

    def test_evaluate_full(self) -> None:
        rag = LLMRAGEvaluator(
            faithfulness_fn=lambda _a, _c: 0.95,
            relevancy_fn=lambda _q, _a: 0.85,
        )
        pe = PipelineEvaluator(rag_evaluator=rag)
        result = pe.evaluate(
            query="Q",
            answer="A",
            retrieved=[_item("a"), _item("b")],
            relevant=["a"],
            contexts=["c1"],
            k=2,
        )
        assert isinstance(result, EvaluationResult)
        assert result.retrieval_metrics is not None
        assert result.retrieval_metrics.precision_at_k == pytest.approx(0.5)
        assert result.rag_metrics is not None
        assert result.rag_metrics.faithfulness == pytest.approx(0.95)

    def test_evaluate_full_without_rag(self) -> None:
        pe = PipelineEvaluator()
        result = pe.evaluate(
            query="Q",
            answer="A",
            retrieved=[_item("a")],
            relevant=["a"],
            contexts=["c1"],
            k=1,
        )
        assert result.retrieval_metrics is not None
        assert result.rag_metrics is None

    def test_evaluate_with_metadata(self) -> None:
        pe = PipelineEvaluator()
        result = pe.evaluate(
            query="Q",
            answer="A",
            retrieved=[],
            relevant=[],
            contexts=[],
            metadata={"run_id": "test-123"},
        )
        assert result.metadata == {"run_id": "test-123"}

    def test_custom_retrieval_calculator(self) -> None:
        calc = RetrievalMetricsCalculator(k=1)
        pe = PipelineEvaluator(retrieval_calculator=calc)
        m = pe.evaluate_retrieval(
            retrieved=[_item("a"), _item("b")],
            relevant=["b"],
            k=1,
        )
        # k=1 means only first item "a" is considered -- "b" is excluded
        assert m.precision_at_k == 0.0
        assert m.hit_rate == 0.0

    def test_evaluate_retrieval_k_override(self) -> None:
        pe = PipelineEvaluator()
        m = pe.evaluate_retrieval(
            retrieved=[_item("x"), _item("a")],
            relevant=["a"],
            k=1,
        )
        # k=1 means only first item "x" -> no hit
        assert m.precision_at_k == 0.0
        assert m.hit_rate == 0.0

    def test_evaluate_full_with_ground_truth(self) -> None:
        rag = LLMRAGEvaluator(
            recall_fn=lambda _q, _c, _gt: 0.77,
        )
        pe = PipelineEvaluator(rag_evaluator=rag)
        result = pe.evaluate(
            query="Q",
            answer="A",
            retrieved=[_item("a")],
            relevant=["a"],
            contexts=["c1"],
            ground_truth="the truth",
            k=1,
        )
        assert result.rag_metrics is not None
        assert result.rag_metrics.context_recall == pytest.approx(0.77)
