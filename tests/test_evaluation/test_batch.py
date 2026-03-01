"""Tests for batch evaluation module."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from astro_context.evaluation.batch import (
    AggregatedMetrics,
    BatchEvaluator,
    EvaluationDataset,
    EvaluationSample,
    _percentile,
)
from astro_context.evaluation.evaluator import PipelineEvaluator
from astro_context.models.context import ContextItem, SourceType
from tests.conftest import FakeRetriever


def _item(doc_id: str) -> ContextItem:
    """Create a minimal ContextItem with a given ID."""
    return ContextItem(id=doc_id, content=f"doc {doc_id}", source=SourceType.RETRIEVAL)


class TestEvaluationSample:
    """Tests for EvaluationSample model."""

    def test_basic_creation(self) -> None:
        sample = EvaluationSample(query="test query", expected_ids=["a", "b"])
        assert sample.query == "test query"
        assert sample.expected_ids == ["a", "b"]
        assert sample.ground_truth_answer is None
        assert sample.contexts == []
        assert sample.metadata == {}

    def test_full_creation(self) -> None:
        sample = EvaluationSample(
            query="test",
            expected_ids=["a"],
            ground_truth_answer="answer",
            contexts=["ctx1"],
            metadata={"key": "val"},
        )
        assert sample.ground_truth_answer == "answer"
        assert sample.contexts == ["ctx1"]
        assert sample.metadata == {"key": "val"}

    def test_frozen(self) -> None:
        sample = EvaluationSample(query="q", expected_ids=["a"])
        with pytest.raises(ValidationError):
            sample.query = "new"  # type: ignore[misc]


class TestEvaluationDataset:
    """Tests for EvaluationDataset model."""

    def test_empty_dataset(self) -> None:
        ds = EvaluationDataset(name="test")
        assert len(ds) == 0

    def test_with_samples(self) -> None:
        samples = [
            EvaluationSample(query="q1", expected_ids=["a", "b"]),
            EvaluationSample(query="q2", expected_ids=["c"]),
        ]
        ds = EvaluationDataset(name="test", samples=samples)
        assert len(ds) == 2

    def test_iteration(self) -> None:
        samples = [
            EvaluationSample(query="q1", expected_ids=["a"]),
            EvaluationSample(query="q2", expected_ids=["b"]),
        ]
        ds = EvaluationDataset(name="test", samples=samples)
        collected = list(ds)
        assert len(collected) == 2
        assert collected[0].query == "q1"
        assert collected[1].query == "q2"

    def test_frozen(self) -> None:
        ds = EvaluationDataset(name="test")
        with pytest.raises(ValidationError):
            ds.name = "other"  # type: ignore[misc]

    def test_default_name(self) -> None:
        ds = EvaluationDataset()
        assert ds.name == "default"

    def test_metadata(self) -> None:
        ds = EvaluationDataset(name="test", metadata={"version": "1.0"})
        assert ds.metadata == {"version": "1.0"}


class TestAggregatedMetrics:
    """Tests for AggregatedMetrics model."""

    def test_frozen(self) -> None:
        metrics = AggregatedMetrics(count=0)
        with pytest.raises(ValidationError):
            metrics.count = 5  # type: ignore[misc]

    def test_defaults(self) -> None:
        metrics = AggregatedMetrics(count=3)
        assert metrics.count == 3
        assert metrics.mean_precision == 0.0
        assert metrics.mean_recall == 0.0
        assert metrics.mean_f1 == 0.0
        assert metrics.mean_mrr == 0.0
        assert metrics.mean_ndcg == 0.0
        assert metrics.mean_hit_rate == 0.0
        assert metrics.p95_precision == 0.0
        assert metrics.p95_recall == 0.0
        assert metrics.min_precision == 0.0
        assert metrics.min_recall == 0.0
        assert metrics.per_sample_results == []
        assert metrics.metadata == {}


class TestPercentile:
    """Tests for the _percentile helper."""

    def test_empty(self) -> None:
        assert _percentile([], 95) == 0.0

    def test_single_value(self) -> None:
        assert _percentile([0.5], 95) == 0.5

    def test_multiple_values(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        p95 = _percentile(values, 95)
        # idx = int(10 * 95 / 100) = 9 -> sorted_v[9] = 1.0
        assert p95 == pytest.approx(1.0)

    def test_p50(self) -> None:
        values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        p50 = _percentile(values, 50)
        # idx = int(10 * 50 / 100) = 5 -> sorted_v[5] = 0.6
        assert p50 == pytest.approx(0.6)

    def test_unsorted_input(self) -> None:
        values = [0.9, 0.1, 0.5, 0.3, 0.7]
        p95 = _percentile(values, 95)
        # sorted: [0.1, 0.3, 0.5, 0.7, 0.9], idx = int(5 * 0.95) = 4 -> 0.9
        assert p95 == pytest.approx(0.9)


class TestBatchEvaluator:
    """Tests for BatchEvaluator."""

    def test_evaluate_single_sample(self) -> None:
        items = [_item("a"), _item("b")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[EvaluationSample(query="q1", expected_ids=["a"])],
        )
        result = batch_eval.evaluate(ds, k=2)

        assert result.count == 1
        assert result.mean_precision == pytest.approx(0.5)
        assert result.mean_recall == pytest.approx(1.0)
        assert len(result.per_sample_results) == 1

    def test_evaluate_multiple_samples(self) -> None:
        items = [_item("a"), _item("b"), _item("c")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
                EvaluationSample(query="q2", expected_ids=["a", "b", "c"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=3)

        assert result.count == 2
        assert len(result.per_sample_results) == 2

    def test_mean_metrics(self) -> None:
        # Sample 1: retrieve [a, b], relevant=[a] -> precision=0.5, recall=1.0
        # Sample 2: retrieve [a, b], relevant=[a, b] -> precision=1.0, recall=1.0
        items = [_item("a"), _item("b")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
                EvaluationSample(query="q2", expected_ids=["a", "b"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=2)

        assert result.mean_precision == pytest.approx(0.75)  # (0.5 + 1.0) / 2
        assert result.mean_recall == pytest.approx(1.0)  # (1.0 + 1.0) / 2

    def test_p95_metrics(self) -> None:
        items = [_item("a"), _item("b")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
                EvaluationSample(query="q2", expected_ids=["a", "b"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=2)

        # Two values: [0.5, 1.0] sorted -> idx = int(2 * 0.95) = 1 -> 1.0
        assert result.p95_precision == pytest.approx(1.0)
        assert result.p95_recall == pytest.approx(1.0)

    def test_min_metrics(self) -> None:
        items = [_item("a"), _item("b")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
                EvaluationSample(query="q2", expected_ids=["a", "b"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=2)

        assert result.min_precision == pytest.approx(0.5)
        assert result.min_recall == pytest.approx(1.0)

    def test_empty_dataset(self) -> None:
        items = [_item("a")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(name="empty")
        result = batch_eval.evaluate(ds)

        assert result.count == 0
        assert result.mean_precision == 0.0
        assert result.mean_recall == 0.0
        assert result.per_sample_results == []

    def test_per_sample_results_preserved(self) -> None:
        items = [_item("a"), _item("b")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
                EvaluationSample(query="q2", expected_ids=["b"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=2)

        assert len(result.per_sample_results) == 2
        for r in result.per_sample_results:
            assert r.retrieval_metrics is not None

    def test_top_k_parameter(self) -> None:
        """Test that top_k controls retriever output."""
        items = [_item("a"), _item("b"), _item("c")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever, top_k=1)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=1)

        # top_k=1 means only item "a" is retrieved
        assert result.mean_precision == pytest.approx(1.0)
        assert result.mean_recall == pytest.approx(1.0)

    def test_no_relevant_items_found(self) -> None:
        """Test when retrieved items do not match expected."""
        items = [_item("x"), _item("y")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(query="q1", expected_ids=["a", "b"]),
            ],
        )
        result = batch_eval.evaluate(ds, k=2)

        assert result.mean_precision == 0.0
        assert result.mean_recall == 0.0
        assert result.mean_hit_rate == 0.0

    def test_sample_metadata_preserved(self) -> None:
        """Test that per-sample metadata is carried through."""
        items = [_item("a")]
        retriever = FakeRetriever(items)
        evaluator = PipelineEvaluator()
        batch_eval = BatchEvaluator(evaluator=evaluator, retriever=retriever)

        ds = EvaluationDataset(
            name="test",
            samples=[
                EvaluationSample(
                    query="q1",
                    expected_ids=["a"],
                    metadata={"category": "test"},
                ),
            ],
        )
        result = batch_eval.evaluate(ds, k=1)

        assert result.per_sample_results[0].metadata == {"category": "test"}
