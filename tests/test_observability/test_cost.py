"""Tests for cost tracking observability."""

from __future__ import annotations

import threading

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.observability.cost import (
    CostEntry,
    CostSummary,
    CostTracker,
    CostTrackingCallback,
)


class TestCostEntryFrozen:
    """CostEntry is a frozen Pydantic model."""

    def test_cost_entry_frozen(self) -> None:
        entry = CostEntry(operation="embed", model="text-embedding-3-small")
        with pytest.raises(Exception):  # noqa: B017
            entry.operation = "other"  # type: ignore[misc]


class TestCostSummaryFrozen:
    """CostSummary is a frozen Pydantic model."""

    def test_cost_summary_frozen(self) -> None:
        summary = CostSummary()
        with pytest.raises(Exception):  # noqa: B017
            summary.total_cost_usd = 999.0  # type: ignore[misc]


class TestCostTracker:
    """Tests for the CostTracker class."""

    def test_record_basic(self) -> None:
        tracker = CostTracker()
        entry = tracker.record(
            operation="embedding",
            model="text-embedding-3-small",
            input_tokens=100,
            output_tokens=0,
            cost_per_input_token=0.00002,
        )
        assert entry.operation == "embedding"
        assert entry.model == "text-embedding-3-small"
        assert entry.input_tokens == 100
        assert entry.output_tokens == 0
        assert entry.cost_usd == pytest.approx(0.002)

    def test_record_multiple(self) -> None:
        tracker = CostTracker()
        tracker.record(
            operation="embedding",
            model="model-a",
            input_tokens=100,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="rerank",
            model="model-b",
            input_tokens=200,
            output_tokens=50,
            cost_per_input_token=0.002,
            cost_per_output_token=0.004,
        )
        assert len(tracker.entries) == 2

    def test_summary_aggregation(self) -> None:
        tracker = CostTracker()
        tracker.record(
            operation="embedding",
            model="model-a",
            input_tokens=100,
            output_tokens=0,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="rerank",
            model="model-b",
            input_tokens=200,
            output_tokens=50,
            cost_per_input_token=0.002,
            cost_per_output_token=0.004,
        )
        summary = tracker.summary()
        assert summary.total_input_tokens == 300
        assert summary.total_output_tokens == 50
        # 100*0.001 + 200*0.002 + 50*0.004 = 0.1 + 0.4 + 0.2 = 0.7
        assert summary.total_cost_usd == pytest.approx(0.7)

    def test_summary_by_model(self) -> None:
        tracker = CostTracker()
        tracker.record(
            operation="embed",
            model="model-a",
            input_tokens=100,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="embed",
            model="model-b",
            input_tokens=200,
            cost_per_input_token=0.002,
        )
        tracker.record(
            operation="embed",
            model="model-a",
            input_tokens=50,
            cost_per_input_token=0.001,
        )
        summary = tracker.summary()
        assert summary.by_model["model-a"] == pytest.approx(0.15)
        assert summary.by_model["model-b"] == pytest.approx(0.4)

    def test_summary_by_operation(self) -> None:
        tracker = CostTracker()
        tracker.record(
            operation="embedding",
            model="m1",
            input_tokens=100,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="rerank",
            model="m2",
            input_tokens=200,
            cost_per_input_token=0.002,
        )
        summary = tracker.summary()
        assert summary.by_operation["embedding"] == pytest.approx(0.1)
        assert summary.by_operation["rerank"] == pytest.approx(0.4)

    def test_reset(self) -> None:
        tracker = CostTracker()
        tracker.record(operation="embed", model="m1", input_tokens=100)
        assert len(tracker.entries) == 1
        tracker.reset()
        assert len(tracker.entries) == 0
        assert tracker.summary().total_cost_usd == 0.0

    def test_thread_safety(self) -> None:
        tracker = CostTracker()
        num_threads = 10
        entries_per_thread = 100

        def record_entries() -> None:
            for _ in range(entries_per_thread):
                tracker.record(
                    operation="embed",
                    model="test-model",
                    input_tokens=1,
                    cost_per_input_token=0.001,
                )

        threads = [threading.Thread(target=record_entries) for _ in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(tracker.entries) == num_threads * entries_per_thread

    def test_entries_property(self) -> None:
        tracker = CostTracker()
        tracker.record(operation="embed", model="m1", input_tokens=10)
        entries_copy = tracker.entries
        # Modifying the copy should not affect the tracker
        entries_copy.clear()
        assert len(tracker.entries) == 1

    def test_repr(self) -> None:
        tracker = CostTracker()
        assert "CostTracker(entries=0)" in repr(tracker)
        tracker.record(operation="embed", model="m1", input_tokens=10)
        assert "CostTracker(entries=1)" in repr(tracker)

    def test_zero_token_cost(self) -> None:
        """Record with 0 input/output tokens should yield cost_usd=0."""
        tracker = CostTracker()
        entry = tracker.record(
            operation="embedding",
            model="model-x",
            input_tokens=0,
            output_tokens=0,
            cost_per_input_token=0.01,
            cost_per_output_token=0.02,
        )
        assert entry.cost_usd == 0.0
        summary = tracker.summary()
        assert summary.total_cost_usd == 0.0
        assert summary.total_input_tokens == 0
        assert summary.total_output_tokens == 0

    def test_summary_single_entry(self) -> None:
        """Summary with a single entry: by_model and by_operation each have 1 key."""
        tracker = CostTracker()
        tracker.record(
            operation="rerank",
            model="reranker-v1",
            input_tokens=50,
            cost_per_input_token=0.002,
        )
        summary = tracker.summary()
        assert len(summary.by_model) == 1
        assert "reranker-v1" in summary.by_model
        assert len(summary.by_operation) == 1
        assert "rerank" in summary.by_operation
        assert len(summary.entries) == 1

    def test_multiple_models_same_operation(self) -> None:
        """Multiple models under the same operation group correctly in by_model."""
        tracker = CostTracker()
        tracker.record(
            operation="embedding",
            model="model-a",
            input_tokens=100,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="embedding",
            model="model-b",
            input_tokens=200,
            cost_per_input_token=0.001,
        )
        tracker.record(
            operation="embedding",
            model="model-a",
            input_tokens=300,
            cost_per_input_token=0.001,
        )
        summary = tracker.summary()
        # by_model should have 2 keys
        assert len(summary.by_model) == 2
        assert summary.by_model["model-a"] == pytest.approx(0.4)  # (100+300)*0.001
        assert summary.by_model["model-b"] == pytest.approx(0.2)  # 200*0.001
        # by_operation should have 1 key (all are "embedding")
        assert len(summary.by_operation) == 1
        assert summary.by_operation["embedding"] == pytest.approx(0.6)

    def test_record_returns_cost_entry(self) -> None:
        """record() should return a CostEntry instance with correct values."""
        tracker = CostTracker()
        entry = tracker.record(
            operation="llm",
            model="gpt-4",
            input_tokens=500,
            output_tokens=200,
            cost_per_input_token=0.03,
            cost_per_output_token=0.06,
            metadata={"request_id": "abc123"},
        )
        assert isinstance(entry, CostEntry)
        assert entry.operation == "llm"
        assert entry.model == "gpt-4"
        assert entry.input_tokens == 500
        assert entry.output_tokens == 200
        assert entry.cost_usd == pytest.approx(500 * 0.03 + 200 * 0.06)
        assert entry.metadata["request_id"] == "abc123"
        assert entry.timestamp is not None

    def test_concurrent_reads_during_writes(self) -> None:
        """Thread reads summary() while another thread writes -- no crash or corruption."""
        tracker = CostTracker()
        errors: list[Exception] = []
        summaries: list[float] = []

        def writer() -> None:
            for _ in range(200):
                tracker.record(
                    operation="embed",
                    model="m1",
                    input_tokens=1,
                    cost_per_input_token=0.001,
                )

        def reader() -> None:
            for _ in range(200):
                try:
                    s = tracker.summary()
                    summaries.append(s.total_cost_usd)
                except Exception as e:
                    errors.append(e)

        t_write = threading.Thread(target=writer)
        t_read = threading.Thread(target=reader)
        t_write.start()
        t_read.start()
        t_write.join()
        t_read.join()

        assert len(errors) == 0
        # Final summary should reflect all 200 entries
        final = tracker.summary()
        assert len(final.entries) == 200
        assert final.total_cost_usd == pytest.approx(0.2)

    def test_large_number_of_entries(self) -> None:
        """1000+ records should still produce a correct summary."""
        tracker = CostTracker()
        n = 1500
        for _ in range(n):
            tracker.record(
                operation="embed",
                model="model-x",
                input_tokens=10,
                output_tokens=5,
                cost_per_input_token=0.001,
                cost_per_output_token=0.002,
            )
        summary = tracker.summary()
        assert len(summary.entries) == n
        assert summary.total_input_tokens == 10 * n
        assert summary.total_output_tokens == 5 * n
        # Each entry cost = 10*0.001 + 5*0.002 = 0.01 + 0.01 = 0.02
        assert summary.total_cost_usd == pytest.approx(0.02 * n)
        assert len(summary.by_model) == 1
        assert len(summary.by_operation) == 1


class TestCostTrackingCallback:
    """Tests for the CostTrackingCallback pipeline callback."""

    def test_cost_tracking_callback(self) -> None:
        tracker = CostTracker()
        callback = CostTrackingCallback(tracker)

        query = QueryBundle(query_str="test query")
        callback.on_pipeline_start(query)

        items_with_cost = [
            ContextItem(
                content="test content",
                source=SourceType.RETRIEVAL,
                metadata={
                    "cost_model": "text-embedding-3-small",
                    "cost_input_tokens": 100,
                    "cost_output_tokens": 0,
                    "cost_per_input_token": 0.00002,
                    "cost_per_output_token": 0.0,
                },
            ),
        ]

        callback.on_step_end("retrieval", items_with_cost, time_ms=50.0)

        entries = tracker.entries
        assert len(entries) == 1
        assert entries[0].operation == "retrieval"
        assert entries[0].model == "text-embedding-3-small"
        assert entries[0].input_tokens == 100
        assert entries[0].cost_usd == pytest.approx(0.002)
        assert entries[0].metadata["time_ms"] == 50.0

    def test_callback_ignores_items_without_cost_metadata(self) -> None:
        tracker = CostTracker()
        callback = CostTrackingCallback(tracker)

        items_no_cost = [
            ContextItem(
                content="test content",
                source=SourceType.RETRIEVAL,
            ),
        ]

        callback.on_step_end("retrieval", items_no_cost, time_ms=50.0)
        assert len(tracker.entries) == 0

    def test_callback_repr(self) -> None:
        tracker = CostTracker()
        callback = CostTrackingCallback(tracker)
        assert "CostTrackingCallback" in repr(callback)
