"""Tests for metrics collectors."""

from __future__ import annotations

from astro_context.observability.metrics import InMemoryMetricsCollector, LoggingMetricsCollector
from astro_context.observability.models import MetricPoint
from astro_context.protocols.observability import MetricsCollector


def _make_metric(
    name: str = "test.metric",
    value: float = 1.0,
    tags: dict[str, str] | None = None,
) -> MetricPoint:
    """Create a test metric with sensible defaults."""
    return MetricPoint(name=name, value=value, tags=tags or {})


class TestInMemoryMetricsCollector:
    """Tests for the InMemoryMetricsCollector."""

    def test_protocol_compliance(self) -> None:
        collector = InMemoryMetricsCollector()
        assert isinstance(collector, MetricsCollector)

    def test_record_and_get(self) -> None:
        collector = InMemoryMetricsCollector()
        metric = _make_metric("latency", 42.0)
        collector.record(metric)

        result = collector.get_metrics()
        assert len(result) == 1
        assert result[0].name == "latency"
        assert result[0].value == 42.0

    def test_get_metrics_by_name(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric("latency", 10.0))
        collector.record(_make_metric("throughput", 100.0))
        collector.record(_make_metric("latency", 20.0))

        latency = collector.get_metrics("latency")
        assert len(latency) == 2
        assert all(m.name == "latency" for m in latency)

    def test_get_metrics_no_match(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric("latency", 10.0))

        result = collector.get_metrics("nonexistent")
        assert result == []

    def test_flush_is_noop(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric())
        collector.flush()
        # Metrics should still be available
        assert len(collector.get_metrics()) == 1

    def test_clear(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric())
        collector.clear()
        assert len(collector.get_metrics()) == 0


class TestInMemoryMetricsSummary:
    """Tests for summary statistics computation."""

    def test_summary_basic(self) -> None:
        collector = InMemoryMetricsCollector()
        for v in [10.0, 20.0, 30.0, 40.0, 50.0]:
            collector.record(_make_metric("latency", v))

        summary = collector.get_summary("latency")
        assert summary["min"] == 10.0
        assert summary["max"] == 50.0
        assert summary["avg"] == 30.0
        assert summary["count"] == 5

    def test_summary_single_value(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric("latency", 42.0))

        summary = collector.get_summary("latency")
        assert summary["min"] == 42.0
        assert summary["max"] == 42.0
        assert summary["avg"] == 42.0
        assert summary["count"] == 1
        assert summary["p50"] == 42.0
        assert summary["p95"] == 42.0

    def test_summary_percentiles(self) -> None:
        collector = InMemoryMetricsCollector()
        # 100 values from 1 to 100
        for i in range(1, 101):
            collector.record(_make_metric("latency", float(i)))

        summary = collector.get_summary("latency")
        assert summary["count"] == 100
        # p50 should be around 50
        assert 49.0 <= summary["p50"] <= 51.0
        # p95 should be around 95
        assert 94.0 <= summary["p95"] <= 96.0

    def test_summary_empty(self) -> None:
        collector = InMemoryMetricsCollector()
        summary = collector.get_summary("nonexistent")
        assert summary == {}

    def test_summary_two_values(self) -> None:
        collector = InMemoryMetricsCollector()
        collector.record(_make_metric("latency", 10.0))
        collector.record(_make_metric("latency", 20.0))

        summary = collector.get_summary("latency")
        assert summary["min"] == 10.0
        assert summary["max"] == 20.0
        assert summary["avg"] == 15.0
        assert summary["count"] == 2


class TestLoggingMetricsCollector:
    """Tests for the LoggingMetricsCollector."""

    def test_protocol_compliance(self) -> None:
        collector = LoggingMetricsCollector()
        assert isinstance(collector, MetricsCollector)

    def test_record_does_not_raise(self) -> None:
        collector = LoggingMetricsCollector()
        collector.record(_make_metric())
        # Smoke test

    def test_flush_does_not_raise(self) -> None:
        collector = LoggingMetricsCollector()
        collector.flush()


class TestMetricPointModel:
    """Tests for the MetricPoint Pydantic model."""

    def test_frozen(self) -> None:
        metric = _make_metric()
        try:
            metric.value = 999.0  # type: ignore[misc]
            raised = False
        except Exception:
            raised = True
        assert raised

    def test_default_timestamp(self) -> None:
        metric = _make_metric()
        assert metric.timestamp is not None

    def test_default_tags(self) -> None:
        metric = _make_metric()
        assert metric.tags == {}

    def test_custom_tags(self) -> None:
        metric = _make_metric(tags={"step": "retrieval"})
        assert metric.tags["step"] == "retrieval"
