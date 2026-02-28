"""Tests for the Tracer class."""

from __future__ import annotations

from astro_context.observability.models import SpanKind
from astro_context.observability.tracer import Tracer


class TestTracerLifecycle:
    """Test trace start/end lifecycle."""

    def test_start_trace_creates_record(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test-pipeline")

        assert trace.trace_id
        assert trace.start_time is not None
        assert trace.end_time is None
        assert trace.metadata["name"] == "test-pipeline"

    def test_start_trace_with_attributes(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test", attributes={"query": "hello"})

        assert trace.metadata["query"] == "hello"
        assert trace.metadata["name"] == "test"

    def test_end_trace_sets_timing(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        ended = tracer.end_trace(trace)

        assert ended.end_time is not None
        assert ended.total_duration_ms is not None
        assert ended.total_duration_ms >= 0

    def test_end_trace_removes_from_active(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        assert tracer.get_trace(trace.trace_id) is not None

        tracer.end_trace(trace)
        assert tracer.get_trace(trace.trace_id) is None

    def test_get_trace_returns_none_for_unknown(self) -> None:
        tracer = Tracer()
        assert tracer.get_trace("nonexistent") is None

    def test_multiple_traces(self) -> None:
        tracer = Tracer()
        t1 = tracer.start_trace("first")
        t2 = tracer.start_trace("second")

        assert tracer.get_trace(t1.trace_id) is not None
        assert tracer.get_trace(t2.trace_id) is not None

        tracer.end_trace(t1)
        assert tracer.get_trace(t1.trace_id) is None
        assert tracer.get_trace(t2.trace_id) is not None


class TestSpanLifecycle:
    """Test span start/end within traces."""

    def test_start_span_creates_span(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(
            trace.trace_id, "retrieval", SpanKind.RETRIEVAL,
        )

        assert span.span_id
        assert span.trace_id == trace.trace_id
        assert span.name == "retrieval"
        assert span.kind == SpanKind.RETRIEVAL
        assert span.start_time is not None
        assert span.end_time is None

    def test_start_span_appends_to_trace(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        tracer.start_span(trace.trace_id, "s1", SpanKind.RETRIEVAL)

        updated_trace = tracer.get_trace(trace.trace_id)
        assert updated_trace is not None
        assert len(updated_trace.spans) == 1
        assert updated_trace.spans[0].name == "s1"

    def test_end_span_sets_timing_and_status(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(trace.trace_id, "s1", SpanKind.RETRIEVAL)
        ended = tracer.end_span(span)

        assert ended.end_time is not None
        assert ended.duration_ms is not None
        assert ended.duration_ms >= 0
        assert ended.status == "ok"

    def test_end_span_with_error_status(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(trace.trace_id, "s1", SpanKind.RETRIEVAL)
        ended = tracer.end_span(span, status="error", attributes={"error": "boom"})

        assert ended.status == "error"
        assert ended.attributes["error"] == "boom"

    def test_end_span_merges_attributes(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(
            trace.trace_id, "s1", SpanKind.RETRIEVAL,
            attributes={"items_in": 5},
        )
        ended = tracer.end_span(span, attributes={"items_out": 3})

        assert ended.attributes["items_in"] == 5
        assert ended.attributes["items_out"] == 3

    def test_end_span_updates_trace(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(trace.trace_id, "s1", SpanKind.RETRIEVAL)
        tracer.end_span(span)

        updated_trace = tracer.get_trace(trace.trace_id)
        assert updated_trace is not None
        assert updated_trace.spans[0].end_time is not None


class TestSpanNesting:
    """Test nested span relationships."""

    def test_child_span_references_parent(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        parent = tracer.start_span(trace.trace_id, "pipeline", SpanKind.PIPELINE)
        child = tracer.start_span(
            trace.trace_id, "retrieval", SpanKind.RETRIEVAL,
            parent_span_id=parent.span_id,
        )

        assert child.parent_span_id == parent.span_id

    def test_multiple_children(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        parent = tracer.start_span(trace.trace_id, "pipeline", SpanKind.PIPELINE)
        c1 = tracer.start_span(
            trace.trace_id, "retrieval", SpanKind.RETRIEVAL,
            parent_span_id=parent.span_id,
        )
        c2 = tracer.start_span(
            trace.trace_id, "reranking", SpanKind.RERANKING,
            parent_span_id=parent.span_id,
        )

        updated = tracer.get_trace(trace.trace_id)
        assert updated is not None
        assert len(updated.spans) == 3  # parent + 2 children
        assert c1.parent_span_id == parent.span_id
        assert c2.parent_span_id == parent.span_id


class TestSpanWithoutActiveTrace:
    """Test span operations when the trace is not active."""

    def test_start_span_with_unknown_trace(self) -> None:
        tracer = Tracer()
        # Should not raise, just returns the span without tracking
        span = tracer.start_span("nonexistent", "s1", SpanKind.RETRIEVAL)
        assert span.trace_id == "nonexistent"

    def test_end_span_with_unknown_trace(self) -> None:
        tracer = Tracer()
        trace = tracer.start_trace("test")
        span = tracer.start_span(trace.trace_id, "s1", SpanKind.RETRIEVAL)
        tracer.end_trace(trace)

        # Trace is gone, but end_span should not raise
        ended = tracer.end_span(span)
        assert ended.end_time is not None
