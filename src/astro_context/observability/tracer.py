"""Core tracing implementation for pipeline observability."""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from typing import Any

from astro_context.observability.models import Span, SpanKind, TraceRecord

logger = logging.getLogger(__name__)


class Tracer:
    """Creates and manages spans within traces.

    The tracer maintains an internal dictionary of active traces keyed by
    ``trace_id``.  Spans are appended to their parent trace as they are
    started, and updated in-place (via ``model_copy``) when ended.

    Note:
        This implementation is **not** thread-safe.  If you need concurrent
        tracing, synchronise externally or use one ``Tracer`` per thread.
    """

    __slots__ = ("_active_traces",)

    def __init__(self) -> None:
        self._active_traces: dict[str, TraceRecord] = {}

    # -- Trace lifecycle --

    def start_trace(
        self,
        name: str,
        attributes: dict[str, Any] | None = None,
    ) -> TraceRecord:
        """Begin a new trace.

        Parameters:
            name: A human-readable name for the trace (e.g. the pipeline query).
            attributes: Optional metadata to attach to the trace.

        Returns:
            A new ``TraceRecord`` registered in the tracer's active set.
        """
        now = datetime.now(UTC)
        metadata: dict[str, Any] = {"name": name}
        if attributes:
            metadata.update(attributes)

        trace = TraceRecord(start_time=now, metadata=metadata)
        self._active_traces[trace.trace_id] = trace
        logger.debug("Started trace %s (%s)", trace.trace_id, name)
        return trace

    def end_trace(self, trace: TraceRecord) -> TraceRecord:
        """Finalise a trace and remove it from the active set.

        Parameters:
            trace: The trace to end.  Its ``end_time`` and ``total_duration_ms``
                are computed from the current wall-clock time.

        Returns:
            An updated (frozen) ``TraceRecord`` with timing information.
        """
        now = datetime.now(UTC)
        duration_ms: float | None = None
        if trace.start_time is not None:
            delta = now - trace.start_time
            duration_ms = delta.total_seconds() * 1000

        ended = trace.model_copy(
            update={
                "end_time": now,
                "total_duration_ms": duration_ms,
            },
        )
        self._active_traces.pop(trace.trace_id, None)
        logger.debug("Ended trace %s (%.2f ms)", trace.trace_id, duration_ms or 0)
        return ended

    # -- Span lifecycle --

    def start_span(
        self,
        trace_id: str,
        name: str,
        kind: SpanKind,
        parent_span_id: str | None = None,
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Create a new span and attach it to the given trace.

        Parameters:
            trace_id: The trace this span belongs to.
            name: A human-readable name for the span.
            kind: The category of operation (retrieval, formatting, etc.).
            parent_span_id: Optional parent span for nested operations.
            attributes: Optional metadata to attach to the span.

        Returns:
            A new ``Span`` that has been appended to the trace's span list.
        """
        span = Span(
            trace_id=trace_id,
            name=name,
            kind=kind,
            parent_span_id=parent_span_id,
            start_time=datetime.now(UTC),
            attributes=attributes or {},
        )

        trace = self._active_traces.get(trace_id)
        if trace is not None:
            updated = trace.model_copy(update={"spans": [*trace.spans, span]})
            self._active_traces[trace_id] = updated

        logger.debug("Started span %s (%s) in trace %s", span.span_id, name, trace_id)
        return span

    def end_span(
        self,
        span: Span,
        status: str = "ok",
        attributes: dict[str, Any] | None = None,
    ) -> Span:
        """Finalise a span with timing and status information.

        Parameters:
            span: The span to end.
            status: The outcome status (``"ok"`` or ``"error"``).
            attributes: Additional attributes to merge into the span.

        Returns:
            An updated (frozen) ``Span`` with ``end_time`` and ``duration_ms``.
        """
        now = datetime.now(UTC)
        delta = now - span.start_time
        duration_ms = delta.total_seconds() * 1000

        merged_attrs = {**span.attributes, **(attributes or {})}
        ended = span.model_copy(
            update={
                "end_time": now,
                "duration_ms": duration_ms,
                "status": status,
                "attributes": merged_attrs,
            },
        )

        # Update the span in the active trace
        trace = self._active_traces.get(span.trace_id)
        if trace is not None:
            updated_spans = [
                ended if s.span_id == span.span_id else s for s in trace.spans
            ]
            self._active_traces[span.trace_id] = trace.model_copy(
                update={"spans": updated_spans},
            )

        logger.debug(
            "Ended span %s (%s) — %.2f ms", span.span_id, span.name, duration_ms,
        )
        return ended

    # -- Query --

    def get_trace(self, trace_id: str) -> TraceRecord | None:
        """Look up an active trace by its ID.

        Parameters:
            trace_id: The trace to look up.

        Returns:
            The ``TraceRecord`` if still active, or ``None``.
        """
        return self._active_traces.get(trace_id)
