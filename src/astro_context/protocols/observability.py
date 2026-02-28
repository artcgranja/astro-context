"""Observability protocols for span export and metrics collection."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from astro_context.observability.models import MetricPoint, Span


@runtime_checkable
class SpanExporter(Protocol):
    """Exports spans to external systems (stdout, file, OTLP, etc.).

    Implementations receive a batch of completed spans and are responsible
    for serialising and delivering them to the target backend.
    """

    def export(self, spans: list[Span]) -> None:
        """Export a batch of spans.

        Parameters:
            spans: The completed spans to export.
        """
        ...


@runtime_checkable
class MetricsCollector(Protocol):
    """Collects and exports metrics.

    Implementations buffer ``MetricPoint`` values and flush them to a
    backend on demand or at regular intervals.
    """

    def record(self, metric: MetricPoint) -> None:
        """Record a single metric measurement.

        Parameters:
            metric: The metric point to record.
        """
        ...

    def flush(self) -> None:
        """Flush buffered metrics to the backend."""
        ...
