"""OTLP exporters for spans and metrics.

Bridges astro-context observability types to OpenTelemetry via OTLP/HTTP.
Requires the ``otlp`` extra: ``pip install astro-context[otlp]``
"""

from __future__ import annotations

import logging
from datetime import UTC
from typing import Any

from astro_context.observability.models import Span, SpanKind, TraceRecord

logger = logging.getLogger(__name__)

__all__ = [
    "OTLPMetricsExporter",
    "OTLPSpanExporter",
]


def _convert_record_to_spans(record: TraceRecord) -> list[dict[str, Any]]:
    """Convert a TraceRecord to a list of span dicts.

    Separates serialisation logic from the OTel SDK so it can be tested
    without installing OpenTelemetry packages.

    Parameters:
        record: The trace record containing spans to convert.

    Returns:
        A list of dictionaries, one per span, with keys suitable for
        constructing OTel span objects.
    """
    return [_convert_span(span) for span in record.spans]


def _convert_span(span: Span) -> dict[str, Any]:
    """Convert a single astro-context Span to a dict for export.

    Parameters:
        span: The span to convert.

    Returns:
        A dictionary with span data in a format suitable for OTel export.
    """
    result: dict[str, Any] = {
        "name": span.name,
        "span_id": span.span_id,
        "trace_id": span.trace_id,
        "parent_span_id": span.parent_span_id,
        "kind": _map_span_kind(span.kind),
        "start_time_ns": _datetime_to_ns(span.start_time),
        "status": span.status,
        "attributes": dict(span.attributes),
    }
    if span.end_time is not None:
        result["end_time_ns"] = _datetime_to_ns(span.end_time)
    if span.duration_ms is not None:
        result["duration_ms"] = span.duration_ms
    if span.events:
        result["events"] = list(span.events)
    return result


_SPAN_KIND_MAP: dict[SpanKind, str] = {
    SpanKind.PIPELINE: "INTERNAL",
    SpanKind.RETRIEVAL: "CLIENT",
    SpanKind.RERANKING: "INTERNAL",
    SpanKind.FORMATTING: "INTERNAL",
    SpanKind.MEMORY: "INTERNAL",
    SpanKind.INGESTION: "PRODUCER",
    SpanKind.QUERY_TRANSFORM: "INTERNAL",
}


def _map_span_kind(kind: SpanKind) -> str:
    """Map an astro-context SpanKind to an OTel span kind string.

    Parameters:
        kind: The astro-context span kind.

    Returns:
        An OpenTelemetry span kind string.
    """
    return _SPAN_KIND_MAP.get(kind, "INTERNAL")


def _datetime_to_ns(dt: Any) -> int:
    """Convert a datetime to nanoseconds since the Unix epoch.

    Parameters:
        dt: A datetime object (timezone-aware or naive).

    Returns:
        Nanoseconds since 1970-01-01 00:00:00 UTC.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return int(dt.timestamp() * 1_000_000_000)


class OTLPSpanExporter:
    """Export spans to an OpenTelemetry collector via OTLP/HTTP.

    Converts astro-context ``Span`` objects into OTLP span format
    and exports them via the OpenTelemetry OTLP HTTP exporter.

    Requires the ``opentelemetry-exporter-otlp-proto-http`` and
    ``opentelemetry-sdk`` packages.  Install via::

        pip install astro-context[otlp]

    Implements the ``SpanExporter`` protocol.

    Parameters:
        endpoint: OTLP collector endpoint URL.
            Default ``"http://localhost:4318"``.
        service_name: Service name for the OTLP resource.
            Default ``"astro-context"``.
        headers: Optional headers dict for authentication.
    """

    __slots__ = ("_endpoint", "_exporter", "_headers", "_provider", "_service_name")

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "astro-context",
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                OTLPSpanExporter as _OTLPExporter,
            )
            from opentelemetry.sdk.resources import Resource
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        except ImportError:
            msg = (
                "OTLPSpanExporter requires opentelemetry packages. "
                "Install with: pip install astro-context[otlp]"
            )
            raise ImportError(msg) from None

        self._endpoint = endpoint
        self._service_name = service_name
        self._headers = headers or {}

        resource = Resource.create({"service.name": service_name})
        exporter = _OTLPExporter(
            endpoint=f"{endpoint}/v1/traces",
            headers=self._headers,
        )
        provider = TracerProvider(resource=resource)
        provider.add_span_processor(SimpleSpanProcessor(exporter))

        self._provider = provider
        self._exporter = exporter

    def export(self, spans: list[Span]) -> None:
        """Export spans to the OTLP collector.

        Each astro-context ``Span`` is converted and sent via the
        underlying OTel exporter.

        Parameters:
            spans: The completed spans to export.
        """
        tracer = self._provider.get_tracer(self._service_name)
        for span in spans:
            converted = _convert_span(span)
            otel_span = tracer.start_span(
                name=converted["name"],
                attributes=converted["attributes"],
            )
            otel_span.end()
        logger.debug("Exported %d span(s) via OTLP", len(spans))

    def shutdown(self) -> None:
        """Flush pending spans and shut down the exporter."""
        self._provider.shutdown()

    def export_record(self, record: TraceRecord) -> None:
        """Export all spans from a TraceRecord.

        Convenience method that extracts spans from a ``TraceRecord``
        and exports them.

        Parameters:
            record: The trace record whose spans to export.
        """
        self.export(list(record.spans))


class OTLPMetricsExporter:
    """Export metrics to an OpenTelemetry collector via OTLP/HTTP.

    Converts astro-context ``MetricPoint`` objects into OTLP metric format
    and exports them via the OpenTelemetry OTLP HTTP metrics exporter.

    Implements the ``MetricsCollector`` protocol.

    Parameters:
        endpoint: OTLP collector endpoint URL.
            Default ``"http://localhost:4318"``.
        service_name: Service name for the OTLP resource.
            Default ``"astro-context"``.
        headers: Optional headers dict for authentication.
    """

    __slots__ = ("_endpoint", "_headers", "_meter", "_provider", "_service_name")

    def __init__(
        self,
        endpoint: str = "http://localhost:4318",
        service_name: str = "astro-context",
        headers: dict[str, str] | None = None,
    ) -> None:
        try:
            from opentelemetry.exporter.otlp.proto.http.metric_exporter import (
                OTLPMetricExporter as _OTLPMetricExporter,
            )
            from opentelemetry.sdk.metrics import MeterProvider
            from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
            from opentelemetry.sdk.resources import Resource
        except ImportError:
            msg = (
                "OTLPMetricsExporter requires opentelemetry packages. "
                "Install with: pip install astro-context[otlp]"
            )
            raise ImportError(msg) from None

        self._endpoint = endpoint
        self._service_name = service_name
        self._headers = headers or {}

        resource = Resource.create({"service.name": service_name})
        exporter = _OTLPMetricExporter(
            endpoint=f"{endpoint}/v1/metrics",
            headers=self._headers,
        )
        reader = PeriodicExportingMetricReader(exporter, export_interval_millis=5000)
        provider = MeterProvider(resource=resource, metric_readers=[reader])

        self._provider = provider
        self._meter = provider.get_meter(service_name)

    def record(self, metric: Any) -> None:
        """Record a single metric measurement.

        Accepts a ``MetricPoint`` instance and records its value as an
        OTel gauge observation.

        Parameters:
            metric: The metric point to record.
        """
        from astro_context.observability.models import MetricPoint

        if not isinstance(metric, MetricPoint):
            msg = f"Expected MetricPoint, got {type(metric).__name__}"
            raise TypeError(msg)

        gauge = self._meter.create_gauge(metric.name)
        gauge.set(metric.value, attributes=dict(metric.tags))
        logger.debug("Recorded metric %s=%s", metric.name, metric.value)

    def flush(self) -> None:
        """Flush buffered metrics to the OTLP collector."""
        self._provider.force_flush()

    def shutdown(self) -> None:
        """Shut down the metrics exporter."""
        self._provider.shutdown()
