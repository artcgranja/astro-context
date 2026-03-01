# Observability API Reference

All classes are importable from `astro_context.observability`.
For usage examples see the [Observability Guide](../guides/observability.md).

---

## SpanKind

String enum representing the category of a traced operation.

```python
from astro_context.observability import SpanKind
```

| Value | String | Description |
|-------|--------|-------------|
| `PIPELINE` | `"pipeline"` | Root pipeline span |
| `RETRIEVAL` | `"retrieval"` | Retrieval / search |
| `RERANKING` | `"reranking"` | Reranking steps |
| `FORMATTING` | `"formatting"` | Output formatting |
| `MEMORY` | `"memory"` | Memory operations |
| `INGESTION` | `"ingestion"` | Indexing / ingestion |
| `QUERY_TRANSFORM` | `"query_transform"` | Query rewriting |

---

## Span

Immutable Pydantic model representing a single traced operation. Spans form a
tree via `parent_span_id`.

```python
from astro_context.observability import Span
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `span_id` | `str` | auto UUID | Unique span identifier |
| `parent_span_id` | `str \| None` | `None` | Parent span for nesting |
| `trace_id` | `str` | -- | Trace this span belongs to |
| `name` | `str` | -- | Human-readable operation name |
| `kind` | `SpanKind` | -- | Category of operation |
| `start_time` | `datetime` | now (UTC) | When the span started |
| `end_time` | `datetime \| None` | `None` | When the span ended |
| `duration_ms` | `float \| None` | `None` | Duration in milliseconds |
| `status` | `str` | `"ok"` | Outcome: `"ok"` or `"error"` |
| `attributes` | `dict[str, Any]` | `{}` | Arbitrary key-value metadata |
| `events` | `list[dict[str, Any]]` | `[]` | Timestamped event log |

---

## TraceRecord

Immutable Pydantic model grouping all spans from a single pipeline execution.

```python
from astro_context.observability import TraceRecord
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `trace_id` | `str` | auto UUID | Unique trace identifier |
| `spans` | `list[Span]` | `[]` | Spans in this trace |
| `start_time` | `datetime \| None` | `None` | Trace start time |
| `end_time` | `datetime \| None` | `None` | Trace end time |
| `total_duration_ms` | `float \| None` | `None` | Total duration in ms |
| `metadata` | `dict[str, Any]` | `{}` | Trace-level metadata |

---

## MetricPoint

Immutable Pydantic model for a single metric measurement.

```python
from astro_context.observability import MetricPoint
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `name` | `str` | -- | Metric name (e.g. `"pipeline.build_time_ms"`) |
| `value` | `float` | -- | Numeric measurement value |
| `timestamp` | `datetime` | now (UTC) | When the measurement was taken |
| `tags` | `dict[str, str]` | `{}` | Labels for filtering and grouping |

---

## Tracer

Creates and manages spans within traces.

```python
from astro_context.observability import Tracer
```

**Constructor:** `Tracer()` -- no parameters.

!!! note
    `Tracer` is **not** thread-safe. Synchronise externally or use one
    instance per thread.

**`start_trace(name, attributes=None) -> TraceRecord`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | -- | Human-readable trace name |
| `attributes` | `dict[str, Any] \| None` | `None` | Metadata to attach |

**`end_trace(trace) -> TraceRecord`** -- finalises a trace, computes `end_time` and `total_duration_ms`, removes it from the active set.

**`start_span(trace_id, name, kind, parent_span_id=None, attributes=None) -> Span`**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `trace_id` | `str` | -- | Trace this span belongs to |
| `name` | `str` | -- | Operation name |
| `kind` | `SpanKind` | -- | Category of operation |
| `parent_span_id` | `str \| None` | `None` | Parent span for nesting |
| `attributes` | `dict[str, Any] \| None` | `None` | Metadata to attach |

**`end_span(span, status="ok", attributes=None) -> Span`** -- finalises a span with `end_time`, `duration_ms`, and merged attributes.

**`get_trace(trace_id) -> TraceRecord | None`** -- look up an active trace by ID.

---

## TracingCallback

Pipeline callback that automatically traces execution via spans.

```python
from astro_context.observability import TracingCallback
```

**Constructor:**

```python
TracingCallback(
    tracer: Tracer | None = None,
    exporters: list[SpanExporter] | None = None,
    metrics_collector: MetricsCollector | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tracer` | `Tracer \| None` | `None` | Custom tracer; creates one if omitted |
| `exporters` | `list[SpanExporter] \| None` | `None` | Span exporters for completed spans |
| `metrics_collector` | `MetricsCollector \| None` | `None` | Collector for timing metrics |

**Properties:**

| Property | Type | Description |
|----------|------|-------------|
| `tracer` | `Tracer` | Underlying tracer instance |
| `last_trace` | `TraceRecord \| None` | Most recently completed trace |

**Callback methods** (called by the pipeline automatically):

- `on_pipeline_start(query: QueryBundle) -> None`
- `on_step_start(step_name: str, items: list[ContextItem]) -> None`
- `on_step_end(step_name: str, items: list[ContextItem], time_ms: float) -> None`
- `on_step_error(step_name: str, error: Exception) -> None`
- `on_pipeline_end(result: ContextResult) -> None`

---

## ConsoleSpanExporter

Exports spans as structured JSON to Python's logging system.

```python
from astro_context.observability import ConsoleSpanExporter
```

**Constructor:** `ConsoleSpanExporter(log_level: int = logging.INFO)`

**`export(spans: list[Span]) -> None`** -- logs each span as a JSON object.

---

## InMemorySpanExporter

Stores spans in memory for testing and debugging.

```python
from astro_context.observability import InMemorySpanExporter
```

**Constructor:** `InMemorySpanExporter()`

**`export(spans: list[Span]) -> None`** -- appends spans to internal buffer.

**`get_spans() -> list[Span]`** -- returns a copy of all exported spans.

**`clear() -> None`** -- removes all stored spans.

---

## FileSpanExporter

Writes spans as JSON-Lines to a file in append mode.

```python
from astro_context.observability import FileSpanExporter
```

**Constructor:** `FileSpanExporter(path: str | Path)`

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str \| Path` | File path to write to; parent directories must exist |

**`export(spans: list[Span]) -> None`** -- appends one JSON object per span.

---

## OTLPSpanExporter

Exports spans to an OpenTelemetry collector via OTLP/HTTP.

```python
from astro_context.observability import OTLPSpanExporter
```

!!! warning
    Requires `opentelemetry-exporter-otlp-proto-http` and
    `opentelemetry-sdk`. Install with `pip install astro-context[otlp]`.

**Constructor:**

```python
OTLPSpanExporter(
    endpoint: str = "http://localhost:4318",
    service_name: str = "astro-context",
    headers: dict[str, str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | `"http://localhost:4318"` | OTLP collector URL |
| `service_name` | `str` | `"astro-context"` | OTel resource service name |
| `headers` | `dict[str, str] \| None` | `None` | Auth headers |

**`export(spans: list[Span]) -> None`** -- converts and exports spans.

**`export_record(record: TraceRecord) -> None`** -- exports all spans from a `TraceRecord`.

**`shutdown() -> None`** -- flushes pending spans and shuts down.

---

## InMemoryMetricsCollector

Stores metric points in memory with summary statistics.

```python
from astro_context.observability import InMemoryMetricsCollector
```

**Constructor:** `InMemoryMetricsCollector()`

**`record(metric: MetricPoint) -> None`** -- stores a metric point.

**`flush() -> None`** -- no-op; metrics remain in memory.

**`get_metrics(name: str | None = None) -> list[MetricPoint]`** -- returns stored metrics, optionally filtered by name.

**`get_summary(name: str) -> dict[str, Any]`** -- returns `{"min", "max", "avg", "count", "p50", "p95"}` for the named metric. Empty dict if no matches.

**`clear() -> None`** -- removes all stored metrics.

---

## LoggingMetricsCollector

Logs each metric as structured JSON via Python's logging module.

```python
from astro_context.observability import LoggingMetricsCollector
```

**Constructor:** `LoggingMetricsCollector(log_level: int = logging.INFO)`

**`record(metric: MetricPoint) -> None`** -- logs the metric immediately.

**`flush() -> None`** -- no-op.

---

## OTLPMetricsExporter

Exports metrics to an OpenTelemetry collector via OTLP/HTTP.

```python
from astro_context.observability import OTLPMetricsExporter
```

!!! warning
    Requires `opentelemetry-exporter-otlp-proto-http` and
    `opentelemetry-sdk`. Install with `pip install astro-context[otlp]`.

**Constructor:**

```python
OTLPMetricsExporter(
    endpoint: str = "http://localhost:4318",
    service_name: str = "astro-context",
    headers: dict[str, str] | None = None,
)
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `endpoint` | `str` | `"http://localhost:4318"` | OTLP collector URL |
| `service_name` | `str` | `"astro-context"` | OTel resource service name |
| `headers` | `dict[str, str] \| None` | `None` | Auth headers |

**`record(metric: MetricPoint) -> None`** -- records as an OTel gauge. Raises `TypeError` if not a `MetricPoint`.

**`flush() -> None`** -- flushes buffered metrics.

**`shutdown() -> None`** -- shuts down the exporter.

---

## CostEntry

Immutable Pydantic model for a single cost record.

```python
from astro_context.observability import CostEntry
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `operation` | `str` | -- | Operation type (e.g. `"embedding"`) |
| `model` | `str` | -- | Model identifier |
| `input_tokens` | `int` | `0` | Input tokens consumed |
| `output_tokens` | `int` | `0` | Output tokens produced |
| `cost_usd` | `float` | `0.0` | Total cost in USD |
| `timestamp` | `datetime` | now (UTC) | When the operation occurred |
| `metadata` | `dict[str, Any]` | `{}` | Arbitrary metadata |

## CostSummary

Immutable Pydantic model for aggregated cost.

```python
from astro_context.observability import CostSummary
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `total_cost_usd` | `float` | `0.0` | Sum of all entry costs |
| `total_input_tokens` | `int` | `0` | Sum of all input tokens |
| `total_output_tokens` | `int` | `0` | Sum of all output tokens |
| `entries` | `list[CostEntry]` | `[]` | Individual cost entries |
| `by_model` | `dict[str, float]` | `{}` | Cost breakdown by model |
| `by_operation` | `dict[str, float]` | `{}` | Cost breakdown by operation |

---

## CostTracker

Thread-safe tracker for accumulating cost entries.

```python
from astro_context.observability import CostTracker
```

**Constructor:** `CostTracker()` -- no parameters.

**Property:** `entries -> list[CostEntry]` -- copy of all recorded entries.

**`record(...) -> CostEntry`**

```python
def record(
    self,
    operation: str,
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    cost_per_input_token: float = 0.0,
    cost_per_output_token: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> CostEntry
```

Computes `cost_usd = input_tokens * cost_per_input_token + output_tokens * cost_per_output_token`.

**`summary() -> CostSummary`** -- aggregates all entries with per-model and per-operation breakdowns.

**`reset() -> None`** -- clears all recorded entries.

---

## CostTrackingCallback

Pipeline callback that records cost entries from step execution metadata.

```python
from astro_context.observability import CostTrackingCallback
```

**Constructor:** `CostTrackingCallback(tracker: CostTracker)`

Records a cost entry on `on_step_end` when items contain `cost_model` in
their metadata. Other callback methods are no-ops.

Expected metadata keys on `ContextItem.metadata`:

| Key | Type | Description |
|-----|------|-------------|
| `cost_model` | `str` | Model identifier (triggers recording) |
| `cost_input_tokens` | `int` | Input tokens consumed |
| `cost_output_tokens` | `int` | Output tokens produced |
| `cost_per_input_token` | `float` | USD per input token |
| `cost_per_output_token` | `float` | USD per output token |
