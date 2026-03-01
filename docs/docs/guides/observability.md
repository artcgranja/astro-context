# Observability

astro-context provides built-in tracing, metrics collection, and cost tracking
for pipeline execution. Every span, metric point, and cost entry is a Pydantic
model -- fully typed and serialisable.

---

## Tracing with Tracer

`Tracer` is the low-level API for creating traces and spans. A **trace**
groups all operations from a single pipeline execution; each operation is
represented as a **span**.

```python
from astro_context.observability import Tracer, SpanKind

tracer = Tracer()

# Start a trace
trace = tracer.start_trace("my-pipeline", attributes={"user": "demo"})

# Start a span within the trace
span = tracer.start_span(
    trace_id=trace.trace_id,
    name="retrieval",
    kind=SpanKind.RETRIEVAL,
    attributes={"top_k": 10},
)

# ... perform retrieval work ...

# End the span
ended_span = tracer.end_span(span, status="ok", attributes={"items": 5})
print(f"Span took {ended_span.duration_ms:.1f} ms")

# End the trace
ended_trace = tracer.end_trace(trace)
print(f"Trace took {ended_trace.total_duration_ms:.1f} ms")
```

### SpanKind values

| Value | When to use |
|-------|-------------|
| `SpanKind.PIPELINE` | Root pipeline span |
| `SpanKind.RETRIEVAL` | Retrieval / search operations |
| `SpanKind.RERANKING` | Reranking steps |
| `SpanKind.FORMATTING` | Output formatting |
| `SpanKind.MEMORY` | Memory / history operations |
| `SpanKind.INGESTION` | Indexing and ingestion |
| `SpanKind.QUERY_TRANSFORM` | Query rewriting / expansion |

!!! note
    `Tracer` is **not** thread-safe. Use one instance per thread or
    synchronise externally for concurrent tracing.

---

## Automatic Pipeline Tracing

`TracingCallback` hooks into the pipeline lifecycle and creates spans
automatically for the overall execution and each individual step.

```python
from astro_context.observability import (
    TracingCallback,
    InMemorySpanExporter,
    InMemoryMetricsCollector,
)

exporter = InMemorySpanExporter()
metrics = InMemoryMetricsCollector()

callback = TracingCallback(
    exporters=[exporter],
    metrics_collector=metrics,
)

# Attach to your pipeline
pipeline.add_callback(callback)

# Run the pipeline -- spans are recorded automatically
result = pipeline.build("What is context engineering?")

# Inspect the trace
trace = callback.last_trace
print(f"Trace ID: {trace.trace_id}")
print(f"Duration: {trace.total_duration_ms:.1f} ms")
print(f"Spans:    {len(trace.spans)}")

# Inspect individual spans
for span in exporter.get_spans():
    print(f"  {span.name}: {span.duration_ms:.1f} ms [{span.status}]")
```

!!! tip
    `TracingCallback` automatically infers `SpanKind` from the step name
    using heuristics (e.g. a step named `"rerank"` maps to
    `SpanKind.RERANKING`).

---

## Span Exporters

Exporters receive a batch of completed spans and deliver them to a backend.
All exporters implement the `SpanExporter` protocol.

### ConsoleSpanExporter

Logs spans as JSON via Python's `logging` module. Useful for development.

```python
import logging
from astro_context.observability import ConsoleSpanExporter

logging.basicConfig(level=logging.INFO)
exporter = ConsoleSpanExporter(log_level=logging.INFO)

callback = TracingCallback(exporters=[exporter])
```

### InMemorySpanExporter

Stores spans in an in-memory list. Ideal for testing and debugging.

```python
from astro_context.observability import InMemorySpanExporter

exporter = InMemorySpanExporter()

# After pipeline execution:
spans = exporter.get_spans()
exporter.clear()  # reset for the next run
```

### FileSpanExporter

Appends spans as JSON-Lines to a file on disk.

```python
from astro_context.observability import FileSpanExporter

exporter = FileSpanExporter(path="traces.jsonl")

callback = TracingCallback(exporters=[exporter])
```

### OTLPSpanExporter

Exports spans to an OpenTelemetry collector via OTLP/HTTP. Requires the
`otlp` extra.

```bash
pip install astro-context[otlp]
```

```python
from astro_context.observability import OTLPSpanExporter

exporter = OTLPSpanExporter(
    endpoint="http://localhost:4318",
    service_name="my-app",
    headers={"Authorization": "Bearer ..."},
)

callback = TracingCallback(exporters=[exporter])

# Shut down cleanly when done
exporter.shutdown()
```

!!! warning
    `OTLPSpanExporter` raises `ImportError` at construction if the
    `opentelemetry-exporter-otlp-proto-http` and `opentelemetry-sdk`
    packages are not installed.

---

## Metrics Collection

Metrics collectors record `MetricPoint` values. `TracingCallback` records
step durations and pipeline build times automatically when a collector is
configured.

### InMemoryMetricsCollector

Stores metrics in memory with summary statistics.

```python
from astro_context.observability import InMemoryMetricsCollector, MetricPoint

collector = InMemoryMetricsCollector()

# Record manually
collector.record(MetricPoint(name="step.duration_ms", value=12.5, tags={"step": "retrieval"}))
collector.record(MetricPoint(name="step.duration_ms", value=8.3, tags={"step": "rerank"}))

# Query metrics
all_step_durations = collector.get_metrics("step.duration_ms")
print(f"Recorded {len(all_step_durations)} duration metrics")

# Get summary statistics
summary = collector.get_summary("step.duration_ms")
print(f"  avg: {summary['avg']:.1f} ms")
print(f"  p95: {summary['p95']:.1f} ms")
print(f"  min: {summary['min']:.1f} ms")
print(f"  max: {summary['max']:.1f} ms")

collector.clear()
```

### LoggingMetricsCollector

Emits each metric as a structured JSON log message immediately on record.

```python
import logging
from astro_context.observability import LoggingMetricsCollector

logging.basicConfig(level=logging.INFO)
collector = LoggingMetricsCollector(log_level=logging.INFO)
```

### OTLPMetricsExporter

Exports metrics to an OpenTelemetry collector via OTLP/HTTP. Requires the
`otlp` extra.

```python
from astro_context.observability import OTLPMetricsExporter

exporter = OTLPMetricsExporter(
    endpoint="http://localhost:4318",
    service_name="my-app",
)

callback = TracingCallback(metrics_collector=exporter)

# Flush and shut down when done
exporter.flush()
exporter.shutdown()
```

---

## Cost Tracking

`CostTracker` accumulates per-operation cost entries (tokens, USD) and
produces aggregated summaries. It is thread-safe.

```python
from astro_context.observability import CostTracker

tracker = CostTracker()

# Record embedding cost
tracker.record(
    operation="embedding",
    model="text-embedding-3-small",
    input_tokens=500,
    cost_per_input_token=0.00002,
)

# Record reranking cost
tracker.record(
    operation="rerank",
    model="rerank-v3",
    input_tokens=2000,
    cost_per_input_token=0.00001,
)

# Get summary
summary = tracker.summary()
print(f"Total cost:    ${summary.total_cost_usd:.4f}")
print(f"Input tokens:  {summary.total_input_tokens}")
print(f"By model:      {summary.by_model}")
print(f"By operation:  {summary.by_operation}")

# Reset for next run
tracker.reset()
```

### CostTrackingCallback

`CostTrackingCallback` is a pipeline callback that automatically records
cost entries when pipeline steps produce items with cost-related metadata.

```python
from astro_context.observability import CostTracker, CostTrackingCallback

tracker = CostTracker()
cost_callback = CostTrackingCallback(tracker=tracker)

# Attach both tracing and cost tracking
pipeline.add_callback(cost_callback)

# After pipeline execution
summary = tracker.summary()
print(f"Pipeline cost: ${summary.total_cost_usd:.4f}")
```

The callback looks for these metadata keys on `ContextItem.metadata`:

| Key | Type | Description |
|-----|------|-------------|
| `cost_model` | `str` | Model identifier (triggers recording) |
| `cost_input_tokens` | `int` | Input tokens consumed |
| `cost_output_tokens` | `int` | Output tokens produced |
| `cost_per_input_token` | `float` | USD per input token |
| `cost_per_output_token` | `float` | USD per output token |

!!! tip
    Combine `TracingCallback` and `CostTrackingCallback` on the same
    pipeline to get both performance traces and cost breakdowns.

---

## Full Example: Tracing + Metrics + Cost

```python
from astro_context.observability import (
    CostTracker,
    CostTrackingCallback,
    InMemoryMetricsCollector,
    InMemorySpanExporter,
    TracingCallback,
)

# Set up all observers
span_exporter = InMemorySpanExporter()
metrics_collector = InMemoryMetricsCollector()
cost_tracker = CostTracker()

tracing_cb = TracingCallback(
    exporters=[span_exporter],
    metrics_collector=metrics_collector,
)
cost_cb = CostTrackingCallback(tracker=cost_tracker)

pipeline.add_callback(tracing_cb)
pipeline.add_callback(cost_cb)

# Run the pipeline
result = pipeline.build("What is context engineering?")

# Inspect results
print("--- Spans ---")
for span in span_exporter.get_spans():
    print(f"  {span.name}: {span.duration_ms:.1f} ms")

print("--- Metrics ---")
summary = metrics_collector.get_summary("pipeline.build_time_ms")
if summary:
    print(f"  Build time avg: {summary['avg']:.1f} ms")

print("--- Cost ---")
cost_summary = cost_tracker.summary()
print(f"  Total: ${cost_summary.total_cost_usd:.4f}")
```

---

## Next Steps

- [Observability API Reference](../api/observability.md) -- full class and method signatures
- [Evaluation Guide](../guides/evaluation.md) -- measure retrieval and RAG quality
- [Pipeline Guide](../guides/pipeline.md) -- build and configure context pipelines
