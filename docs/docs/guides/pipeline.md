# Pipeline Guide

The `ContextPipeline` is the heart of astro-context. Think of it as an **assembly
line** for LLM context: raw materials (documents, memories, system prompts) enter
one end, pass through a series of processing steps, and a fully-assembled,
token-aware context window exits the other end.

## Basic Usage

The simplest pipeline needs only a token limit and a query:

```python
from astro_context import ContextPipeline

pipeline = ContextPipeline(max_tokens=8192)
result = pipeline.build("What is context engineering?")

print(result.formatted_output)
print(f"Tokens used: {result.window.used_tokens}")
print(f"Build time: {result.build_time_ms:.1f}ms")
```

You can also pass a `QueryBundle` for richer queries:

```python
from astro_context import ContextPipeline, QueryBundle

query = QueryBundle(
    query_str="What is context engineering?",
    metadata={"user_id": "u-42"},
)
result = pipeline.build(query)
```

## Adding Steps

Steps are the building blocks of a pipeline. Each step receives the current list
of `ContextItem` objects and the query, then returns a (potentially modified) list.

### Factory Functions

The fastest way to add steps is with the built-in factory functions.

**Retriever step** -- appends items from a retriever:

```python
from astro_context import (
    ContextPipeline, ContextItem, QueryBundle,
    retriever_step, SourceType,
)

class MyRetriever:
    """Retriever protocol: must have a retrieve(query, top_k) method."""
    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return [
            ContextItem(content="Context engineering is...", source=SourceType.RETRIEVAL, score=0.95),
        ]

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(retriever_step("search", MyRetriever(), top_k=5))
)
result = pipeline.build("What is context engineering?")
print(result.window.items[0].content)
```

**Filter step** -- removes items that fail a predicate:

```python
from astro_context import ContextPipeline, filter_step

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(filter_step("high-score", lambda item: item.score > 0.5))
)
```

**Postprocessor step** -- transforms items through a `PostProcessor`:

```python
from astro_context import postprocessor_step
```

**Reranker step** -- re-scores and reorders items:

```python
from astro_context import reranker_step
```

**Query transform step** -- expands a query into variants, retrieves for each,
and merges results using Reciprocal Rank Fusion:

```python
from astro_context import query_transform_step
```

**Classified retriever step** -- classifies the query and routes to the
appropriate retriever:

```python
from astro_context import classified_retriever_step
```

!!! tip "All factory functions"
    See the [Pipeline API Reference](../api/pipeline.md) for full signatures of
    every factory function including `async_retriever_step`,
    `async_postprocessor_step`, `async_reranker_step`, `auto_promotion_step`,
    `graph_retrieval_step`, and `create_eviction_promoter`.

## Decorator API

For quick prototyping, register steps directly with decorators. This is inspired
by Pydantic AI's `@agent.tool` pattern.

### `@pipeline.step`

Register a **synchronous** function as a pipeline step:

```python
from astro_context import ContextPipeline, ContextItem, QueryBundle, SourceType

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def add_knowledge(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    return items + [
        ContextItem(
            content=f"Relevant fact for: {query.query_str}",
            source=SourceType.RETRIEVAL,
            score=0.9,
        )
    ]

result = pipeline.build("Tell me about black holes")
print(result.window.items[0].content)
```

You can also pass options:

```python
@pipeline.step(name="optional-enrichment", on_error="skip")
def enrich(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    # If this step raises, the pipeline continues without it
    return items
```

### `@pipeline.async_step`

Register an **asynchronous** function. Async steps can only run via `abuild()`:

```python
import asyncio
from astro_context import ContextPipeline, ContextItem, QueryBundle, SourceType

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.async_step
async def fetch_from_api(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    # Simulate an async database call
    await asyncio.sleep(0.01)
    return items + [
        ContextItem(content="Async result", source=SourceType.RETRIEVAL, score=0.8),
    ]

result = asyncio.run(pipeline.abuild("async query"))
print(result.window.items[0].content)
```

!!! warning
    Passing an async function to `@pipeline.step` raises `TypeError`.
    Passing a sync function to `@pipeline.async_step` also raises `TypeError`.
    Use the correct decorator for each function type.

## Chaining Configuration

`ContextPipeline` uses a fluent API -- every configuration method returns `self`
so you can chain calls:

```python
from astro_context import (
    ContextPipeline, ContextItem, QueryBundle, SourceType,
    filter_step, GenericTextFormatter, default_rag_budget,
)

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(filter_step("score-gate", lambda item: item.score > 0.3))
    .add_system_prompt("You are a helpful research assistant.", priority=10)
    .with_formatter(GenericTextFormatter())
    .with_budget(default_rag_budget(8192))
)
```

### `.add_system_prompt(content, priority=10)`

Adds a system-level instruction as a high-priority `ContextItem` with
`source=SourceType.SYSTEM`. Multiple system prompts are supported.

### `.with_memory(memory_provider)`

Attaches a memory provider. Any object with a
`get_context_items() -> list[ContextItem]` method works.

### `.with_formatter(formatter)`

Sets the output formatter. Ships with `GenericTextFormatter`,
`AnthropicFormatter`, and `OpenAIFormatter`.

### `.with_budget(budget)`

Attaches a `TokenBudget` for fine-grained per-source allocation:

```python
from astro_context import ContextPipeline, default_chat_budget

pipeline = ContextPipeline(max_tokens=8192).with_budget(default_chat_budget(8192))
```

See the [Models API Reference](../api/models.md) for `TokenBudget` and budget factories.

### `.add_callback(callback)`

Registers a `PipelineCallback` for observability. Implement any subset of:
`on_pipeline_start`, `on_step_start`, `on_step_end`, `on_step_error`,
`on_pipeline_end`.

### `.with_query_enricher(enricher)`

Attaches a query enricher that rewrites the query using memory context
**before** pipeline steps execute:

```python
from astro_context import ContextPipeline, MemoryContextEnricher

enricher = MemoryContextEnricher(max_items=3)
pipeline = ContextPipeline(max_tokens=8192).with_query_enricher(enricher)
```

## Async Pipeline

Use `abuild()` when your pipeline contains async steps or when you want
to run inside an async application:

```python
import asyncio
from astro_context import ContextPipeline, ContextItem, QueryBundle, SourceType

pipeline = ContextPipeline(max_tokens=4096)

@pipeline.async_step
async def async_lookup(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    await asyncio.sleep(0.01)  # simulate I/O
    return items + [
        ContextItem(content="From async DB", source=SourceType.RETRIEVAL, score=0.9),
    ]

async def main():
    result = await pipeline.abuild("async query")
    print(result.formatted_output)

asyncio.run(main())
```

!!! note
    `abuild()` supports **both** sync and async steps in the same pipeline.
    Sync steps are called directly; async steps are awaited. This lets you mix
    fast in-memory filters with async database lookups.

## Error Handling

Each step has an `on_error` policy:

- `"raise"` (default) -- propagates the error, wrapping unknown exceptions
  in `PipelineExecutionError`
- `"skip"` -- logs the error and continues with the items from **before**
  the failing step

```python
from astro_context import ContextPipeline, ContextItem, QueryBundle, SourceType

pipeline = ContextPipeline(max_tokens=4096)

@pipeline.step(name="flaky-step", on_error="skip")
def flaky(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    raise RuntimeError("transient failure")

result = pipeline.build("test")
print(result.diagnostics.get("skipped_steps"))  # ['flaky-step']
```

!!! warning
    When `on_error="raise"`, the `PipelineExecutionError` carries partial
    diagnostics in its `diagnostics` attribute, so you can still inspect what
    happened before the failure.

## Diagnostics

Every `ContextResult` includes a `diagnostics` dictionary with detailed
information about the build:

```python
from astro_context import ContextPipeline, ContextItem, QueryBundle, SourceType

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def add_items(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    return items + [
        ContextItem(content="Item A", source=SourceType.RETRIEVAL, score=0.9, token_count=10),
        ContextItem(content="Item B", source=SourceType.RETRIEVAL, score=0.7, token_count=15),
    ]

result = pipeline.build("test")
diag = result.diagnostics

print(f"Steps: {diag['steps']}")
#   [{'name': 'add_items', 'items_after': 2, 'time_ms': 0.05}]
print(f"Items included: {diag['items_included']}")
print(f"Items overflow: {diag['items_overflow']}")
print(f"Token utilization: {diag['token_utilization']:.2%}")
```

The `diagnostics` dictionary (`PipelineDiagnostics`) can contain:

| Key | Type | Description |
|---|---|---|
| `steps` | `list[StepDiagnostic]` | Per-step timing and item counts |
| `memory_items` | `int` | Number of memory items injected |
| `total_items_considered` | `int` | Items entering the assembly phase |
| `items_included` | `int` | Items that fit in the window |
| `items_overflow` | `int` | Items that exceeded the token budget |
| `token_utilization` | `float` | Fraction of budget used (0.0--1.0) |
| `token_usage_by_source` | `dict[str, int]` | Tokens per source type (budget mode) |
| `skipped_steps` | `list[str]` | Steps skipped due to `on_error="skip"` |
| `failed_step` | `str` | Name of the step that caused a fatal error |
| `query_enriched` | `bool` | Whether query enrichment was applied |

## Complete Example

A full pipeline combining retrieval, filtering, system prompts, and budgets:

```python
from astro_context import (
    ContextPipeline, ContextItem, QueryBundle, SourceType,
    retriever_step, filter_step, default_rag_budget, GenericTextFormatter,
)

class DocRetriever:
    def __init__(self, docs: list[str]):
        self._docs = docs

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        results = []
        for doc in self._docs:
            if any(w in doc.lower() for w in query.query_str.lower().split()):
                results.append(ContextItem(
                    content=doc, source=SourceType.RETRIEVAL,
                    score=0.85, token_count=len(doc.split()),
                ))
        return results[:top_k]

docs = [
    "Context engineering is the discipline of building dynamic systems.",
    "Token budgets control how much context fits in a prompt.",
    "Retrieval augmented generation combines search with LLMs.",
]

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_system_prompt("You are a technical assistant.")
    .add_step(retriever_step("docs", DocRetriever(docs), top_k=5))
    .add_step(filter_step("score-gate", lambda item: item.score > 0.5))
    .with_budget(default_rag_budget(8192))
    .with_formatter(GenericTextFormatter())
)

result = pipeline.build("What is context engineering?")
print(f"Items: {len(result.window.items)}, Overflow: {len(result.overflow_items)}")
print(f"Tokens: {result.window.used_tokens}/{result.window.max_tokens}")
```

## Next Steps

- [Pipeline API Reference](../api/pipeline.md) -- full constructor and method signatures
- [Models API Reference](../api/models.md) -- `ContextItem`, `TokenBudget`, and all data models
- [Exceptions Reference](../api/exceptions.md) -- error hierarchy
