# astro-context

**Context engineering toolkit for AI applications.**

*Stop duct-taping RAG, memory, and tools together. Build intelligent context pipelines in minutes.*

---

## Why astro-context?

Most AI frameworks focus on the LLM call. But the real challenge is assembling the
right **context** -- the system prompt, conversation memory, retrieved documents,
and tool outputs that the model actually sees. astro-context gives you a
single, composable pipeline that manages all of it within a strict token budget.

---

## Features

**Hybrid RAG** -- Dense embeddings + BM25 sparse retrieval with Reciprocal Rank Fusion.
Combine multiple retrieval strategies in a single pipeline for higher recall and precision.

**Smart Memory** -- Token-aware sliding window with automatic eviction.
Oldest turns are evicted when the conversation exceeds its budget, so you never
lose recent context.

**Token Budgets** -- Never exceed your context window.
Priority-ranked assembly fills from highest-priority items down. Per-source
allocations let you reserve tokens for system prompts, memory, retrieval, and
LLM responses independently.

**Provider Agnostic** -- Anthropic, OpenAI, or plain text.
Format the assembled context for any LLM provider with a single method call.
Swap providers without changing your pipeline.

**Protocol-Based** -- Plug in anything via PEP 544.
Every extension point (retriever, tokenizer, reranker, memory store) is defined
as a structural protocol. Bring your own implementations without inheriting from
base classes.

**Type-Safe** -- Pydantic v2 throughout.
All models are frozen Pydantic v2 dataclasses with full `py.typed` support.
Catch integration errors at type-check time, not at runtime.

---

## Quick Start

Build your first context pipeline in 30 seconds:

```python
from astro_context import ContextPipeline, MemoryManager, AnthropicFormatter

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(MemoryManager(conversation_tokens=4096))
    .with_formatter(AnthropicFormatter())
    .add_system_prompt("You are a helpful assistant.")
)

result = pipeline.build("What is context engineering?")
print(result.formatted_output)   # Ready for the Anthropic API
print(result.diagnostics)        # Token usage, timing, overflow info
```

!!! tip
    `build()` accepts either a plain `str` or a `QueryBundle` object.
    Plain strings are automatically wrapped in a `QueryBundle` for you.

---

## How It Works

```
ContextPipeline
  |
  |-- System Prompts (priority=10)
  |-- Memory Manager (priority=7)
  |-- Pipeline Steps
  |     |-- Retriever Steps (append items)
  |     |-- PostProcessor Steps (transform items)
  |     |-- Filter Steps (filter items)
  |
  v
ContextWindow (token-aware, priority-ranked)
  |
  v
Formatter (Anthropic / OpenAI / Generic)
  |
  v
ContextResult (formatted output + diagnostics)
```

Every `ContextItem` carries a `priority` (1--10). When the total exceeds
`max_tokens`, the pipeline fills from highest priority down. Items that do not
fit are tracked in `result.overflow_items`.

---

## Comparison

| Feature | LangChain | LlamaIndex | mem0 | **astro-context** |
|---------|:---------:|:----------:|:----:|:-----------------:|
| Hybrid RAG (Dense + BM25 + RRF) | partial | yes | no | **yes** |
| Token-aware Memory | partial | no | yes | **yes** |
| Token Budget Management | no | no | no | **yes** |
| Provider-agnostic Formatting | no | no | no | **yes** |
| Protocol-based Plugins | no | partial | no | **yes** |
| Zero-config Defaults | no | no | yes | **yes** |

---

## Installation

```bash
pip install astro-context
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add astro-context
```

### Optional Extras

```bash
pip install astro-context[bm25]   # BM25 sparse retrieval (rank-bm25)
pip install astro-context[cli]    # CLI tools (typer + rich)
pip install astro-context[all]    # Everything
```

---

## Hybrid Retrieval Example

```python
import math
from astro_context import (
    ContextPipeline, ContextItem, QueryBundle, SourceType,
    DenseRetriever, InMemoryContextStore, InMemoryVectorStore,
    retriever_step,
)

def my_embed_fn(text: str) -> list[float]:
    """Simple deterministic embedding for demonstration."""
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw

dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

items = [
    ContextItem(content="Python is great for data science.", source=SourceType.RETRIEVAL),
    ContextItem(content="RAG combines retrieval with generation.", source=SourceType.RETRIEVAL),
]
dense.index(items)

pipeline = ContextPipeline(max_tokens=8192).add_step(
    retriever_step("search", dense, top_k=5)
)
query = QueryBundle(
    query_str="How does RAG work?",
    embedding=my_embed_fn("How does RAG work?"),
)
result = pipeline.build(query)
```

---

## Token Budgets

For fine-grained control over how tokens are allocated across sources,
use the preset budget factories:

```python
from astro_context import ContextPipeline, default_chat_budget

budget = default_chat_budget(max_tokens=8192)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

Three presets are available:

- `default_chat_budget` -- Optimized for conversational apps (60% conversation)
- `default_rag_budget` -- Optimized for RAG-heavy apps (40% retrieval)
- `default_agent_budget` -- Optimized for agentic apps (balanced allocation)

!!! note
    Each budget automatically reserves 15% of tokens for the LLM response.
    Per-source overflow strategies (`"truncate"` or `"drop"`) control what
    happens when a source exceeds its cap.

---

## Decorator API

Register pipeline steps with decorators instead of factory functions:

```python
from astro_context import ContextPipeline, ContextItem, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def boost_recent(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Boost the score of recent items."""
    return [
        item.model_copy(update={"score": min(1.0, item.score * 1.5)})
        if item.metadata.get("recent")
        else item
        for item in items
    ]

result = pipeline.build("What is context engineering?")
```

!!! tip
    Use `@pipeline.async_step` for async functions and call `abuild()`
    instead of `build()`.

---

## Next Steps

- **[Getting Started](getting-started.md)** -- Installation, first pipeline, and all the basics
- **[Architecture](concepts/architecture.md)** -- How the pipeline, window, and priority system work
- **[Pipeline Guide](guides/pipeline.md)** -- Steps, callbacks, decorators, and error handling
- **[API Reference](api/pipeline.md)** -- Full API documentation for `ContextPipeline`
