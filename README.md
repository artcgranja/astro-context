# astro-context

**Context engineering toolkit for AI applications.**

> "Context is the product. The LLM is just the consumer."

Stop duct-taping RAG, memory, and tools together. Build intelligent context pipelines in minutes.

## Install

```bash
pip install astro-context
```

## Documentation

Full documentation at [arthurgranja.github.io/astro-context](https://arthurgranja.github.io/astro-context/)

- [Getting Started](https://arthurgranja.github.io/astro-context/getting-started/)
- [Architecture](https://arthurgranja.github.io/astro-context/concepts/architecture/)
- [Pipeline Guide](https://arthurgranja.github.io/astro-context/guides/pipeline/)
- [API Reference](https://arthurgranja.github.io/astro-context/api/pipeline/)
- [Examples](https://arthurgranja.github.io/astro-context/examples/rag-pipeline/)

## 30 Seconds to Your First Context Pipeline

```python
from astro_context import ContextPipeline, QueryBundle, MemoryManager, AnthropicFormatter

pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(MemoryManager(conversation_tokens=4096))
    .with_formatter(AnthropicFormatter())
    .add_system_prompt("You are a helpful assistant.")
)

result = pipeline.build("What is context engineering?")   # Plain strings work too
print(result.formatted_output)   # Ready for Claude API
print(result.diagnostics)        # Token usage, timing, overflow info
```

> **Tip:** `build()` accepts either a plain string or a `QueryBundle` object:
> ```python
> result = pipeline.build("What is context engineering?")
> # equivalent to:
> result = pipeline.build(QueryBundle(query_str="What is context engineering?"))
> ```

## Why astro-context?

| Feature | LangChain | LlamaIndex | mem0 | **astro-context** |
|---------|:---------:|:----------:|:----:|:-----------------:|
| Hybrid RAG (Dense + BM25 + RRF) | partial | yes | no | **yes** |
| Token-aware Memory | partial | no | yes | **yes** |
| Token Budget Management | no | no | no | **yes** |
| Provider-agnostic Formatting | no | no | no | **yes** |
| Protocol-based Plugins | no | partial | no | **yes** |
| Zero-config Defaults | no | no | yes | **yes** |

## Features

- **Hybrid RAG** -- Dense embeddings + BM25 sparse retrieval with Reciprocal Rank Fusion
- **Smart Memory** -- Token-aware sliding window with automatic eviction
- **Token Budgets** -- Priority-ranked context assembly that never exceeds your window
- **Provider Agnostic** -- Format output for Anthropic, OpenAI, or plain text
- **Protocol-Based** -- Plug in any vector store, tokenizer, or retriever via PEP 544 Protocols
- **Type-Safe** -- Pydantic v2 models throughout, full `py.typed` support
- **Model-Agnostic** -- Core never calls an LLM; you provide the embedding function
- **Rich Diagnostics** -- Per-step timing, token utilization, overflow tracking

## Hybrid Retrieval

```python
import math
from astro_context import (
    ContextPipeline, ContextItem, QueryBundle, SourceType,
    DenseRetriever, HybridRetriever,
    InMemoryContextStore, InMemoryVectorStore,
    retriever_step,
)

# You provide the embedding function (any model or API)
def my_embed_fn(text: str) -> list[float]:
    """Simple deterministic embedding for demonstration."""
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw

# Create retriever with in-memory stores
dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

# Create ContextItem objects for your documents
items = [
    ContextItem(content="Python is great for data science.", source=SourceType.RETRIEVAL),
    ContextItem(content="RAG combines retrieval with generation.", source=SourceType.RETRIEVAL),
]

# Index your documents
dense.index(items)

# Build the pipeline (also supports SparseRetriever + HybridRetriever with RRF)
pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(retriever_step("search", dense, top_k=5))
)
query = QueryBundle(query_str="How does RAG work?", embedding=my_embed_fn("How does RAG work?"))
result = pipeline.build(query)
```

## Memory Management

```python
from astro_context import ContextPipeline, QueryBundle, MemoryManager

memory = MemoryManager(conversation_tokens=4096)
memory.add_user_message("Help me migrate from MySQL to Postgres")
memory.add_assistant_message("Sure! What MySQL version are you using?")
memory.add_user_message("MySQL 8.0, about 50GB")

pipeline = ContextPipeline(max_tokens=8192).with_memory(memory)
result = pipeline.build(QueryBundle(query_str="What are the risks?"))
# Memory is automatically included, oldest turns evicted if over budget
```

## Provider Formatting

```python
from astro_context import AnthropicFormatter, OpenAIFormatter

# Anthropic: {"system": "...", "messages": [...]}
result = pipeline.with_formatter(AnthropicFormatter()).build(query)

# OpenAI: {"messages": [{"role": "system", ...}, ...]}
result = pipeline.with_formatter(OpenAIFormatter()).build(query)
```

## Decorator API

Instead of `add_step()` with factory functions, you can use the `@pipeline.step` decorator to register pipeline steps directly:

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

@pipeline.step(name="quality-filter")
def filter_low_quality(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Remove items below a quality threshold."""
    return [item for item in items if item.score > 0.3]

result = pipeline.build("What is context engineering?")
```

## Async Pipeline

For pipelines that include async steps (e.g., database lookups, API calls), use `abuild()` and `@pipeline.async_step`:

```python
import asyncio
from astro_context import ContextPipeline, ContextItem, SourceType, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.async_step
async def fetch_from_db(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Fetch relevant context from an async database."""
    results = await my_async_db_search(query.query_str)  # your async function
    new_items = [
        ContextItem(content=r["text"], source=SourceType.RETRIEVAL, score=r["score"])
        for r in results
    ]
    return items + new_items

@pipeline.step
def filter_results(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Sync steps and async steps can be mixed in the same pipeline."""
    return [item for item in items if item.score > 0.5]

# Use abuild() instead of build() to run the async pipeline
result = asyncio.run(pipeline.abuild("What is context engineering?"))
```

## Architecture

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

## Priority System

Every `ContextItem` has a `priority` field (1--10) that controls placement order in the context window. Higher priority items are placed first and are never evicted in favor of lower priority items.

| Priority | Source | Usage |
|----------|--------|-------|
| 10 | System prompts | Instructions, persona, rules |
| 8 | Persistent memory | Long-term facts from `MemoryManager.add_fact()` |
| 7 | Conversation memory | Recent chat turns from `SlidingWindowMemory` |
| 5 | Retrieval (default) | RAG results from retrievers |
| 1--4 | Custom | Low-priority supplementary context |

When the total context exceeds `max_tokens`, the pipeline fills from highest priority down. Items that do not fit are tracked in `result.overflow_items`.

## Token Budgets

For fine-grained control over how tokens are allocated across sources, use `TokenBudget`:

```python
from astro_context import ContextPipeline, TokenBudget, default_chat_budget

# Use a preset budget (allocates tokens across system, memory, retrieval, etc.)
budget = default_chat_budget(max_tokens=8192)
pipeline = ContextPipeline(max_tokens=8192).with_budget(budget)
```

Three preset factories are available:

- `default_chat_budget(max_tokens)` -- Optimized for conversational apps (60% conversation)
- `default_rag_budget(max_tokens)` -- Optimized for RAG-heavy apps (40% retrieval)
- `default_agent_budget(max_tokens)` -- Optimized for agentic apps (balanced allocation)

Each budget supports `reserve_tokens` to guarantee room for the LLM response, and per-source overflow strategies (`"truncate"` or `"drop"`).

## Optional Dependencies

```bash
pip install astro-context[bm25]   # BM25 sparse retrieval (rank-bm25)
pip install astro-context[cli]    # CLI tools (typer + rich)
pip install astro-context[all]    # Everything
```

## CLI

```bash
pip install astro-context[cli]
astro-context info       # Show installation info
astro-context --help     # See all commands
```

## Development

```bash
git clone https://github.com/arthurgranja/astro-context.git
cd astro-context
uv sync           # Install all dependencies
uv run pytest     # Run tests (1088 tests, 94% coverage)
uv run ruff check src/ tests/  # Lint
```

## Roadmap

- **v0.1.0** (current) -- Hybrid RAG + Memory + Pipeline + Formatters + Async pipeline + Decorator API + [Full documentation site](https://arthurgranja.github.io/astro-context/)
- **v0.2.0** -- MCP Bridge, progressive summarization, persistent storage backends
- **v0.3.0** -- GraphRAG, multi-modal context, LangChain/LlamaIndex adapters
- **v1.0.0** -- Production-grade APIs, plugin ecosystem

## License

MIT
