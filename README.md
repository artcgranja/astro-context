# astro-context

**Context engineering toolkit for AI applications.**

> "Context is the product. The LLM is just the consumer."

Stop duct-taping RAG, memory, and tools together. Build intelligent context pipelines in minutes.

## Install

```bash
pip install astro-context
```

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
from astro_context import (
    ContextPipeline, QueryBundle, DenseRetriever, SparseRetriever,
    HybridRetriever, InMemoryContextStore, InMemoryVectorStore,
)
from astro_context.pipeline import retriever_step

# Dense + BM25 with Reciprocal Rank Fusion
dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,  # you provide this
)
sparse = SparseRetriever()

# Index your documents
dense.index(items)
sparse.index(items)

# Fuse with weighted RRF
hybrid = HybridRetriever(retrievers=[dense, sparse], weights=[0.6, 0.4])

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(retriever_step("search", hybrid, top_k=10))
)
result = pipeline.build(QueryBundle(query_str="How to optimize queries?"))
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
uv run pytest     # Run tests (961 tests, 94% coverage)
uv run ruff check src/ tests/  # Lint
```

## Roadmap

- **v0.1.0** (current) -- Hybrid RAG + Memory + Pipeline + Formatters + Async pipeline + Decorator API
- **v0.2.0** -- MCP Bridge, progressive summarization, persistent storage backends
- **v0.3.0** -- GraphRAG, multi-modal context, LangChain/LlamaIndex adapters
- **v1.0.0** -- Production-grade APIs, plugin ecosystem

## License

MIT
