# Getting Started

## Installation

```bash
pip install astro-context
```

Or with [uv](https://docs.astral.sh/uv/):

```bash
uv add astro-context
```

## Your First Pipeline

The simplest pipeline uses memory and a system prompt:

```python
from astro_context import ContextPipeline, QueryBundle, MemoryManager

# Create pipeline with memory
memory = MemoryManager(conversation_tokens=4096)
pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(memory)
    .add_system_prompt("You are a helpful coding assistant.")
)

# Add conversation history
memory.add_user_message("Help me write a Python function")
memory.add_assistant_message("Sure! What should the function do?")

# Build context for the next query
result = pipeline.build(QueryBundle(query_str="It should sort a list"))
print(result.formatted_output)
```

## Adding Retrieval

Add semantic search with dense retrieval:

```python
from astro_context import (
    ContextPipeline,
    QueryBundle,
    ContextItem,
    SourceType,
    DenseRetriever,
    InMemoryContextStore,
    InMemoryVectorStore,
)
from astro_context.pipeline import retriever_step

# You provide the embedding function (any provider works)
def my_embed_fn(text: str) -> list[float]:
    # Replace with your actual embedding call
    # e.g., openai.embeddings.create(model="text-embedding-3-small", input=text)
    return [0.0] * 384

# Set up retriever
retriever = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=my_embed_fn,
)

# Index some documents
docs = [
    ContextItem(content="Python lists support .sort() and sorted().", source=SourceType.RETRIEVAL),
    ContextItem(content="Use lambda for custom sort keys.", source=SourceType.RETRIEVAL),
]
retriever.index(docs)

# Build pipeline with retrieval
pipeline = (
    ContextPipeline(max_tokens=4096)
    .add_step(retriever_step("search", retriever, top_k=5))
)

result = pipeline.build(QueryBundle(query_str="How to sort in Python?"))
print(f"Found {len(result.window.items)} context items")
print(f"Used {result.window.used_tokens}/{result.window.max_tokens} tokens")
```

## Hybrid Retrieval (Dense + BM25)

Combine dense and sparse retrieval with Reciprocal Rank Fusion:

```python
from astro_context import (
    DenseRetriever, SparseRetriever, HybridRetriever,
    InMemoryVectorStore, InMemoryContextStore,
)
from astro_context.pipeline import retriever_step

# Create individual retrievers
vector_store = InMemoryVectorStore()
context_store = InMemoryContextStore()
dense = DenseRetriever(vector_store=vector_store, context_store=context_store, embed_fn=my_embed_fn)
sparse = SparseRetriever()

# Index documents in both
dense.index(items)
sparse.index(items)

# Combine with RRF
hybrid = HybridRetriever(
    retrievers=[dense, sparse],
    weights=[0.6, 0.4],  # 60% dense, 40% sparse
)

pipeline = (
    ContextPipeline(max_tokens=8192)
    .add_step(retriever_step("hybrid_search", hybrid, top_k=10))
)
```

## Formatting for Different Providers

```python
from astro_context import AnthropicFormatter, OpenAIFormatter, GenericTextFormatter

# Anthropic format: {"system": "...", "messages": [...]}
pipeline.with_formatter(AnthropicFormatter())

# OpenAI format: {"messages": [...]}
pipeline.with_formatter(OpenAIFormatter())

# Plain text with section headers
pipeline.with_formatter(GenericTextFormatter())
```

## Diagnostics

Every `build()` call includes diagnostics:

```python
result = pipeline.build(query)
print(result.diagnostics)
# {
#   "steps": [{"name": "search", "items_after": 15, "time_ms": 2.1}],
#   "total_items_considered": 15,
#   "items_included": 10,
#   "items_overflow": 5,
#   "token_utilization": 0.87,
# }
print(f"Build time: {result.build_time_ms}ms")
```

## Decorator API for Pipeline Steps

Instead of using `add_step()` with factory functions, you can register steps using the
`@pipeline.step` decorator. This is especially convenient for custom post-processing logic:

```python
from astro_context import ContextPipeline, ContextItem, QueryBundle

pipeline = ContextPipeline(max_tokens=8192)

@pipeline.step
def boost_recent(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Boost scores of recent items."""
    return [
        item.model_copy(update={"score": item.score * 1.5})
        if item.metadata.get("recent") else item
        for item in items
    ]

@pipeline.step(name="quality-filter")
def remove_low_quality(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    """Filter out low-scoring items."""
    return [item for item in items if item.score > 0.3]

result = pipeline.build("How to sort in Python?")
```

For async steps (e.g., database lookups), use `@pipeline.async_step` and call `abuild()`:

```python
@pipeline.async_step
async def fetch_from_db(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
    results = await my_async_search(query.query_str)
    return items + results

result = await pipeline.abuild("How to sort in Python?")
```
