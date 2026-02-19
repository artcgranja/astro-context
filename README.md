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

result = pipeline.build(QueryBundle(query_str="What is context engineering?"))
print(result.formatted_output)   # Ready for Claude API
print(result.diagnostics)        # Token usage, timing, overflow info
```

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
git clone https://github.com/artcgranja/astro-context.git
cd astro-context
uv sync           # Install all dependencies
uv run pytest     # Run tests (188 tests, 89% coverage)
uv run ruff check src/ tests/  # Lint
```

## Roadmap

- **v0.1.0** (current) -- Hybrid RAG + Memory + Pipeline + Formatters
- **v0.2.0** -- Async pipeline, MCP Bridge, progressive summarization
- **v0.3.0** -- GraphRAG, multi-modal context, LangChain/LlamaIndex adapters
- **v1.0.0** -- Production-grade APIs, plugin ecosystem

## License

MIT
