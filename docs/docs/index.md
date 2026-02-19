# astro-context

**Context engineering toolkit for AI applications.**

> Stop duct-taping RAG, memory, and tools together.
> Build intelligent context pipelines in minutes.

## Quick Start

```python
from astro_context import (
    ContextPipeline,
    QueryBundle,
    MemoryManager,
    AnthropicFormatter,
)

# Create a pipeline with memory and formatting
pipeline = (
    ContextPipeline(max_tokens=8192)
    .with_memory(MemoryManager(conversation_tokens=4096))
    .with_formatter(AnthropicFormatter())
    .add_system_prompt("You are a helpful assistant.")
)

# Build context for a query
result = pipeline.build(QueryBundle(query_str="What is context engineering?"))

# Use the result
print(result.formatted_output)  # Ready for Anthropic API
print(result.diagnostics)       # Token usage, timing, overflow info
```

## Features

- **Hybrid RAG** -- Dense + Sparse (BM25) retrieval with Reciprocal Rank Fusion
- **Smart Memory** -- Token-aware sliding window conversation memory
- **Token Budgets** -- Never exceed your context window
- **Provider Agnostic** -- Format for Claude, GPT, or any LLM
- **Protocol-Based** -- Pluggable storage, retrievers, and tokenizers via PEP 544
- **Type-Safe** -- Pydantic v2 models throughout

## Install

```bash
pip install astro-context
# or with uv
uv add astro-context
```

### Optional Extras

```bash
pip install astro-context[bm25]   # BM25 sparse retrieval
pip install astro-context[cli]    # CLI tools
pip install astro-context[all]    # Everything
```
