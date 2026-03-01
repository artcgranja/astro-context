# Context Engineering

> "Context is the product. The LLM is just the consumer."

Context engineering is the discipline of designing and assembling the
information that an LLM receives in its prompt. A well-engineered context
window directly determines the quality of the model's output -- regardless
of which model you use.

## Why Context Matters

Large language models do not have memory. Every request starts from a blank
slate. The only way an LLM "knows" anything about your user, your data, or
your application is through the context you provide in the prompt.

This makes context the single most important lever for AI application
quality. A mediocre model with excellent context will outperform a frontier
model with poor context.

Context engineering answers questions like:

- Which documents should the model see for this query?
- How much conversation history fits in the window?
- Should system instructions or retrieval results get priority when space
  is tight?
- How do you prevent one source from crowding out another?

## The Problem with Current Approaches

Most AI frameworks treat context assembly as an afterthought. RAG pipelines
concatenate retrieval results into a prompt string. Chat frameworks append
messages until the context window overflows. Memory systems operate in
isolation from retrieval.

The result is context that is assembled _ad hoc_ -- without token
awareness, without priority ranking, and without diagnostics.

| Feature | LangChain | LlamaIndex | mem0 | **astro-context** |
|---------|:---------:|:----------:|:----:|:-----------------:|
| Hybrid RAG (Dense + BM25 + RRF) | partial | yes | no | **yes** |
| Token-aware Memory | partial | no | yes | **yes** |
| Token Budget Management | no | no | no | **yes** |
| Provider-agnostic Formatting | no | no | no | **yes** |
| Protocol-based Plugins | no | partial | no | **yes** |
| Zero-config Defaults | no | no | yes | **yes** |

## The astro-context Philosophy

astro-context is built on three design principles that set it apart.

### 1. Model-Agnostic by Design

The core library never calls an LLM. It does not import `openai`, `anthropic`,
or any model SDK. You provide the embedding function, the tokenizer, and the
retrieval backend. The library handles assembly, ranking, and budgeting.

This means you can swap models, providers, or even embedding strategies
without changing your context pipeline.

```python
import math

# You own the embedding function -- any model, any API, any dimension
def my_embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw
```

### 2. Token-Aware Everything

Every component in astro-context is token-aware:

- **ContextItem** carries a `token_count` field.
- **ContextWindow** tracks `used_tokens` and `remaining_tokens`.
- **MemoryManager** evicts oldest turns when the conversation exceeds its
  token budget.
- **TokenBudget** allocates portions of the window to different source
  types with per-source caps and overflow strategies.

Nothing is concatenated blindly. The library always knows how many tokens
remain and what was dropped.

### 3. Protocol-Based Plugins

Instead of deep inheritance hierarchies, astro-context uses Python Protocols
(PEP 544) for all extension points. Any object with the right method
signatures works -- no base class required.

```python
from astro_context.protocols import Retriever

# This class satisfies the Retriever protocol without inheriting anything
class MyRetriever:
    def retrieve(self, query, top_k=10):
        return [...]  # Return list of ContextItem
```

This makes it trivial to integrate existing code, wrap third-party APIs, or
test with simple stubs.

!!! tip "Duck typing for AI"
    Protocols give you the benefits of interfaces (type safety, IDE
    autocompletion, runtime checking with `isinstance`) without the rigidity
    of inheritance. See the [Protocols](protocols.md) concept page for
    details.

## What astro-context Is Not

astro-context is **not**:

- An LLM wrapper -- it never generates text.
- A vector database -- it provides in-memory stores for development and
  protocols for plugging in any backend.
- An agent framework -- it assembles context for agents, but does not
  manage tool execution or planning.

It sits between your data sources and your LLM call, ensuring the model
receives the best possible context within its token budget.

## Core Data Flow

```
 Your Data Sources          astro-context              Your LLM Call
 +-----------------+     +------------------+     +------------------+
 | Vector DB       | --> |                  |     |                  |
 | Conversation    | --> | ContextPipeline  | --> | Anthropic / OpenAI|
 | System Prompts  | --> |   .build()       |     |   API call       |
 | Tool Results    | --> |                  |     |                  |
 +-----------------+     +------------------+     +------------------+
                          - Priority ranking
                          - Token budgets
                          - Overflow tracking
                          - Provider formatting
                          - Diagnostics
```

## Getting Started

To build your first context pipeline, see the
[Getting Started](../getting-started.md) guide. For deeper architectural
details, read the [Architecture](architecture.md) page.

## See Also

- [Architecture](architecture.md) -- six-stage pipeline internals
- [Protocols](protocols.md) -- PEP 544 structural subtyping
- [Token Budgets](token-budgets.md) -- per-source allocation and overflow
