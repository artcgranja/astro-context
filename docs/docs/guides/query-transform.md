# Query Transformation Guide

Query transformation rewrites or expands a user query **before** retrieval to
improve recall and relevance. A single natural-language question may not align
well with the embedding space; transforming it into one or more alternative
phrasings helps the retriever surface better results.

```
User Query --> Transform --> [Query 1, Query 2, ...] --> Retrieve --> Merge
```

All transformers accept **callback functions** for LLM generation so that
`astro-context` never calls an LLM directly. You supply your own generation
function and the transformer handles orchestration.

## Transformer Overview

| Transformer                  | Strategy                                          | Output      |
|------------------------------|---------------------------------------------------|-------------|
| `HyDETransformer`           | Generate a hypothetical answer to the query       | 1 query     |
| `MultiQueryTransformer`     | Generate N alternative phrasings                  | N+1 queries |
| `DecompositionTransformer`  | Break a complex query into sub-questions          | N queries   |
| `StepBackTransformer`       | Generate a broader, more abstract version         | 2 queries   |
| `ConversationRewriter`      | Rewrite using chat history                        | 1 query     |
| `ContextualQueryTransformer`| Prepend conversation context, then delegate       | varies      |

---

## HyDETransformer

Hypothetical Document Embeddings (HyDE) generates a plausible answer to the
query and uses that as the retrieval query. The intuition is that a hypothetical
answer is closer in embedding space to the real answer than the question itself.

```python
from astro_context.query import HyDETransformer
from astro_context.models.query import QueryBundle

def generate_hypothetical(query: str) -> str:
    # In production, call your LLM here
    return f"A hypothetical answer about: {query}"

hyde = HyDETransformer(generate_fn=generate_hypothetical)

query = QueryBundle(query_str="What is retrieval augmented generation?")
expanded = hyde.transform(query)

print(expanded[0].query_str)
# "A hypothetical answer about: What is retrieval augmented generation?"
print(expanded[0].metadata["transform"])
# "hyde"
```

## MultiQueryTransformer

Generates multiple query variations for broader retrieval coverage. The original
query is always included as the first element.

```python
from astro_context.query import MultiQueryTransformer
from astro_context.models.query import QueryBundle

def generate_variations(query: str, count: int) -> list[str]:
    return [f"variation {i}: {query}" for i in range(count)]

multi = MultiQueryTransformer(generate_fn=generate_variations, num_queries=3)

query = QueryBundle(query_str="How does RAG work?")
expanded = multi.transform(query)

print(len(expanded))  # 4 (original + 3 variations)
print(expanded[0].query_str)  # "How does RAG work?" (original)
```

## DecompositionTransformer

Breaks a complex query into simpler sub-questions. Useful for multi-hop or
compound questions where answering the original requires synthesizing information
from multiple sources.

```python
from astro_context.query import DecompositionTransformer
from astro_context.models.query import QueryBundle

def decompose(query: str) -> list[str]:
    return [
        "What is retrieval augmented generation?",
        "What are the benefits of RAG over fine-tuning?",
    ]

decomp = DecompositionTransformer(generate_fn=decompose)

query = QueryBundle(query_str="Compare RAG and fine-tuning approaches")
sub_queries = decomp.transform(query)

for sq in sub_queries:
    print(sq.query_str)
    print(f"  parent_query: {sq.metadata['parent_query']}")
```

## StepBackTransformer

Generates a more abstract version of the query alongside the original. The
"step-back" technique retrieves high-level context that helps answer specific
questions.

```python
from astro_context.query import StepBackTransformer
from astro_context.models.query import QueryBundle

def step_back(query: str) -> str:
    return "What are the general principles of information retrieval?"

sb = StepBackTransformer(generate_fn=step_back)

query = QueryBundle(query_str="How does BM25 scoring work?")
expanded = sb.transform(query)

print(len(expanded))  # 2: [original, step-back]
print(expanded[0].query_str)  # "How does BM25 scoring work?"
print(expanded[1].query_str)  # "What are the general principles of information retrieval?"
```

---

## Chaining Transformers with QueryTransformPipeline

`QueryTransformPipeline` chains multiple transformers in sequence. Each transformer
is applied to every query produced by the previous stage, producing a flat list of
unique queries at the end (deduplicated by `query_str`).

```python
from astro_context.query import (
    QueryTransformPipeline,
    MultiQueryTransformer,
    StepBackTransformer,
)
from astro_context.models.query import QueryBundle

def gen_variations(query: str, count: int) -> list[str]:
    return [f"v{i}: {query}" for i in range(count)]

def gen_stepback(query: str) -> str:
    return f"General: {query}"

pipeline = QueryTransformPipeline(
    transformers=[
        MultiQueryTransformer(generate_fn=gen_variations, num_queries=2),
        StepBackTransformer(generate_fn=gen_stepback),
    ]
)

query = QueryBundle(query_str="How does vector search work?")
all_queries = pipeline.transform(query)

print(f"Produced {len(all_queries)} unique queries")
for q in all_queries:
    print(f"  - {q.query_str}")
```

!!! note
    `QueryTransformPipeline` also supports `atransform()` for async execution.
    Transformers that implement `AsyncQueryTransformer` are called via
    `atransform`; others fall back to the synchronous `transform` method.

---

## Conversation-Aware Transformers

### ConversationRewriter

Rewrites a query using conversation history via a user-supplied callback. When
the `QueryBundle` carries non-empty `chat_history`, the `rewrite_fn` is called.
If history is empty, the original query passes through unchanged.

```python
from astro_context.query import ConversationRewriter
from astro_context.models.query import QueryBundle
from astro_context.models.memory import ConversationTurn

def rewrite(query: str, history: list[ConversationTurn]) -> str:
    context = "; ".join(f"{t.role}: {t.content}" for t in history)
    return f"{query} (context: {context})"

rewriter = ConversationRewriter(rewrite_fn=rewrite)

query = QueryBundle(
    query_str="What about the second approach?",
    chat_history=[
        ConversationTurn(role="user", content="Compare RAG and fine-tuning"),
        ConversationTurn(role="assistant", content="RAG retrieves documents..."),
    ],
)
result = rewriter.transform(query)
print(result[0].query_str)
```

### ContextualQueryTransformer

Wraps another transformer, prepending a summary of conversation history to the
query string before delegation.

```python
from astro_context.query import ContextualQueryTransformer, HyDETransformer
from astro_context.models.query import QueryBundle
from astro_context.models.memory import ConversationTurn

def gen_hyde(query: str) -> str:
    return f"Hypothetical: {query}"

inner = HyDETransformer(generate_fn=gen_hyde)
contextual = ContextualQueryTransformer(
    inner=inner,
    context_prefix="Given the conversation context: ",
)

query = QueryBundle(
    query_str="Tell me more",
    chat_history=[
        ConversationTurn(role="user", content="What is RAG?"),
    ],
)
result = contextual.transform(query)
print(result[0].query_str)
# Contains the conversation context prepended to the HyDE output
```

---

## Integration with Pipeline Steps

Use `query_transform_step()` to plug any transformer directly into a
retrieval pipeline. The step transforms the query, retrieves for each variant
using the provided retriever, and merges results via Reciprocal Rank Fusion.

```python
from astro_context.query import MultiQueryTransformer
from astro_context.pipeline import query_transform_step

def gen_variations(query: str, count: int) -> list[str]:
    return [f"v{i}: {query}" for i in range(count)]

transformer = MultiQueryTransformer(
    generate_fn=gen_variations,
    num_queries=3,
)

# `my_retriever` is any object implementing the Retriever protocol
step = query_transform_step(
    name="multi-query-retrieval",
    transformer=transformer,
    retriever=my_retriever,
    top_k=10,
)
```

The step can then be added to a `ContextPipeline`:

```python
from astro_context.pipeline import ContextPipeline

pipeline = ContextPipeline(steps=[step])
```

!!! tip
    Combine `QueryTransformPipeline` with `query_transform_step` to chain
    multiple transformers and use the combined output for retrieval in a
    single pipeline step.

!!! warning
    Each expanded query triggers a separate retrieval call. With
    `MultiQueryTransformer(num_queries=3)` you get 4 retrieval calls
    (original + 3 variations). Keep `top_k` and `num_queries` reasonable
    to control latency and cost.
