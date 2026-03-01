# RAG Pipeline

Build a complete Retrieval-Augmented Generation system with dense retrieval,
hybrid search, reranking, filtering, and provider-formatted output.

---

## Overview

This example demonstrates:

- Creating `DenseRetriever` with in-memory stores
- Building a `HybridRetriever` combining dense and sparse search
- Adding a reranker stage and a score filter
- Querying the pipeline and inspecting diagnostics
- Formatting output for the Anthropic API

## Full Example

```python
import math

from astro_context import (
    ContextItem,
    ContextPipeline,
    CrossEncoderReranker,
    DenseRetriever,
    HybridRetriever,
    InMemoryContextStore,
    InMemoryVectorStore,
    QueryBundle,
    SourceType,
    AnthropicFormatter,
    GenericTextFormatter,
    filter_step,
    reranker_step,
    retriever_step,
)

# ---------------------------------------------------------------
# 1. Deterministic embedding function (no API key needed)
# ---------------------------------------------------------------
def embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw


# ---------------------------------------------------------------
# 2. Sample documents
# ---------------------------------------------------------------
documents = [
    "Python is a versatile programming language used for web development, "
    "data science, machine learning, and automation.",
    "RAG (Retrieval-Augmented Generation) combines information retrieval "
    "with language model generation for grounded answers.",
    "Vector databases store embeddings and enable similarity search "
    "for semantic retrieval in AI applications.",
    "Context engineering is the discipline of building systems that "
    "provide the right context to language models.",
    "BM25 is a ranking function used in information retrieval that "
    "scores documents based on term frequency.",
    "Hybrid retrieval combines dense embedding search with sparse "
    "keyword search using Reciprocal Rank Fusion (RRF).",
    "Chunking strategies include fixed-size, recursive, sentence-based, "
    "and semantic splitting approaches.",
    "Reranking uses a cross-encoder to re-score retrieved documents "
    "for improved relevance after initial retrieval.",
]

items = [
    ContextItem(content=doc, source=SourceType.RETRIEVAL)
    for doc in documents
]

# ---------------------------------------------------------------
# 3. Create a DenseRetriever and index documents
# ---------------------------------------------------------------
dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn,
)
indexed = dense.index(items)
print(f"Indexed {indexed} documents into DenseRetriever")

# ---------------------------------------------------------------
# 4. Create a reranker (deterministic scoring for demo)
# ---------------------------------------------------------------
def simple_scorer(query: str, doc: str) -> float:
    """Score based on word overlap (no external model needed)."""
    query_words = set(query.lower().split())
    doc_words = set(doc.lower().split())
    overlap = len(query_words & doc_words)
    return min(1.0, overlap / max(len(query_words), 1))

reranker = CrossEncoderReranker(score_fn=simple_scorer, top_k=5)

# ---------------------------------------------------------------
# 5. Build the pipeline with retriever + reranker + filter
# ---------------------------------------------------------------
pipeline = (
    ContextPipeline(max_tokens=4096)
    .add_step(retriever_step("dense-search", dense, top_k=10))
    .add_step(reranker_step("rerank", reranker, top_k=5))
    .add_step(filter_step("quality-filter", lambda item: item.score > 0.1))
    .with_formatter(GenericTextFormatter())
    .add_system_prompt("You are a helpful AI assistant specializing in RAG.")
)

# ---------------------------------------------------------------
# 6. Query the pipeline
# ---------------------------------------------------------------
query = QueryBundle(
    query_str="How does hybrid retrieval work?",
    embedding=embed_fn("How does hybrid retrieval work?"),
)
result = pipeline.build(query)

print("\n=== Retrieved Context ===\n")
for item in result.window.items:
    if item.source == SourceType.RETRIEVAL:
        score = f"{item.score:.3f}"
        method = item.metadata.get("retrieval_method", "unknown")
        print(f"  [{score}] ({method}) {item.content[:80]}...")

# ---------------------------------------------------------------
# 7. Inspect diagnostics
# ---------------------------------------------------------------
print("\n=== Pipeline Diagnostics ===\n")
diag = result.diagnostics
print(f"  Build time:        {result.build_time_ms:.1f} ms")
print(f"  Items considered:  {diag.get('total_items_considered', 0)}")
print(f"  Items included:    {diag.get('items_included', 0)}")
print(f"  Items overflowed:  {diag.get('items_overflow', 0)}")
print(f"  Token utilization: {diag.get('token_utilization', 0):.1%}")

print("\n  Step timings:")
for step in diag.get("steps", []):
    print(f"    {step['name']}: {step['time_ms']:.1f} ms "
          f"({step['items_after']} items)")

# ---------------------------------------------------------------
# 8. Format for Anthropic API
# ---------------------------------------------------------------
print("\n=== Anthropic-Formatted Output ===\n")
anthropic_pipeline = pipeline.with_formatter(AnthropicFormatter())
anthropic_result = anthropic_pipeline.build(query)
formatted = anthropic_result.formatted_output

# formatted is a dict with "system" and "messages" keys
print(f"  Format type: {anthropic_result.format_type}")
if isinstance(formatted, dict):
    print(f"  System blocks: {len(formatted.get('system', []))}")
    print(f"  Messages: {len(formatted.get('messages', []))}")

# ---------------------------------------------------------------
# 9. Using the decorator API instead of factory functions
# ---------------------------------------------------------------
print("\n=== Decorator API ===\n")
decorator_pipeline = ContextPipeline(max_tokens=4096)

@decorator_pipeline.step
def custom_retrieval(items, query):
    """Retrieve using our dense retriever."""
    results = dense.retrieve(query, top_k=5)
    return items + results

@decorator_pipeline.step(name="boost-relevant")
def boost_relevant(items, query):
    """Boost items that mention 'retrieval' in content."""
    boosted = []
    for item in items:
        if "retrieval" in item.content.lower():
            item = item.model_copy(update={"score": min(1.0, item.score * 1.5)})
        boosted.append(item)
    return boosted

result = decorator_pipeline.build(query)
print(f"  Items from decorator pipeline: {len(result.window.items)}")
```

## Key Concepts

### Pipeline Step Types

| Factory Function | Purpose | Protocol |
|------------------|---------|----------|
| `retriever_step()` | Fetch items from a retriever | `Retriever` |
| `reranker_step()` | Re-score and re-order items | `Reranker` |
| `filter_step()` | Remove items by predicate | `Callable` |
| `postprocessor_step()` | Transform items arbitrarily | `PostProcessor` |
| `query_transform_step()` | Expand query, retrieve per variant | `QueryTransformer` |

### Hybrid Retrieval

Combine dense and sparse retrievers with Reciprocal Rank Fusion:

```python
from astro_context import HybridRetriever, SparseRetriever

sparse = SparseRetriever()  # requires pip install astro-context[bm25]
sparse.index(items)

hybrid = HybridRetriever(
    retrievers=[dense, sparse],
    weights=[0.7, 0.3],  # weight dense higher
    rrf_k=60,            # RRF smoothing constant
)
```

!!! tip "Embedding at Query Time"
    When using `DenseRetriever` with an `embed_fn`, you can pass either a
    plain string or a `QueryBundle` with a pre-computed embedding. If you
    pass a string, the retriever calls `embed_fn` automatically.

!!! note "Reranker vs PostProcessor"
    `reranker_step()` expects an object with a `rerank(query, items, top_k)`
    method. `postprocessor_step()` expects `process(items, query)`. Use
    `CrossEncoderReranker` for the reranker protocol, or `ScoreReranker`
    for the postprocessor protocol.

## Next Steps

- [Document Ingestion](document-ingestion.md) -- ingest real files into your pipeline
- [Custom Retriever](custom-retriever.md) -- implement your own retriever
- [Evaluation Workflow](evaluation-workflow.md) -- measure retrieval quality
