# Custom Retriever

Implement the `Retriever` protocol to create your own retriever and
integrate it seamlessly with `ContextPipeline` and `HybridRetriever`.

---

## Overview

This example demonstrates:

- The `Retriever` protocol definition and what it requires
- Implementing a dictionary-based retriever from scratch
- Using the custom retriever in a pipeline
- Combining it with `HybridRetriever` alongside a `DenseRetriever`

## The Retriever Protocol

The `Retriever` protocol requires a single method:

```python
from astro_context import Retriever, ContextItem, QueryBundle

class Retriever:
    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        ...
```

Any class with a `retrieve` method matching this signature automatically
satisfies the protocol -- no inheritance or registration needed. This is
structural subtyping via [PEP 544](https://peps.python.org/pep-0544/).

## Full Example

```python
import math

from astro_context import (
    ContextItem,
    ContextPipeline,
    DenseRetriever,
    HybridRetriever,
    InMemoryContextStore,
    InMemoryVectorStore,
    QueryBundle,
    Retriever,
    SourceType,
    retriever_step,
)

# ---------------------------------------------------------------
# 1. Deterministic embedding function
# ---------------------------------------------------------------
def embed_fn(text: str) -> list[float]:
    seed = sum(ord(c) for c in text) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(64)]
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if norm else raw


# ---------------------------------------------------------------
# 2. Custom retriever: keyword-based dictionary lookup
# ---------------------------------------------------------------
class KeywordDictRetriever:
    """A simple retriever that matches query keywords against a dictionary.

    This demonstrates the minimal interface needed to satisfy the
    Retriever protocol: just a `retrieve` method.
    """

    def __init__(self) -> None:
        self._docs: list[ContextItem] = []
        self._keyword_index: dict[str, list[int]] = {}

    def index(self, items: list[ContextItem]) -> int:
        """Build a keyword index from items."""
        self._docs = list(items)
        self._keyword_index.clear()
        for idx, item in enumerate(items):
            words = set(item.content.lower().split())
            for word in words:
                # Strip punctuation
                clean = word.strip(".,!?;:()[]{}\"'")
                if len(clean) > 2:
                    self._keyword_index.setdefault(clean, []).append(idx)
        return len(items)

    def retrieve(
        self, query: QueryBundle, top_k: int = 10
    ) -> list[ContextItem]:
        """Retrieve items matching query keywords, scored by match count."""
        query_words = set(query.query_str.lower().split())
        scores: dict[int, int] = {}

        for word in query_words:
            clean = word.strip(".,!?;:()[]{}\"'")
            for idx in self._keyword_index.get(clean, []):
                scores[idx] = scores.get(idx, 0) + 1

        # Sort by match count descending
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        max_score = ranked[0][1] if ranked else 1

        results: list[ContextItem] = []
        for idx, count in ranked[:top_k]:
            item = self._docs[idx]
            normalized_score = count / max_score
            scored = item.model_copy(update={
                "source": SourceType.RETRIEVAL,
                "score": normalized_score,
                "metadata": {
                    **item.metadata,
                    "retrieval_method": "keyword_dict",
                    "keyword_matches": count,
                },
            })
            results.append(scored)

        return results


# ---------------------------------------------------------------
# 3. Verify it satisfies the protocol
# ---------------------------------------------------------------
my_retriever = KeywordDictRetriever()
print(f"Is Retriever? {isinstance(my_retriever, Retriever)}")  # True

# ---------------------------------------------------------------
# 4. Index sample documents
# ---------------------------------------------------------------
documents = [
    "Python is great for data science and machine learning.",
    "RAG combines retrieval with language model generation.",
    "Vector databases enable fast similarity search.",
    "Context engineering builds intelligent AI pipelines.",
    "BM25 uses term frequency for document ranking.",
    "Hybrid search combines dense and sparse retrieval.",
]

items = [
    ContextItem(content=doc, source=SourceType.RETRIEVAL)
    for doc in documents
]

my_retriever.index(items)
print(f"Indexed {len(items)} documents")

# ---------------------------------------------------------------
# 5. Use in a ContextPipeline
# ---------------------------------------------------------------
print("\n=== Custom Retriever in Pipeline ===\n")

pipeline = (
    ContextPipeline(max_tokens=2048)
    .add_step(retriever_step("keyword-search", my_retriever, top_k=3))
    .add_system_prompt("Answer based on the context provided.")
)

query = QueryBundle(query_str="How does retrieval work in RAG?")
result = pipeline.build(query)

for item in result.window.items:
    if item.source == SourceType.RETRIEVAL:
        matches = item.metadata.get("keyword_matches", 0)
        method = item.metadata.get("retrieval_method", "?")
        print(f"  [{item.score:.2f}] ({method}, {matches} matches) "
              f"{item.content[:60]}...")

# ---------------------------------------------------------------
# 6. Combine with DenseRetriever via HybridRetriever
# ---------------------------------------------------------------
print("\n=== Hybrid: Custom + Dense ===\n")

dense = DenseRetriever(
    vector_store=InMemoryVectorStore(),
    context_store=InMemoryContextStore(),
    embed_fn=embed_fn,
)
dense.index(items)

hybrid = HybridRetriever(
    retrievers=[my_retriever, dense],
    weights=[0.4, 0.6],  # weight dense retriever higher
    rrf_k=60,
)

query = QueryBundle(
    query_str="How does hybrid search work?",
    embedding=embed_fn("How does hybrid search work?"),
)
hybrid_results = hybrid.retrieve(query, top_k=3)

for item in hybrid_results:
    method = item.metadata.get("retrieval_method", "?")
    rrf = item.metadata.get("rrf_raw_score", 0)
    print(f"  [{item.score:.3f}] (rrf={rrf:.4f}, orig={method}) "
          f"{item.content[:60]}...")

# ---------------------------------------------------------------
# 7. Custom retriever in a pipeline with the decorator API
# ---------------------------------------------------------------
print("\n=== Decorator API with Custom Retriever ===\n")

pipeline2 = ContextPipeline(max_tokens=2048)

@pipeline2.step
def keyword_search(items, query):
    results = my_retriever.retrieve(query, top_k=5)
    return items + results

@pipeline2.step(name="dedup")
def deduplicate(items, query):
    seen = set()
    deduped = []
    for item in items:
        if item.id not in seen:
            seen.add(item.id)
            deduped.append(item)
    return deduped

result2 = pipeline2.build(QueryBundle(query_str="data science python"))
print(f"  Results: {len(result2.window.items)} items")
for item in result2.window.items:
    print(f"    [{item.score:.2f}] {item.content[:60]}...")
```

## Protocol Reference

### Retriever (sync)

```python
class Retriever(Protocol):
    def retrieve(
        self, query: QueryBundle, top_k: int = 10
    ) -> list[ContextItem]:
        ...
```

### AsyncRetriever (async)

```python
class AsyncRetriever(Protocol):
    async def aretrieve(
        self, query: QueryBundle, top_k: int = 10
    ) -> list[ContextItem]:
        ...
```

Use `async_retriever_step()` to add an async retriever to the pipeline,
and call `pipeline.abuild()` instead of `pipeline.build()`.

!!! tip "Protocol Compliance"
    You can verify your class satisfies the protocol at runtime:
    ```python
    from astro_context import Retriever
    assert isinstance(my_retriever, Retriever)
    ```

!!! note "No Inheritance Required"
    Unlike abstract base classes, protocols use structural subtyping.
    Your class does not need to inherit from `Retriever` -- it just
    needs a `retrieve` method with the correct signature.

!!! warning "Return ContextItem Objects"
    The `retrieve` method must return `list[ContextItem]`. The pipeline
    expects these specific model objects, not plain dicts or strings.

## Next Steps

- [RAG Pipeline](rag-pipeline.md) -- use your retriever in a full pipeline
- [Evaluation Workflow](evaluation-workflow.md) -- benchmark your retriever
- [Document Ingestion](document-ingestion.md) -- feed documents to your retriever
