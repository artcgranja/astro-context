"""Example: Hybrid RAG Pipeline. Run with: python examples/hybrid_rag.py

Demonstrates building a hybrid retrieval pipeline that combines dense
embedding retrieval with sparse BM25 retrieval using Reciprocal Rank
Fusion (RRF).

Uses a simple whitespace tokenizer and sin-based fake embeddings so the
example runs without any external dependencies beyond astro-context.
"""

from __future__ import annotations

import math

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import retriever_step
from astro_context.retrieval.dense import DenseRetriever
from astro_context.retrieval.hybrid import HybridRetriever
from astro_context.storage.memory_store import InMemoryContextStore, InMemoryVectorStore

# ---------------------------------------------------------------------------
# Simple helpers (no external dependencies)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words."""

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return " ".join(words[:max_tokens])


def fake_embed(text: str, dim: int = 64) -> list[float]:
    """Deterministic sin-based embedding for demonstration purposes."""
    seed = sum(ord(c) * (i + 1) for i, c in enumerate(text[:50])) % 10000
    raw = [math.sin(seed * 1000 + i) for i in range(dim)]
    norm = math.sqrt(sum(x * x for x in raw))
    if norm == 0:
        return raw
    return [x / norm for x in raw]


# ---------------------------------------------------------------------------
# Build sample documents
# ---------------------------------------------------------------------------

DOCUMENTS = [
    "Python is a high-level programming language known for readability.",
    "Machine learning models require large amounts of training data.",
    "Context engineering is the art of assembling the right information for LLMs.",
    "Vector databases enable efficient similarity search over embeddings.",
    "Retrieval-augmented generation combines search with language model generation.",
    "BM25 is a probabilistic ranking function used in information retrieval.",
    "Reciprocal rank fusion merges results from multiple retrieval signals.",
]


def build_items(tokenizer: WhitespaceTokenizer) -> list[ContextItem]:
    """Create ContextItem objects from sample documents."""
    items: list[ContextItem] = []
    for i, text in enumerate(DOCUMENTS):
        items.append(
            ContextItem(
                id=f"doc-{i}",
                content=text,
                source=SourceType.RETRIEVAL,
                score=0.0,
                priority=5,
                token_count=tokenizer.count_tokens(text),
            )
        )
    return items


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------


def main() -> None:
    tokenizer = WhitespaceTokenizer()
    items = build_items(tokenizer)

    # 1. Create stores
    vector_store = InMemoryVectorStore()
    context_store = InMemoryContextStore()

    # 2. Create dense retriever and index documents
    dense = DenseRetriever(
        vector_store=vector_store,
        context_store=context_store,
        embed_fn=fake_embed,
    )
    dense.index(items)

    # 3. Create hybrid retriever (dense only in this example since
    #    SparseRetriever requires the optional bm25 dependency)
    hybrid = HybridRetriever(
        retrievers=[dense],
        weights=[1.0],
    )

    # 4. Build the pipeline
    pipeline = (
        ContextPipeline(max_tokens=4096, tokenizer=tokenizer)
        .add_system_prompt("You are a helpful assistant specializing in AI.")
        .add_step(retriever_step("hybrid-search", hybrid, top_k=3))
    )

    # 5. Run a query
    query = QueryBundle(
        query_str="How does retrieval-augmented generation work?",
        embedding=fake_embed("How does retrieval-augmented generation work?"),
    )
    result = pipeline.build(query)

    # 6. Print results
    print("=== Hybrid RAG Pipeline Results ===")
    print(f"Format type: {result.format_type}")
    print(f"Build time: {result.build_time_ms:.1f}ms")
    print(f"Items in context: {len(result.window.items)}")
    print(f"Token utilization: {result.window.utilization:.1%}")
    print()

    print("--- Retrieved Items ---")
    for item in result.window.items:
        print(f"  [{item.source.value}] (score={item.score:.3f}) {item.content[:80]}")

    print()
    print(f"Overflow items: {len(result.overflow_items)}")
    print(f"Diagnostics: {result.diagnostics}")


if __name__ == "__main__":
    main()
