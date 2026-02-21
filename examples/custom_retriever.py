"""Example: Custom Retriever Protocol. Run with: python examples/custom_retriever.py

Shows how to implement a custom retriever that satisfies the Retriever
protocol (PEP 544) and plug it into a ContextPipeline via retriever_step().

No inheritance is needed -- any object with a
``retrieve(query: QueryBundle, top_k: int) -> list[ContextItem]`` method works.
"""

from __future__ import annotations

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import retriever_step

# ---------------------------------------------------------------------------
# Simple whitespace tokenizer (avoids tiktoken dependency)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal tokenizer for demonstration."""

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return " ".join(words[:max_tokens])


# ---------------------------------------------------------------------------
# Custom retriever
# ---------------------------------------------------------------------------

# A simple knowledge base (in practice this could be a database, API, etc.)
KNOWLEDGE_BASE = [
    "Context engineering focuses on what information reaches the LLM.",
    "Token budgets prevent context windows from overflowing.",
    "Priority-ranked items ensure the most important context comes first.",
    "Sliding window memory keeps recent conversation turns available.",
    "Hybrid retrieval combines dense and sparse signals for better recall.",
]


class KeywordRetriever:
    """A simple keyword-matching retriever that satisfies the Retriever protocol.

    This demonstrates the protocol-based extensibility of astro-context:
    no base class inheritance is required. Just implement the ``retrieve``
    method with the correct signature.
    """

    def __init__(self, documents: list[str]) -> None:
        self._documents = documents

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve documents matching query keywords.

        Scores each document by the fraction of query terms it contains.
        """
        query_terms = set(query.query_str.lower().split())
        if not query_terms:
            return []

        scored: list[tuple[float, str]] = []
        for doc in self._documents:
            doc_lower = doc.lower()
            matches = sum(1 for term in query_terms if term in doc_lower)
            score = matches / len(query_terms)
            if score > 0:
                scored.append((score, doc))

        # Sort by score descending, take top_k
        scored.sort(key=lambda x: x[0], reverse=True)

        tokenizer = WhitespaceTokenizer()
        return [
            ContextItem(
                content=doc,
                source=SourceType.RETRIEVAL,
                score=score,
                priority=5,
                token_count=tokenizer.count_tokens(doc),
            )
            for score, doc in scored[:top_k]
        ]


# ---------------------------------------------------------------------------
# Main example
# ---------------------------------------------------------------------------


def main() -> None:
    tokenizer = WhitespaceTokenizer()

    # 1. Create the custom retriever
    retriever = KeywordRetriever(KNOWLEDGE_BASE)

    # 2. Build a pipeline with the custom retriever
    pipeline = (
        ContextPipeline(max_tokens=4096, tokenizer=tokenizer)
        .add_system_prompt("You are a context engineering expert.")
        .add_step(retriever_step("keyword-search", retriever, top_k=3))
    )

    # 3. Run a query
    result = pipeline.build("How do token budgets work?")

    # 4. Print results
    print("=== Custom Retriever Pipeline Results ===")
    print(f"Items in context: {len(result.window.items)}")
    print()

    for item in result.window.items:
        label = item.source.value
        print(f"  [{label}] (score={item.score:.3f}, prio={item.priority}) {item.content}")

    print()
    print(f"Build time: {result.build_time_ms:.1f}ms")

    # 5. Verify it satisfies the protocol (optional runtime check)
    from astro_context.protocols.retriever import Retriever

    print(f"Satisfies Retriever protocol: {isinstance(retriever, Retriever)}")


if __name__ == "__main__":
    main()
