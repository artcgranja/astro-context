"""In-memory storage implementations for development and testing.

These are the default backends -- no external dependencies needed.
Production users provide their own implementations (Redis, Postgres, etc.)
that satisfy the storage protocols.
"""

from __future__ import annotations

import heapq
from typing import Any

from astro_context.models.context import ContextItem


class InMemoryContextStore:
    """Dict-backed context store. Implements ContextStore protocol."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[str, ContextItem] = {}

    def add(self, item: ContextItem) -> None:
        self._items[item.id] = item

    def get(self, item_id: str) -> ContextItem | None:
        return self._items.get(item_id)

    def get_all(self) -> list[ContextItem]:
        return list(self._items.values())

    def delete(self, item_id: str) -> bool:
        return self._items.pop(item_id, None) is not None

    def clear(self) -> None:
        self._items.clear()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(items={len(self._items)})"


class InMemoryVectorStore:
    """Brute-force cosine similarity vector store.

    For development/testing only. Production use should provide
    FAISS, Chroma, Qdrant, etc. via the VectorStore protocol.
    """

    __slots__ = ("_embeddings", "_metadata")

    def __init__(self) -> None:
        self._embeddings: dict[str, list[float]] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        self._embeddings[item_id] = embedding
        if metadata:
            self._metadata[item_id] = metadata

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        if not self._embeddings:
            return []
        results: list[tuple[str, float]] = []
        for item_id, emb in self._embeddings.items():
            score = self._cosine_similarity(query_embedding, emb)
            results.append((item_id, score))
        return heapq.nlargest(top_k, results, key=lambda x: x[1])

    def delete(self, item_id: str) -> bool:
        removed = self._embeddings.pop(item_id, None) is not None
        self._metadata.pop(item_id, None)
        return removed

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors without numpy."""
        dot = sum((x * y for x, y in zip(a, b, strict=True)), 0.0)
        norm_a = sum((x * x for x in a), 0.0) ** 0.5
        norm_b = sum((x * x for x in b), 0.0) ** 0.5
        if norm_a == 0 or norm_b == 0:
            return 0.0
        similarity: float = dot / (norm_a * norm_b)
        return max(-1.0, min(1.0, similarity))

    def __repr__(self) -> str:
        return f"{type(self).__name__}(embeddings={len(self._embeddings)})"


class InMemoryDocumentStore:
    """Dict-backed document store. Implements DocumentStore protocol."""

    __slots__ = ("_documents", "_metadata")

    def __init__(self) -> None:
        self._documents: dict[str, str] = {}
        self._metadata: dict[str, dict[str, Any]] = {}

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        self._documents[doc_id] = content
        if metadata:
            self._metadata[doc_id] = metadata

    def get_document(self, doc_id: str) -> str | None:
        return self._documents.get(doc_id)

    def list_documents(self) -> list[str]:
        return list(self._documents.keys())

    def delete_document(self, doc_id: str) -> bool:
        removed = self._documents.pop(doc_id, None) is not None
        self._metadata.pop(doc_id, None)
        return removed

    def __repr__(self) -> str:
        return f"{type(self).__name__}(documents={len(self._documents)})"
