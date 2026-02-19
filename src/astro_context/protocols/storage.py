"""Storage protocol definitions.

All storage backends implement these protocols using structural subtyping (PEP 544).
Users can provide any object that matches the interface -- no inheritance required.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from astro_context.models.context import ContextItem


@runtime_checkable
class ContextStore(Protocol):
    """Protocol for storing and retrieving context items."""

    def add(self, item: ContextItem) -> None: ...
    def get(self, item_id: str) -> ContextItem | None: ...
    def get_all(self) -> list[ContextItem]: ...
    def delete(self, item_id: str) -> bool: ...
    def clear(self) -> None: ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector similarity search backends.

    Implementations might wrap FAISS, Chroma, Qdrant, Pinecone, etc.
    """

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None: ...

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]: ...

    def delete(self, item_id: str) -> bool: ...


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage (raw text before chunking/indexing)."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None: ...
    def get_document(self, doc_id: str) -> str | None: ...
    def list_documents(self) -> list[str]: ...
    def delete_document(self, doc_id: str) -> bool: ...
