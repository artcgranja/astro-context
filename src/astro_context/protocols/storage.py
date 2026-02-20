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

    def add(self, item: ContextItem) -> None:
        """Persist a single context item to the store.

        Parameters:
            item: The context item to store. If an item with the same
                ``id`` already exists, the implementation should overwrite it.

        Side Effects:
            The item is durably persisted (or held in memory, depending on
            the backend) and will be visible to subsequent ``get`` /
            ``get_all`` calls.
        """
        ...

    def get(self, item_id: str) -> ContextItem | None:
        """Retrieve a single context item by its unique identifier.

        Parameters:
            item_id: The unique string identifier of the item.

        Returns:
            The matching ``ContextItem``, or ``None`` if no item with the
            given id exists.
        """
        ...

    def get_all(self) -> list[ContextItem]:
        """Return every context item currently held in the store.

        Returns:
            A list of all stored ``ContextItem`` objects.  The order is
            implementation-defined.  Returns an empty list when the store
            is empty.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove a context item by id.

        Parameters:
            item_id: The unique identifier of the item to delete.

        Returns:
            ``True`` if an item was found and deleted, ``False`` if no
            item with the given id existed.

        Side Effects:
            The item is permanently removed from the store.
        """
        ...

    def clear(self) -> None:
        """Remove all items from the store.

        Side Effects:
            The store is left empty.  This operation is irreversible.
        """
        ...


@runtime_checkable
class VectorStore(Protocol):
    """Protocol for vector similarity search backends.

    Implementations might wrap FAISS, Chroma, Qdrant, Pinecone, etc.
    """

    def add_embedding(
        self, item_id: str, embedding: list[float], metadata: dict[str, Any] | None = None
    ) -> None:
        """Store an embedding vector associated with an item id.

        Parameters:
            item_id: A unique identifier linking the embedding back to its
                source context item or document.
            embedding: The dense vector representation of the item.
            metadata: Optional key-value metadata attached to the vector
                entry (e.g., source filename, chunk index).

        Side Effects:
            The embedding is persisted in the vector index.  If an entry
            for ``item_id`` already exists, the implementation should
            overwrite it.
        """
        ...

    def search(
        self, query_embedding: list[float], top_k: int = 10
    ) -> list[tuple[str, float]]:
        """Find the most similar embeddings to a query vector.

        Parameters:
            query_embedding: The dense vector to compare against stored
                embeddings.
            top_k: Maximum number of results to return.

        Returns:
            A list of ``(item_id, score)`` tuples ordered by descending
            similarity.  The score semantics (cosine, dot-product, etc.)
            are implementation-defined.
        """
        ...

    def delete(self, item_id: str) -> bool:
        """Remove the embedding associated with an item id.

        Parameters:
            item_id: The identifier of the embedding to remove.

        Returns:
            ``True`` if the embedding was found and removed, ``False``
            otherwise.

        Side Effects:
            The embedding is permanently removed from the vector index.
        """
        ...


@runtime_checkable
class DocumentStore(Protocol):
    """Protocol for document storage (raw text before chunking/indexing)."""

    def add_document(
        self, doc_id: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Store a raw document (pre-chunking / pre-indexing).

        Parameters:
            doc_id: A unique identifier for the document.
            content: The full text content of the document.
            metadata: Optional key-value metadata (e.g., source URL,
                author, ingestion timestamp).

        Side Effects:
            The document is persisted and becomes visible to subsequent
            ``get_document`` and ``list_documents`` calls.
        """
        ...

    def get_document(self, doc_id: str) -> str | None:
        """Retrieve a document's text content by its unique identifier.

        Parameters:
            doc_id: The identifier assigned when the document was added.

        Returns:
            The full text content of the document, or ``None`` if no
            document with the given id exists.
        """
        ...

    def list_documents(self) -> list[str]:
        """Return the ids of all stored documents.

        Returns:
            A list of document id strings.  The order is
            implementation-defined.  Returns an empty list when the store
            contains no documents.
        """
        ...

    def delete_document(self, doc_id: str) -> bool:
        """Remove a document by its id.

        Parameters:
            doc_id: The identifier of the document to delete.

        Returns:
            ``True`` if a document was found and deleted, ``False`` if no
            document with the given id existed.

        Side Effects:
            The document is permanently removed from the store.
        """
        ...
