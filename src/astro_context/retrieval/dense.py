"""Dense (embedding-based) retrieval."""

from __future__ import annotations

from collections.abc import Callable

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.protocols.storage import ContextStore, VectorStore
from astro_context.tokens.counter import get_default_counter


class DenseRetriever:
    """Retrieves context items via embedding similarity search.

    Requires a VectorStore backend and a ContextStore to resolve IDs to items.
    The embed_fn is user-provided -- astro-context never calls an LLM directly.

    Implements the Retriever protocol.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        context_store: ContextStore,
        embed_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._vector_store = vector_store
        self._context_store = context_store
        self._embed_fn = embed_fn
        self._counter = get_default_counter()

    def index(self, items: list[ContextItem]) -> int:
        """Index items into vector and context stores. Returns count indexed."""
        if self._embed_fn is None:
            msg = "embed_fn must be provided to index items"
            raise ValueError(msg)
        count = 0
        for item in items:
            embedding = self._embed_fn(item.content)
            self._vector_store.add_embedding(item.id, embedding, item.metadata)
            self._context_store.add(item)
            count += 1
        return count

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        """Retrieve items most similar to the query embedding."""
        if query.embedding is not None:
            query_embedding = query.embedding
        elif self._embed_fn is not None:
            query_embedding = self._embed_fn(query.query_str)
        else:
            msg = "Either provide query.embedding or set embed_fn on the retriever"
            raise ValueError(msg)

        results = self._vector_store.search(query_embedding, top_k=top_k)
        items: list[ContextItem] = []
        for item_id, score in results:
            item = self._context_store.get(item_id)
            if item is not None:
                scored_item = ContextItem(
                    id=item.id,
                    content=item.content,
                    source=SourceType.RETRIEVAL,
                    score=min(1.0, max(0.0, score)),
                    priority=item.priority,
                    token_count=item.token_count or self._counter.count_tokens(item.content),
                    metadata={**item.metadata, "retrieval_method": "dense"},
                    created_at=item.created_at,
                )
                items.append(scored_item)
        return items
