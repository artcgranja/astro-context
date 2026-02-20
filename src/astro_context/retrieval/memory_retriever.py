"""Multi-signal memory retriever combining recency, relevance, and importance."""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_context.protocols.memory import MemoryDecay
    from astro_context.protocols.storage import MemoryEntryStore, VectorStore

from astro_context.models.memory import MemoryEntry


class ScoredMemoryRetriever:
    """Multi-signal memory retriever combining recency, relevance, and importance.

    Computes a composite score for each memory entry::

        score = alpha * recency + beta * relevance + gamma * importance

    Where:

    - **recency**: From a ``MemoryDecay`` protocol implementation (if provided)
      or a simple exponential time-based decay.
    - **relevance**: Cosine similarity from a ``VectorStore`` (if both
      ``embed_fn`` and ``vector_store`` are provided), or a basic keyword
      overlap score.
    - **importance**: The entry's ``relevance_score`` field (0.0 -- 1.0).

    The retriever does **not** call an LLM. The ``embed_fn`` is user-provided.

    Example::

        retriever = ScoredMemoryRetriever(
            store=my_store,
            embed_fn=my_embed,
            vector_store=my_vectors,
            alpha=0.3,
            beta=0.5,
            gamma=0.2,
        )
        top_memories = retriever.retrieve("what did the user say about testing?")
    """

    __slots__ = (
        "_alpha",
        "_beta",
        "_decay",
        "_embed_fn",
        "_gamma",
        "_store",
        "_vector_store",
    )

    def __init__(
        self,
        store: MemoryEntryStore,
        embed_fn: Callable[[str], list[float]] | None = None,
        vector_store: VectorStore | None = None,
        decay: MemoryDecay | None = None,
        alpha: float = 0.3,
        beta: float = 0.5,
        gamma: float = 0.2,
    ) -> None:
        self._store = store
        self._embed_fn = embed_fn
        self._vector_store = vector_store
        self._decay = decay
        self._alpha = alpha
        self._beta = beta
        self._gamma = gamma

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        *,
        user_id: str | None = None,
        memory_type: str | None = None,
    ) -> list[MemoryEntry]:
        """Retrieve top-k memory entries scored by the multi-signal function.

        Parameters:
            query: The search query text.
            top_k: Maximum number of entries to return.
            user_id: If provided, only entries with this user_id are returned.
            memory_type: If provided, only entries with this memory_type are
                returned (matched as string value).

        Returns:
            A list of up to ``top_k`` memory entries sorted by composite
            score (descending).
        """
        # Step 1: get candidate entries and optional relevance scores from vector search
        relevance_map: dict[str, float] = {}

        if self._vector_store is not None and self._embed_fn is not None:
            query_embedding = self._embed_fn(query)
            # Request more candidates than top_k to allow for filtering
            vector_results = self._vector_store.search(query_embedding, top_k=top_k * 3)
            relevance_map = dict(vector_results)

        # Step 2: get all entries from store
        candidates = self._store.list_all()

        # Step 3: filter by user_id and memory_type
        now = datetime.now(UTC)
        filtered: list[MemoryEntry] = []
        for entry in candidates:
            # Skip expired entries
            if entry.expires_at is not None and entry.expires_at <= now:
                continue
            if user_id is not None and entry.user_id != user_id:
                continue
            if memory_type is not None and str(entry.memory_type) != memory_type:
                continue
            filtered.append(entry)

        # Step 4: score each entry
        scored: list[tuple[float, MemoryEntry]] = []
        for entry in filtered:
            recency = self._compute_recency(entry)
            relevance = self._compute_relevance(query, entry, relevance_map)
            importance = entry.relevance_score

            composite = (
                self._alpha * recency
                + self._beta * relevance
                + self._gamma * importance
            )
            scored.append((composite, entry))

        # Step 5: sort by score descending, return top_k
        scored.sort(key=lambda x: x[0], reverse=True)
        return [entry for _, entry in scored[:top_k]]

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add an entry to the store and optionally index its embedding.

        If both ``embed_fn`` and ``vector_store`` are provided, the entry's
        content is embedded and stored in the vector index.
        """
        self._store.add(entry)
        if self._embed_fn is not None and self._vector_store is not None:
            embedding = self._embed_fn(entry.content)
            self._vector_store.add_embedding(entry.id, embedding)

    # ------------------------------------------------------------------
    # Scoring helpers
    # ------------------------------------------------------------------

    def _compute_recency(self, entry: MemoryEntry) -> float:
        """Compute a recency score from 0.0 (very old) to 1.0 (just now).

        Uses the ``MemoryDecay`` protocol if provided, otherwise falls back
        to an exponential decay with a half-life of 7 days.
        """
        if self._decay is not None:
            return self._decay.compute_retention(entry)

        # Exponential decay: score = exp(-lambda * age_seconds)
        # Half-life of 7 days: lambda = ln(2) / (7 * 86400)
        half_life_seconds = 7 * 86400.0
        decay_lambda = math.log(2) / half_life_seconds

        now = datetime.now(UTC)
        last_accessed = entry.last_accessed
        # Ensure timezone-aware comparison
        if last_accessed.tzinfo is None:
            age_seconds = (now - last_accessed.replace(tzinfo=UTC)).total_seconds()
        else:
            age_seconds = (now - last_accessed).total_seconds()

        age_seconds = max(0.0, age_seconds)
        return math.exp(-decay_lambda * age_seconds)

    def _compute_relevance(
        self,
        query: str,
        entry: MemoryEntry,
        relevance_map: dict[str, float],
    ) -> float:
        """Compute a relevance score for the entry against the query.

        Uses pre-computed vector similarity from ``relevance_map`` if the
        entry was in the vector search results. Otherwise falls back to a
        basic keyword overlap score.
        """
        # Use vector similarity if available
        if entry.id in relevance_map:
            score = relevance_map[entry.id]
            return max(0.0, min(1.0, score))

        # Fallback: keyword overlap (Jaccard-like)
        return self._keyword_overlap(query, entry.content)

    @staticmethod
    def _keyword_overlap(query: str, content: str) -> float:
        """Compute a simple keyword overlap score between query and content.

        Returns a value from 0.0 (no overlap) to 1.0 (full overlap of
        query terms in content).
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return 0.0
        content_lower = content.lower()
        matches = sum(1 for term in query_terms if term in content_lower)
        return matches / len(query_terms)

    def __repr__(self) -> str:
        return (
            f"ScoredMemoryRetriever("
            f"alpha={self._alpha}, beta={self._beta}, gamma={self._gamma}, "
            f"embed_fn={'set' if self._embed_fn is not None else 'None'}, "
            f"vector_store={'set' if self._vector_store is not None else 'None'}, "
            f"decay={'set' if self._decay is not None else 'None'})"
        )
