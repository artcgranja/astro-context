"""Tests for astro_context.retrieval.memory_retriever.ScoredMemoryRetriever."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from astro_context.models.memory import MemoryEntry, MemoryType
from astro_context.retrieval.memory_retriever import ScoredMemoryRetriever
from astro_context.storage.json_memory_store import InMemoryEntryStore
from astro_context.storage.memory_store import InMemoryVectorStore
from tests.conftest import make_embedding

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_entry(
    *,
    entry_id: str = "e1",
    content: str = "some memory content",
    relevance_score: float = 0.5,
    user_id: str | None = None,
    memory_type: MemoryType = MemoryType.SEMANTIC,
    tags: list[str] | None = None,
    last_accessed: datetime | None = None,
    created_at: datetime | None = None,
    expires_at: datetime | None = None,
) -> MemoryEntry:
    """Build a MemoryEntry with sensible test defaults."""
    kwargs: dict = {
        "id": entry_id,
        "content": content,
        "relevance_score": relevance_score,
        "memory_type": memory_type,
    }
    if user_id is not None:
        kwargs["user_id"] = user_id
    if tags is not None:
        kwargs["tags"] = tags
    if last_accessed is not None:
        kwargs["last_accessed"] = last_accessed
    if created_at is not None:
        kwargs["created_at"] = created_at
    if expires_at is not None:
        kwargs["expires_at"] = expires_at
    return MemoryEntry(**kwargs)


def _fake_embed(text: str) -> list[float]:
    """Deterministic embedding based on text length for testing."""
    seed = len(text) % 100
    return make_embedding(seed)


class _ConstantDecay:
    """A trivial MemoryDecay that always returns a fixed retention value."""

    def __init__(self, value: float = 0.8) -> None:
        self._value = value

    def compute_retention(self, entry: MemoryEntry) -> float:
        return self._value


class _ImportanceOnlyDecay:
    """A MemoryDecay that returns the entry's relevance_score (to test custom scoring)."""

    def compute_retention(self, entry: MemoryEntry) -> float:
        return entry.relevance_score


@pytest.fixture
def entry_store() -> InMemoryEntryStore:
    """Return a fresh InMemoryEntryStore."""
    return InMemoryEntryStore()


@pytest.fixture
def vector_store() -> InMemoryVectorStore:
    """Return a fresh InMemoryVectorStore."""
    return InMemoryVectorStore()


# ---------------------------------------------------------------------------
# Basic retrieval (store only, no vector store)
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverBasic:
    """Basic retrieval with store only, no vector store."""

    def test_retrieve_with_store_only(self, entry_store: InMemoryEntryStore) -> None:
        entry_store.add(_make_entry(entry_id="b1", content="testing basics"))
        entry_store.add(_make_entry(entry_id="b2", content="another memory"))

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("testing")
        assert len(results) == 2

    def test_retrieve_returns_entries_sorted_by_composite_score(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        """Higher importance (relevance_score) should produce higher composite score."""
        entry_store.add(
            _make_entry(entry_id="lo", content="memory item", relevance_score=0.1)
        )
        entry_store.add(
            _make_entry(entry_id="hi", content="memory item", relevance_score=0.9)
        )

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("memory")
        # With default alpha=0.3, beta=0.5, gamma=0.2 and similar recency/relevance,
        # the importance difference (gamma * 0.9 vs gamma * 0.1) should differentiate
        assert results[0].id == "hi"

    def test_retrieve_respects_top_k(self, entry_store: InMemoryEntryStore) -> None:
        for i in range(10):
            entry_store.add(_make_entry(entry_id=f"k{i}", content=f"memory number {i}"))

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("memory", top_k=3)
        assert len(results) == 3


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverFiltering:
    """Filtering by user_id, memory_type, and expiry."""

    def test_filters_by_user_id(self, entry_store: InMemoryEntryStore) -> None:
        entry_store.add(_make_entry(entry_id="u1", content="user mem", user_id="alice"))
        entry_store.add(_make_entry(entry_id="u2", content="user mem", user_id="bob"))

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("user", user_id="alice")
        assert len(results) == 1
        assert results[0].id == "u1"

    def test_filters_by_memory_type(self, entry_store: InMemoryEntryStore) -> None:
        entry_store.add(
            _make_entry(
                entry_id="m1", content="type mem", memory_type=MemoryType.PROCEDURAL
            )
        )
        entry_store.add(
            _make_entry(
                entry_id="m2", content="type mem", memory_type=MemoryType.SEMANTIC
            )
        )

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("type", memory_type="procedural")
        assert len(results) == 1
        assert results[0].id == "m1"

    def test_filters_expired_entries(self, entry_store: InMemoryEntryStore) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        future = datetime.now(UTC) + timedelta(hours=1)

        entry_store.add(
            _make_entry(entry_id="exp", content="expiry test", expires_at=past)
        )
        entry_store.add(
            _make_entry(entry_id="valid", content="expiry test", expires_at=future)
        )
        entry_store.add(
            _make_entry(entry_id="no_exp", content="expiry test")
        )

        retriever = ScoredMemoryRetriever(store=entry_store)
        results = retriever.retrieve("expiry")
        ids = {e.id for e in results}
        assert "exp" not in ids
        assert "valid" in ids
        assert "no_exp" in ids


# ---------------------------------------------------------------------------
# add_entry
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverAddEntry:
    """add_entry() stores and optionally indexes embeddings."""

    def test_add_entry_stores_in_entry_store(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        retriever = ScoredMemoryRetriever(store=entry_store)
        entry = _make_entry(entry_id="ae1", content="added entry")
        retriever.add_entry(entry)
        assert entry_store.get("ae1") is not None

    def test_add_entry_indexes_embedding_when_embed_fn_and_vector_store(
        self,
        entry_store: InMemoryEntryStore,
        vector_store: InMemoryVectorStore,
    ) -> None:
        retriever = ScoredMemoryRetriever(
            store=entry_store,
            embed_fn=_fake_embed,
            vector_store=vector_store,
        )
        entry = _make_entry(entry_id="ae2", content="embedded entry")
        retriever.add_entry(entry)

        # Verify embedding was stored in vector store
        results = vector_store.search(_fake_embed("embedded entry"), top_k=5)
        found_ids = [r[0] for r in results]
        assert "ae2" in found_ids

    def test_add_entry_skips_embedding_when_no_vector_store(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        retriever = ScoredMemoryRetriever(
            store=entry_store, embed_fn=_fake_embed
        )
        entry = _make_entry(entry_id="ae3", content="no vector")
        retriever.add_entry(entry)
        # Should not raise, just adds to store
        assert entry_store.get("ae3") is not None

    def test_add_entry_skips_embedding_when_no_embed_fn(
        self,
        entry_store: InMemoryEntryStore,
        vector_store: InMemoryVectorStore,
    ) -> None:
        retriever = ScoredMemoryRetriever(
            store=entry_store, vector_store=vector_store
        )
        entry = _make_entry(entry_id="ae4", content="no embed fn")
        retriever.add_entry(entry)
        assert entry_store.get("ae4") is not None


# ---------------------------------------------------------------------------
# Multi-signal scoring with custom alpha/beta/gamma
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverCustomWeights:
    """Multi-signal scoring with custom alpha/beta/gamma."""

    def test_importance_only_scoring(self, entry_store: InMemoryEntryStore) -> None:
        """With gamma=1.0 and alpha=beta=0.0, only importance matters."""
        entry_store.add(
            _make_entry(entry_id="lo", content="weight test", relevance_score=0.2)
        )
        entry_store.add(
            _make_entry(entry_id="hi", content="weight test", relevance_score=0.9)
        )

        retriever = ScoredMemoryRetriever(
            store=entry_store, alpha=0.0, beta=0.0, gamma=1.0
        )
        results = retriever.retrieve("weight")
        assert results[0].id == "hi"
        assert results[1].id == "lo"

    def test_recency_only_scoring(self, entry_store: InMemoryEntryStore) -> None:
        """With alpha=1.0 and beta=gamma=0.0, only recency matters."""
        now = datetime.now(UTC)
        entry_store.add(
            _make_entry(
                entry_id="old",
                content="recency test",
                last_accessed=now - timedelta(days=30),
                relevance_score=0.9,
            )
        )
        entry_store.add(
            _make_entry(
                entry_id="new",
                content="recency test",
                last_accessed=now,
                relevance_score=0.1,
            )
        )

        retriever = ScoredMemoryRetriever(
            store=entry_store, alpha=1.0, beta=0.0, gamma=0.0
        )
        results = retriever.retrieve("recency")
        assert results[0].id == "new"

    def test_relevance_only_scoring_with_keyword_fallback(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        """With beta=1.0 and alpha=gamma=0.0, only relevance matters."""
        entry_store.add(
            _make_entry(entry_id="match", content="python programming language")
        )
        entry_store.add(
            _make_entry(entry_id="no_match", content="java enterprise beans")
        )

        retriever = ScoredMemoryRetriever(
            store=entry_store, alpha=0.0, beta=1.0, gamma=0.0
        )
        # Query "python programming" has 2/2 keyword overlap with first entry
        results = retriever.retrieve("python programming")
        assert results[0].id == "match"


# ---------------------------------------------------------------------------
# Retrieval with vector store and embed_fn
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverWithVectorStore:
    """Retrieval with vector store and embed_fn for relevance scoring."""

    def test_uses_vector_similarity_for_relevance(
        self,
        entry_store: InMemoryEntryStore,
        vector_store: InMemoryVectorStore,
    ) -> None:
        # Add entries and embeddings
        e1 = _make_entry(entry_id="v1", content="vector similar to query")
        e2 = _make_entry(entry_id="v2", content="completely different topic")
        entry_store.add(e1)
        entry_store.add(e2)

        # Use distinct embeddings: v1 matches the query embedding, v2 does not
        query_emb = make_embedding(seed=42)
        vector_store.add_embedding("v1", make_embedding(seed=42))   # identical to query
        vector_store.add_embedding("v2", make_embedding(seed=999))  # far from query

        def embed_fn(text: str) -> list[float]:
            return query_emb

        retriever = ScoredMemoryRetriever(
            store=entry_store,
            embed_fn=embed_fn,
            vector_store=vector_store,
            alpha=0.0,
            beta=1.0,
            gamma=0.0,
        )
        results = retriever.retrieve("any query")
        assert len(results) == 2
        assert results[0].id == "v1"

    def test_entries_not_in_vector_store_use_keyword_fallback(
        self,
        entry_store: InMemoryEntryStore,
        vector_store: InMemoryVectorStore,
    ) -> None:
        """Entries without embeddings should still get a keyword-based relevance score."""
        e1 = _make_entry(entry_id="indexed", content="keyword overlap test")
        e2 = _make_entry(entry_id="not_indexed", content="keyword overlap test")
        entry_store.add(e1)
        entry_store.add(e2)

        # Only index e1 in vector store
        vector_store.add_embedding("indexed", make_embedding(seed=1))

        retriever = ScoredMemoryRetriever(
            store=entry_store,
            embed_fn=_fake_embed,
            vector_store=vector_store,
        )
        results = retriever.retrieve("keyword overlap")
        # Both should be returned (non-indexed uses keyword fallback)
        ids = {e.id for e in results}
        assert "indexed" in ids
        assert "not_indexed" in ids


# ---------------------------------------------------------------------------
# Custom MemoryDecay
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverCustomDecay:
    """Retrieval with a custom MemoryDecay implementation."""

    def test_uses_custom_decay_for_recency(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        entry_store.add(
            _make_entry(entry_id="d1", content="decay test", relevance_score=0.5)
        )
        entry_store.add(
            _make_entry(entry_id="d2", content="decay test", relevance_score=0.5)
        )

        # ConstantDecay always returns 0.8 for recency
        decay = _ConstantDecay(value=0.8)
        retriever = ScoredMemoryRetriever(
            store=entry_store,
            decay=decay,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
        )
        results = retriever.retrieve("decay")
        # Both get same recency (0.8), both should be returned
        assert len(results) == 2

    def test_custom_decay_affects_ranking(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        """ImportanceOnlyDecay uses relevance_score as recency, effectively
        doubling the importance signal when alpha + gamma are both used."""
        entry_store.add(
            _make_entry(entry_id="lo", content="decay rank test", relevance_score=0.1)
        )
        entry_store.add(
            _make_entry(entry_id="hi", content="decay rank test", relevance_score=0.9)
        )

        decay = _ImportanceOnlyDecay()
        retriever = ScoredMemoryRetriever(
            store=entry_store,
            decay=decay,
            alpha=1.0,
            beta=0.0,
            gamma=0.0,
        )
        results = retriever.retrieve("decay rank")
        assert results[0].id == "hi"


# ---------------------------------------------------------------------------
# Keyword overlap (fallback relevance)
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverKeywordOverlap:
    """Fallback keyword relevance scoring when no vector store is present."""

    def test_keyword_overlap_full_match(self) -> None:
        score = ScoredMemoryRetriever._keyword_overlap(
            "python machine learning", "Python Machine Learning Basics"
        )
        assert score == pytest.approx(1.0)

    def test_keyword_overlap_partial_match(self) -> None:
        score = ScoredMemoryRetriever._keyword_overlap(
            "python java rust", "Python is great for data science"
        )
        # Only "python" matches out of 3 terms
        assert score == pytest.approx(1.0 / 3.0)

    def test_keyword_overlap_no_match(self) -> None:
        score = ScoredMemoryRetriever._keyword_overlap(
            "completely unrelated", "Python machine learning"
        )
        assert score == pytest.approx(0.0)

    def test_keyword_overlap_empty_query(self) -> None:
        score = ScoredMemoryRetriever._keyword_overlap("", "some content")
        assert score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# repr
# ---------------------------------------------------------------------------


class TestScoredMemoryRetrieverRepr:
    """Test __repr__ for debugging."""

    def test_repr_shows_parameters(self, entry_store: InMemoryEntryStore) -> None:
        retriever = ScoredMemoryRetriever(store=entry_store, alpha=0.4, beta=0.4, gamma=0.2)
        r = repr(retriever)
        assert "ScoredMemoryRetriever" in r
        assert "0.4" in r
        assert "0.2" in r

    def test_repr_shows_embed_fn_status(
        self,
        entry_store: InMemoryEntryStore,
        vector_store: InMemoryVectorStore,
    ) -> None:
        retriever = ScoredMemoryRetriever(
            store=entry_store,
            embed_fn=_fake_embed,
            vector_store=vector_store,
        )
        r = repr(retriever)
        assert "embed_fn=set" in r
        assert "vector_store=set" in r

    def test_repr_shows_none_when_no_extras(
        self, entry_store: InMemoryEntryStore
    ) -> None:
        retriever = ScoredMemoryRetriever(store=entry_store)
        r = repr(retriever)
        assert "embed_fn=None" in r
        assert "vector_store=None" in r
        assert "decay=None" in r
