"""Tests for astro_context.retrieval.reranker.ScoreReranker."""

from __future__ import annotations

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.protocols.postprocessor import PostProcessor
from astro_context.retrieval.reranker import ScoreReranker

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    """Create a ContextItem for testing."""
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        priority=5,
        token_count=len(content.split()),
    )


def _simple_scorer(query: str, doc: str) -> float:
    """Score based on keyword overlap fraction."""
    query_words = set(query.lower().split())
    doc_words = set(doc.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & doc_words) / len(query_words)


# ---------------------------------------------------------------------------
# Basic reranking
# ---------------------------------------------------------------------------


class TestScoreRerankerBasic:
    """Basic reranking with a simple score function."""

    def test_reranks_items_by_score(self) -> None:
        items = [
            _make_item("low", "unrelated content here"),
            _make_item("high", "python programming language"),
        ]
        reranker = ScoreReranker(score_fn=_simple_scorer)
        query = QueryBundle(query_str="python programming")
        result = reranker.process(items, query)

        assert result[0].id == "high"
        assert result[1].id == "low"

    def test_scores_are_updated_on_items(self) -> None:
        items = [_make_item("a", "python programming")]
        reranker = ScoreReranker(score_fn=_simple_scorer)
        query = QueryBundle(query_str="python programming")
        result = reranker.process(items, query)

        assert result[0].score == 1.0  # full overlap

    def test_items_sorted_descending_by_score(self) -> None:
        items = [
            _make_item("a", "no match at all"),
            _make_item("b", "partial python match"),
            _make_item("c", "python programming language complete"),
        ]

        def scorer(query: str, doc: str) -> float:
            # Deterministic scoring: c > b > a
            if "complete" in doc:
                return 0.9
            if "partial" in doc:
                return 0.5
            return 0.1

        reranker = ScoreReranker(score_fn=scorer)
        query = QueryBundle(query_str="python")
        result = reranker.process(items, query)

        assert [item.id for item in result] == ["c", "b", "a"]

    def test_score_function_receives_correct_arguments(self) -> None:
        """Verify the score function gets (query_str, item.content)."""
        received_args: list[tuple[str, str]] = []

        def tracking_scorer(query: str, doc: str) -> float:
            received_args.append((query, doc))
            return 0.5

        items = [_make_item("x", "document content")]
        reranker = ScoreReranker(score_fn=tracking_scorer)
        query = QueryBundle(query_str="my query")
        reranker.process(items, query)

        assert len(received_args) == 1
        assert received_args[0] == ("my query", "document content")


# ---------------------------------------------------------------------------
# top_k truncation
# ---------------------------------------------------------------------------


class TestScoreRerankerTopK:
    """top_k truncation."""

    def test_top_k_truncates_results(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5, top_k=3)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert len(result) == 3

    def test_top_k_none_returns_all(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5, top_k=None)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert len(result) == 10

    def test_top_k_larger_than_items(self) -> None:
        items = [_make_item("a", "content a"), _make_item("b", "content b")]
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5, top_k=100)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Score clamping
# ---------------------------------------------------------------------------


class TestScoreRerankerClamping:
    """Scores outside 0.0-1.0 are clamped."""

    def test_score_above_one_is_clamped(self) -> None:
        items = [_make_item("a", "content")]
        reranker = ScoreReranker(score_fn=lambda q, d: 2.5)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert result[0].score == 1.0

    def test_score_below_zero_is_clamped(self) -> None:
        items = [_make_item("a", "content")]
        reranker = ScoreReranker(score_fn=lambda q, d: -0.5)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert result[0].score == 0.0

    def test_score_within_range_unchanged(self) -> None:
        items = [_make_item("a", "content")]
        reranker = ScoreReranker(score_fn=lambda q, d: 0.7)
        query = QueryBundle(query_str="test")
        result = reranker.process(items, query)

        assert result[0].score == 0.7


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestScoreRerankerEdgeCases:
    """Edge cases: empty items, None query."""

    def test_empty_items_returns_empty(self) -> None:
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5)
        query = QueryBundle(query_str="test")
        result = reranker.process([], query)

        assert result == []

    def test_none_query_returns_items_unchanged(self) -> None:
        items = [_make_item("a", "content", score=0.3)]
        reranker = ScoreReranker(score_fn=lambda q, d: 0.9)
        result = reranker.process(items, query=None)

        # With None query, process returns items as-is
        assert len(result) == 1
        assert result[0].id == "a"
        assert result[0].score == 0.3  # unchanged

    def test_empty_items_with_none_query(self) -> None:
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5)
        result = reranker.process([], query=None)

        assert result == []


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestScoreRerankerProtocol:
    """PostProcessor protocol conformance."""

    def test_isinstance_check(self) -> None:
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5)
        assert isinstance(reranker, PostProcessor)

    def test_no_format_type_property(self) -> None:
        """ScoreReranker is a PostProcessor, not a Formatter -- no format_type."""
        reranker = ScoreReranker(score_fn=lambda q, d: 0.5)
        assert not hasattr(reranker, "format_type")
