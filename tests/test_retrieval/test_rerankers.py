"""Tests for astro_context.retrieval.rerankers (advanced reranker implementations)."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from astro_context.exceptions import RetrieverError
from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.step import reranker_step
from astro_context.protocols.reranker import Reranker
from astro_context.retrieval.rerankers import (
    CohereReranker,
    CrossEncoderReranker,
    FlashRankReranker,
    RerankerPipeline,
    RoundRobinReranker,
)

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


def _overlap_scorer(query: str, doc: str) -> float:
    """Score based on keyword overlap fraction."""
    query_words = set(query.lower().split())
    doc_words = set(doc.lower().split())
    if not query_words:
        return 0.0
    return len(query_words & doc_words) / len(query_words)


# ---------------------------------------------------------------------------
# CrossEncoderReranker
# ---------------------------------------------------------------------------


class TestCrossEncoderReranker:
    """Tests for CrossEncoderReranker."""

    def test_reranks_by_score_descending(self) -> None:
        items = [
            _make_item("low", "unrelated content here"),
            _make_item("high", "python programming language"),
        ]
        reranker = CrossEncoderReranker(score_fn=_overlap_scorer, top_k=10)
        query = QueryBundle(query_str="python programming")
        result = reranker.rerank(query, items)

        assert result[0].id == "high"
        assert result[1].id == "low"

    def test_scores_updated_on_items(self) -> None:
        items = [_make_item("a", "python programming")]
        reranker = CrossEncoderReranker(score_fn=_overlap_scorer, top_k=10)
        query = QueryBundle(query_str="python programming")
        result = reranker.rerank(query, items)

        assert result[0].score == 1.0

    def test_top_k_truncates(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=3)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 3

    def test_score_clamped_above_one(self) -> None:
        items = [_make_item("a", "content")]
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 2.5, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert result[0].score == 1.0

    def test_score_clamped_below_zero(self) -> None:
        items = [_make_item("a", "content")]
        reranker = CrossEncoderReranker(score_fn=lambda q, d: -0.5, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert result[0].score == 0.0

    def test_empty_items(self) -> None:
        reranker = CrossEncoderReranker(score_fn=_overlap_scorer, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, [])

        assert result == []

    def test_single_item(self) -> None:
        items = [_make_item("only", "the only item")]
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.7, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 1
        assert result[0].id == "only"
        assert result[0].score == 0.7

    def test_score_fn_receives_correct_args(self) -> None:
        received: list[tuple[str, str]] = []

        def tracker(query: str, doc: str) -> float:
            received.append((query, doc))
            return 0.5

        items = [_make_item("x", "doc content")]
        reranker = CrossEncoderReranker(score_fn=tracker, top_k=10)
        query = QueryBundle(query_str="my query")
        reranker.rerank(query, items)

        assert len(received) == 1
        assert received[0] == ("my query", "doc content")

    def test_repr(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=5)
        assert "CrossEncoderReranker" in repr(reranker)
        assert "top_k=5" in repr(reranker)

    def test_protocol_compliance(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=10)
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# CohereReranker
# ---------------------------------------------------------------------------


class TestCohereReranker:
    """Tests for CohereReranker."""

    def test_reranks_using_callback(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            # Reverse order with scores
            return [(i, 1.0 - i * 0.1) for i in reversed(range(min(top_k, len(docs))))]

        items = [
            _make_item("a", "first doc"),
            _make_item("b", "second doc"),
            _make_item("c", "third doc"),
        ]
        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 3

    def test_top_k_limits_results(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return [(i, 0.9 - i * 0.1) for i in range(min(top_k, len(docs)))]

        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=3)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 3

    def test_scores_updated(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return [(0, 0.95)]

        items = [_make_item("a", "content")]
        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert result[0].score == 0.95

    def test_score_clamped(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return [(0, 1.5)]

        items = [_make_item("a", "content")]
        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert result[0].score == 1.0

    def test_callback_receives_correct_args(self) -> None:
        received: list[tuple[str, list[str], int]] = []

        def tracking_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            received.append((query, docs, top_k))
            return [(0, 0.5)]

        items = [_make_item("x", "doc text")]
        reranker = CohereReranker(rerank_fn=tracking_rerank, top_k=10)
        query = QueryBundle(query_str="my query")
        reranker.rerank(query, items)

        assert len(received) == 1
        assert received[0][0] == "my query"
        assert received[0][1] == ["doc text"]

    def test_empty_items(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return []

        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, [])

        assert result == []

    def test_invalid_index_skipped(self) -> None:
        def mock_rerank(
            query: str, docs: list[str], top_k: int
        ) -> list[tuple[int, float]]:
            return [(0, 0.9), (99, 0.8)]  # index 99 is out of range

        items = [_make_item("a", "content")]
        reranker = CohereReranker(rerank_fn=mock_rerank, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 1
        assert result[0].id == "a"

    def test_repr(self) -> None:
        reranker = CohereReranker(
            rerank_fn=lambda q, d, k: [], top_k=5
        )
        assert "CohereReranker" in repr(reranker)
        assert "top_k=5" in repr(reranker)

    def test_protocol_compliance(self) -> None:
        reranker = CohereReranker(
            rerank_fn=lambda q, d, k: [], top_k=10
        )
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# FlashRankReranker
# ---------------------------------------------------------------------------


class TestFlashRankReranker:
    """Tests for FlashRankReranker."""

    def test_import_error_on_rerank(self) -> None:
        reranker = FlashRankReranker(top_k=10)
        items = [_make_item("a", "content")]
        query = QueryBundle(query_str="test")

        with patch.dict("sys.modules", {"flashrank": None}), pytest.raises(
            RetrieverError, match="flashrank is required"
        ):
            reranker.rerank(query, items)

    def test_empty_items(self) -> None:
        reranker = FlashRankReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, [])

        assert result == []

    def test_repr(self) -> None:
        reranker = FlashRankReranker(model_name="my-model", top_k=5)
        assert "FlashRankReranker" in repr(reranker)
        assert "my-model" in repr(reranker)
        assert "top_k=5" in repr(reranker)

    def test_protocol_compliance(self) -> None:
        reranker = FlashRankReranker(top_k=10)
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# RoundRobinReranker
# ---------------------------------------------------------------------------


class TestRoundRobinReranker:
    """Tests for RoundRobinReranker."""

    def test_rerank_sorts_by_score(self) -> None:
        items = [
            _make_item("low", "content", score=0.1),
            _make_item("high", "content", score=0.9),
            _make_item("mid", "content", score=0.5),
        ]
        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert [item.id for item in result] == ["high", "mid", "low"]

    def test_rerank_truncates_to_top_k(self) -> None:
        items = [_make_item(f"item-{i}", "content", score=0.5) for i in range(10)]
        reranker = RoundRobinReranker(top_k=3)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 3

    def test_rerank_empty_items(self) -> None:
        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, [])

        assert result == []

    def test_rerank_multiple_interleaves(self) -> None:
        set_a = [_make_item("a1", "content a1"), _make_item("a2", "content a2")]
        set_b = [_make_item("b1", "content b1"), _make_item("b2", "content b2")]
        set_c = [_make_item("c1", "content c1")]

        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank_multiple(query, [set_a, set_b, set_c])

        # Round-robin: a1, b1, c1, a2, b2
        assert [item.id for item in result] == ["a1", "b1", "c1", "a2", "b2"]

    def test_rerank_multiple_deduplicates(self) -> None:
        shared = _make_item("shared", "shared content")
        set_a = [shared, _make_item("a1", "content a1")]
        set_b = [shared, _make_item("b1", "content b1")]

        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank_multiple(query, [set_a, set_b])

        ids = [item.id for item in result]
        assert ids.count("shared") == 1
        assert len(result) == 3  # shared, a1, b1

    def test_rerank_multiple_respects_top_k(self) -> None:
        set_a = [_make_item(f"a{i}", f"content a{i}") for i in range(5)]
        set_b = [_make_item(f"b{i}", f"content b{i}") for i in range(5)]

        reranker = RoundRobinReranker(top_k=4)
        query = QueryBundle(query_str="test")
        result = reranker.rerank_multiple(query, [set_a, set_b])

        assert len(result) == 4

    def test_rerank_multiple_empty_sets(self) -> None:
        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank_multiple(query, [])

        assert result == []

    def test_rerank_multiple_custom_top_k(self) -> None:
        set_a = [_make_item(f"a{i}", f"content a{i}") for i in range(5)]
        set_b = [_make_item(f"b{i}", f"content b{i}") for i in range(5)]

        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank_multiple(query, [set_a, set_b], top_k=2)

        assert len(result) == 2

    def test_repr(self) -> None:
        reranker = RoundRobinReranker(top_k=5)
        assert "RoundRobinReranker" in repr(reranker)
        assert "top_k=5" in repr(reranker)

    def test_protocol_compliance(self) -> None:
        reranker = RoundRobinReranker(top_k=10)
        assert isinstance(reranker, Reranker)


# ---------------------------------------------------------------------------
# RerankerPipeline
# ---------------------------------------------------------------------------


class TestRerankerPipeline:
    """Tests for RerankerPipeline."""

    def test_chains_rerankers(self) -> None:
        # First reranker: identity scoring
        reranker1 = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=100)
        # Second reranker: scores based on content length
        reranker2 = CrossEncoderReranker(
            score_fn=lambda q, d: min(1.0, len(d) / 50.0), top_k=100
        )

        pipeline = RerankerPipeline(rerankers=[reranker1, reranker2], top_k=10)
        items = [
            _make_item("short", "hi"),
            _make_item("long", "this is a much longer document with many words"),
        ]
        query = QueryBundle(query_str="test")
        result = pipeline.rerank(query, items)

        # The second reranker should rank the longer doc higher
        assert result[0].id == "long"

    def test_final_top_k_applied(self) -> None:
        reranker1 = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=100)
        pipeline = RerankerPipeline(rerankers=[reranker1], top_k=2)

        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        query = QueryBundle(query_str="test")
        result = pipeline.rerank(query, items)

        assert len(result) == 2

    def test_empty_items(self) -> None:
        reranker1 = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=10)
        pipeline = RerankerPipeline(rerankers=[reranker1], top_k=10)

        query = QueryBundle(query_str="test")
        result = pipeline.rerank(query, [])

        assert result == []

    def test_single_reranker(self) -> None:
        reranker1 = CrossEncoderReranker(
            score_fn=lambda q, d: 0.8 if "target" in d else 0.2, top_k=100
        )
        pipeline = RerankerPipeline(rerankers=[reranker1], top_k=10)

        items = [
            _make_item("miss", "some content"),
            _make_item("hit", "target document"),
        ]
        query = QueryBundle(query_str="test")
        result = pipeline.rerank(query, items)

        assert result[0].id == "hit"

    def test_requires_at_least_one_reranker(self) -> None:
        with pytest.raises(ValueError, match="At least one reranker"):
            RerankerPipeline(rerankers=[], top_k=10)

    def test_repr(self) -> None:
        reranker1 = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=10)
        pipeline = RerankerPipeline(rerankers=[reranker1], top_k=5)

        assert "RerankerPipeline" in repr(pipeline)
        assert "rerankers=1" in repr(pipeline)
        assert "top_k=5" in repr(pipeline)

    def test_protocol_compliance(self) -> None:
        reranker1 = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=10)
        pipeline = RerankerPipeline(rerankers=[reranker1], top_k=10)
        assert isinstance(pipeline, Reranker)

    def test_intermediate_stages_pass_all_items(self) -> None:
        """Intermediate stages should not truncate prematurely."""
        call_counts: dict[str, int] = {"first": 0, "second": 0}

        def first_scorer(query: str, doc: str) -> float:
            call_counts["first"] += 1
            return 0.5

        def second_scorer(query: str, doc: str) -> float:
            call_counts["second"] += 1
            return 0.8 if "special" in doc else 0.2

        reranker1 = CrossEncoderReranker(score_fn=first_scorer, top_k=100)
        reranker2 = CrossEncoderReranker(score_fn=second_scorer, top_k=100)
        pipeline = RerankerPipeline(rerankers=[reranker1, reranker2], top_k=5)

        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        items.append(_make_item("special-item", "special content"))
        query = QueryBundle(query_str="test")
        result = pipeline.rerank(query, items)

        # All 11 items should pass through first reranker
        assert call_counts["first"] == 11
        # All 11 items should pass through second reranker
        assert call_counts["second"] == 11
        # Final result capped at top_k=5
        assert len(result) == 5


# ---------------------------------------------------------------------------
# Pipeline integration (reranker_step)
# ---------------------------------------------------------------------------


class TestRerankerStep:
    """Tests for the reranker_step pipeline factory."""

    def test_creates_pipeline_step(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=10)
        step = reranker_step("my-reranker", reranker, top_k=5)

        assert step.name == "my-reranker"
        assert step.is_async is False

    def test_step_reranks_items(self) -> None:
        def scorer(query: str, doc: str) -> float:
            return 0.9 if "target" in doc else 0.1

        reranker = CrossEncoderReranker(score_fn=scorer, top_k=10)
        step = reranker_step("test-reranker", reranker, top_k=10)

        items = [
            _make_item("miss", "some content"),
            _make_item("hit", "target document"),
        ]
        query = QueryBundle(query_str="test")
        result = step.execute(items, query)

        assert result[0].id == "hit"

    def test_step_top_k(self) -> None:
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.5, top_k=100)
        step = reranker_step("test-reranker", reranker, top_k=2)

        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        query = QueryBundle(query_str="test")
        result = step.execute(items, query)

        assert len(result) == 2


# ---------------------------------------------------------------------------
# Items with no scores (default 0.0)
# ---------------------------------------------------------------------------


class TestRerankerWithUnscoredItems:
    """Test rerankers handle items with default score (0.0)."""

    def test_cross_encoder_handles_zero_score_items(self) -> None:
        items = [
            ContextItem(
                id="unscored",
                content="some content",
                source=SourceType.RETRIEVAL,
            ),
        ]
        reranker = CrossEncoderReranker(score_fn=lambda q, d: 0.7, top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        assert len(result) == 1
        assert result[0].score == 0.7

    def test_round_robin_handles_zero_score_items(self) -> None:
        items = [
            ContextItem(id="a", content="content a", source=SourceType.RETRIEVAL),
            ContextItem(id="b", content="content b", source=SourceType.RETRIEVAL),
        ]
        reranker = RoundRobinReranker(top_k=10)
        query = QueryBundle(query_str="test")
        result = reranker.rerank(query, items)

        # Both have score 0.0, so order should be stable
        assert len(result) == 2
