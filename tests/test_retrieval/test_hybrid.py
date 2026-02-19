"""Tests for astro_context.retrieval.hybrid."""

from __future__ import annotations

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.retrieval.hybrid import HybridRetriever


class FakeRetriever:
    """Fake retriever that returns pre-configured results for testing."""

    def __init__(self, results: list[ContextItem]) -> None:
        self._results = results

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._results[:top_k]


def _make_item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        priority=5,
        token_count=5,
    )


class TestHybridRetrieverCombinesResults:
    """HybridRetriever combines results from multiple retrievers."""

    def test_combines_two_retrievers(self) -> None:
        r1 = FakeRetriever([_make_item("a", "item A"), _make_item("b", "item B")])
        r2 = FakeRetriever([_make_item("c", "item C"), _make_item("a", "item A")])

        hybrid = HybridRetriever(retrievers=[r1, r2])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        result_ids = [item.id for item in results]
        assert "a" in result_ids
        assert "b" in result_ids
        assert "c" in result_ids

    def test_deduplicates_items(self) -> None:
        r1 = FakeRetriever([_make_item("a", "item A")])
        r2 = FakeRetriever([_make_item("a", "item A")])

        hybrid = HybridRetriever(retrievers=[r1, r2])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        result_ids = [item.id for item in results]
        assert result_ids.count("a") == 1


class TestHybridRetrieverRRFScoring:
    """RRF scoring gives higher scores to items appearing in multiple lists."""

    def test_items_in_multiple_lists_score_higher(self) -> None:
        # Item "a" appears in both retrievers, "b" only in first, "c" only in second
        r1 = FakeRetriever([_make_item("a", "item A"), _make_item("b", "item B")])
        r2 = FakeRetriever([_make_item("a", "item A"), _make_item("c", "item C")])

        hybrid = HybridRetriever(retrievers=[r1, r2])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        # Item "a" should be ranked first because it appears in both lists
        assert results[0].id == "a"
        assert results[0].metadata.get("retrieval_method") == "hybrid_rrf"

    def test_rrf_scores_are_normalized_zero_to_one(self) -> None:
        r1 = FakeRetriever([_make_item("a", "A"), _make_item("b", "B")])
        r2 = FakeRetriever([_make_item("c", "C"), _make_item("a", "A")])

        hybrid = HybridRetriever(retrievers=[r1, r2])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)
        for item in results:
            assert 0.0 <= item.score <= 1.0


class TestHybridRetrieverWeighted:
    """Weighted RRF biases toward the weighted retriever."""

    def test_weighted_rrf(self) -> None:
        # r1 has item "a" ranked first, r2 has item "b" ranked first
        r1 = FakeRetriever([_make_item("a", "item A"), _make_item("b", "item B")])
        r2 = FakeRetriever([_make_item("b", "item B"), _make_item("a", "item A")])

        # Heavily weight r2 so "b" (r2's top item) should win
        hybrid = HybridRetriever(retrievers=[r1, r2], weights=[0.1, 10.0])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)
        assert results[0].id == "b"

    def test_equal_weights_same_as_unweighted(self) -> None:
        r1 = FakeRetriever([_make_item("a", "A"), _make_item("b", "B")])
        r2 = FakeRetriever([_make_item("a", "A"), _make_item("c", "C")])

        hybrid_weighted = HybridRetriever(retrievers=[r1, r2], weights=[1.0, 1.0])
        hybrid_unweighted = HybridRetriever(retrievers=[r1, r2])

        results_w = hybrid_weighted.retrieve(QueryBundle(query_str="test"), top_k=10)
        results_u = hybrid_unweighted.retrieve(QueryBundle(query_str="test"), top_k=10)

        assert [r.id for r in results_w] == [r.id for r in results_u]


class TestHybridRetrieverValidation:
    """HybridRetriever validation."""

    def test_empty_retrievers_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="At least one retriever"):
            HybridRetriever(retrievers=[])

    def test_mismatched_weights_raises_value_error(self) -> None:
        r1 = FakeRetriever([])
        with pytest.raises(ValueError, match="weights must have same length"):
            HybridRetriever(retrievers=[r1], weights=[1.0, 2.0])

    def test_single_retriever_works(self) -> None:
        r1 = FakeRetriever([_make_item("a", "A")])
        hybrid = HybridRetriever(retrievers=[r1])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)
        assert len(results) == 1
        assert results[0].id == "a"

    def test_top_k_limits_results(self) -> None:
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]
        r1 = FakeRetriever(items)
        hybrid = HybridRetriever(retrievers=[r1])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=3)
        assert len(results) == 3

    def test_empty_results_from_all_retrievers(self) -> None:
        r1 = FakeRetriever([])
        r2 = FakeRetriever([])
        hybrid = HybridRetriever(retrievers=[r1, r2])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)
        assert results == []
