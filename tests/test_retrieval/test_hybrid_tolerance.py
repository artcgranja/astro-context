"""Tests for HybridRetriever failure tolerance.

Verifies that HybridRetriever handles sub-retriever failures gracefully:
- Partial failures still return results from working retrievers
- All retrievers failing raises RetrieverError
"""

from __future__ import annotations

import pytest

from astro_context.exceptions import RetrieverError
from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.retrieval.hybrid import HybridRetriever
from tests.conftest import FakeRetriever

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


class FailingRetriever:
    """A retriever that always raises an exception."""

    def __init__(self, error_msg: str = "retriever failed") -> None:
        self._error_msg = error_msg

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        msg = self._error_msg
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Partial failure: some retrievers work, some fail
# ---------------------------------------------------------------------------


class TestHybridRetrieverPartialFailure:
    """When some sub-retrievers fail, the hybrid retriever still returns results."""

    def test_one_failing_one_working(self) -> None:
        failing = FailingRetriever("boom")
        working = FakeRetriever([_make_item("a", "item A"), _make_item("b", "item B")])

        hybrid = HybridRetriever(retrievers=[failing, working])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        result_ids = {item.id for item in results}
        assert "a" in result_ids
        assert "b" in result_ids

    def test_two_failing_one_working(self) -> None:
        failing1 = FailingRetriever("error 1")
        failing2 = FailingRetriever("error 2")
        working = FakeRetriever([_make_item("c", "item C")])

        hybrid = HybridRetriever(retrievers=[failing1, failing2, working])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        assert len(results) == 1
        assert results[0].id == "c"

    def test_failing_retriever_before_working(self) -> None:
        """Failing retriever listed first does not block working retriever."""
        failing = FailingRetriever("first fails")
        working = FakeRetriever([_make_item("x", "item X")])

        hybrid = HybridRetriever(retrievers=[failing, working])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        assert len(results) == 1
        assert results[0].id == "x"

    def test_failing_retriever_after_working(self) -> None:
        """Failing retriever listed last does not affect results."""
        working = FakeRetriever([_make_item("y", "item Y")])
        failing = FailingRetriever("last fails")

        hybrid = HybridRetriever(retrievers=[working, failing])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        assert len(results) == 1
        assert results[0].id == "y"

    def test_results_have_rrf_metadata(self) -> None:
        """Even with one failed retriever, results have hybrid_rrf metadata."""
        failing = FailingRetriever("boom")
        working = FakeRetriever([_make_item("m", "item M")])

        hybrid = HybridRetriever(retrievers=[failing, working])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        assert results[0].metadata.get("retrieval_method") == "hybrid_rrf"

    def test_scores_normalized_with_partial_failure(self) -> None:
        """Scores are still normalized 0-1 with partial failures."""
        failing = FailingRetriever("boom")
        items = [_make_item(f"n{i}", f"item {i}") for i in range(5)]
        working = FakeRetriever(items)

        hybrid = HybridRetriever(retrievers=[failing, working])
        results = hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

        for item in results:
            assert 0.0 <= item.score <= 1.0


# ---------------------------------------------------------------------------
# Total failure: all retrievers fail
# ---------------------------------------------------------------------------


class TestHybridRetrieverTotalFailure:
    """When all sub-retrievers fail, RetrieverError is raised."""

    def test_all_retrievers_fail_raises_retriever_error(self) -> None:
        failing1 = FailingRetriever("error 1")
        failing2 = FailingRetriever("error 2")

        hybrid = HybridRetriever(retrievers=[failing1, failing2])
        with pytest.raises(RetrieverError, match="All sub-retrievers failed"):
            hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

    def test_single_failing_retriever_raises_retriever_error(self) -> None:
        failing = FailingRetriever("only one and it fails")

        hybrid = HybridRetriever(retrievers=[failing])
        with pytest.raises(RetrieverError, match="All sub-retrievers failed"):
            hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)

    def test_all_three_fail(self) -> None:
        retrievers = [FailingRetriever(f"error {i}") for i in range(3)]

        hybrid = HybridRetriever(retrievers=retrievers)
        with pytest.raises(RetrieverError, match="All sub-retrievers failed"):
            hybrid.retrieve(QueryBundle(query_str="test"), top_k=10)
