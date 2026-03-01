"""Tests for pipeline step factory functions."""

from __future__ import annotations

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.step import query_transform_step
from tests.conftest import FakeRetriever


def _make_item(item_id: str, content: str, score: float = 0.5) -> ContextItem:
    return ContextItem(
        id=item_id,
        content=content,
        source=SourceType.RETRIEVAL,
        score=score,
        priority=5,
        token_count=5,
    )


class FakeTransformer:
    """Fake QueryTransformer that returns pre-configured query variants."""

    def __init__(self, queries: list[QueryBundle]) -> None:
        self._queries = queries

    def transform(self, query: QueryBundle) -> list[QueryBundle]:
        return self._queries


class TestQueryTransformStepUsesRRF:
    """query_transform_step should fuse results from multiple queries via RRF."""

    def test_rrf_fusion_deduplicates(self) -> None:
        """Items appearing in multiple query results are deduplicated."""
        q1 = QueryBundle(query_str="variant 1")
        q2 = QueryBundle(query_str="variant 2")

        # Both queries return item "a"; q1 also returns "b", q2 returns "c"
        retriever_results: dict[str, list[ContextItem]] = {
            "variant 1": [_make_item("a", "A"), _make_item("b", "B")],
            "variant 2": [_make_item("a", "A"), _make_item("c", "C")],
        }

        class QueryAwareRetriever:
            def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                return retriever_results.get(query.query_str, [])[:top_k]

        transformer = FakeTransformer([q1, q2])
        step = query_transform_step("test-rrf", transformer, QueryAwareRetriever(), top_k=10)

        result = step.execute([], QueryBundle(query_str="original"))
        result_ids = [r.id for r in result]

        # "a" should appear only once
        assert result_ids.count("a") == 1
        # All unique items present
        assert set(result_ids) == {"a", "b", "c"}

    def test_rrf_fusion_ranks_overlapping_higher(self) -> None:
        """Items in multiple query results should rank higher via RRF."""
        q1 = QueryBundle(query_str="variant 1")
        q2 = QueryBundle(query_str="variant 2")

        retriever_results: dict[str, list[ContextItem]] = {
            "variant 1": [_make_item("a", "A"), _make_item("b", "B")],
            "variant 2": [_make_item("a", "A"), _make_item("c", "C")],
        }

        class QueryAwareRetriever:
            def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
                return retriever_results.get(query.query_str, [])[:top_k]

        transformer = FakeTransformer([q1, q2])
        step = query_transform_step("test-rrf", transformer, QueryAwareRetriever(), top_k=10)

        result = step.execute([], QueryBundle(query_str="original"))
        # "a" appears in both lists, should be ranked first
        assert result[0].id == "a"

    def test_rrf_metadata_present(self) -> None:
        """Fused items should have RRF metadata."""
        q1 = QueryBundle(query_str="variant 1")
        items = [_make_item("a", "A")]

        transformer = FakeTransformer([q1])
        retriever = FakeRetriever(items)
        step = query_transform_step("test-rrf", transformer, retriever, top_k=10)

        result = step.execute([], QueryBundle(query_str="original"))
        assert len(result) == 1
        assert result[0].metadata.get("retrieval_method") == "rrf"
        assert "rrf_raw_score" in result[0].metadata

    def test_existing_items_not_duplicated(self) -> None:
        """Items already in the pipeline should not be duplicated."""
        q1 = QueryBundle(query_str="variant 1")
        existing = [_make_item("a", "A")]
        new_items = [_make_item("a", "A"), _make_item("b", "B")]

        transformer = FakeTransformer([q1])
        retriever = FakeRetriever(new_items)
        step = query_transform_step("test-rrf", transformer, retriever, top_k=10)

        result = step.execute(existing, QueryBundle(query_str="original"))
        result_ids = [r.id for r in result]
        assert result_ids.count("a") == 1
        assert "b" in result_ids

    def test_top_k_limits_fused_results(self) -> None:
        """top_k should limit the number of fused results."""
        q1 = QueryBundle(query_str="variant 1")
        items = [_make_item(f"item-{i}", f"content {i}") for i in range(10)]

        transformer = FakeTransformer([q1])
        retriever = FakeRetriever(items)
        step = query_transform_step("test-rrf", transformer, retriever, top_k=3)

        result = step.execute([], QueryBundle(query_str="original"))
        assert len(result) == 3
