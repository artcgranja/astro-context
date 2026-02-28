"""Tests for query transformation pipeline."""

from __future__ import annotations

import pytest

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle
from astro_context.pipeline.step import query_transform_step
from astro_context.query.pipeline import QueryTransformPipeline
from astro_context.query.transformers import (
    DecompositionTransformer,
    HyDETransformer,
    MultiQueryTransformer,
    StepBackTransformer,
)


class TestQueryTransformPipeline:
    """Tests for the QueryTransformPipeline class."""

    def test_single_transformer(self) -> None:
        transformer = StepBackTransformer(generate_fn=lambda q: f"broader: {q}")
        pipeline = QueryTransformPipeline(transformers=[transformer])
        query = QueryBundle(query_str="specific question")
        result = pipeline.transform(query)
        assert len(result) == 2
        assert result[0].query_str == "specific question"
        assert result[1].query_str == "broader: specific question"

    def test_chained_transformers(self) -> None:
        step_back = StepBackTransformer(generate_fn=lambda q: f"abstract: {q}")
        multi = MultiQueryTransformer(
            generate_fn=lambda q, n: [f"var-{i}: {q}" for i in range(n)],
            num_queries=1,
        )
        pipeline = QueryTransformPipeline(transformers=[step_back, multi])
        query = QueryBundle(query_str="original")
        result = pipeline.transform(query)
        # step_back: [original, abstract: original]
        # multi on each: [original, var-0: original] + [abstract: original, var-0: abstract: original]
        assert len(result) == 4

    def test_deduplication(self) -> None:
        """Ensure duplicate query strings are removed."""
        # A transformer that always returns the same query string
        multi = MultiQueryTransformer(
            generate_fn=lambda q, n: [q for _ in range(n)],
            num_queries=3,
        )
        pipeline = QueryTransformPipeline(transformers=[multi])
        query = QueryBundle(query_str="test")
        result = pipeline.transform(query)
        # Original "test" + 3 duplicates of "test" -> deduplicated to 1
        assert len(result) == 1
        assert result[0].query_str == "test"

    def test_empty_transformers_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one transformer"):
            QueryTransformPipeline(transformers=[])

    def test_deduplication_preserves_order(self) -> None:
        multi = MultiQueryTransformer(
            generate_fn=lambda q, n: ["alpha", "beta", "alpha"],
            num_queries=3,
        )
        pipeline = QueryTransformPipeline(transformers=[multi])
        query = QueryBundle(query_str="original")
        result = pipeline.transform(query)
        query_strs = [r.query_str for r in result]
        assert query_strs == ["original", "alpha", "beta"]

    def test_repr(self) -> None:
        t1 = HyDETransformer(generate_fn=lambda q: q)
        t2 = StepBackTransformer(generate_fn=lambda q: q)
        pipeline = QueryTransformPipeline(transformers=[t1, t2])
        r = repr(pipeline)
        assert "HyDETransformer" in r
        assert "StepBackTransformer" in r
        assert "QueryTransformPipeline" in r

    async def test_atransform_sync_fallback(self) -> None:
        """Async pipeline falls back to sync transform for sync transformers."""
        transformer = StepBackTransformer(generate_fn=lambda q: f"broader: {q}")
        pipeline = QueryTransformPipeline(transformers=[transformer])
        query = QueryBundle(query_str="specific")
        result = await pipeline.atransform(query)
        assert len(result) == 2
        assert result[0].query_str == "specific"
        assert result[1].query_str == "broader: specific"

    def test_complex_chain(self) -> None:
        """Test a realistic multi-step transformation chain."""
        decompose = DecompositionTransformer(
            generate_fn=lambda q: [f"sub1 of {q}", f"sub2 of {q}"],
        )
        hyde = HyDETransformer(generate_fn=lambda q: f"hypothetical: {q}")
        pipeline = QueryTransformPipeline(transformers=[decompose, hyde])
        query = QueryBundle(query_str="complex question")
        result = pipeline.transform(query)
        # decompose: [sub1, sub2]
        # hyde on each: [hypothetical: sub1, hypothetical: sub2]
        assert len(result) == 2
        assert all(r.query_str.startswith("hypothetical:") for r in result)


class TestQueryTransformStep:
    """Tests for the query_transform_step pipeline integration."""

    def _make_retriever(self, items: list[ContextItem]) -> _FakeRetriever:
        return _FakeRetriever(items)

    def test_basic_transform_and_retrieve(self) -> None:
        items = [
            ContextItem(id="1", content="doc 1", source="retrieval"),
            ContextItem(id="2", content="doc 2", source="retrieval"),
        ]
        retriever = self._make_retriever(items)
        transformer = StepBackTransformer(generate_fn=lambda q: f"broader: {q}")
        step = query_transform_step("test", transformer, retriever)
        result = step.execute([], QueryBundle(query_str="specific"))
        # Both queries retrieve the same 2 items, but dedup by ID -> 2 unique
        assert len(result) == 2

    def test_deduplicates_by_item_id(self) -> None:
        items = [ContextItem(id="shared", content="doc", source="retrieval")]
        retriever = self._make_retriever(items)
        multi = MultiQueryTransformer(
            generate_fn=lambda q, n: [f"v{i}" for i in range(n)],
            num_queries=3,
        )
        step = query_transform_step("test", multi, retriever)
        result = step.execute([], QueryBundle(query_str="test"))
        # All 4 queries return the same item with id="shared"
        assert len(result) == 1
        assert result[0].id == "shared"

    def test_preserves_existing_items(self) -> None:
        existing = [ContextItem(id="existing", content="already here", source="system")]
        new_items = [ContextItem(id="new", content="new doc", source="retrieval")]
        retriever = self._make_retriever(new_items)
        transformer = HyDETransformer(generate_fn=lambda q: f"hyp: {q}")
        step = query_transform_step("test", transformer, retriever)
        result = step.execute(existing, QueryBundle(query_str="test"))
        assert len(result) == 2
        assert result[0].id == "existing"
        assert result[1].id == "new"

    def test_no_duplicates_with_existing(self) -> None:
        """Items already in the list are not added again."""
        existing = [ContextItem(id="dup", content="existing", source="system")]
        retriever = self._make_retriever(
            [ContextItem(id="dup", content="from retrieval", source="retrieval")]
        )
        transformer = HyDETransformer(generate_fn=lambda q: q)
        step = query_transform_step("test", transformer, retriever)
        result = step.execute(existing, QueryBundle(query_str="test"))
        assert len(result) == 1
        assert result[0].id == "dup"

    def test_top_k_respected(self) -> None:
        items = [
            ContextItem(id=f"item-{i}", content=f"doc {i}", source="retrieval")
            for i in range(10)
        ]
        retriever = self._make_retriever(items)
        transformer = HyDETransformer(generate_fn=lambda q: q)
        step = query_transform_step("test", transformer, retriever, top_k=3)
        result = step.execute([], QueryBundle(query_str="test"))
        assert len(result) == 3

    def test_step_name(self) -> None:
        retriever = self._make_retriever([])
        transformer = HyDETransformer(generate_fn=lambda q: q)
        step = query_transform_step("my-step", transformer, retriever)
        assert step.name == "my-step"


class _FakeRetriever:
    """Minimal retriever for testing query_transform_step."""

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items

    def retrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._items[:top_k]
