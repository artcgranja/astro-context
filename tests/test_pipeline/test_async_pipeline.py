"""Tests for async pipeline support (abuild, async steps, mixed sync/async)."""

from __future__ import annotations

import pytest

from astro_context.models.context import ContextItem, ContextResult, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.step import (
    PipelineStep,
    async_postprocessor_step,
    async_retriever_step,
    filter_step,
    retriever_step,
)
from tests.conftest import FakeRetriever
from tests.test_pipeline.conftest import make_items, make_pipeline


class FakeAsyncRetriever:
    """Async retriever for testing."""

    def __init__(self, items: list[ContextItem]) -> None:
        self._items = items

    async def aretrieve(self, query: QueryBundle, top_k: int = 10) -> list[ContextItem]:
        return self._items[:top_k]


class FakeAsyncPostProcessor:
    """Async post-processor for testing."""

    def __init__(self, min_score: float = 0.5) -> None:
        self._min_score = min_score

    async def aprocess(
        self, items: list[ContextItem], query: QueryBundle | None = None
    ) -> list[ContextItem]:
        return [item for item in items if item.score >= self._min_score]


class TestAbuildWithSyncSteps:
    """abuild() works with sync-only steps."""

    @pytest.mark.asyncio
    async def test_abuild_empty_pipeline(self) -> None:
        pipeline = make_pipeline()
        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert isinstance(result, ContextResult)
        assert len(result.window.items) == 0

    @pytest.mark.asyncio
    async def test_abuild_with_system_prompt(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("You are helpful.")
        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert len(result.window.items) == 1
        assert result.window.items[0].source == SourceType.SYSTEM

    @pytest.mark.asyncio
    async def test_abuild_with_sync_retriever(self) -> None:
        items = make_items()
        retriever = FakeRetriever(items)
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert len(result.window.items) == 3

    @pytest.mark.asyncio
    async def test_abuild_diagnostics_match_build(self) -> None:
        items = make_items()
        retriever = FakeRetriever(items)
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert "steps" in result.diagnostics
        assert result.diagnostics["steps"][0]["name"] == "search"
        assert "total_items_considered" in result.diagnostics


class TestAbuildWithAsyncSteps:
    """abuild() with async pipeline steps."""

    @pytest.mark.asyncio
    async def test_async_retriever_step(self) -> None:
        items = make_items()
        retriever = FakeAsyncRetriever(items)
        pipeline = make_pipeline()
        pipeline.add_step(async_retriever_step("async-search", retriever, top_k=2))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert len(result.window.items) == 2

    @pytest.mark.asyncio
    async def test_async_postprocessor_step(self) -> None:
        items = make_items()
        retriever = FakeRetriever(items)
        processor = FakeAsyncPostProcessor(min_score=0.6)
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever))
        pipeline.add_step(async_postprocessor_step("filter", processor))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert all(item.score >= 0.6 for item in result.window.items)


class TestMixedSyncAsyncPipeline:
    """Pipeline with both sync and async steps."""

    @pytest.mark.asyncio
    async def test_mixed_steps_in_abuild(self) -> None:
        """Sync retriever -> async postprocessor -> sync filter."""
        items = make_items(5)
        retriever = FakeRetriever(items)
        processor = FakeAsyncPostProcessor(min_score=0.5)

        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("sync-search", retriever))
        pipeline.add_step(async_postprocessor_step("async-filter", processor))
        pipeline.add_step(filter_step("score-gate", lambda item: item.score >= 0.6))

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        assert all(item.score >= 0.6 for item in result.window.items)
        assert len(result.diagnostics["steps"]) == 3


class TestAsyncStepSyncError:
    """Async steps raise TypeError when called synchronously."""

    def test_async_step_raises_in_sync_build(self) -> None:
        items = make_items()
        retriever = FakeAsyncRetriever(items)
        pipeline = make_pipeline()
        pipeline.add_step(async_retriever_step("async-search", retriever))

        with pytest.raises(TypeError, match=r"async.*cannot be called synchronously"):
            pipeline.build(QueryBundle(query_str="test"))


class TestPipelineStepAexecute:
    """aexecute() on individual pipeline steps."""

    @pytest.mark.asyncio
    async def test_sync_step_via_aexecute(self) -> None:
        """Sync functions should work through aexecute."""

        def noop(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        step = PipelineStep(name="noop", fn=noop)
        items = make_items()
        result = await step.aexecute(items, QueryBundle(query_str="test"))
        assert result == items

    @pytest.mark.asyncio
    async def test_async_step_via_aexecute(self) -> None:
        """Async functions should be awaited by aexecute."""

        async def async_noop(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        step = PipelineStep(name="async-noop", fn=async_noop, is_async=True)
        items = make_items()
        result = await step.aexecute(items, QueryBundle(query_str="test"))
        assert result == items
