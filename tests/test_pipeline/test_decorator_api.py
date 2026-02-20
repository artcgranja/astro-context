"""Tests for the decorator-based pipeline step registration API."""

from __future__ import annotations

import pytest

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.query import QueryBundle
from tests.conftest import FakeTokenizer
from tests.test_pipeline.conftest import make_pipeline


class TestStepDecorator:
    """@pipeline.step decorator for sync functions."""

    def test_bare_decorator(self) -> None:
        """@pipeline.step without arguments."""
        pipeline = make_pipeline()

        @pipeline.step
        def my_filter(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return [i for i in items if i.score > 0.5]

        # Step should be registered
        assert len(pipeline._steps) == 1
        assert pipeline._steps[0].name == "my_filter"
        assert pipeline._steps[0].is_async is False

    def test_decorator_with_name(self) -> None:
        """@pipeline.step(name="custom-name")."""
        pipeline = make_pipeline()

        @pipeline.step(name="custom-name")
        def my_filter(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        assert pipeline._steps[0].name == "custom-name"

    def test_decorator_returns_original_function(self) -> None:
        """The decorator should return the original function unchanged."""
        pipeline = make_pipeline()

        @pipeline.step
        def my_fn(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        # Should be callable as the original function
        result = my_fn([], QueryBundle(query_str="test"))
        assert result == []

    def test_decorator_step_executes_in_pipeline(self) -> None:
        """Step registered via decorator should execute in build()."""
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System prompt")

        @pipeline.step
        def add_item(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            tokenizer = FakeTokenizer()
            new_item = ContextItem(
                id="added",
                content="Added by decorator step",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=tokenizer.count_tokens("Added by decorator step"),
            )
            return [*items, new_item]

        result = pipeline.build(QueryBundle(query_str="test"))
        contents = [item.content for item in result.window.items]
        assert "Added by decorator step" in contents

    def test_multiple_decorated_steps(self) -> None:
        """Multiple @pipeline.step decorators register steps in order."""
        pipeline = make_pipeline()

        @pipeline.step
        def step_one(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        @pipeline.step
        def step_two(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        @pipeline.step(name="step-three")
        def step_three(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        assert len(pipeline._steps) == 3
        assert pipeline._steps[0].name == "step_one"
        assert pipeline._steps[1].name == "step_two"
        assert pipeline._steps[2].name == "step-three"


class TestAsyncStepDecorator:
    """@pipeline.async_step decorator for async functions."""

    def test_bare_async_decorator(self) -> None:
        pipeline = make_pipeline()

        @pipeline.async_step
        async def my_async_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        assert len(pipeline._steps) == 1
        assert pipeline._steps[0].name == "my_async_step"
        assert pipeline._steps[0].is_async is True

    def test_async_decorator_with_name(self) -> None:
        pipeline = make_pipeline()

        @pipeline.async_step(name="custom-async")
        async def my_async_step(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        assert pipeline._steps[0].name == "custom-async"
        assert pipeline._steps[0].is_async is True

    def test_async_decorator_rejects_sync_function(self) -> None:
        """@pipeline.async_step should raise TypeError for sync functions."""
        pipeline = make_pipeline()

        with pytest.raises(TypeError, match="async function"):

            @pipeline.async_step
            def not_async(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:  # type: ignore[type-var]
                return items

    @pytest.mark.asyncio
    async def test_async_step_executes_in_abuild(self) -> None:
        """Async step registered via decorator should execute in abuild()."""
        pipeline = make_pipeline()

        @pipeline.async_step
        async def async_add(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            tokenizer = FakeTokenizer()
            new_item = ContextItem(
                id="async-added",
                content="Added by async decorator step",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=tokenizer.count_tokens("Added by async decorator step"),
            )
            return [*items, new_item]

        result = await pipeline.abuild(QueryBundle(query_str="test"))
        contents = [item.content for item in result.window.items]
        assert "Added by async decorator step" in contents


class TestMixedDecoratorAndExplicitSteps:
    """Mix of decorator-registered and explicitly added steps."""

    def test_decorator_and_add_step_order(self) -> None:
        """Steps are registered in the order they appear (decorator or add_step)."""
        pipeline = make_pipeline()

        @pipeline.step
        def first(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        from astro_context.pipeline.step import PipelineStep

        pipeline.add_step(PipelineStep(name="second", fn=lambda items, q: items))

        @pipeline.step(name="third")
        def third(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
            return items

        assert [s.name for s in pipeline._steps] == ["first", "second", "third"]
