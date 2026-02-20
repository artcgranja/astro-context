"""Tests for astro_context.pipeline.pipeline."""

from __future__ import annotations

from astro_context.formatters.anthropic import AnthropicFormatter
from astro_context.formatters.generic import GenericTextFormatter
from astro_context.models.context import ContextItem, ContextResult, SourceType
from astro_context.models.query import QueryBundle
from astro_context.pipeline.pipeline import ContextPipeline
from astro_context.pipeline.step import PipelineStep, retriever_step
from tests.conftest import FakeRetriever, FakeTokenizer, make_memory_manager
from tests.test_pipeline.conftest import make_pipeline


class TestPipelineEmpty:
    """Empty pipeline with just system prompt."""

    def test_empty_pipeline_returns_context_result(self) -> None:
        pipeline = make_pipeline()
        result = pipeline.build(QueryBundle(query_str="test"))
        assert isinstance(result, ContextResult)
        assert len(result.window.items) == 0

    def test_pipeline_with_system_prompt_only(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("You are a helpful assistant.")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.window.items) == 1
        assert result.window.items[0].source == SourceType.SYSTEM
        assert result.window.items[0].content == "You are a helpful assistant."

    def test_system_prompt_has_high_priority(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System prompt", priority=10)
        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.window.items[0].priority == 10

    def test_system_prompt_has_token_count(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("You are a helpful assistant.")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.window.items[0].token_count > 0


class TestPipelineWithRetriever:
    """Pipeline with retriever step."""

    def test_retriever_items_included(self) -> None:
        tokenizer = FakeTokenizer()
        retrieval_items = [
            ContextItem(
                id="r1",
                content="Retrieved document about Python.",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=5,
                token_count=tokenizer.count_tokens("Retrieved document about Python."),
            ),
        ]
        retriever = FakeRetriever(retrieval_items)
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever, top_k=5))

        result = pipeline.build(QueryBundle(query_str="Python"))
        assert len(result.window.items) == 1
        assert result.window.items[0].content == "Retrieved document about Python."

    def test_diagnostics_include_step_info(self) -> None:
        retriever = FakeRetriever([])
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert "steps" in result.diagnostics
        assert len(result.diagnostics["steps"]) == 1
        assert result.diagnostics["steps"][0]["name"] == "search"


class TestPipelineWithMemory:
    """Pipeline with memory manager."""

    def test_memory_items_included(self) -> None:
        memory = make_memory_manager()
        memory.add_user_message("Hello")
        memory.add_assistant_message("Hi there!")

        pipeline = make_pipeline()
        pipeline.with_memory(memory)

        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.window.items) == 2
        assert result.diagnostics.get("memory_items") == 2

    def test_memory_items_have_memory_source(self) -> None:
        memory = make_memory_manager()
        memory.add_user_message("Hello")

        pipeline = make_pipeline()
        pipeline.with_memory(memory)

        result = pipeline.build(QueryBundle(query_str="test"))
        assert all(item.source == SourceType.MEMORY for item in result.window.items)


class TestPipelineWithFormatter:
    """Pipeline with formatter."""

    def test_generic_formatter_default(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("System message.")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.format_type == "generic"
        assert isinstance(result.formatted_output, str)

    def test_anthropic_formatter(self) -> None:
        pipeline = make_pipeline()
        pipeline.with_formatter(AnthropicFormatter())
        pipeline.add_system_prompt("System message.")

        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.format_type == "anthropic"
        assert isinstance(result.formatted_output, dict)
        assert "system" in result.formatted_output
        assert "messages" in result.formatted_output


class TestPipelineBuild:
    """build() returns ContextResult with diagnostics."""

    def test_build_returns_context_result(self) -> None:
        pipeline = make_pipeline()
        result = pipeline.build(QueryBundle(query_str="test"))
        assert isinstance(result, ContextResult)

    def test_build_time_is_positive(self) -> None:
        pipeline = make_pipeline()
        pipeline.add_system_prompt("Hello")
        result = pipeline.build(QueryBundle(query_str="test"))
        assert result.build_time_ms >= 0.0

    def test_diagnostics_contain_expected_keys(self) -> None:
        tokenizer = FakeTokenizer()
        retrieval_items = [
            ContextItem(
                id="r1",
                content="doc1",
                source=SourceType.RETRIEVAL,
                score=0.9,
                priority=3,
                token_count=tokenizer.count_tokens("doc1"),
            ),
        ]
        memory = make_memory_manager()
        memory.add_user_message("Hello")

        pipeline = (
            make_pipeline()
            .add_system_prompt("System")
            .with_memory(memory)
            .add_step(retriever_step("search", FakeRetriever(retrieval_items)))
        )
        result = pipeline.build(QueryBundle(query_str="test"))

        diag = result.diagnostics
        assert "steps" in diag
        assert "memory_items" in diag
        assert "total_items_considered" in diag
        assert "items_included" in diag
        assert "items_overflow" in diag
        assert "token_utilization" in diag

    def test_overflow_items_tracked(self) -> None:
        tokenizer = FakeTokenizer()
        # Create items that exceed the max_tokens budget
        big_content = "word " * 500  # 500 tokens with FakeTokenizer
        items = [
            ContextItem(
                id=f"big-{i}",
                content=big_content,
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=5,
                token_count=tokenizer.count_tokens(big_content),
            )
            for i in range(5)
        ]
        retriever = FakeRetriever(items)
        pipeline = make_pipeline(max_tokens=200)
        pipeline.add_step(retriever_step("search", retriever, top_k=5))

        result = pipeline.build(QueryBundle(query_str="test"))
        assert len(result.overflow_items) > 0

    def test_items_without_token_count_get_counted(self) -> None:
        """Items with token_count=0 get their tokens counted by the pipeline."""
        items = [
            ContextItem(
                id="no-count",
                content="This item has no token count set",
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=5,
                token_count=0,  # Will be counted by pipeline
            ),
        ]
        retriever = FakeRetriever(items)
        pipeline = make_pipeline()
        pipeline.add_step(retriever_step("search", retriever))

        result = pipeline.build(QueryBundle(query_str="test"))
        # The item should have a real token count after pipeline processing
        assert result.window.items[0].token_count > 0


class TestPipelinePriorityOrdering:
    """Priority ordering: system > memory > retrieval."""

    def test_system_before_memory_before_retrieval(self) -> None:
        tokenizer = FakeTokenizer()
        retrieval_items = [
            ContextItem(
                id="ret",
                content="retrieved",
                source=SourceType.RETRIEVAL,
                score=0.5,
                priority=3,
                token_count=tokenizer.count_tokens("retrieved"),
            ),
        ]

        memory = make_memory_manager()
        memory.add_user_message("memory msg")

        pipeline = (
            make_pipeline()
            .add_system_prompt("system prompt", priority=10)
            .with_memory(memory)
            .add_step(retriever_step("search", FakeRetriever(retrieval_items)))
        )

        result = pipeline.build(QueryBundle(query_str="test"))
        items = result.window.items
        # System (priority 10) should come before memory (priority 7) which should come
        # before retrieval (priority 3)
        assert items[0].source == SourceType.SYSTEM
        assert items[1].source == SourceType.MEMORY
        assert items[2].source == SourceType.RETRIEVAL


class TestPipelineMethodChaining:
    """Method chaining returns self."""

    def test_add_step_returns_self(self) -> None:
        pipeline = make_pipeline()
        step = PipelineStep(name="noop", fn=lambda items, q: items)
        result = pipeline.add_step(step)
        assert result is pipeline

    def test_with_memory_returns_self(self) -> None:
        pipeline = make_pipeline()
        result = pipeline.with_memory(make_memory_manager())
        assert result is pipeline

    def test_with_formatter_returns_self(self) -> None:
        pipeline = make_pipeline()
        result = pipeline.with_formatter(GenericTextFormatter())
        assert result is pipeline

    def test_add_system_prompt_returns_self(self) -> None:
        pipeline = make_pipeline()
        result = pipeline.add_system_prompt("Hello")
        assert result is pipeline

    def test_full_chain(self) -> None:
        pipeline = (
            make_pipeline()
            .add_system_prompt("System")
            .with_memory(make_memory_manager())
            .with_formatter(GenericTextFormatter())
            .add_step(PipelineStep(name="noop", fn=lambda items, q: items))
        )
        assert isinstance(pipeline, ContextPipeline)
