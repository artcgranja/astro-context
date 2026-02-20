"""ContextPipeline -- the main orchestrator for astro-context."""

from __future__ import annotations

import inspect
import time
from collections.abc import Callable
from typing import Any, cast, overload

from astro_context.exceptions import AstroContextError, FormatterError
from astro_context.formatters.base import Formatter
from astro_context.formatters.generic import GenericTextFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.models.context import (
    ContextItem,
    ContextResult,
    ContextWindow,
    PipelineDiagnostics,
    SourceType,
)
from astro_context.models.query import QueryBundle
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

from .step import AsyncStepFn, PipelineStep, SyncStepFn


class ContextPipeline:
    """The main orchestrator that assembles context from multiple sources.

    Usage::

        pipeline = (
            ContextPipeline(max_tokens=8192)
            .add_step(retriever_step("search", my_retriever))
            .with_memory(my_memory_manager)
            .with_formatter(AnthropicFormatter())
            .add_system_prompt("You are a helpful assistant.")
        )
        result = pipeline.build(QueryBundle(query_str="What is context engineering?"))

    For async pipelines::

        result = await pipeline.abuild(QueryBundle(query_str="..."))

    The pipeline follows this flow:
        1. Collect system items (system prompts, instructions)
        2. Collect memory items (conversation history)
        3. Execute pipeline steps (retrieval, post-processing)
        4. Assemble into ContextWindow (priority-ranked, token-aware)
        5. Format for target LLM provider
        6. Return ContextResult with diagnostics
    """

    def __init__(
        self,
        max_tokens: int = 8192,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer or get_default_counter()
        self._steps: list[PipelineStep] = []
        self._memory: MemoryManager | None = None
        self._formatter: Formatter = GenericTextFormatter()
        self._system_items: list[ContextItem] = []

    # -- Read-only properties --

    @property
    def max_tokens(self) -> int:
        """The maximum token budget for the context window."""
        return self._max_tokens

    @property
    def formatter(self) -> Formatter:
        """The current output formatter."""
        return self._formatter

    @property
    def steps(self) -> list[PipelineStep]:
        """A copy of the registered pipeline steps."""
        return list(self._steps)

    @property
    def system_items(self) -> list[ContextItem]:
        """A copy of the registered system items."""
        return list(self._system_items)

    def __repr__(self) -> str:
        return (
            f"ContextPipeline("
            f"max_tokens={self._max_tokens}, "
            f"steps={len(self._steps)}, "
            f"formatter='{self._formatter.format_type}')"
        )

    def add_step(self, step: PipelineStep) -> ContextPipeline:
        """Add a pipeline step. Returns self for chaining."""
        self._steps.append(step)
        return self

    def with_memory(self, memory: MemoryManager) -> ContextPipeline:
        """Attach a memory manager. Returns self for chaining.

        .. note::

            A ``MemoryProvider`` protocol is being added to
            ``astro_context.protocols``; once available the *memory*
            parameter will accept any ``MemoryProvider`` implementation.
        """
        # TODO(protocols): accept MemoryProvider protocol in addition to MemoryManager
        self._memory = memory
        return self

    def with_formatter(self, formatter: Formatter) -> ContextPipeline:
        """Set the output formatter. Returns self for chaining."""
        self._formatter = formatter
        return self

    def add_system_prompt(self, content: str, priority: int = 10) -> ContextPipeline:
        """Add a system prompt as a high-priority context item."""
        token_count = self._tokenizer.count_tokens(content)
        item = ContextItem(
            content=content,
            source=SourceType.SYSTEM,
            score=1.0,
            priority=priority,
            token_count=token_count,
        )
        self._system_items.append(item)
        return self

    # -- Decorator-based step registration (inspired by @agent.tool from Pydantic AI) --

    @overload
    def step(self, fn: SyncStepFn, /) -> SyncStepFn: ...

    @overload
    def step(self, /, *, name: str | None = ...) -> Callable[[SyncStepFn], SyncStepFn]: ...

    def step(
        self,
        fn: SyncStepFn | None = None,
        /,
        *,
        name: str | None = None,
    ) -> SyncStepFn | Callable[[SyncStepFn], SyncStepFn]:
        """Decorator to register a function as a pipeline step.

        Can be used with or without arguments::

            pipeline = ContextPipeline()

            @pipeline.step
            def my_filter(items, query):
                return [i for i in items if i.score > 0.5]

            @pipeline.step(name="custom-name")
            def another_step(items, query):
                return items
        """

        def _register(func: SyncStepFn) -> SyncStepFn:
            if inspect.iscoroutinefunction(func):
                msg = f"'{func.__name__}' is async -- use @pipeline.async_step instead"
                raise TypeError(msg)
            step_name: str = name or str(getattr(func, "__name__", "unnamed_step"))
            pipeline_step = PipelineStep(name=step_name, fn=func)
            self._steps.append(pipeline_step)
            return func

        if fn is not None:
            return _register(fn)
        return _register

    @overload
    def async_step(self, fn: AsyncStepFn, /) -> AsyncStepFn: ...

    @overload
    def async_step(
        self, /, *, name: str | None = ...
    ) -> Callable[[AsyncStepFn], AsyncStepFn]: ...

    def async_step(
        self,
        fn: AsyncStepFn | None = None,
        /,
        *,
        name: str | None = None,
    ) -> AsyncStepFn | Callable[[AsyncStepFn], AsyncStepFn]:
        """Decorator to register an async function as a pipeline step.

        Usage::

            @pipeline.async_step
            async def my_async_retriever(items, query):
                results = await fetch_from_db(query)
                return items + results

            @pipeline.async_step(name="db-lookup")
            async def db_step(items, query):
                ...
        """

        def _register(func: AsyncStepFn) -> AsyncStepFn:
            if not inspect.iscoroutinefunction(func):
                fn_name = getattr(func, "__name__", repr(func))
                msg = f"@async_step requires an async function, got {fn_name}"
                raise TypeError(msg)
            step_name: str = name or str(getattr(func, "__name__", "unnamed_async_step"))
            pipeline_step = PipelineStep(name=step_name, fn=func, is_async=True)
            self._steps.append(pipeline_step)
            return func

        if fn is not None:
            return _register(fn)
        return _register

    def _collect_pre_step_items(self, diagnostics: dict[str, Any]) -> list[ContextItem]:
        """Gather system + memory items before executing pipeline steps."""
        all_items: list[ContextItem] = list(self._system_items)

        if self._memory is not None:
            memory_items = self._memory.get_context_items()
            all_items.extend(memory_items)
            diagnostics["memory_items"] = len(memory_items)

        return all_items

    def _count_tokens(self, items: list[ContextItem]) -> list[ContextItem]:
        """Ensure all items have token counts."""
        counted: list[ContextItem] = []
        for item in items:
            if item.token_count == 0:
                token_count = self._tokenizer.count_tokens(item.content)
                item = item.model_copy(update={"token_count": token_count})
            counted.append(item)
        return counted

    def _assemble_result(
        self,
        counted_items: list[ContextItem],
        diagnostics: dict[str, Any],
        start_time: float,
    ) -> ContextResult:
        """Assemble items into a ContextWindow and format the output."""
        window = ContextWindow(max_tokens=self._max_tokens)
        overflow = window.add_items_by_priority(counted_items)

        try:
            formatted = self._formatter.format(window)
        except Exception as e:
            msg = (
                f"Formatter '{self._formatter.format_type}' failed to format "
                f"context window ({len(window.items)} items)"
            )
            raise FormatterError(msg) from e

        build_time = (time.monotonic() - start_time) * 1000
        diagnostics["total_items_considered"] = len(counted_items)
        diagnostics["items_included"] = len(window.items)
        diagnostics["items_overflow"] = len(overflow)
        diagnostics["token_utilization"] = round(window.utilization, 4)

        return ContextResult(
            window=window,
            formatted_output=formatted,
            format_type=self._formatter.format_type,
            overflow_items=overflow,
            diagnostics=cast(PipelineDiagnostics, diagnostics),
            build_time_ms=round(build_time, 2),
        )

    def build(self, query: str | QueryBundle) -> ContextResult:
        """Execute the full pipeline synchronously and return assembled context."""
        if isinstance(query, str):
            query = QueryBundle(query_str=query)
        start_time = time.monotonic()
        diagnostics: dict[str, Any] = {"steps": []}

        all_items = self._collect_pre_step_items(diagnostics)

        for step in self._steps:
            step_start = time.monotonic()
            try:
                all_items = step.execute(all_items, query)
            except (AstroContextError, TypeError, ValueError):
                raise
            except Exception as e:
                diagnostics["failed_step"] = step.name
                msg = f"Pipeline failed at step '{step.name}'"
                raise AstroContextError(msg) from e
            step_time = (time.monotonic() - step_start) * 1000
            diagnostics["steps"].append({
                "name": step.name,
                "items_after": len(all_items),
                "time_ms": round(step_time, 2),
            })

        counted_items = self._count_tokens(all_items)
        return self._assemble_result(counted_items, diagnostics, start_time)

    async def abuild(self, query: str | QueryBundle) -> ContextResult:
        """Execute the full pipeline asynchronously and return assembled context.

        Supports both sync and async pipeline steps. Sync steps are called
        directly; async steps are awaited. This allows mixing sync retrievers
        (e.g., in-memory BM25) with async retrievers (e.g., database lookups)
        in the same pipeline.
        """
        if isinstance(query, str):
            query = QueryBundle(query_str=query)
        start_time = time.monotonic()
        diagnostics: dict[str, Any] = {"steps": []}

        all_items = self._collect_pre_step_items(diagnostics)

        for step in self._steps:
            step_start = time.monotonic()
            try:
                all_items = await step.aexecute(all_items, query)
            except (AstroContextError, TypeError, ValueError):
                raise
            except Exception as e:
                diagnostics["failed_step"] = step.name
                msg = f"Pipeline failed at step '{step.name}'"
                raise AstroContextError(msg) from e
            step_time = (time.monotonic() - step_start) * 1000
            diagnostics["steps"].append({
                "name": step.name,
                "items_after": len(all_items),
                "time_ms": round(step_time, 2),
            })

        counted_items = self._count_tokens(all_items)
        return self._assemble_result(counted_items, diagnostics, start_time)
