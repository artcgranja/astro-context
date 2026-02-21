"""ContextPipeline -- the main orchestrator for astro-context."""

from __future__ import annotations

import inspect
import logging
import time
from collections.abc import Callable
from typing import Any, Literal, cast, overload

from astro_context._callbacks import fire_callbacks
from astro_context.exceptions import AstroContextError, FormatterError, PipelineExecutionError
from astro_context.formatters.base import Formatter
from astro_context.formatters.generic import GenericTextFormatter
from astro_context.models.budget import TokenBudget
from astro_context.models.context import (
    ContextItem,
    ContextResult,
    ContextWindow,
    PipelineDiagnostics,
    SourceType,
)
from astro_context.models.query import QueryBundle
from astro_context.protocols.memory import MemoryProvider
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

from .callbacks import PipelineCallback
from .enrichment import ContextQueryEnricher
from .step import AsyncStepFn, PipelineStep, SyncStepFn

logger = logging.getLogger(__name__)


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
        budget: TokenBudget | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer or get_default_counter()
        self._steps: list[PipelineStep] = []
        self._memory: MemoryProvider | None = None
        self._formatter: Formatter = GenericTextFormatter()
        self._system_items: list[ContextItem] = []
        self._budget: TokenBudget | None = budget
        self._callbacks: list[PipelineCallback] = []
        self._query_enricher: ContextQueryEnricher | None = None

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

    @property
    def budget(self) -> TokenBudget | None:
        """The optional token budget for fine-grained allocation."""
        return self._budget

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

    def with_memory(self, memory: MemoryProvider) -> ContextPipeline:
        """Attach a memory provider. Returns self for chaining.

        Any object satisfying the ``MemoryProvider`` protocol (i.e. having a
        ``get_context_items(priority: int) -> list[ContextItem]`` method) is
        accepted.  ``MemoryManager`` is the canonical implementation.
        """
        self._memory = memory
        return self

    def with_budget(self, budget: TokenBudget) -> ContextPipeline:
        """Attach a token budget for fine-grained allocation. Returns self for chaining."""
        self._budget = budget
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

    def add_callback(self, callback: PipelineCallback) -> ContextPipeline:
        """Register an event callback for pipeline observability. Returns self for chaining."""
        self._callbacks.append(callback)
        return self

    def with_query_enricher(self, enricher: ContextQueryEnricher) -> ContextPipeline:
        """Attach a query enricher for memory-aware query expansion.

        The enricher is called after memory items are collected but before
        pipeline steps execute.  It receives the query string and the memory
        context items, and returns an enriched query string that replaces
        ``query.query_str`` for downstream steps.

        Returns self for chaining.
        """
        self._query_enricher = enricher
        return self

    def _fire(self, method: str, *args: Any) -> None:
        """Invoke a callback method on all registered callbacks, swallowing errors."""
        fire_callbacks(
            self._callbacks, method, *args, logger=logger, log_level=logging.WARNING,
        )

    # -- Decorator-based step registration (inspired by @agent.tool from Pydantic AI) --

    @overload
    def step(self, fn: SyncStepFn, /) -> SyncStepFn: ...

    @overload
    def step(
        self,
        /,
        *,
        name: str | None = ...,
        on_error: Literal["raise", "skip"] = ...,
    ) -> Callable[[SyncStepFn], SyncStepFn]: ...

    def step(
        self,
        fn: SyncStepFn | None = None,
        /,
        *,
        name: str | None = None,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> SyncStepFn | Callable[[SyncStepFn], SyncStepFn]:
        """Decorator to register a function as a pipeline step.

        Can be used with or without arguments::

            pipeline = ContextPipeline()

            @pipeline.step
            def my_filter(items, query):
                return [i for i in items if i.score > 0.5]

            @pipeline.step(name="custom-name", on_error="skip")
            def another_step(items, query):
                return items
        """

        def _register(func: SyncStepFn) -> SyncStepFn:
            if inspect.iscoroutinefunction(func):
                msg = f"'{func.__name__}' is async -- use @pipeline.async_step instead"
                raise TypeError(msg)
            step_name: str = name or str(getattr(func, "__name__", "unnamed_step"))
            pipeline_step = PipelineStep(name=step_name, fn=func, on_error=on_error)
            self._steps.append(pipeline_step)
            return func

        if fn is not None:
            return _register(fn)
        return _register

    @overload
    def async_step(self, fn: AsyncStepFn, /) -> AsyncStepFn: ...

    @overload
    def async_step(
        self,
        /,
        *,
        name: str | None = ...,
        on_error: Literal["raise", "skip"] = ...,
    ) -> Callable[[AsyncStepFn], AsyncStepFn]: ...

    def async_step(
        self,
        fn: AsyncStepFn | None = None,
        /,
        *,
        name: str | None = None,
        on_error: Literal["raise", "skip"] = "raise",
    ) -> AsyncStepFn | Callable[[AsyncStepFn], AsyncStepFn]:
        """Decorator to register an async function as a pipeline step.

        Usage::

            @pipeline.async_step
            async def my_async_retriever(items, query):
                results = await fetch_from_db(query)
                return items + results

            @pipeline.async_step(name="db-lookup", on_error="skip")
            async def db_step(items, query):
                ...
        """

        def _register(func: AsyncStepFn) -> AsyncStepFn:
            if not inspect.iscoroutinefunction(func):
                fn_name = getattr(func, "__name__", repr(func))
                msg = f"@async_step requires an async function, got {fn_name}"
                raise TypeError(msg)
            step_name: str = name or str(getattr(func, "__name__", "unnamed_async_step"))
            pipeline_step = PipelineStep(
                name=step_name, fn=func, is_async=True, on_error=on_error,
            )
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

    @staticmethod
    def _apply_drop_strategy(
        source_items: list[ContextItem],
        cap: int,
    ) -> tuple[list[ContextItem], list[ContextItem]]:
        """Apply the 'drop' overflow strategy: keep all or drop all."""
        total_tokens = sum(item.token_count for item in source_items)
        if total_tokens <= cap:
            return source_items, []
        return [], source_items

    @staticmethod
    def _apply_truncate_strategy(
        source_items: list[ContextItem],
        cap: int,
    ) -> tuple[list[ContextItem], list[ContextItem]]:
        """Apply the 'truncate' overflow strategy: keep items up to cap."""
        accepted: list[ContextItem] = []
        overflowed: list[ContextItem] = []
        used = 0
        for item in source_items:
            if used + item.token_count <= cap:
                accepted.append(item)
                used += item.token_count
            else:
                overflowed.append(item)
        return accepted, overflowed

    def _apply_source_budgets(
        self,
        items: list[ContextItem],
        diagnostics: dict[str, Any],
    ) -> tuple[list[ContextItem], list[ContextItem]]:
        """Pre-filter items by per-source-type token budgets.

        Groups items by ``source``, applies the per-source ``max_tokens``
        cap from ``budget.allocations``, and returns (accepted, overflow).
        Items whose source has no explicit allocation are passed through
        uncapped (they compete in the shared pool during window assembly).

        The per-allocation ``overflow_strategy`` controls behaviour when a
        source exceeds its cap:

        * ``"truncate"`` (default): include items up to the cap sorted by
          priority/score, overflow the rest.
        * ``"drop"``: if the total for that source exceeds the cap, **all**
          items for that source are dropped to overflow.
        """
        budget = self._budget
        if budget is None:
            msg = "_apply_source_budgets called without a budget configured"
            raise PipelineExecutionError(msg)

        # Build lookups of source -> max_tokens and source -> overflow_strategy
        source_caps: dict[SourceType, int] = {}
        source_strategies: dict[SourceType, str] = {}
        for alloc in budget.allocations:
            source_caps[alloc.source] = alloc.max_tokens
            source_strategies[alloc.source] = alloc.overflow_strategy

        # Group items by source, preserving priority/score ordering
        by_source: dict[SourceType, list[ContextItem]] = {}
        for item in sorted(items, key=lambda x: (-x.priority, -x.score)):
            by_source.setdefault(item.source, []).append(item)

        accepted: list[ContextItem] = []
        overflow: list[ContextItem] = []
        overflow_by_source: dict[str, int] = {}

        for source, source_items in by_source.items():
            if source not in source_caps:
                accepted.extend(source_items)
                continue

            cap = source_caps[source]
            if source_strategies[source] == "drop":
                kept, dropped = self._apply_drop_strategy(source_items, cap)
            else:
                kept, dropped = self._apply_truncate_strategy(source_items, cap)

            accepted.extend(kept)
            overflow.extend(dropped)
            if dropped:
                overflow_by_source[source.value] = len(dropped)

        if overflow_by_source:
            diagnostics["budget_overflow_by_source"] = overflow_by_source

        return accepted, overflow

    def _assemble_result(
        self,
        counted_items: list[ContextItem],
        diagnostics: dict[str, Any],
        start_time: float,
    ) -> ContextResult:
        """Assemble items into a ContextWindow and format the output."""
        effective_max = self._max_tokens
        if self._budget is not None:
            effective_max = self._max_tokens - self._budget.reserve_tokens
            if effective_max <= 0:
                msg = (
                    f"reserve_tokens ({self._budget.reserve_tokens}) must be less than "
                    f"pipeline max_tokens ({self._max_tokens})"
                )
                raise PipelineExecutionError(msg)

        # Apply per-source budgets when allocations are defined
        budget_overflow: list[ContextItem] = []
        if self._budget is not None and self._budget.allocations:
            counted_items, budget_overflow = self._apply_source_budgets(
                counted_items, diagnostics,
            )

        window = ContextWindow(max_tokens=effective_max)
        window_overflow = window.add_items_by_priority(counted_items)
        overflow = budget_overflow + window_overflow

        try:
            formatted = self._formatter.format(window)
        except Exception as e:
            msg = (
                f"Formatter '{self._formatter.format_type}' failed to format "
                f"context window ({len(window.items)} items)"
            )
            raise FormatterError(msg) from e

        build_time = (time.monotonic() - start_time) * 1000
        diagnostics["total_items_considered"] = len(counted_items) + len(budget_overflow)
        diagnostics["items_included"] = len(window.items)
        diagnostics["items_overflow"] = len(overflow)
        diagnostics["token_utilization"] = round(window.utilization, 4)

        if self._budget is not None:
            usage_by_source: dict[str, int] = {}
            for item in window.items:
                key = item.source.value
                usage_by_source[key] = usage_by_source.get(key, 0) + item.token_count
            diagnostics["token_usage_by_source"] = usage_by_source

            # Compute shared pool usage: tokens used by sources without allocations
            allocated_sources = {a.source for a in self._budget.allocations}
            shared_tokens = sum(
                item.token_count
                for item in window.items
                if item.source not in allocated_sources
            )
            diagnostics["shared_pool_usage"] = shared_tokens

        return ContextResult(
            window=window,
            formatted_output=formatted,
            format_type=self._formatter.format_type,
            overflow_items=overflow,
            diagnostics=cast(PipelineDiagnostics, diagnostics),
            build_time_ms=round(build_time, 2),
        )

    # -- Shared build helpers (DRY for build/abuild) --

    def _prepare_build(
        self, query_input: str | QueryBundle
    ) -> tuple[QueryBundle, list[ContextItem], dict[str, Any], float]:
        """Shared pre-step logic for build() and abuild().

        Normalises the query, initialises diagnostics, collects pre-step items,
        and enriches the query with memory context when an enricher is attached.

        Returns ``(query, all_items, diagnostics, start_time)``.
        """
        query = QueryBundle(query_str=query_input) if isinstance(query_input, str) else query_input
        start_time = time.monotonic()
        diagnostics: dict[str, Any] = {"steps": []}

        self._fire("on_pipeline_start", query)

        all_items = self._collect_pre_step_items(diagnostics)

        # Enrich query with memory context before retrieval steps
        if self._query_enricher is not None:
            memory_items = [
                i
                for i in all_items
                if i.source in (SourceType.MEMORY, SourceType.CONVERSATION)
            ]
            if memory_items:
                enriched_text = self._query_enricher.enrich(query.query_str, memory_items)
                query = query.model_copy(update={"query_str": enriched_text})
                diagnostics["query_enriched"] = True

        return query, all_items, diagnostics, start_time

    def _handle_step_error(
        self,
        step: PipelineStep,
        exc: Exception,
        items_before: list[ContextItem],
        diagnostics: dict[str, Any],
        *,
        is_known: bool,
    ) -> list[ContextItem]:
        """Handle an exception raised during step execution.

        *is_known* is ``True`` when the exception is an expected type
        (``AstroContextError``, ``TypeError``, ``ValueError``) that should
        be re-raised as-is if the step policy is ``"raise"``.

        Returns the rollback item list when the step is skipped.  When the
        step policy is ``"raise"`` this method always raises and never
        returns.
        """
        self._fire("on_step_error", step.name, exc)
        if step.on_error == "skip":
            logger.warning("Step '%s' failed; skipping (on_error='skip')", step.name)
            diagnostics.setdefault("skipped_steps", []).append(step.name)
            return items_before
        if is_known:
            raise exc
        diagnostics["failed_step"] = step.name
        msg = f"Pipeline failed at step '{step.name}'"
        raise PipelineExecutionError(msg, diagnostics=diagnostics) from exc

    def _record_step_success(
        self,
        step: PipelineStep,
        all_items: list[ContextItem],
        step_start: float,
        diagnostics: dict[str, Any],
    ) -> None:
        """Record timing and item count for a successfully completed step."""
        step_time = (time.monotonic() - step_start) * 1000
        self._fire("on_step_end", step.name, all_items, step_time)
        diagnostics["steps"].append({
            "name": step.name,
            "items_after": len(all_items),
            "time_ms": round(step_time, 2),
        })

    def _finalize_build(
        self,
        all_items: list[ContextItem],
        diagnostics: dict[str, Any],
        start_time: float,
    ) -> ContextResult:
        """Shared post-step logic: count tokens, assemble result, fire callback."""
        counted_items = self._count_tokens(all_items)
        result = self._assemble_result(counted_items, diagnostics, start_time)
        self._fire("on_pipeline_end", result)
        return result

    # -- Public build entry points --

    def build(self, query: str | QueryBundle) -> ContextResult:
        """Execute the full pipeline synchronously and return assembled context."""
        resolved_query, all_items, diagnostics, start_time = self._prepare_build(query)

        for step in self._steps:
            step_start = time.monotonic()
            items_before = all_items
            self._fire("on_step_start", step.name, list(all_items))
            try:
                all_items = step.execute(all_items, resolved_query)
            except (AstroContextError, TypeError, ValueError) as exc:
                all_items = self._handle_step_error(
                    step, exc, items_before, diagnostics, is_known=True,
                )
                continue
            except Exception as e:
                all_items = self._handle_step_error(
                    step, e, items_before, diagnostics, is_known=False,
                )
                continue
            self._record_step_success(step, all_items, step_start, diagnostics)

        return self._finalize_build(all_items, diagnostics, start_time)

    async def abuild(self, query: str | QueryBundle) -> ContextResult:
        """Execute the full pipeline asynchronously and return assembled context.

        Supports both sync and async pipeline steps. Sync steps are called
        directly; async steps are awaited. This allows mixing sync retrievers
        (e.g., in-memory BM25) with async retrievers (e.g., database lookups)
        in the same pipeline.
        """
        resolved_query, all_items, diagnostics, start_time = self._prepare_build(query)

        for step in self._steps:
            step_start = time.monotonic()
            items_before = all_items
            self._fire("on_step_start", step.name, list(all_items))
            try:
                all_items = await step.aexecute(all_items, resolved_query)
            except (AstroContextError, TypeError, ValueError) as exc:
                all_items = self._handle_step_error(
                    step, exc, items_before, diagnostics, is_known=True,
                )
                continue
            except Exception as e:
                all_items = self._handle_step_error(
                    step, e, items_before, diagnostics, is_known=False,
                )
                continue
            self._record_step_success(step, all_items, step_start, diagnostics)

        return self._finalize_build(all_items, diagnostics, start_time)
