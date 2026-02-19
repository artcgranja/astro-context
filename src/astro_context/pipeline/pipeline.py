"""ContextPipeline -- the main orchestrator for astro-context."""

from __future__ import annotations

import time
from typing import Any

from astro_context.formatters.base import BaseFormatter
from astro_context.formatters.generic import GenericTextFormatter
from astro_context.memory.manager import MemoryManager
from astro_context.models.budget import TokenBudget
from astro_context.models.context import ContextItem, ContextResult, ContextWindow, SourceType
from astro_context.models.query import QueryBundle
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

from .step import PipelineStep


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
        budget: TokenBudget | None = None,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._max_tokens = max_tokens
        self._budget = budget
        self._tokenizer = tokenizer or get_default_counter()
        self._steps: list[PipelineStep] = []
        self._memory: MemoryManager | None = None
        self._formatter: BaseFormatter = GenericTextFormatter()
        self._system_items: list[ContextItem] = []

    def add_step(self, step: PipelineStep) -> ContextPipeline:
        """Add a pipeline step. Returns self for chaining."""
        self._steps.append(step)
        return self

    def with_memory(self, memory: MemoryManager) -> ContextPipeline:
        """Attach a memory manager. Returns self for chaining."""
        self._memory = memory
        return self

    def with_formatter(self, formatter: BaseFormatter) -> ContextPipeline:
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

    def build(self, query: QueryBundle) -> ContextResult:
        """Execute the full pipeline and return assembled context."""
        start_time = time.monotonic()
        diagnostics: dict[str, Any] = {"steps": []}

        # 1. Start with system items
        all_items: list[ContextItem] = list(self._system_items)

        # 2. Add memory items
        if self._memory is not None:
            memory_items = self._memory.get_context_items()
            all_items.extend(memory_items)
            diagnostics["memory_items"] = len(memory_items)

        # 3. Execute pipeline steps
        for step in self._steps:
            step_start = time.monotonic()
            all_items = step.execute(all_items, query)
            step_time = (time.monotonic() - step_start) * 1000
            diagnostics["steps"].append({
                "name": step.name,
                "items_after": len(all_items),
                "time_ms": round(step_time, 2),
            })

        # 4. Ensure all items have token counts
        counted_items: list[ContextItem] = []
        for item in all_items:
            if item.token_count == 0:
                token_count = self._tokenizer.count_tokens(item.content)
                item = ContextItem(
                    id=item.id,
                    content=item.content,
                    source=item.source,
                    score=item.score,
                    priority=item.priority,
                    token_count=token_count,
                    metadata=item.metadata,
                    created_at=item.created_at,
                )
            counted_items.append(item)

        # 5. Assemble into ContextWindow
        window = ContextWindow(max_tokens=self._max_tokens)
        overflow = window.add_items_by_priority(counted_items)

        # 6. Format output
        formatted = self._formatter.format(window)

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
            diagnostics=diagnostics,
            build_time_ms=round(build_time, 2),
        )
