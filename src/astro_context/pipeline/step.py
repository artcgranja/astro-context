"""Composable pipeline steps with sync and async support."""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from astro_context.exceptions import AstroContextError
from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle
from astro_context.protocols.postprocessor import AsyncPostProcessor, PostProcessor
from astro_context.protocols.retriever import AsyncRetriever, Retriever

SyncStepFn = Callable[[list[ContextItem], QueryBundle], list[ContextItem]]
AsyncStepFn = Callable[[list[ContextItem], QueryBundle], Awaitable[list[ContextItem]]]
StepFn = SyncStepFn | AsyncStepFn


@dataclass(slots=True)
class PipelineStep:
    """A single step in the context pipeline.

    Steps are composable and execute in order. Each step receives the
    current list of context items and the query, and returns a
    (potentially modified) list of context items.

    Supports both sync and async execution functions. If an async fn
    is provided, the step can only be executed via ``aexecute()``.
    """

    name: str
    fn: StepFn
    is_async: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def execute(self, items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        """Execute this step synchronously."""
        if self.is_async:
            msg = (
                f"Step '{self.name}' is async and cannot be called synchronously. "
                "Use pipeline.abuild() or step.aexecute() instead."
            )
            raise TypeError(msg)
        try:
            result = self.fn(items, query)
        except AstroContextError:
            raise
        except Exception as e:
            msg = f"Step '{self.name}' failed during execution"
            raise AstroContextError(msg) from e
        if not isinstance(result, list):
            msg = f"Step '{self.name}' must return a list of ContextItem"
            raise TypeError(msg)
        return result

    async def aexecute(
        self, items: list[ContextItem], query: QueryBundle
    ) -> list[ContextItem]:
        """Execute this step asynchronously.

        Works for both sync and async step functions -- sync functions
        are called directly (they are typically fast in-memory operations).
        """
        try:
            result = self.fn(items, query)
            if inspect.isawaitable(result):
                result = await result
        except AstroContextError:
            raise
        except Exception as e:
            msg = f"Step '{self.name}' failed during execution"
            raise AstroContextError(msg) from e
        if not isinstance(result, list):
            msg = f"Step '{self.name}' must return a list of ContextItem"
            raise TypeError(msg)
        return result


def retriever_step(name: str, retriever: Retriever, top_k: int = 10) -> PipelineStep:
    """Create a pipeline step from a Retriever protocol implementation."""

    def _retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        retrieved = retriever.retrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_retrieve)


def async_retriever_step(
    name: str, retriever: AsyncRetriever, top_k: int = 10
) -> PipelineStep:
    """Create an async pipeline step from an AsyncRetriever implementation."""

    async def _aretrieve(
        items: list[ContextItem], query: QueryBundle
    ) -> list[ContextItem]:
        retrieved = await retriever.aretrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_aretrieve, is_async=True)


def postprocessor_step(name: str, processor: PostProcessor) -> PipelineStep:
    """Create a pipeline step from a PostProcessor protocol implementation."""

    def _process(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return processor.process(items, query)

    return PipelineStep(name=name, fn=_process)


def async_postprocessor_step(name: str, processor: AsyncPostProcessor) -> PipelineStep:
    """Create an async pipeline step from an AsyncPostProcessor implementation."""

    async def _aprocess(
        items: list[ContextItem], query: QueryBundle
    ) -> list[ContextItem]:
        return await processor.aprocess(items, query)

    return PipelineStep(name=name, fn=_aprocess, is_async=True)


def filter_step(name: str, predicate: Callable[[ContextItem], bool]) -> PipelineStep:
    """Create a pipeline step that filters items by a predicate."""

    def _filter(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return [item for item in items if predicate(item)]

    return PipelineStep(name=name, fn=_filter)
