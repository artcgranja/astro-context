"""Composable pipeline steps."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from astro_context.models.context import ContextItem
from astro_context.models.query import QueryBundle


@dataclass
class PipelineStep:
    """A single step in the context pipeline.

    Steps are composable and execute in order. Each step receives the
    current list of context items and the query, and returns a
    (potentially modified) list of context items.
    """

    name: str
    fn: Callable[[list[ContextItem], QueryBundle], list[ContextItem]]
    metadata: dict[str, Any] = field(default_factory=dict)

    def execute(self, items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        """Execute this step on the given items."""
        return self.fn(items, query)


def retriever_step(name: str, retriever: Any, top_k: int = 10) -> PipelineStep:
    """Create a pipeline step from a Retriever protocol implementation."""

    def _retrieve(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        retrieved = retriever.retrieve(query, top_k=top_k)
        return items + retrieved

    return PipelineStep(name=name, fn=_retrieve)


def postprocessor_step(name: str, processor: Any) -> PipelineStep:
    """Create a pipeline step from a PostProcessor protocol implementation."""

    def _process(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return processor.process(items, query)

    return PipelineStep(name=name, fn=_process)


def filter_step(name: str, predicate: Callable[[ContextItem], bool]) -> PipelineStep:
    """Create a pipeline step that filters items by a predicate."""

    def _filter(items: list[ContextItem], query: QueryBundle) -> list[ContextItem]:
        return [item for item in items if predicate(item)]

    return PipelineStep(name=name, fn=_filter)
