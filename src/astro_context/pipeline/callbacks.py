"""Pipeline callback protocol for observability and event hooks."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from astro_context.models.context import ContextItem, ContextResult
from astro_context.models.query import QueryBundle


@runtime_checkable
class PipelineCallback(Protocol):
    """Protocol for pipeline event callbacks.

    Implement this protocol to receive events during pipeline execution.
    All methods have default no-op implementations so you only need to
    override the ones you care about.
    """

    def on_pipeline_start(self, query: QueryBundle) -> None: ...
    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None: ...
    def on_step_end(self, step_name: str, items: list[ContextItem], time_ms: float) -> None: ...
    def on_step_error(self, step_name: str, error: Exception) -> None: ...
    def on_pipeline_end(self, result: ContextResult) -> None: ...
