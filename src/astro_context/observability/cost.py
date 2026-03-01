"""Cost tracking for LLM operations and pipeline observability."""

from __future__ import annotations

import logging
import threading
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astro_context.models.context import ContextItem, ContextResult
from astro_context.models.query import QueryBundle

logger = logging.getLogger(__name__)


class CostEntry(BaseModel):
    """A single cost record for an LLM or embedding operation.

    Parameters:
        operation: The type of operation (e.g. ``"embedding"``, ``"rerank"``).
        model: The model identifier used for the operation.
        input_tokens: Number of input tokens consumed.
        output_tokens: Number of output tokens produced.
        cost_usd: Total cost in USD for this operation.
        timestamp: When the operation occurred.
        metadata: Arbitrary key-value metadata for the entry.
    """

    model_config = ConfigDict(frozen=True)

    operation: str
    model: str
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class CostSummary(BaseModel):
    """Aggregated cost summary across multiple operations.

    Parameters:
        total_cost_usd: Sum of all entry costs in USD.
        total_input_tokens: Sum of all input tokens across entries.
        total_output_tokens: Sum of all output tokens across entries.
        entries: The individual cost entries included in this summary.
        by_model: Cost breakdown grouped by model identifier.
        by_operation: Cost breakdown grouped by operation type.
    """

    model_config = ConfigDict(frozen=True)

    total_cost_usd: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    entries: list[CostEntry] = Field(default_factory=list)
    by_model: dict[str, float] = Field(default_factory=dict)
    by_operation: dict[str, float] = Field(default_factory=dict)


class CostTracker:
    """Thread-safe tracker for accumulating cost entries.

    Records individual cost entries and provides summary aggregation.
    All mutating operations are protected by a threading lock.

    Usage::

        tracker = CostTracker()
        tracker.record(
            operation="embedding",
            model="text-embedding-3-small",
            input_tokens=500,
            output_tokens=0,
            cost_per_input_token=0.00002,
        )
        summary = tracker.summary()
        print(summary.total_cost_usd)
    """

    __slots__ = ("_entries", "_lock")

    def __init__(self) -> None:
        self._entries: list[CostEntry] = []
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        with self._lock:
            count = len(self._entries)
        return f"CostTracker(entries={count})"

    def record(
        self,
        operation: str,
        model: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_per_input_token: float = 0.0,
        cost_per_output_token: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ) -> CostEntry:
        """Record a cost entry for an operation.

        Parameters:
            operation: The type of operation (e.g. ``"embedding"``).
            model: The model identifier used.
            input_tokens: Number of input tokens consumed.
            output_tokens: Number of output tokens produced.
            cost_per_input_token: Cost in USD per input token.
            cost_per_output_token: Cost in USD per output token.
            metadata: Optional key-value metadata.

        Returns:
            The created ``CostEntry``.
        """
        cost_usd = input_tokens * cost_per_input_token + output_tokens * cost_per_output_token
        entry = CostEntry(
            operation=operation,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
            metadata=metadata or {},
        )
        with self._lock:
            self._entries.append(entry)
        return entry

    def summary(self) -> CostSummary:
        """Compute an aggregated summary of all recorded entries.

        Returns:
            A ``CostSummary`` with totals and per-model/per-operation breakdowns.
        """
        with self._lock:
            entries = list(self._entries)

        total_cost = 0.0
        total_input = 0
        total_output = 0
        by_model: dict[str, float] = {}
        by_operation: dict[str, float] = {}

        for entry in entries:
            total_cost += entry.cost_usd
            total_input += entry.input_tokens
            total_output += entry.output_tokens
            by_model[entry.model] = by_model.get(entry.model, 0.0) + entry.cost_usd
            by_operation[entry.operation] = by_operation.get(entry.operation, 0.0) + entry.cost_usd

        return CostSummary(
            total_cost_usd=total_cost,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            entries=entries,
            by_model=by_model,
            by_operation=by_operation,
        )

    def reset(self) -> None:
        """Clear all recorded entries."""
        with self._lock:
            self._entries.clear()

    @property
    def entries(self) -> list[CostEntry]:
        """Return a copy of all recorded entries.

        Returns:
            A new list containing all ``CostEntry`` objects.
        """
        with self._lock:
            return list(self._entries)


class CostTrackingCallback:
    """Pipeline callback that records cost metadata from step execution.

    Listens for ``on_step_end`` events and records cost entries when
    the step items contain cost-related metadata.

    Implements the ``PipelineCallback`` protocol.

    Parameters:
        tracker: The ``CostTracker`` to record entries into.
    """

    __slots__ = ("_tracker",)

    def __init__(self, tracker: CostTracker) -> None:
        self._tracker = tracker

    def __repr__(self) -> str:
        return f"CostTrackingCallback(tracker={self._tracker!r})"

    def on_pipeline_start(self, query: QueryBundle) -> None:
        """Called when the pipeline starts. No-op for cost tracking."""

    def on_step_start(self, step_name: str, items: list[ContextItem]) -> None:
        """Called when a pipeline step starts. No-op for cost tracking."""

    def on_step_end(self, step_name: str, items: list[ContextItem], time_ms: float) -> None:
        """Called when a pipeline step ends.

        Records a cost entry if any item contains cost metadata
        (keys ``cost_model``, ``cost_input_tokens``, ``cost_output_tokens``,
        ``cost_per_input_token``, ``cost_per_output_token``).

        Parameters:
            step_name: Name of the completed step.
            items: Context items after the step.
            time_ms: Duration of the step in milliseconds.
        """
        for item in items:
            meta = item.metadata
            if "cost_model" in meta:
                self._tracker.record(
                    operation=step_name,
                    model=meta.get("cost_model", "unknown"),
                    input_tokens=int(meta.get("cost_input_tokens", 0)),
                    output_tokens=int(meta.get("cost_output_tokens", 0)),
                    cost_per_input_token=float(meta.get("cost_per_input_token", 0.0)),
                    cost_per_output_token=float(meta.get("cost_per_output_token", 0.0)),
                    metadata={"time_ms": time_ms},
                )

    def on_step_error(self, step_name: str, error: Exception) -> None:
        """Called when a pipeline step errors. No-op for cost tracking."""

    def on_pipeline_end(self, result: ContextResult) -> None:
        """Called when the pipeline ends. No-op for cost tracking."""
