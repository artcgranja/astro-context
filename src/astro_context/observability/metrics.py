"""Built-in metrics collectors for observability."""

from __future__ import annotations

import json
import logging
from typing import Any

from astro_context.observability.models import MetricPoint

logger = logging.getLogger(__name__)


class InMemoryMetricsCollector:
    """Stores metric points in memory for testing and debugging.

    Provides helpers to query stored metrics and compute summary statistics
    (min, max, avg, count, p50, p95).
    """

    __slots__ = ("_metrics",)

    def __init__(self) -> None:
        self._metrics: list[MetricPoint] = []

    def record(self, metric: MetricPoint) -> None:
        """Record a single metric measurement.

        Parameters:
            metric: The metric point to store.
        """
        self._metrics.append(metric)

    def flush(self) -> None:
        """No-op for the in-memory collector.

        Metrics remain available via ``get_metrics()`` after flushing.
        """

    def get_metrics(self, name: str | None = None) -> list[MetricPoint]:
        """Return stored metrics, optionally filtered by name.

        Parameters:
            name: If provided, only return metrics with this name.

        Returns:
            A list of matching ``MetricPoint`` objects.
        """
        if name is None:
            return list(self._metrics)
        return [m for m in self._metrics if m.name == name]

    def get_summary(self, name: str) -> dict[str, Any]:
        """Compute summary statistics for a named metric.

        Parameters:
            name: The metric name to summarise.

        Returns:
            A dictionary with keys ``min``, ``max``, ``avg``, ``count``,
            ``p50``, and ``p95``.  Returns an empty dict if no metrics
            match the given name.
        """
        values = [m.value for m in self._metrics if m.name == name]
        if not values:
            return {}

        sorted_values = sorted(values)
        count = len(sorted_values)

        return {
            "min": sorted_values[0],
            "max": sorted_values[-1],
            "avg": sum(sorted_values) / count,
            "count": count,
            "p50": _percentile(sorted_values, 50),
            "p95": _percentile(sorted_values, 95),
        }

    def clear(self) -> None:
        """Remove all stored metrics."""
        self._metrics.clear()


class LoggingMetricsCollector:
    """Logs metric points via the standard ``logging`` module.

    Each recorded metric is emitted as a structured JSON log message.
    ``flush()`` is a no-op because metrics are emitted immediately.
    """

    __slots__ = ("_log_level",)

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    def record(self, metric: MetricPoint) -> None:
        """Log a single metric measurement.

        Parameters:
            metric: The metric point to log.
        """
        data = {
            "name": metric.name,
            "value": metric.value,
            "timestamp": metric.timestamp.isoformat(),
            "tags": metric.tags,
        }
        logger.log(self._log_level, json.dumps(data, default=str))

    def flush(self) -> None:
        """No-op; metrics are logged immediately on ``record()``."""


def _percentile(sorted_values: list[float], pct: int) -> float:
    """Compute a percentile from pre-sorted values using the nearest-rank method.

    Parameters:
        sorted_values: A non-empty, pre-sorted list of floats.
        pct: The percentile to compute (0-100).

    Returns:
        The value at the given percentile.
    """
    if len(sorted_values) == 1:
        return sorted_values[0]
    # Use Python's statistics.quantiles for accurate interpolation
    # quantiles returns n-1 cut points for n quantiles
    # For p50 and p95 we compute directly
    k = pct / 100
    idx = k * (len(sorted_values) - 1)
    lower = int(idx)
    upper = lower + 1
    if upper >= len(sorted_values):
        return sorted_values[-1]
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight
