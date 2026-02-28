"""Built-in span exporters for observability."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from astro_context.observability.models import Span

logger = logging.getLogger(__name__)


class ConsoleSpanExporter:
    """Exports spans as structured JSON to the standard logging system.

    Each span is serialised to a JSON string and emitted at the configured
    log level.  Useful for development and debugging.
    """

    __slots__ = ("_log_level",)

    def __init__(self, log_level: int = logging.INFO) -> None:
        self._log_level = log_level

    def export(self, spans: list[Span]) -> None:
        """Export spans by logging each one as a JSON object.

        Parameters:
            spans: The completed spans to export.
        """
        for span in spans:
            data = _span_to_dict(span)
            logger.log(self._log_level, json.dumps(data, default=str))


class InMemorySpanExporter:
    """Stores exported spans in an in-memory list for testing and debugging.

    Provides ``get_spans()`` and ``clear()`` helpers.
    """

    __slots__ = ("_spans",)

    def __init__(self) -> None:
        self._spans: list[Span] = []

    def export(self, spans: list[Span]) -> None:
        """Append spans to the in-memory buffer.

        Parameters:
            spans: The completed spans to store.
        """
        self._spans.extend(spans)

    def get_spans(self) -> list[Span]:
        """Return a copy of all exported spans.

        Returns:
            A list of all spans that have been exported so far.
        """
        return list(self._spans)

    def clear(self) -> None:
        """Remove all stored spans."""
        self._spans.clear()


class FileSpanExporter:
    """Writes spans as JSON-Lines to a file on disk.

    Each call to ``export()`` appends one JSON object per span, separated
    by newlines.  The file is opened in append mode so multiple exports
    accumulate.

    Parameters:
        path: The file path to write to.  Parent directories must exist.
    """

    __slots__ = ("_path",)

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)

    def export(self, spans: list[Span]) -> None:
        """Append spans as JSON lines to the configured file.

        Parameters:
            spans: The completed spans to export.
        """
        with self._path.open("a") as fh:
            for span in spans:
                data = _span_to_dict(span)
                fh.write(json.dumps(data, default=str))
                fh.write("\n")


def _span_to_dict(span: Span) -> dict[str, Any]:
    """Convert a Span to a JSON-serialisable dictionary.

    Parameters:
        span: The span to convert.

    Returns:
        A plain dictionary representation of the span.
    """
    return {
        "span_id": span.span_id,
        "parent_span_id": span.parent_span_id,
        "trace_id": span.trace_id,
        "name": span.name,
        "kind": span.kind.value,
        "start_time": span.start_time.isoformat(),
        "end_time": span.end_time.isoformat() if span.end_time else None,
        "duration_ms": span.duration_ms,
        "status": span.status,
        "attributes": span.attributes,
        "events": span.events,
    }
