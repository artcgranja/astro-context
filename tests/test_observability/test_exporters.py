"""Tests for span exporters."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from astro_context.observability.exporters import (
    ConsoleSpanExporter,
    FileSpanExporter,
    InMemorySpanExporter,
)
from astro_context.observability.models import Span, SpanKind
from astro_context.protocols.observability import SpanExporter


def _make_span(
    name: str = "test-span",
    kind: SpanKind = SpanKind.RETRIEVAL,
    trace_id: str = "trace-1",
    status: str = "ok",
    duration_ms: float | None = 42.0,
) -> Span:
    """Create a test span with sensible defaults."""
    now = datetime.now(UTC)
    return Span(
        trace_id=trace_id,
        name=name,
        kind=kind,
        start_time=now,
        end_time=now,
        duration_ms=duration_ms,
        status=status,
    )


class TestInMemorySpanExporter:
    """Tests for the InMemorySpanExporter."""

    def test_protocol_compliance(self) -> None:
        exporter = InMemorySpanExporter()
        assert isinstance(exporter, SpanExporter)

    def test_export_stores_spans(self) -> None:
        exporter = InMemorySpanExporter()
        spans = [_make_span("s1"), _make_span("s2")]
        exporter.export(spans)

        stored = exporter.get_spans()
        assert len(stored) == 2
        assert stored[0].name == "s1"
        assert stored[1].name == "s2"

    def test_export_accumulates(self) -> None:
        exporter = InMemorySpanExporter()
        exporter.export([_make_span("s1")])
        exporter.export([_make_span("s2")])

        assert len(exporter.get_spans()) == 2

    def test_get_spans_returns_copy(self) -> None:
        exporter = InMemorySpanExporter()
        exporter.export([_make_span()])

        result = exporter.get_spans()
        result.clear()
        assert len(exporter.get_spans()) == 1

    def test_clear(self) -> None:
        exporter = InMemorySpanExporter()
        exporter.export([_make_span()])
        exporter.clear()

        assert len(exporter.get_spans()) == 0

    def test_export_empty_list(self) -> None:
        exporter = InMemorySpanExporter()
        exporter.export([])
        assert len(exporter.get_spans()) == 0


class TestConsoleSpanExporter:
    """Tests for the ConsoleSpanExporter."""

    def test_protocol_compliance(self) -> None:
        exporter = ConsoleSpanExporter()
        assert isinstance(exporter, SpanExporter)

    def test_export_does_not_raise(self, caplog: object) -> None:
        exporter = ConsoleSpanExporter()
        exporter.export([_make_span()])
        # Smoke test — just ensure no exception

    def test_export_empty_list(self) -> None:
        exporter = ConsoleSpanExporter()
        exporter.export([])


class TestFileSpanExporter:
    """Tests for the FileSpanExporter."""

    def test_protocol_compliance(self, tmp_path: Path) -> None:
        exporter = FileSpanExporter(tmp_path / "spans.jsonl")
        assert isinstance(exporter, SpanExporter)

    def test_export_writes_jsonlines(self, tmp_path: Path) -> None:
        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)
        exporter.export([_make_span("s1"), _make_span("s2")])

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

        data = json.loads(lines[0])
        assert data["name"] == "s1"
        assert data["kind"] == "retrieval"

    def test_export_appends(self, tmp_path: Path) -> None:
        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)
        exporter.export([_make_span("s1")])
        exporter.export([_make_span("s2")])

        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_export_empty_list(self, tmp_path: Path) -> None:
        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)
        exporter.export([])

        # File should not exist or be empty
        if path.exists():
            assert path.read_text() == ""

    def test_span_without_end_time(self, tmp_path: Path) -> None:
        path = tmp_path / "spans.jsonl"
        exporter = FileSpanExporter(path)
        span = _make_span()
        # Create span without end_time
        unfinished = span.model_copy(update={"end_time": None, "duration_ms": None})
        exporter.export([unfinished])

        data = json.loads(path.read_text().strip())
        assert data["end_time"] is None
        assert data["duration_ms"] is None
