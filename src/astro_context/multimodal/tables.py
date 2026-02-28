"""Table extraction utilities for Markdown and HTML sources."""

from __future__ import annotations

import logging
import re
from html.parser import HTMLParser
from pathlib import Path

from astro_context.multimodal.models import ModalityType, MultiModalContent

logger = logging.getLogger(__name__)

# Matches a contiguous block of lines where every line starts with '|'.
_MD_TABLE_RE = re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*$\n?){2,}",
    re.MULTILINE,
)


class MarkdownTableParser:
    """Extracts tables from Markdown text using regex.

    Returns each table found as a ``MultiModalContent`` with
    ``modality=TABLE``.  Implements the ``TableExtractor`` protocol.
    """

    __slots__ = ()

    def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]:
        """Extract Markdown tables from a source.

        Parameters:
            source: A file path to a Markdown file, or raw bytes / string
                encoded as UTF-8.

        Returns:
            A list of ``MultiModalContent`` objects representing each
            table found.
        """
        text = self._read_source(source)
        tables: list[MultiModalContent] = []
        for match in _MD_TABLE_RE.finditer(text):
            table_text = match.group(0).strip()
            tables.append(
                MultiModalContent(
                    modality=ModalityType.TABLE,
                    content=table_text,
                    metadata={"format": "markdown"},
                )
            )
        return tables

    @staticmethod
    def _read_source(source: Path | bytes) -> str:
        """Read the source into a string."""
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")
        return source.decode("utf-8")


class _HTMLTableContentHandler(HTMLParser):
    """Internal HTML parser that collects <table> elements."""

    __slots__ = ("_current_cell", "_current_row", "_depth", "_rows", "tables")

    def __init__(self) -> None:
        super().__init__()
        self.tables: list[list[list[str]]] = []
        self._rows: list[list[str]] = []
        self._current_row: list[str] = []
        self._current_cell: list[str] = []
        self._depth: int = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        tag_lower = tag.lower()
        if tag_lower == "table":
            self._depth += 1
            if self._depth == 1:
                self._rows = []
        elif tag_lower == "tr" and self._depth == 1:
            self._current_row = []
        elif tag_lower in {"td", "th"} and self._depth == 1:
            self._current_cell = []

    def handle_endtag(self, tag: str) -> None:
        tag_lower = tag.lower()
        if tag_lower == "table":
            if self._depth == 1 and self._rows:
                self.tables.append(self._rows)
                self._rows = []
            self._depth = max(0, self._depth - 1)
        elif tag_lower == "tr" and self._depth == 1:
            if self._current_row:
                self._rows.append(self._current_row)
            self._current_row = []
        elif tag_lower in {"td", "th"} and self._depth == 1:
            self._current_row.append("".join(self._current_cell).strip())
            self._current_cell = []

    def handle_data(self, data: str) -> None:
        if self._depth >= 1:
            self._current_cell.append(data)


def _table_to_markdown(rows: list[list[str]]) -> str:
    """Convert a list of rows into a Markdown table string."""
    if not rows:
        return ""
    # Ensure all rows have the same column count
    max_cols = max(len(r) for r in rows)
    normalised = [r + [""] * (max_cols - len(r)) for r in rows]

    lines: list[str] = []
    header = normalised[0]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in normalised[1:]:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


class HTMLTableParser:
    """Extracts tables from HTML using the stdlib ``html.parser``.

    Converts each ``<table>`` element into a Markdown representation
    and returns it as a ``MultiModalContent`` with ``modality=TABLE``.
    Implements the ``TableExtractor`` protocol.
    """

    __slots__ = ()

    def extract_tables(self, source: Path | bytes) -> list[MultiModalContent]:
        """Extract HTML tables from a source.

        Parameters:
            source: A file path to an HTML file, or raw bytes encoded
                as UTF-8.

        Returns:
            A list of ``MultiModalContent`` objects representing each
            table found, with content converted to Markdown format.
        """
        text = self._read_source(source)
        handler = _HTMLTableContentHandler()
        handler.feed(text)

        tables: list[MultiModalContent] = []
        for raw_table in handler.tables:
            md = _table_to_markdown(raw_table)
            if md:
                tables.append(
                    MultiModalContent(
                        modality=ModalityType.TABLE,
                        content=md,
                        metadata={"format": "html", "original_format": "html"},
                    )
                )
        return tables

    @staticmethod
    def _read_source(source: Path | bytes) -> str:
        """Read the source into a string."""
        if isinstance(source, Path):
            return source.read_text(encoding="utf-8")
        return source.decode("utf-8")
