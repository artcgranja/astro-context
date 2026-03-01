"""Table-aware chunking strategy for document ingestion.

Detects markdown and HTML tables in text, preserves them as atomic
units where possible, and delegates the remaining prose to an inner
chunker.

Implements the ``Chunker`` protocol.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from astro_context.ingestion.chunkers import RecursiveCharacterChunker
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter

logger = logging.getLogger(__name__)

# Matches consecutive lines that look like markdown table rows (start and end with ``|``).
_MARKDOWN_TABLE_RE = re.compile(
    r"(?:^[ \t]*\|.+\|[ \t]*$\n?){2,}",
    re.MULTILINE,
)

# Matches ``<table>...</table>`` blocks (non-greedy, case-insensitive).
_HTML_TABLE_RE = re.compile(
    r"<table[\s>].*?</table>",
    re.DOTALL | re.IGNORECASE,
)


class TableAwareChunker:
    """Preserve tables as atomic chunks while delegating prose to an inner chunker.

    Markdown and HTML tables are extracted from the document, replaced
    with unique placeholders, and the remaining text is chunked by the
    inner chunker.  Each table is then either kept whole (if it fits
    within ``chunk_size`` tokens) or split row-by-row, preserving the
    header row for markdown tables.

    Implements the ``Chunker`` protocol.
    """

    __slots__ = ("_chunk_size", "_inner", "_tokenizer")

    def __init__(
        self,
        inner_chunker: Any | None = None,
        chunk_size: int = 512,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        self._chunk_size = chunk_size
        self._tokenizer = tokenizer or get_default_counter()
        self._inner = inner_chunker or RecursiveCharacterChunker(
            chunk_size=chunk_size, overlap=0, tokenizer=self._tokenizer
        )

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Chunk text while preserving tables as atomic units.

        Parameters:
            text: The document text to chunk.
            metadata: Optional metadata forwarded to the inner chunker.

        Returns:
            A list of text chunks with tables preserved or row-split.
        """
        if not text or not text.strip():
            return []

        tables: list[str] = []
        processed = text

        # Extract HTML tables first (they may contain pipes that look
        # like markdown table rows).
        def _replace_html(m: re.Match[str]) -> str:
            idx = len(tables)
            tables.append(m.group(0))
            return f"__TABLE_{idx}__"

        processed = _HTML_TABLE_RE.sub(_replace_html, processed)

        # Extract markdown tables.
        def _replace_md(m: re.Match[str]) -> str:
            idx = len(tables)
            tables.append(m.group(0))
            return f"__TABLE_{idx}__"

        processed = _MARKDOWN_TABLE_RE.sub(_replace_md, processed)

        # Chunk the remaining text (with placeholders).
        text_chunks = self._inner.chunk(processed, metadata) if processed.strip() else []

        # Expand placeholders and build final chunk list.
        result: list[str] = []
        for tc in text_chunks:
            expanded = self._expand_placeholders(tc, tables)
            result.extend(expanded)

        # Any tables referenced by placeholders that did NOT appear in
        # any text chunk (e.g. the entire document was a table) are
        # appended directly.
        referenced: set[int] = set()
        for tc in text_chunks:
            for i in range(len(tables)):
                if f"__TABLE_{i}__" in tc:
                    referenced.add(i)
        for i, table in enumerate(tables):
            if i not in referenced:
                result.extend(self._chunk_table(table))

        return result

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _expand_placeholders(self, chunk: str, tables: list[str]) -> list[str]:
        """Replace placeholders in *chunk* with table content.

        If a chunk is *only* a placeholder, the table is returned
        (possibly row-split).  If it is mixed, the placeholder is
        replaced inline only when the result still fits within the
        token budget; otherwise the table is emitted as a separate
        chunk.
        """
        parts: list[str] = []

        # Check if the chunk is just a placeholder (possibly with whitespace).
        placeholder_only = re.fullmatch(r"\s*__TABLE_(\d+)__\s*", chunk)
        if placeholder_only:
            idx = int(placeholder_only.group(1))
            return self._chunk_table(tables[idx])

        # Otherwise try inline replacement.
        current = chunk
        for i, table in enumerate(tables):
            tag = f"__TABLE_{i}__"
            if tag in current:
                replaced = current.replace(tag, table)
                if self._tokenizer.count_tokens(replaced) <= self._chunk_size:
                    current = replaced
                else:
                    # Emit text before placeholder, then table separately.
                    before, after = current.split(tag, 1)
                    if before.strip():
                        parts.append(before.strip())
                    parts.extend(self._chunk_table(table))
                    current = after

        remaining = current.strip()
        if remaining:
            parts.append(remaining)
        return parts if parts else [chunk]

    def _chunk_table(self, table: str) -> list[str]:
        """Return the table as-is or split by rows if too large."""
        if self._tokenizer.count_tokens(table) <= self._chunk_size:
            return [table.strip()] if table.strip() else []

        # Determine if it's a markdown or HTML table.
        if table.strip().startswith("<table") or table.strip().startswith("<TABLE"):
            return self._split_html_table(table)
        return self._split_markdown_table(table)

    def _split_markdown_table(self, table: str) -> list[str]:
        """Split a markdown table by rows, preserving the header."""
        lines = [ln for ln in table.strip().splitlines() if ln.strip()]
        if len(lines) < 2:
            return [table.strip()]

        # First line is the header, second is the separator.
        header_lines: list[str] = [lines[0]]
        data_start = 1
        if len(lines) > 1 and re.match(r"^\s*\|[\s\-:|]+\|\s*$", lines[1]):
            header_lines.append(lines[1])
            data_start = 2

        header = "\n".join(header_lines)
        chunks: list[str] = []
        current = header

        for line in lines[data_start:]:
            candidate = current + "\n" + line
            if self._tokenizer.count_tokens(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = header + "\n" + line

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def _split_html_table(self, table: str) -> list[str]:
        """Naively split an HTML table by ``<tr>`` rows."""
        rows = re.split(r"(?=<tr[\s>])", table, flags=re.IGNORECASE)
        if len(rows) <= 1:
            return [table.strip()]

        # The first element is everything before the first <tr>.
        preamble = rows[0]
        # Find closing tag.
        closing = "</table>"
        chunks: list[str] = []
        current = preamble

        for row in rows[1:]:
            candidate = current + row
            if self._tokenizer.count_tokens(candidate + closing) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append((current + closing).strip())
                current = preamble + row

        if current.strip():
            chunks.append((current + closing).strip())

        return chunks

    def __repr__(self) -> str:
        return f"TableAwareChunker(inner={self._inner!r}, chunk_size={self._chunk_size})"
