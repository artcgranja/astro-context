"""Ingestion protocol definitions.

Any object with ``chunk`` / ``parse`` methods matching these signatures
can be used in the ingestion pipeline -- no inheritance required.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Chunker(Protocol):
    """Protocol for text chunking strategies.

    Chunkers split a single text string into smaller pieces suitable
    for embedding and retrieval.  They operate on raw text and return
    raw text -- the orchestrator (``DocumentIngester``) handles
    ``ContextItem`` construction.

    Parameters accepted by ``chunk`` are deliberately minimal so that
    implementations stay focused on splitting logic.
    """

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split *text* into a list of chunks.

        Parameters:
            text: The full document text to be chunked.
            metadata: Optional document-level metadata that the chunker
                may use to influence splitting (e.g. language hints).

        Returns:
            A list of text chunks.  May return an empty list if the
            input text is empty.
        """
        ...


@runtime_checkable
class DocumentParser(Protocol):
    """Protocol for document parsers.

    Parsers convert a file (or raw bytes) into plain text plus
    document-level metadata (title, author, page count, etc.).
    """

    def parse(self, source: Path | bytes) -> tuple[str, dict[str, Any]]:
        """Parse a document and extract its text content and metadata.

        Parameters:
            source: Either a filesystem ``Path`` to the document or
                the raw file bytes.

        Returns:
            A ``(text, metadata)`` tuple where *text* is the extracted
            plain-text content and *metadata* is a dict of document-level
            fields (e.g. ``title``, ``author``, ``page_count``).
        """
        ...

    @property
    def supported_extensions(self) -> list[str]:
        """File extensions this parser can handle (e.g. ``[".md", ".markdown"]``)."""
        ...
