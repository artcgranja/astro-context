"""Metadata generation and enrichment for ingested documents."""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)


def generate_doc_id(content: str, source_path: str | None = None) -> str:
    """Generate a deterministic document ID from content and optional path.

    Parameters:
        content: The full document text.
        source_path: Optional file path used as a salt for uniqueness.

    Returns:
        A 16-character hex string derived from SHA-256.
    """
    h = hashlib.sha256()
    if source_path:
        h.update(source_path.encode("utf-8"))
    h.update(content.encode("utf-8"))
    return h.hexdigest()[:16]


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a chunk ID from a document ID and chunk index.

    Parameters:
        doc_id: The parent document's ID.
        chunk_index: Zero-based index of the chunk within the document.

    Returns:
        A string in the format ``"{doc_id}-chunk-{chunk_index}"``.
    """
    return f"{doc_id}-chunk-{chunk_index}"


def extract_chunk_metadata(
    chunk_text: str,
    chunk_index: int,
    total_chunks: int,
    doc_id: str,
    doc_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build standard metadata for a single chunk.

    Parameters:
        chunk_text: The text content of this chunk.
        chunk_index: Zero-based position of this chunk.
        total_chunks: Total number of chunks in the document.
        doc_id: The parent document's ID.
        doc_metadata: Optional document-level metadata to propagate.

    Returns:
        A metadata dict with standard fields plus propagated doc metadata.
    """
    meta: dict[str, Any] = {
        "parent_doc_id": doc_id,
        "chunk_index": chunk_index,
        "total_chunks": total_chunks,
        "word_count": len(chunk_text.split()),
        "char_count": len(chunk_text),
    }
    if doc_metadata:
        for key, value in doc_metadata.items():
            prefixed = f"doc_{key}" if not key.startswith("doc_") else key
            meta[prefixed] = value
    return meta


class MetadataEnricher:
    """Chain of user-provided metadata enrichment functions.

    Each enricher function receives ``(text, chunk_index, total_chunks, metadata)``
    and returns an updated metadata dict.
    """

    __slots__ = ("_enrichers",)

    def __init__(
        self,
        enrichers: list[Callable[[str, int, int, dict[str, Any]], dict[str, Any]]] | None = None,
    ) -> None:
        self._enrichers = list(enrichers) if enrichers else []

    def add(
        self, fn: Callable[[str, int, int, dict[str, Any]], dict[str, Any]]
    ) -> None:
        """Register an enricher function.

        Parameters:
            fn: A callable ``(text, chunk_index, total_chunks, metadata) -> metadata``.
        """
        self._enrichers.append(fn)

    def enrich(
        self,
        text: str,
        chunk_index: int,
        total_chunks: int,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        """Run all enrichers in order, threading metadata through.

        Parameters:
            text: The chunk text.
            chunk_index: Zero-based chunk position.
            total_chunks: Total chunks in the document.
            metadata: The initial metadata dict.

        Returns:
            The enriched metadata dict after all enrichers have run.
        """
        result = dict(metadata)
        for fn in self._enrichers:
            result = fn(text, chunk_index, total_chunks, result)
        return result

    def __repr__(self) -> str:
        return f"MetadataEnricher(enrichers={len(self._enrichers)})"
