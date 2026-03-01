"""Code-aware chunking strategy for document ingestion.

Splits source code at function, class, and other top-level definition
boundaries, merging small blocks until they reach the token budget.

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

# Map file extensions to language names for auto-detection.
_EXTENSION_MAP: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
}

# Regex patterns for identifying top-level code boundaries.
_LANGUAGE_PATTERNS: dict[str, re.Pattern[str]] = {
    "python": re.compile(r"^(?:def |class |async def )", re.MULTILINE),
    "javascript": re.compile(
        r"^(?:function |class |const \w+ = |export (?:default )?(?:function|class))",
        re.MULTILINE,
    ),
    "typescript": re.compile(
        r"^(?:function |class |const \w+ = |export (?:default )?(?:function|class))",
        re.MULTILINE,
    ),
    "go": re.compile(r"^func ", re.MULTILINE),
    "rust": re.compile(r"^(?:fn |impl |pub fn |pub impl )", re.MULTILINE),
}


class CodeChunker:
    """Split source code at definition boundaries.

    Identifies top-level constructs (functions, classes, impls) using
    language-specific regex patterns and groups adjacent small blocks
    until they fill the token budget.  When no code boundaries are
    found, falls back to ``RecursiveCharacterChunker``.

    Implements the ``Chunker`` protocol.
    """

    __slots__ = ("_chunk_size", "_fallback", "_language", "_overlap", "_patterns", "_tokenizer")

    def __init__(
        self,
        language: str | None = None,
        chunk_size: int = 512,
        overlap: int = 50,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if chunk_size <= 0:
            msg = f"chunk_size must be positive, got {chunk_size}"
            raise ValueError(msg)
        if overlap < 0:
            msg = f"overlap must be non-negative, got {overlap}"
            raise ValueError(msg)
        if overlap >= chunk_size:
            msg = f"overlap ({overlap}) must be less than chunk_size ({chunk_size})"
            raise ValueError(msg)
        self._language = language
        self._chunk_size = chunk_size
        self._overlap = overlap
        self._tokenizer = tokenizer or get_default_counter()
        self._patterns = _LANGUAGE_PATTERNS
        self._fallback = RecursiveCharacterChunker(
            chunk_size=chunk_size, overlap=overlap, tokenizer=self._tokenizer
        )

    def chunk(self, text: str, metadata: dict[str, Any] | None = None) -> list[str]:
        """Split source code at definition boundaries.

        Parameters:
            text: The source code to chunk.
            metadata: Optional metadata; ``extension`` key is used for
                language auto-detection when no explicit language is set.

        Returns:
            A list of code chunks.
        """
        if not text or not text.strip():
            return []

        lang = self._resolve_language(metadata)
        pattern = self._patterns.get(lang) if lang else None

        if pattern is None:
            logger.debug("No code pattern for language %r; falling back to recursive chunker", lang)
            return self._fallback.chunk(text, metadata)

        boundaries = [m.start() for m in pattern.finditer(text)]

        if not boundaries:
            logger.debug("No boundaries found in text; falling back to recursive chunker")
            return self._fallback.chunk(text, metadata)

        return self._split_and_merge(text, boundaries)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve_language(self, metadata: dict[str, Any] | None) -> str | None:
        """Determine the language from explicit setting or metadata extension."""
        if self._language:
            return self._language
        if metadata:
            ext = metadata.get("extension", "")
            return _EXTENSION_MAP.get(ext)
        return None

    def _split_and_merge(self, text: str, boundaries: list[int]) -> list[str]:
        """Split text at boundary positions, then merge small blocks."""
        # Build blocks between boundaries
        blocks: list[str] = []
        positions = [0, *boundaries] if boundaries[0] != 0 else list(boundaries)
        for i, pos in enumerate(positions):
            end = positions[i + 1] if i + 1 < len(positions) else len(text)
            block = text[pos:end]
            if block.strip():
                blocks.append(block)

        # Merge small adjacent blocks
        chunks: list[str] = []
        current = ""
        for block in blocks:
            candidate = current + block if current else block
            if self._tokenizer.count_tokens(candidate) <= self._chunk_size:
                current = candidate
            else:
                if current.strip():
                    chunks.append(current.strip())
                current = block

        if current.strip():
            chunks.append(current.strip())

        return chunks

    def __repr__(self) -> str:
        return (
            f"CodeChunker(language={self._language!r}, "
            f"chunk_size={self._chunk_size}, overlap={self._overlap})"
        )
