"""Tokenizer protocol for token counting abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting.

    The default implementation uses tiktoken, but users can provide
    any tokenizer (e.g., HuggingFace tokenizers, sentencepiece).
    """

    def count_tokens(self, text: str) -> int: ...
    def truncate_to_tokens(self, text: str, max_tokens: int) -> str: ...
