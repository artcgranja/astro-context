"""Token counting implementations."""

from __future__ import annotations

import threading

import tiktoken


class TiktokenCounter:
    """Token counter using OpenAI's tiktoken library.

    Default encoding is cl100k_base (used by GPT-4, Claude tokenizers
    are similar enough for budget estimation purposes).

    Implements the Tokenizer protocol via structural subtyping.
    """

    def __init__(self, encoding_name: str = "cl100k_base") -> None:
        self._encoding = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        return len(self._encoding.encode(text))

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token limit."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])


_default_counter: TiktokenCounter | None = None
_counter_lock = threading.Lock()


def get_default_counter() -> TiktokenCounter:
    """Get or create the default TiktokenCounter singleton (thread-safe)."""
    global _default_counter
    if _default_counter is None:
        with _counter_lock:
            if _default_counter is None:
                _default_counter = TiktokenCounter()
    return _default_counter
