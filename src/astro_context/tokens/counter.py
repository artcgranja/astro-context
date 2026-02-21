"""Token counting implementations."""

from __future__ import annotations

import functools


class TiktokenCounter:
    """Token counter using OpenAI's tiktoken library.

    Default encoding is cl100k_base (used by GPT-4, Claude tokenizers
    are similar enough for budget estimation purposes).

    Implements the Tokenizer protocol via structural subtyping.

    The tiktoken import is deferred to ``__init__`` so that importing this
    module does not trigger BPE data loading when callers supply their own
    :class:`~astro_context.protocols.tokenizer.Tokenizer` implementation.
    """

    __slots__ = ("_cache", "_encoding", "_max_cache_size")

    def __init__(
        self, encoding_name: str = "cl100k_base", max_cache_size: int = 10_000
    ) -> None:
        try:
            import tiktoken
        except ImportError:
            msg = (
                "tiktoken is required for the default tokenizer. "
                "Install it with: pip install astro-context[tiktoken] "
                "or pip install tiktoken"
            )
            raise ImportError(msg) from None

        self._encoding = tiktoken.get_encoding(encoding_name)
        self._max_cache_size = max_cache_size
        self._cache: dict[str, int] = {}

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string."""
        if text in self._cache:
            return self._cache[text]
        count = len(self._encoding.encode(text))
        # Only cache strings under 10k chars to avoid memory bloat
        if len(text) < 10_000:
            if len(self._cache) >= self._max_cache_size:
                self._cache.clear()
            self._cache[text] = count
        return count

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within a token limit."""
        tokens = self._encoding.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._encoding.decode(tokens[:max_tokens])

    def __repr__(self) -> str:
        return f"{type(self).__name__}(encoding={self._encoding.name!r})"


@functools.cache
def get_default_counter() -> TiktokenCounter:
    """Get or create the default TiktokenCounter singleton.

    Uses :func:`functools.cache` which is inherently thread-safe via the
    GIL and is the idiomatic Python 3.9+ pattern for singleton factories.

    Call ``get_default_counter.cache_clear()`` to reset the singleton
    (useful in tests).

    Raises:
        ImportError: If tiktoken is not installed.
    """
    try:
        return TiktokenCounter()
    except ImportError:
        msg = (
            "tiktoken is required for the default tokenizer. "
            "Install it with: pip install astro-context[tiktoken] "
            "or pip install tiktoken"
        )
        raise ImportError(msg) from None
