"""Tokenizer protocol for token counting abstraction."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    """Protocol for token counting.

    The default implementation uses tiktoken, but users can provide
    any tokenizer (e.g., HuggingFace tokenizers, sentencepiece).
    """

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text string.

        Parameters:
            text: The input text to tokenize and count.

        Returns:
            The total number of tokens as determined by the underlying
            tokenization scheme (e.g., BPE, SentencePiece, whitespace).
        """
        ...

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text so that it contains at most ``max_tokens`` tokens.

        The truncation preserves a prefix of the original text; trailing
        tokens beyond the limit are removed.

        Parameters:
            text: The input text to truncate.
            max_tokens: The maximum number of tokens allowed in the
                returned string.

        Returns:
            A string whose token count is less than or equal to
            ``max_tokens``.  If the original text is already within the
            limit, it is returned unchanged.
        """
        ...
