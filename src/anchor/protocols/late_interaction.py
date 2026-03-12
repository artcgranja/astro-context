"""Protocol for token-level encoders used in late interaction models."""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class TokenLevelEncoder(Protocol):
    """Encodes text into per-token embeddings for late interaction scoring.

    Unlike standard encoders that produce a single vector per document,
    token-level encoders produce one embedding per token, enabling
    fine-grained matching (e.g. ColBERT MaxSim).

    Parameters:
        text: The text to encode.

    Returns:
        A list of embeddings, one per token. Each embedding is a list of floats.
    """

    def encode_tokens(self, text: str) -> list[list[float]]:
        """Encode text into per-token embeddings."""
        ...
