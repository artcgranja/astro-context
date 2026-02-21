"""Shared math utilities for astro-context."""

from __future__ import annotations


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors without numpy.

    Returns a value in [-1.0, 1.0]. Returns 0.0 if either vector has zero norm.
    Raises ``ValueError`` if the vectors have different dimensionality.
    """
    if len(a) != len(b):
        msg = "vectors must have the same dimensionality"
        raise ValueError(msg)

    dot = sum((x * y for x, y in zip(a, b, strict=True)), 0.0)
    norm_a = sum((x * x for x in a), 0.0) ** 0.5
    norm_b = sum((x * x for x in b), 0.0) ** 0.5

    if norm_a == 0 or norm_b == 0:
        return 0.0

    similarity: float = dot / (norm_a * norm_b)
    return max(-1.0, min(1.0, similarity))
