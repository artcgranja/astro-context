"""Memory decay and recency scoring implementations.

Provides pluggable decay curves for long-term memory retention scoring
and recency scorers for ordering conversation turns within a window.
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_context.models.memory import MemoryEntry


class EbbinghausDecay:
    """Ebbinghaus forgetting curve: R = e^(-t/S).

    Strength ``S`` grows with each access:
    ``S = base_strength + access_count * reinforcement_factor``.
    Time ``t`` is measured in hours since last access.
    """

    __slots__ = ("_base_strength", "_reinforcement_factor")

    def __init__(
        self,
        base_strength: float = 1.0,
        reinforcement_factor: float = 0.5,
    ) -> None:
        if base_strength <= 0:
            msg = "base_strength must be positive"
            raise ValueError(msg)
        if reinforcement_factor < 0:
            msg = "reinforcement_factor must be non-negative"
            raise ValueError(msg)
        self._base_strength = base_strength
        self._reinforcement_factor = reinforcement_factor

    def compute_retention(self, entry: MemoryEntry) -> float:
        """Compute retention score using the Ebbinghaus forgetting curve.

        Parameters:
            entry: A memory entry with ``last_accessed`` and ``access_count``.

        Returns:
            A float in [0.0, 1.0] representing how well the memory is retained.
        """
        now = datetime.now(UTC)
        elapsed_hours = (now - entry.last_accessed).total_seconds() / 3600.0
        strength = self._base_strength + entry.access_count * self._reinforcement_factor
        retention = math.exp(-elapsed_hours / strength)
        return max(0.0, min(1.0, retention))


class LinearDecay:
    """Linear decay from 1.0 to 0.0 over a configurable half-life.

    At ``half_life_hours`` the retention is 0.5. At twice the half-life
    the retention reaches 0.0.
    """

    __slots__ = ("_half_life_hours",)

    def __init__(self, half_life_hours: float = 168.0) -> None:
        if half_life_hours <= 0:
            msg = "half_life_hours must be positive"
            raise ValueError(msg)
        self._half_life_hours = half_life_hours

    def compute_retention(self, entry: MemoryEntry) -> float:
        """Compute retention score using linear interpolation.

        Parameters:
            entry: A memory entry with ``last_accessed``.

        Returns:
            A float in [0.0, 1.0] where 1.0 means just accessed
            and 0.0 means fully decayed.
        """
        now = datetime.now(UTC)
        elapsed_hours = (now - entry.last_accessed).total_seconds() / 3600.0
        # At half_life_hours -> 0.5, at 2*half_life_hours -> 0.0
        retention = 1.0 - (elapsed_hours / (2.0 * self._half_life_hours))
        return max(0.0, min(1.0, retention))


class ExponentialRecencyScorer:
    """Exponential recency scoring with steeper recent-bias than linear.

    Uses ``(e^(rate * normalized) - 1) / (e^(rate) - 1)`` to map
    position to score. The oldest item gets 0.0, the newest gets 1.0,
    with an exponential curve in between.
    """

    __slots__ = ("_decay_rate",)

    def __init__(self, decay_rate: float = 2.0) -> None:
        if decay_rate <= 0:
            msg = "decay_rate must be positive"
            raise ValueError(msg)
        self._decay_rate = decay_rate

    def score(self, index: int, total: int) -> float:
        """Compute exponential recency score for a turn at a given position.

        Parameters:
            index: Zero-based position of the turn (0 = oldest).
            total: Total number of turns in the window.

        Returns:
            A float in [0.0, 1.0] with exponential bias toward recent turns.
        """
        if total <= 1:
            return 1.0
        normalized = index / max(1, total - 1)
        denominator = math.exp(self._decay_rate) - 1.0
        if denominator == 0:
            return normalized
        return (math.exp(self._decay_rate * normalized) - 1.0) / denominator


class LinearRecencyScorer:
    """Linear recency scoring matching the current astro-context default.

    Produces scores from ``min_score`` (oldest) to 1.0 (newest) with
    linear interpolation.
    """

    __slots__ = ("_min_score",)

    def __init__(self, min_score: float = 0.5) -> None:
        if not 0.0 <= min_score < 1.0:
            msg = "min_score must be in [0.0, 1.0)"
            raise ValueError(msg)
        self._min_score = min_score

    def score(self, index: int, total: int) -> float:
        """Compute linear recency score for a turn at a given position.

        Parameters:
            index: Zero-based position of the turn (0 = oldest).
            total: Total number of turns in the window.

        Returns:
            A float in [min_score, 1.0] with linear interpolation.
        """
        if total <= 1:
            return 1.0
        return self._min_score + (1.0 - self._min_score) * (index / max(1, total - 1))
