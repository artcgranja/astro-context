"""Pluggable eviction policies for conversation memory.

Each policy implements the ``EvictionPolicy`` protocol by providing a
``select_for_eviction`` method that returns indices of turns to remove
when the memory window exceeds its token budget.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from astro_context.models.memory import ConversationTurn


class FIFOEviction:
    """Default FIFO eviction -- evict oldest turns first.

    This matches the built-in behaviour of ``SlidingWindowMemory`` and
    is provided as an explicit, composable implementation of the
    ``EvictionPolicy`` protocol.
    """

    __slots__ = ()

    def select_for_eviction(
        self,
        turns: list[ConversationTurn],
        tokens_to_free: int,
    ) -> list[int]:
        """Select oldest turns until enough tokens are freed.

        Parameters:
            turns: The current list of conversation turns (oldest first).
            tokens_to_free: Minimum number of tokens to reclaim.

        Returns:
            A list of zero-based indices into *turns* that should be evicted.
        """
        indices: list[int] = []
        freed = 0
        for i, turn in enumerate(turns):
            if freed >= tokens_to_free:
                break
            indices.append(i)
            freed += turn.token_count
        return indices


class ImportanceEviction:
    """Evict turns with the lowest importance scores first.

    A user-provided ``importance_fn`` assigns a numeric score to each
    turn. Turns with the lowest scores are evicted until the required
    number of tokens has been freed.
    """

    __slots__ = ("_importance_fn",)

    def __init__(self, importance_fn: Callable[[ConversationTurn], float]) -> None:
        self._importance_fn = importance_fn

    def select_for_eviction(
        self,
        turns: list[ConversationTurn],
        tokens_to_free: int,
    ) -> list[int]:
        """Select least-important turns until enough tokens are freed.

        Parameters:
            turns: The current list of conversation turns.
            tokens_to_free: Minimum number of tokens to reclaim.

        Returns:
            A list of zero-based indices into *turns* ordered by ascending
            importance (least important first).
        """
        scored = sorted(
            enumerate(turns),
            key=lambda pair: self._importance_fn(pair[1]),
        )
        indices: list[int] = []
        freed = 0
        for idx, turn in scored:
            if freed >= tokens_to_free:
                break
            indices.append(idx)
            freed += turn.token_count
        return indices


class PairedEviction:
    """Evict user+assistant turn pairs together to prevent orphaned context.

    Consecutive turns with roles ``"user"`` followed by ``"assistant"``
    are treated as a single pair. A lone turn at the boundary that does
    not form a complete pair is still eligible for eviction on its own.
    Pairs are evicted oldest-first.
    """

    __slots__ = ()

    def select_for_eviction(
        self,
        turns: list[ConversationTurn],
        tokens_to_free: int,
    ) -> list[int]:
        """Select oldest turn-pairs until enough tokens are freed.

        Parameters:
            turns: The current list of conversation turns (oldest first).
            tokens_to_free: Minimum number of tokens to reclaim.

        Returns:
            A list of zero-based indices into *turns* that should be evicted,
            keeping user/assistant pairs intact.
        """
        # Build groups of (indices, total_tokens) representing pairs or singles
        groups: list[tuple[list[int], int]] = []
        i = 0
        while i < len(turns):
            if (
                turns[i].role == "user"
                and i + 1 < len(turns)
                and turns[i + 1].role == "assistant"
            ):
                pair_tokens = turns[i].token_count + turns[i + 1].token_count
                groups.append(([i, i + 1], pair_tokens))
                i += 2
            else:
                groups.append(([i], turns[i].token_count))
                i += 1

        indices: list[int] = []
        freed = 0
        for group_indices, group_tokens in groups:
            if freed >= tokens_to_free:
                break
            indices.extend(group_indices)
            freed += group_tokens
        return indices
