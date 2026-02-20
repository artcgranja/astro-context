"""Token-aware sliding window memory for conversation history."""

from __future__ import annotations

from collections import deque

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.memory import ConversationTurn, Role
from astro_context.protocols.tokenizer import Tokenizer
from astro_context.tokens.counter import get_default_counter


class SlidingWindowMemory:
    """Maintains a rolling window of conversation turns within a token budget.

    When adding turns that would exceed the token limit, oldest turns are
    evicted first (FIFO). This is the simplest and most common memory strategy.
    """

    __slots__ = ("_max_tokens", "_tokenizer", "_total_tokens", "_turns")

    def __init__(
        self,
        max_tokens: int = 4096,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)
        self._max_tokens = max_tokens
        self._tokenizer = tokenizer or get_default_counter()
        self._turns: deque[ConversationTurn] = deque()
        self._total_tokens: int = 0

    def __repr__(self) -> str:
        return (
            f"SlidingWindowMemory(turns={len(self._turns)}, "
            f"tokens={self._total_tokens}/{self._max_tokens})"
        )

    @property
    def turns(self) -> list[ConversationTurn]:
        return list(self._turns)

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    def add_turn(self, role: Role, content: str, **metadata: object) -> ConversationTurn:
        """Add a conversation turn, evicting old turns if necessary."""
        token_count = self._tokenizer.count_tokens(content)
        turn = ConversationTurn(
            role=role,
            content=content,
            token_count=token_count,
            metadata=dict(metadata),
        )

        # If this single turn exceeds the budget, truncate it
        if token_count > self._max_tokens:
            truncated = self._tokenizer.truncate_to_tokens(content, self._max_tokens)
            content = truncated
            token_count = self._max_tokens
            turn = ConversationTurn(
                role=role,
                content=content,
                token_count=token_count,
                metadata={**dict(metadata), "truncated": True},
            )

        # Evict oldest turns until the new turn fits
        while self._turns and (self._total_tokens + turn.token_count > self._max_tokens):
            evicted = self._turns.popleft()
            self._total_tokens -= evicted.token_count

        self._turns.append(turn)
        self._total_tokens += turn.token_count
        return turn

    def to_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Convert conversation turns to ContextItems for the pipeline.

        The content is stored as-is without a role prefix. The role is
        available in the item's metadata under the ``"role"`` key so that
        downstream formatters (Anthropic, OpenAI) can set the role via
        their own message structure and avoid the double-role bug
        (e.g. ``{"role": "user", "content": "user: Hello"}``).
        """
        items: list[ContextItem] = []
        num_turns = len(self._turns)
        for i, turn in enumerate(self._turns):
            # Recency-weighted score: oldest=0.5, newest=1.0
            recency_score = 0.5 + 0.5 * (i / max(1, num_turns - 1))
            item = ContextItem(
                content=turn.content,
                source=SourceType.MEMORY,
                score=round(recency_score, 4),
                priority=priority,
                token_count=turn.token_count,
                metadata={"role": turn.role, **turn.metadata},
                created_at=turn.timestamp,
            )
            items.append(item)
        return items

    def clear(self) -> None:
        self._turns.clear()
        self._total_tokens = 0
