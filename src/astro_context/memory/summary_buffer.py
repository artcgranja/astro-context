"""Two-tier memory: recent turns verbatim plus a running summary of evicted turns.

``SummaryBufferMemory`` wraps a ``SlidingWindowMemory`` and uses its
``on_evict`` hook to capture evicted turns. The evicted turns are
passed to a user-provided compaction function that produces a summary
string. The summary is stored alongside the live window and emitted
as a ``ContextItem`` with a configurable priority (default 6, between
memory at 7 and retrieval at 5).

Two summarization strategies are supported via mutually exclusive
constructor parameters:

- ``compact_fn`` -- simple summarization that receives only the newly
  evicted turns.
- ``progressive_compact_fn`` -- progressive summarization that also
  receives the previous summary so the user can incrementally refine it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from astro_context.models.context import ContextItem, SourceType
from astro_context.models.memory import ConversationTurn, Role
from astro_context.tokens.counter import get_default_counter

from .sliding_window import SlidingWindowMemory

if TYPE_CHECKING:
    from astro_context.protocols.tokenizer import Tokenizer


class SummaryBufferMemory:
    """A two-tier memory that keeps recent turns verbatim AND a running summary.

    When the internal ``SlidingWindowMemory`` evicts turns to stay within
    its token budget, this class intercepts the eviction and calls the
    user-provided compaction function to produce a summary of the
    evicted content. The summary is stored as a single string and
    replaced on each subsequent eviction.

    Exactly one of ``compact_fn`` or ``progressive_compact_fn`` must be
    provided:

    - ``compact_fn(turns) -> str`` receives only the newly evicted turns.
    - ``progressive_compact_fn(turns, previous_summary) -> str`` also
      receives the current summary (or ``None`` on the first eviction),
      enabling progressive summarization patterns.
    """

    __slots__ = (
        "_compact_fn",
        "_progressive_compact_fn",
        "_summary",
        "_summary_priority",
        "_summary_tokens",
        "_tokenizer",
        "_window",
    )

    def __init__(
        self,
        max_tokens: int,
        compact_fn: Callable[[list[ConversationTurn]], str] | None = None,
        progressive_compact_fn: (
            Callable[[list[ConversationTurn], str | None], str] | None
        ) = None,
        tokenizer: Tokenizer | None = None,
        summary_priority: int = 6,
    ) -> None:
        if compact_fn is None and progressive_compact_fn is None:
            msg = "exactly one of compact_fn or progressive_compact_fn must be provided"
            raise ValueError(msg)
        if compact_fn is not None and progressive_compact_fn is not None:
            msg = "exactly one of compact_fn or progressive_compact_fn must be provided"
            raise ValueError(msg)
        if max_tokens <= 0:
            msg = "max_tokens must be a positive integer"
            raise ValueError(msg)

        self._compact_fn = compact_fn
        self._progressive_compact_fn = progressive_compact_fn
        self._tokenizer: Tokenizer = tokenizer or get_default_counter()
        self._summary_priority = summary_priority
        self._summary: str | None = None
        self._summary_tokens: int = 0
        self._window = SlidingWindowMemory(
            max_tokens=max_tokens,
            tokenizer=self._tokenizer,
            on_evict=self._handle_eviction,
        )

    def __repr__(self) -> str:
        mode = "progressive" if self._progressive_compact_fn is not None else "simple"
        return (
            f"SummaryBufferMemory(mode={mode!r}, "
            f"window={self._window!r}, "
            f"summary_tokens={self._summary_tokens})"
        )

    def _handle_eviction(self, evicted_turns: list[ConversationTurn]) -> None:
        """Callback invoked by the sliding window when turns are evicted."""
        if self._progressive_compact_fn is not None:
            self._summary = self._progressive_compact_fn(evicted_turns, self._summary)
        elif self._compact_fn is not None:
            self._summary = self._compact_fn(evicted_turns)

        if self._summary is not None:
            self._summary_tokens = self._tokenizer.count_tokens(self._summary)
        else:
            self._summary_tokens = 0

    def add_turn(self, turn: ConversationTurn) -> None:
        """Add a pre-built conversation turn to the memory.

        If the turn causes the sliding window to exceed its token budget,
        oldest turns are evicted and the compaction function is called
        to update the running summary.

        Parameters:
            turn: A ``ConversationTurn`` to append.
        """
        self._window.add_turn(
            role=turn.role,
            content=turn.content,
            **turn.metadata,
        )

    def add_message(self, role: Role, content: str, **metadata: object) -> ConversationTurn:
        """Add a message by role and content, returning the created turn.

        This is a convenience method matching ``SlidingWindowMemory.add_turn``.

        Parameters:
            role: The speaker role (``"user"``, ``"assistant"``, etc.).
            content: The message text.
            **metadata: Arbitrary metadata attached to the turn.

        Returns:
            The ``ConversationTurn`` that was actually stored (may be
            truncated if it exceeds the window budget).
        """
        return self._window.add_turn(role, content, **metadata)

    def to_context_items(self, priority: int = 7) -> list[ContextItem]:
        """Convert the summary and live window into context items.

        The summary (if present) is emitted first with
        ``summary_priority``. Window items follow with the given
        *priority* and recency-weighted scores.

        Parameters:
            priority: Priority for the live conversation items.

        Returns:
            A list of ``ContextItem`` instances ready for the pipeline.
        """
        items: list[ContextItem] = []

        if self._summary is not None:
            summary_item = ContextItem(
                content=self._summary,
                source=SourceType.CONVERSATION,
                score=0.5,
                priority=self._summary_priority,
                token_count=self._summary_tokens,
                metadata={"role": "system", "summary": True},
            )
            items.append(summary_item)

        items.extend(self._window.to_context_items(priority=priority))
        return items

    @property
    def summary(self) -> str | None:
        """The current running summary of evicted turns, or ``None``."""
        return self._summary

    @property
    def summary_tokens(self) -> int:
        """Token count of the current summary."""
        return self._summary_tokens

    @property
    def turns(self) -> list[ConversationTurn]:
        """The live conversation turns still in the sliding window."""
        return self._window.turns

    @property
    def total_tokens(self) -> int:
        """Total tokens across the live window (excludes summary)."""
        return self._window.total_tokens

    def clear(self) -> None:
        """Clear both the sliding window and the running summary."""
        self._window.clear()
        self._summary = None
        self._summary_tokens = 0
