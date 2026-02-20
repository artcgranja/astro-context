"""Tests for SlidingWindowMemory with pluggable eviction policies and recency scorers."""

from __future__ import annotations

from astro_context.memory.decay import ExponentialRecencyScorer, LinearRecencyScorer
from astro_context.memory.eviction import FIFOEviction, ImportanceEviction, PairedEviction
from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.models.context import SourceType
from astro_context.models.memory import ConversationTurn
from tests.conftest import FakeTokenizer


def _make_memory(
    max_tokens: int = 100,
    eviction_policy: object = None,
    recency_scorer: object = None,
    on_evict: object = None,
) -> SlidingWindowMemory:
    """Create a SlidingWindowMemory with FakeTokenizer and optional policy/scorer."""
    return SlidingWindowMemory(
        max_tokens=max_tokens,
        tokenizer=FakeTokenizer(),
        on_evict=on_evict,  # type: ignore[arg-type]
        eviction_policy=eviction_policy,  # type: ignore[arg-type]
        recency_scorer=recency_scorer,  # type: ignore[arg-type]
    )


# ---- Default behavior preserved ----


class TestDefaultBehaviorPreserved:
    """When no policy or scorer is provided, behavior matches the original implementation."""

    def test_default_fifo_eviction_without_policy(self) -> None:
        """Without an eviction policy, oldest turns are evicted first (FIFO)."""
        mem = _make_memory(max_tokens=3)
        mem.add_turn("user", "first")        # 1 token
        mem.add_turn("assistant", "second")   # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens -> evict first two

        remaining = [t.content for t in mem.turns]
        assert "first" not in remaining
        assert "second" not in remaining
        assert "three four five" in remaining

    def test_default_linear_recency_scoring(self) -> None:
        """Without a recency scorer, scores range from 0.5 (oldest) to 1.0 (newest)."""
        mem = _make_memory(max_tokens=100)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        assert len(items) == 3
        assert items[0].score == 0.5
        assert items[2].score == 1.0
        # Middle should be 0.75
        assert abs(items[1].score - 0.75) < 0.01

    def test_single_turn_default_score(self) -> None:
        """A single turn gets score 0.5 with default scoring (0.5 + 0.5 * 0/0)."""
        mem = _make_memory(max_tokens=100)
        mem.add_turn("user", "hello")
        items = mem.to_context_items()
        # Default formula: 0.5 + 0.5 * (0 / max(1, 0)) = 0.5
        assert items[0].score == 0.5


# ---- FIFOEviction policy ----


class TestFIFOEvictionPolicy:
    """SlidingWindowMemory with explicit FIFOEviction behaves identically to default."""

    def test_fifo_policy_evicts_oldest(self) -> None:
        policy = FIFOEviction()
        mem = _make_memory(max_tokens=3, eviction_policy=policy)
        mem.add_turn("user", "first")        # 1 token
        mem.add_turn("assistant", "second")   # 1 token
        mem.add_turn("user", "three four five")  # 3 tokens

        remaining = [t.content for t in mem.turns]
        assert "first" not in remaining
        assert "second" not in remaining
        assert "three four five" in remaining

    def test_fifo_policy_preserves_newest(self) -> None:
        policy = FIFOEviction()
        mem = _make_memory(max_tokens=5, eviction_policy=policy)
        mem.add_turn("user", "a")
        mem.add_turn("assistant", "b")
        mem.add_turn("user", "c")
        mem.add_turn("assistant", "d")
        # 4 tokens total. Adding a 3-token message needs to free 2 tokens.
        mem.add_turn("user", "e f g")  # 3 tokens

        assert mem.turns[-1].content == "e f g"
        assert mem.total_tokens <= 5

    def test_fifo_policy_no_eviction_when_fits(self) -> None:
        policy = FIFOEviction()
        mem = _make_memory(max_tokens=100, eviction_policy=policy)
        mem.add_turn("user", "hello")
        mem.add_turn("assistant", "world")
        assert len(mem.turns) == 2


# ---- ImportanceEviction policy ----


class TestImportanceEvictionPolicy:
    """SlidingWindowMemory with ImportanceEviction evicts least important turns."""

    def test_importance_evicts_least_important(self) -> None:
        # Importance by content: "important"=10, "trivial"=1, "medium"=5
        importance_map = {"important": 10.0, "trivial": 1.0, "medium": 5.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: importance_map.get(t.content, 0.0)
        )
        mem = _make_memory(max_tokens=3, eviction_policy=policy)
        mem.add_turn("user", "important")    # 1 token, importance=10
        mem.add_turn("assistant", "trivial") # 1 token, importance=1
        mem.add_turn("user", "medium")       # 1 token, importance=5
        # Now at 3 tokens. Adding a 2-token message needs to free 2 tokens.
        mem.add_turn("assistant", "new message")  # 2 tokens

        remaining = [t.content for t in mem.turns]
        # "trivial" (importance=1) should be evicted first, then "medium" (5)
        assert "trivial" not in remaining
        assert "new message" in remaining
        # "important" should survive (highest importance)
        assert "important" in remaining

    def test_importance_keeps_high_value_turns(self) -> None:
        # All turns have different importance
        scores = {"a": 1.0, "b": 5.0, "c": 3.0, "d": 4.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: scores.get(t.content, 0.0)
        )
        mem = _make_memory(max_tokens=3, eviction_policy=policy)
        mem.add_turn("user", "a")       # importance=1
        mem.add_turn("assistant", "b")   # importance=5
        mem.add_turn("user", "c")        # importance=3
        # 3 tokens. Adding a 2-token message needs to free 2.
        mem.add_turn("assistant", "d e")  # 2 tokens

        remaining = [t.content for t in mem.turns]
        # "a" (1) and "c" (3) should be evicted, "b" (5) should survive
        assert "b" in remaining
        assert "d e" in remaining


# ---- PairedEviction policy ----


class TestPairedEvictionPolicy:
    """SlidingWindowMemory with PairedEviction evicts user/assistant pairs together."""

    def test_paired_evicts_full_pair(self) -> None:
        policy = PairedEviction()
        mem = _make_memory(max_tokens=5, eviction_policy=policy)
        mem.add_turn("user", "q1")         # 1 token
        mem.add_turn("assistant", "a1")     # 1 token
        mem.add_turn("user", "q2")          # 1 token
        mem.add_turn("assistant", "a2")     # 1 token
        # 4 tokens. Adding 3-token message needs to free 2 tokens.
        mem.add_turn("user", "big question here")  # 3 tokens

        remaining = [t.content for t in mem.turns]
        # First pair (q1, a1) should be evicted together
        assert "q1" not in remaining
        assert "a1" not in remaining
        # Second pair should survive
        assert "q2" in remaining
        assert "a2" in remaining
        assert "big question here" in remaining

    def test_paired_evicts_multiple_pairs_if_needed(self) -> None:
        policy = PairedEviction()
        mem = _make_memory(max_tokens=5, eviction_policy=policy)
        mem.add_turn("user", "q1")
        mem.add_turn("assistant", "a1")
        mem.add_turn("user", "q2")
        mem.add_turn("assistant", "a2")
        # 4 tokens. Adding 5-token message needs to free 4 tokens.
        mem.add_turn("user", "one two three four five")  # 5 tokens

        remaining = [t.content for t in mem.turns]
        # Both pairs should be evicted
        assert "q1" not in remaining
        assert "a1" not in remaining
        assert "q2" not in remaining
        assert "a2" not in remaining
        assert "one two three four five" in remaining

    def test_paired_handles_lone_turns(self) -> None:
        policy = PairedEviction()
        mem = _make_memory(max_tokens=4, eviction_policy=policy)
        mem.add_turn("system", "you are helpful")  # 3 tokens (lone)
        mem.add_turn("user", "hi")                 # 1 token
        # 4 tokens. Adding 2-token message needs to free 2 tokens.
        # System turn is lone (3 tokens), evicting it frees enough.
        mem.add_turn("assistant", "hello there")  # 2 tokens

        remaining = [t.content for t in mem.turns]
        assert "you are helpful" not in remaining
        assert "hi" in remaining
        assert "hello there" in remaining


# ---- ExponentialRecencyScorer ----


class TestExponentialRecencyScorer:
    """SlidingWindowMemory with ExponentialRecencyScorer uses exponential scoring."""

    def test_exponential_scorer_newest_is_one(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        # Newest item should have score 1.0
        assert items[-1].score == 1.0

    def test_exponential_scorer_oldest_is_zero(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        # Oldest item should have score 0.0 (exponential formula: (e^0 - 1)/(e^2 - 1) = 0)
        assert items[0].score == 0.0

    def test_exponential_scorer_steeper_than_linear(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        # Middle score should be less than 0.5 (exponential bias toward recent)
        # Linear would give 0.5 at the midpoint, exponential gives less
        assert items[1].score < 0.5

    def test_exponential_single_turn(self) -> None:
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "only")

        items = mem.to_context_items()
        assert items[0].score == 1.0


# ---- LinearRecencyScorer ----


class TestLinearRecencyScorer:
    """SlidingWindowMemory with LinearRecencyScorer uses configurable linear scoring."""

    def test_linear_scorer_default_matches_builtin(self) -> None:
        """LinearRecencyScorer(min_score=0.5) should match the built-in default."""
        scorer = LinearRecencyScorer(min_score=0.5)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        assert items[0].score == 0.5
        assert items[2].score == 1.0
        assert abs(items[1].score - 0.75) < 0.01

    def test_linear_scorer_custom_min_score(self) -> None:
        """LinearRecencyScorer with min_score=0.0 gives full 0.0-1.0 range."""
        scorer = LinearRecencyScorer(min_score=0.0)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "third")

        items = mem.to_context_items()
        assert items[0].score == 0.0
        assert items[2].score == 1.0
        assert abs(items[1].score - 0.5) < 0.01

    def test_linear_scorer_single_turn(self) -> None:
        scorer = LinearRecencyScorer(min_score=0.3)
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "only")

        items = mem.to_context_items()
        assert items[0].score == 1.0


# ---- Combined eviction + scorer ----


class TestCombinedPolicyAndScorer:
    """SlidingWindowMemory with both custom eviction policy AND custom recency scorer."""

    def test_importance_eviction_with_exponential_scoring(self) -> None:
        scores = {"important": 10.0, "trivial": 1.0, "medium": 5.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: scores.get(t.content, 0.0)
        )
        scorer = ExponentialRecencyScorer(decay_rate=2.0)
        mem = _make_memory(max_tokens=3, eviction_policy=policy, recency_scorer=scorer)

        mem.add_turn("user", "important")     # importance=10
        mem.add_turn("assistant", "trivial")   # importance=1
        mem.add_turn("user", "medium")         # importance=5
        # Evict to fit new message
        mem.add_turn("assistant", "new message")  # 2 tokens

        remaining = [t.content for t in mem.turns]
        assert "important" in remaining
        assert "trivial" not in remaining
        assert "new message" in remaining

        # Verify exponential scoring is applied
        items = mem.to_context_items()
        if len(items) >= 2:
            # Newest should be 1.0
            assert items[-1].score == 1.0

    def test_paired_eviction_with_linear_scoring(self) -> None:
        policy = PairedEviction()
        scorer = LinearRecencyScorer(min_score=0.2)
        mem = _make_memory(max_tokens=5, eviction_policy=policy, recency_scorer=scorer)

        mem.add_turn("user", "q1")
        mem.add_turn("assistant", "a1")
        mem.add_turn("user", "q2")
        mem.add_turn("assistant", "a2")
        mem.add_turn("user", "big question here")  # 3 tokens -> evict pair

        remaining = [t.content for t in mem.turns]
        assert "q1" not in remaining
        assert "a1" not in remaining

        # Verify linear scoring with custom min_score is applied
        items = mem.to_context_items()
        assert items[0].score >= 0.2
        assert items[-1].score == 1.0


# ---- on_evict callback with custom policies ----


class TestOnEvictWithCustomPolicy:
    """on_evict callback fires correctly when using custom eviction policies."""

    def test_on_evict_fires_with_fifo_policy(self) -> None:
        evicted_batches: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted_batches.append(turns)

        policy = FIFOEviction()
        mem = _make_memory(max_tokens=3, eviction_policy=policy, on_evict=on_evict)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        mem.add_turn("user", "three four five")

        assert len(evicted_batches) == 1
        evicted_contents = [t.content for t in evicted_batches[0]]
        assert "first" in evicted_contents
        assert "second" in evicted_contents

    def test_on_evict_fires_with_importance_policy(self) -> None:
        evicted_batches: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted_batches.append(turns)

        scores = {"important": 10.0, "trivial": 1.0, "medium": 5.0}
        policy = ImportanceEviction(
            importance_fn=lambda t: scores.get(t.content, 0.0)
        )
        # Budget of 3. important=1, trivial=1, medium=1 -> 3 tokens total.
        # Adding "new" (1 token) needs to free 1 token.
        mem = _make_memory(max_tokens=3, eviction_policy=policy, on_evict=on_evict)
        mem.add_turn("user", "important")
        mem.add_turn("assistant", "trivial")
        mem.add_turn("user", "medium")
        mem.add_turn("assistant", "new")  # 1 token -> needs to free 1

        assert len(evicted_batches) == 1
        evicted_contents = [t.content for t in evicted_batches[0]]
        # "trivial" (importance=1) should be the one evicted
        assert "trivial" in evicted_contents
        assert "important" not in evicted_contents

    def test_on_evict_fires_with_paired_policy(self) -> None:
        evicted_batches: list[list[ConversationTurn]] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            evicted_batches.append(turns)

        policy = PairedEviction()
        mem = _make_memory(max_tokens=5, eviction_policy=policy, on_evict=on_evict)
        mem.add_turn("user", "q1")
        mem.add_turn("assistant", "a1")
        mem.add_turn("user", "q2")
        mem.add_turn("assistant", "a2")
        mem.add_turn("user", "big question here")  # 3 tokens

        assert len(evicted_batches) == 1
        evicted_contents = [t.content for t in evicted_batches[0]]
        assert "q1" in evicted_contents
        assert "a1" in evicted_contents

    def test_no_callback_when_no_eviction_with_policy(self) -> None:
        called: list[bool] = []

        def on_evict(turns: list[ConversationTurn]) -> None:
            called.append(True)

        policy = FIFOEviction()
        mem = _make_memory(max_tokens=100, eviction_policy=policy, on_evict=on_evict)
        mem.add_turn("user", "hello")
        mem.add_turn("assistant", "world")
        assert len(called) == 0

    def test_callback_error_caught_with_custom_policy(self) -> None:
        """on_evict errors are caught even when using a custom policy."""

        def bad_callback(turns: list[ConversationTurn]) -> None:
            msg = "callback exploded"
            raise ValueError(msg)

        policy = FIFOEviction()
        mem = _make_memory(max_tokens=3, eviction_policy=policy, on_evict=bad_callback)
        mem.add_turn("user", "first")
        mem.add_turn("assistant", "second")
        # Should NOT raise despite callback error
        mem.add_turn("user", "three four five")
        assert mem.turns[-1].content == "three four five"


# ---- Context items source and metadata ----


class TestContextItemsWithPolicyAndScorer:
    """Context items have correct source, priority, and metadata regardless of policy/scorer."""

    def test_context_items_have_correct_source(self) -> None:
        scorer = ExponentialRecencyScorer()
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("user", "Hello")
        items = mem.to_context_items()
        assert items[0].source in (SourceType.MEMORY, SourceType.CONVERSATION)

    def test_context_items_have_correct_priority(self) -> None:
        policy = FIFOEviction()
        scorer = LinearRecencyScorer(min_score=0.3)
        mem = _make_memory(max_tokens=100, eviction_policy=policy, recency_scorer=scorer)
        mem.add_turn("user", "Hello")
        items = mem.to_context_items(priority=9)
        assert items[0].priority == 9

    def test_context_items_have_role_metadata(self) -> None:
        scorer = ExponentialRecencyScorer()
        mem = _make_memory(max_tokens=100, recency_scorer=scorer)
        mem.add_turn("assistant", "response text")
        items = mem.to_context_items()
        assert items[0].metadata["role"] == "assistant"

    def test_empty_memory_returns_empty_list(self) -> None:
        policy = ImportanceEviction(importance_fn=lambda t: 0.0)
        scorer = ExponentialRecencyScorer()
        mem = _make_memory(max_tokens=100, eviction_policy=policy, recency_scorer=scorer)
        assert mem.to_context_items() == []
