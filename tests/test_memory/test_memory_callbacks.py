"""Tests for MemoryCallback protocol and _fire_memory_callback helper."""

from __future__ import annotations

from typing import Any

from astro_context.memory.callbacks import MemoryCallback, _fire_memory_callback
from astro_context.models.memory import ConversationTurn, MemoryEntry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_turns(n: int = 2) -> list[ConversationTurn]:
    """Create a list of dummy conversation turns."""
    return [
        ConversationTurn(role="user", content=f"turn-{i}")
        for i in range(n)
    ]


def _make_entries(n: int = 2) -> list[MemoryEntry]:
    """Create a list of dummy memory entries."""
    return [
        MemoryEntry(content=f"entry-{i}")
        for i in range(n)
    ]


class RecordingCallback:
    """Callback that records every invocation for assertion."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []

    def on_eviction(
        self, turns: list[ConversationTurn], remaining_tokens: int
    ) -> None:
        self.calls.append(("on_eviction", (turns, remaining_tokens), {}))

    def on_compaction(
        self,
        evicted_turns: list[ConversationTurn],
        summary: str,
        previous_summary: str | None,
    ) -> None:
        self.calls.append(
            ("on_compaction", (evicted_turns, summary, previous_summary), {})
        )

    def on_extraction(
        self, turns: list[ConversationTurn], entries: list[MemoryEntry]
    ) -> None:
        self.calls.append(("on_extraction", (turns, entries), {}))

    def on_consolidation(
        self,
        action: str,
        new_entry: MemoryEntry | None,
        existing_entry: MemoryEntry | None,
    ) -> None:
        self.calls.append(
            ("on_consolidation", (action, new_entry, existing_entry), {})
        )

    def on_decay_prune(
        self, pruned_entries: list[MemoryEntry], threshold: float
    ) -> None:
        self.calls.append(("on_decay_prune", (pruned_entries, threshold), {}))

    def on_expiry_prune(self, pruned_entries: list[MemoryEntry]) -> None:
        self.calls.append(("on_expiry_prune", (pruned_entries,), {}))


class PartialCallback:
    """Callback that only implements a subset of methods."""

    def __init__(self) -> None:
        self.eviction_calls: list[tuple[list[ConversationTurn], int]] = []

    def on_eviction(
        self, turns: list[ConversationTurn], remaining_tokens: int
    ) -> None:
        self.eviction_calls.append((turns, remaining_tokens))


class ExplodingCallback:
    """Callback that raises on every method."""

    def on_eviction(
        self, turns: list[ConversationTurn], remaining_tokens: int
    ) -> None:
        msg = "boom in on_eviction"
        raise RuntimeError(msg)

    def on_compaction(
        self,
        evicted_turns: list[ConversationTurn],
        summary: str,
        previous_summary: str | None,
    ) -> None:
        msg = "boom in on_compaction"
        raise RuntimeError(msg)

    def on_extraction(
        self, turns: list[ConversationTurn], entries: list[MemoryEntry]
    ) -> None:
        msg = "boom in on_extraction"
        raise RuntimeError(msg)

    def on_consolidation(
        self,
        action: str,
        new_entry: MemoryEntry | None,
        existing_entry: MemoryEntry | None,
    ) -> None:
        msg = "boom in on_consolidation"
        raise RuntimeError(msg)

    def on_decay_prune(
        self, pruned_entries: list[MemoryEntry], threshold: float
    ) -> None:
        msg = "boom in on_decay_prune"
        raise RuntimeError(msg)

    def on_expiry_prune(self, pruned_entries: list[MemoryEntry]) -> None:
        msg = "boom in on_expiry_prune"
        raise RuntimeError(msg)


# ---------------------------------------------------------------------------
# Tests: Protocol
# ---------------------------------------------------------------------------


class TestMemoryCallbackProtocol:
    """MemoryCallback protocol is runtime_checkable and works structurally."""

    def test_recording_callback_satisfies_protocol(self) -> None:
        cb = RecordingCallback()
        assert isinstance(cb, MemoryCallback)

    def test_exploding_callback_satisfies_protocol(self) -> None:
        cb = ExplodingCallback()
        assert isinstance(cb, MemoryCallback)

    def test_partial_callback_does_not_satisfy_isinstance(self) -> None:
        """A class that only implements on_eviction does NOT pass isinstance
        because runtime_checkable requires ALL protocol methods to be present.
        However, _fire_memory_callback still works via getattr."""
        cb = PartialCallback()
        # runtime_checkable isinstance checks all methods
        assert not isinstance(cb, MemoryCallback)

    def test_arbitrary_object_does_not_satisfy_protocol(self) -> None:
        assert not isinstance("not a callback", MemoryCallback)
        assert not isinstance(42, MemoryCallback)


# ---------------------------------------------------------------------------
# Tests: _fire_memory_callback
# ---------------------------------------------------------------------------


class TestFireMemoryCallback:
    """_fire_memory_callback dispatches to all callbacks and swallows errors."""

    def test_calls_all_callbacks(self) -> None:
        cb1 = RecordingCallback()
        cb2 = RecordingCallback()
        turns = _make_turns()

        _fire_memory_callback([cb1, cb2], "on_eviction", turns, 100)

        assert len(cb1.calls) == 1
        assert cb1.calls[0][0] == "on_eviction"
        assert cb1.calls[0][1] == (turns, 100)

        assert len(cb2.calls) == 1
        assert cb2.calls[0][0] == "on_eviction"

    def test_swallows_errors(self) -> None:
        """An exploding callback does not prevent subsequent callbacks."""
        exploder = ExplodingCallback()
        recorder = RecordingCallback()
        turns = _make_turns()

        # Should not raise
        _fire_memory_callback(
            [exploder, recorder], "on_eviction", turns, 50
        )

        # The recorder should still have been called
        assert len(recorder.calls) == 1

    def test_swallows_error_and_continues_all(self) -> None:
        """Even if the first callback explodes, all remaining are called."""
        r1 = RecordingCallback()
        r2 = RecordingCallback()
        exploder = ExplodingCallback()

        _fire_memory_callback(
            [r1, exploder, r2], "on_eviction", _make_turns(), 0
        )

        assert len(r1.calls) == 1
        assert len(r2.calls) == 1

    def test_empty_callback_list(self) -> None:
        """Calling with no callbacks is a no-op."""
        _fire_memory_callback([], "on_eviction", _make_turns(), 10)

    def test_missing_method_is_swallowed(self) -> None:
        """If a callback lacks the method entirely, the AttributeError
        is caught and swallowed."""

        class Bare:
            pass

        _fire_memory_callback([Bare()], "on_eviction", _make_turns(), 0)  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# Tests: Individual callback methods receive correct data
# ---------------------------------------------------------------------------


class TestOnEvictionCallback:
    def test_receives_turns_and_remaining_tokens(self) -> None:
        cb = RecordingCallback()
        turns = _make_turns(3)

        _fire_memory_callback([cb], "on_eviction", turns, 42)

        assert len(cb.calls) == 1
        name, args, _ = cb.calls[0]
        assert name == "on_eviction"
        assert args[0] is turns
        assert args[1] == 42


class TestOnCompactionCallback:
    def test_receives_summary_and_previous(self) -> None:
        cb = RecordingCallback()
        turns = _make_turns(2)

        _fire_memory_callback(
            [cb], "on_compaction", turns, "new summary", "old summary"
        )

        name, args, _ = cb.calls[0]
        assert name == "on_compaction"
        assert args[0] is turns
        assert args[1] == "new summary"
        assert args[2] == "old summary"

    def test_receives_none_previous_summary(self) -> None:
        cb = RecordingCallback()
        _fire_memory_callback(
            [cb], "on_compaction", _make_turns(), "first summary", None
        )
        _, args, _ = cb.calls[0]
        assert args[2] is None


class TestOnExtractionCallback:
    def test_receives_turns_and_entries(self) -> None:
        cb = RecordingCallback()
        turns = _make_turns(2)
        entries = _make_entries(3)

        _fire_memory_callback([cb], "on_extraction", turns, entries)

        name, args, _ = cb.calls[0]
        assert name == "on_extraction"
        assert args[0] is turns
        assert args[1] is entries


class TestOnConsolidationCallback:
    def test_receives_action_and_entries(self) -> None:
        cb = RecordingCallback()
        new_entry = MemoryEntry(content="new")
        existing_entry = MemoryEntry(content="old")

        _fire_memory_callback(
            [cb], "on_consolidation", "update", new_entry, existing_entry
        )

        name, args, _ = cb.calls[0]
        assert name == "on_consolidation"
        assert args[0] == "update"
        assert args[1] is new_entry
        assert args[2] is existing_entry

    def test_add_action_with_no_existing(self) -> None:
        cb = RecordingCallback()
        new_entry = MemoryEntry(content="brand new")

        _fire_memory_callback(
            [cb], "on_consolidation", "add", new_entry, None
        )

        _, args, _ = cb.calls[0]
        assert args[0] == "add"
        assert args[2] is None


class TestOnDecayPruneCallback:
    def test_receives_pruned_entries_and_threshold(self) -> None:
        cb = RecordingCallback()
        entries = _make_entries(2)

        _fire_memory_callback([cb], "on_decay_prune", entries, 0.15)

        name, args, _ = cb.calls[0]
        assert name == "on_decay_prune"
        assert args[0] is entries
        assert args[1] == 0.15


class TestOnExpiryPruneCallback:
    def test_receives_pruned_entries(self) -> None:
        cb = RecordingCallback()
        entries = _make_entries(3)

        _fire_memory_callback([cb], "on_expiry_prune", entries)

        name, args, _ = cb.calls[0]
        assert name == "on_expiry_prune"
        assert args[0] is entries


# ---------------------------------------------------------------------------
# Tests: Partial implementation
# ---------------------------------------------------------------------------


class TestPartialImplementation:
    def test_partial_callback_only_handles_implemented_methods(self) -> None:
        """A partial callback works for its implemented methods and
        gracefully handles (via no-op Ellipsis) unimplemented methods."""
        cb = PartialCallback()
        turns = _make_turns()

        _fire_memory_callback([cb], "on_eviction", turns, 99)

        assert len(cb.eviction_calls) == 1
        assert cb.eviction_calls[0] == (turns, 99)

    def test_partial_callback_unimplemented_method_does_not_crash(self) -> None:
        """Calling an unimplemented method on a partial callback uses
        the default no-op from the protocol via getattr fallback."""
        cb = PartialCallback()

        # on_compaction is not implemented on PartialCallback.
        # getattr will raise AttributeError which is swallowed.
        _fire_memory_callback(
            [cb], "on_compaction", _make_turns(), "summary", None  # type: ignore[list-item]
        )
        # Should not crash
