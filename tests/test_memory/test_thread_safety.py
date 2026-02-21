"""Thread safety tests for SlidingWindowMemory and SummaryBufferMemory."""

from __future__ import annotations

import threading
from collections.abc import Callable

from astro_context.memory.sliding_window import SlidingWindowMemory
from astro_context.memory.summary_buffer import SummaryBufferMemory
from astro_context.models.memory import ConversationTurn
from tests.conftest import FakeTokenizer

NUM_THREADS = 10
TURNS_PER_THREAD = 100


def _make_sliding_window(max_tokens: int = 100_000) -> SlidingWindowMemory:
    """Create a SlidingWindowMemory with a large token budget for concurrency tests."""
    return SlidingWindowMemory(max_tokens=max_tokens, tokenizer=FakeTokenizer())


def _run_threads(
    targets: list[Callable[[], None]],
) -> None:
    """Start all callables as threads and wait for them to finish."""
    threads = [threading.Thread(target=fn) for fn in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


def _sliding_window_writer(
    mem: SlidingWindowMemory,
    barrier: threading.Barrier,
    thread_id: int,
    errors: list[Exception] | None = None,
) -> None:
    """Write TURNS_PER_THREAD turns into a SlidingWindowMemory."""
    barrier.wait()
    for i in range(TURNS_PER_THREAD):
        try:
            mem.add_turn("user", f"t{thread_id}-{i}")
        except Exception as exc:
            if errors is not None:
                errors.append(exc)


def _sliding_window_reader(
    mem: SlidingWindowMemory,
    barrier: threading.Barrier,
    errors: list[Exception],
) -> None:
    """Read from a SlidingWindowMemory repeatedly."""
    barrier.wait()
    for _ in range(TURNS_PER_THREAD):
        try:
            _ = mem.turns
            _ = mem.total_tokens
            _ = mem.to_context_items()
        except Exception as exc:
            errors.append(exc)


def _summary_buffer_writer(
    mem: SummaryBufferMemory,
    barrier: threading.Barrier,
    thread_id: int,
    errors: list[Exception] | None = None,
) -> None:
    """Write TURNS_PER_THREAD messages into a SummaryBufferMemory."""
    barrier.wait()
    for i in range(TURNS_PER_THREAD):
        try:
            mem.add_message("user", f"t{thread_id}-{i}")
        except Exception as exc:
            if errors is not None:
                errors.append(exc)


def _summary_buffer_reader(
    mem: SummaryBufferMemory,
    barrier: threading.Barrier,
    errors: list[Exception],
) -> None:
    """Read from a SummaryBufferMemory repeatedly."""
    barrier.wait()
    for _ in range(TURNS_PER_THREAD):
        try:
            _ = mem.turns
            _ = mem.total_tokens
            _ = mem.summary
            _ = mem.to_context_items()
        except Exception as exc:
            errors.append(exc)


class TestSlidingWindowThreadSafety:
    """Verify SlidingWindowMemory is safe under concurrent access."""

    def test_concurrent_add_turn_no_lost_turns(self) -> None:
        """10 threads each add 100 turns; none should be lost."""
        mem = _make_sliding_window()
        barrier = threading.Barrier(NUM_THREADS)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(TURNS_PER_THREAD):
                role = "user" if i % 2 == 0 else "assistant"
                mem.add_turn(role, f"t{thread_id}-msg{i}")

        _run_threads([lambda tid=t: writer(tid) for t in range(NUM_THREADS)])

        expected_turns = NUM_THREADS * TURNS_PER_THREAD
        assert len(mem.turns) == expected_turns

    def test_concurrent_add_turn_token_count_consistent(self) -> None:
        """Total tokens must equal the sum of individual turn token counts."""
        mem = _make_sliding_window()
        barrier = threading.Barrier(NUM_THREADS)

        _run_threads([
            lambda tid=t: _sliding_window_writer(mem, barrier, tid)
            for t in range(NUM_THREADS)
        ])

        turns = mem.turns
        expected_tokens = sum(t.token_count for t in turns)
        assert mem.total_tokens == expected_tokens

    def test_concurrent_reads_and_writes(self) -> None:
        """Concurrent readers and writers must not crash."""
        mem = _make_sliding_window()
        barrier = threading.Barrier(NUM_THREADS)
        errors: list[Exception] = []

        targets: list[Callable[[], None]] = [
            lambda tid=t: _sliding_window_writer(mem, barrier, tid, errors)
            for t in range(NUM_THREADS // 2)
        ]
        targets.extend(
            lambda: _sliding_window_reader(mem, barrier, errors)
            for _ in range(NUM_THREADS // 2)
        )
        _run_threads(targets)

        assert errors == [], f"Concurrent read/write errors: {errors}"

    def test_concurrent_add_turn_with_eviction(self) -> None:
        """Concurrent adds with a small budget trigger eviction without corruption."""
        mem = _make_sliding_window(max_tokens=100)
        barrier = threading.Barrier(NUM_THREADS)

        _run_threads([
            lambda tid=t: _sliding_window_writer(mem, barrier, tid)
            for t in range(NUM_THREADS)
        ])

        # Token count must not exceed budget
        assert mem.total_tokens <= mem.max_tokens
        # Token count must match sum of stored turns
        turns = mem.turns
        expected_tokens = sum(t.token_count for t in turns)
        assert mem.total_tokens == expected_tokens


class TestSummaryBufferThreadSafety:
    """Verify SummaryBufferMemory is safe under concurrent access."""

    def test_concurrent_add_message_no_crash(self) -> None:
        """10 threads each add 100 messages; no crash or data corruption."""
        tokenizer = FakeTokenizer()
        mem = SummaryBufferMemory(
            max_tokens=100_000,
            compact_fn=lambda turns: "; ".join(t.content for t in turns),
            tokenizer=tokenizer,
        )
        barrier = threading.Barrier(NUM_THREADS)

        _run_threads([
            lambda tid=t: _summary_buffer_writer(mem, barrier, tid)
            for t in range(NUM_THREADS)
        ])

        expected_turns = NUM_THREADS * TURNS_PER_THREAD
        assert len(mem.turns) == expected_turns

    def test_concurrent_add_turn_prebuilt(self) -> None:
        """Concurrent add_turn with pre-built ConversationTurn objects."""
        tokenizer = FakeTokenizer()
        mem = SummaryBufferMemory(
            max_tokens=100_000,
            compact_fn=lambda turns: "; ".join(t.content for t in turns),
            tokenizer=tokenizer,
        )
        barrier = threading.Barrier(NUM_THREADS)

        def writer(thread_id: int) -> None:
            barrier.wait()
            for i in range(TURNS_PER_THREAD):
                turn = ConversationTurn(
                    role="user",
                    content=f"t{thread_id}-{i}",
                    token_count=tokenizer.count_tokens(f"t{thread_id}-{i}"),
                )
                mem.add_turn(turn)

        _run_threads([lambda tid=t: writer(tid) for t in range(NUM_THREADS)])

        expected_turns = NUM_THREADS * TURNS_PER_THREAD
        assert len(mem.turns) == expected_turns

    def test_concurrent_reads_and_writes_summary_buffer(self) -> None:
        """Concurrent readers and writers must not crash on SummaryBufferMemory."""
        tokenizer = FakeTokenizer()
        mem = SummaryBufferMemory(
            max_tokens=100_000,
            compact_fn=lambda turns: "; ".join(t.content for t in turns),
            tokenizer=tokenizer,
        )
        barrier = threading.Barrier(NUM_THREADS)
        errors: list[Exception] = []

        targets: list[Callable[[], None]] = [
            lambda tid=t: _summary_buffer_writer(mem, barrier, tid, errors)
            for t in range(NUM_THREADS // 2)
        ]
        targets.extend(
            lambda: _summary_buffer_reader(mem, barrier, errors)
            for _ in range(NUM_THREADS // 2)
        )
        _run_threads(targets)

        assert errors == [], f"Concurrent read/write errors: {errors}"

    def test_concurrent_add_with_eviction_and_summary(self) -> None:
        """Concurrent adds with eviction must produce a valid summary."""
        tokenizer = FakeTokenizer()

        def capture_compact(turns: list[ConversationTurn]) -> str:
            return "; ".join(t.content for t in turns)

        # Small budget to force eviction
        mem = SummaryBufferMemory(
            max_tokens=100,
            compact_fn=capture_compact,
            tokenizer=tokenizer,
        )
        barrier = threading.Barrier(NUM_THREADS)

        _run_threads([
            lambda tid=t: _summary_buffer_writer(mem, barrier, tid)
            for t in range(NUM_THREADS)
        ])

        # Token count must not exceed budget
        assert mem.total_tokens <= 100
        # Summary should have been created (evictions happened)
        assert mem.summary is not None
        # Token count must match stored turns
        turns = mem.turns
        expected_tokens = sum(t.token_count for t in turns)
        assert mem.total_tokens == expected_tokens
