#!/usr/bin/env python3
"""Memory Showcase: all memory features in astro-context.

Run with:  python examples/memory_showcase.py

Demonstrates SlidingWindowMemory, eviction policies, SummaryBufferMemory,
MemoryManager, and recency scorers -- all without any API keys or external
services.
"""

from __future__ import annotations

from astro_context import (
    ExponentialRecencyScorer,
    ImportanceEviction,
    LinearRecencyScorer,
    MemoryManager,
    PairedEviction,
    SlidingWindowMemory,
    SummaryBufferMemory,
)
from astro_context.models.memory import ConversationTurn

# ---------------------------------------------------------------------------
# Shared tokenizer (no external dependency)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal tokenizer that counts whitespace-separated words."""

    def count_tokens(self, text: str) -> int:
        return len(text.split()) if text.strip() else 0

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        return " ".join(text.split()[:max_tokens])


tokenizer = WhitespaceTokenizer()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def header(number: int, title: str) -> None:
    """Print a boxed section header."""
    inner = f"  {number}. {title}"
    width = max(len(inner) + 2, 40)
    inner = inner.ljust(width - 2)
    print()
    print(f"\u2554{'═' * width}\u2557")
    print(f"\u2551{inner}\u2551")
    print(f"\u255a{'═' * width}\u255d")
    print()


def subheader(title: str) -> None:
    print(f"--- {title} ---")


# ===========================================================================
# 1. Sliding Window Memory
# ===========================================================================


def demo_sliding_window() -> None:
    header(1, "Sliding Window Memory")

    print("Creating SlidingWindowMemory with max_tokens=100 (word-based).\n")

    evicted_log: list[str] = []

    def on_evict(turns: list[ConversationTurn]) -> None:
        for t in turns:
            evicted_log.append(f"[{t.role}] {t.content[:50]}")

    mem = SlidingWindowMemory(
        max_tokens=100,
        tokenizer=tokenizer,
        on_evict=on_evict,
    )

    conversation = [
        ("user", "Hello! I want to learn about context engineering."),
        (
            "assistant",
            "Great topic! Context engineering is the practice of"
            " assembling information for LLMs.",
        ),
        ("user", "What are the key components?"),
        (
            "assistant",
            "The key components include token budgets, memory"
            " management, retrieval, and formatting.",
        ),
        ("user", "How does memory management work in practice?"),
        (
            "assistant",
            "Memory management uses sliding windows with eviction"
            " policies to keep conversation history within token"
            " limits while preserving the most important context.",
        ),
        ("user", "Can you explain eviction policies?"),
        (
            "assistant",
            "Eviction policies determine which turns to remove"
            " when the token budget is exceeded. FIFO removes"
            " the oldest turns first.",
        ),
        ("user", "What about importance-based eviction?"),
        (
            "assistant",
            "Importance-based eviction scores each turn and"
            " removes the least important ones first rather"
            " than simply the oldest.",
        ),
    ]

    for role, content in conversation:
        mem.add_turn(role, content)
        token_count = tokenizer.count_tokens(content)
        print(
            f"  Added [{role:>9}] ({token_count:2} tokens)"
            f" | Window: {mem.total_tokens}/{mem.max_tokens} tokens,"
            f" {len(mem.turns)} turns"
        )

    print()
    subheader("Evicted turns (FIFO)")
    if evicted_log:
        for entry in evicted_log:
            print(f"  EVICTED: {entry}")
    else:
        print("  (no evictions yet)")

    print()
    subheader("Recency scores on context items")
    items = mem.to_context_items()
    for item in items:
        role = item.metadata.get("role", "?")
        print(f"  score={item.score:.4f}  [{role:>9}]  {item.content[:60]}...")

    print(f"\nTotal context items: {len(items)}")


# ===========================================================================
# 2. Custom EvictionPolicy -- ImportanceEviction
# ===========================================================================


def demo_importance_eviction() -> None:
    header(2, "ImportanceEviction (Custom Policy)")

    print(
        "ImportanceEviction keeps longer (more detailed) messages"
        " and evicts shorter ones first.\n"
    )

    def importance_fn(turn: ConversationTurn) -> float:
        """Longer messages are considered more important."""
        return float(turn.token_count)

    policy = ImportanceEviction(importance_fn=importance_fn)

    mem = SlidingWindowMemory(
        max_tokens=60,
        tokenizer=tokenizer,
        eviction_policy=policy,
    )

    messages = [
        ("user", "Hi"),
        ("assistant", "Hello! How can I help you today?"),
        (
            "user",
            "Tell me about context engineering and all its"
            " components in detail",
        ),
        (
            "assistant",
            "Sure! Context engineering involves careful assembly"
            " of prompts, memory, retrieval results, and system"
            " instructions into a coherent context window for"
            " language models.",
        ),
        ("user", "Thanks for the detailed explanation!"),
    ]

    for role, content in messages:
        mem.add_turn(role, content)
        tokens = tokenizer.count_tokens(content)
        print(
            f"  Added [{role:>9}] ({tokens:2} tokens)"
            f" | Window: {mem.total_tokens}/{mem.max_tokens} tokens"
        )

    print()
    subheader("Remaining turns (short messages evicted first)")
    for turn in mem.turns:
        print(f"  [{turn.role:>9}] ({turn.token_count:2} tokens) {turn.content[:70]}...")


# ===========================================================================
# 3. PairedEviction
# ===========================================================================


def demo_paired_eviction() -> None:
    header(3, "PairedEviction")

    print("PairedEviction evicts user+assistant turn pairs together,")
    print("preventing orphaned context (a question without its answer).\n")

    evicted_pairs: list[str] = []

    def on_evict(turns: list[ConversationTurn]) -> None:
        for t in turns:
            evicted_pairs.append(f"[{t.role}] {t.content[:50]}")

    policy = PairedEviction()
    mem = SlidingWindowMemory(
        max_tokens=50,
        tokenizer=tokenizer,
        eviction_policy=policy,
        on_evict=on_evict,
    )

    pairs = [
        ("user", "What is RAG?"),
        ("assistant", "RAG stands for Retrieval-Augmented Generation."),
        ("user", "How does it work?"),
        ("assistant", "It retrieves relevant documents and includes them in the context."),
        ("user", "What about hybrid RAG?"),
        ("assistant", "Hybrid RAG combines dense and sparse retrieval for better results."),
    ]

    for role, content in pairs:
        mem.add_turn(role, content)

    print("Remaining turns in window:")
    for turn in mem.turns:
        print(f"  [{turn.role:>9}] {turn.content}")

    print()
    print("Evicted turns (always in pairs):")
    for entry in evicted_pairs:
        print(f"  EVICTED: {entry}")

    print()
    print("Notice: user questions and their assistant answers are evicted together.")


# ===========================================================================
# 4. SummaryBufferMemory
# ===========================================================================


def demo_summary_buffer() -> None:
    header(4, "SummaryBufferMemory")

    print("SummaryBufferMemory keeps recent turns verbatim and compacts")
    print("evicted turns into a running summary.\n")

    def compact_fn(evicted: list[ConversationTurn]) -> str:
        """Simple compaction: join evicted turn contents with semicolons."""
        return "Summary of earlier conversation: " + "; ".join(
            f"{t.role} said '{t.content}'" for t in evicted
        )

    sbm = SummaryBufferMemory(
        max_tokens=30,
        compact_fn=compact_fn,
        tokenizer=tokenizer,
    )

    messages = [
        ("user", "What is astro-context?"),
        ("assistant", "A context engineering toolkit for AI apps."),
        ("user", "What does it do?"),
        ("assistant", "It assembles and manages context for LLMs."),
        ("user", "Does it call any LLM?"),
        ("assistant", "No, the library never calls an LLM itself."),
        ("user", "What about memory features?"),
        ("assistant", "It has sliding window memory and summary buffers."),
    ]

    for role, content in messages:
        sbm.add_message(role, content)
        summary_status = "yes" if sbm.summary else "no"
        print(f"  Added [{role:>9}] | Live turns: {len(sbm.turns)}, Summary: {summary_status}")

    print()
    subheader("Running summary (compacted from evicted turns)")
    if sbm.summary:
        print(f"  {sbm.summary[:120]}...")
        print(f"  Summary tokens: {sbm.summary_tokens}")
    else:
        print("  (no summary yet)")

    print()
    subheader("Live turns still in the window")
    for turn in sbm.turns:
        print(f"  [{turn.role:>9}] {turn.content}")

    print()
    subheader("Context items (summary + live turns)")
    items = sbm.to_context_items()
    for item in items:
        is_summary = item.metadata.get("summary", False)
        role = item.metadata.get("role", "?")
        label = "[SUMMARY]" if is_summary else f"[{role:>9}]"
        print(f"  prio={item.priority} score={item.score:.4f}  {label}  {item.content[:60]}...")


# ===========================================================================
# 5. MemoryManager Facade
# ===========================================================================


def demo_memory_manager() -> None:
    header(5, "MemoryManager Facade")

    print("MemoryManager provides a high-level API that coordinates")
    print("conversation memory and produces context items.\n")

    manager = MemoryManager(
        conversation_tokens=80,
        tokenizer=tokenizer,
    )

    print(f"  Type: {manager.conversation_type}")
    print(f"  {manager!r}\n")

    manager.add_user_message("What is the capital of France?")
    manager.add_assistant_message("The capital of France is Paris.")
    manager.add_user_message("And what about Germany?")
    manager.add_assistant_message("The capital of Germany is Berlin.")
    manager.add_user_message("Which one has a larger population?")

    items = manager.get_context_items()

    subheader("Context items from MemoryManager")
    for item in items:
        role = item.metadata.get("role", "?")
        print(
            f"  prio={item.priority}  score={item.score:.4f}  "
            f"[{role:>9}]  {item.content[:60]}"
        )

    print(f"\nTotal items: {len(items)}")

    print()
    subheader("Using SummaryBufferMemory as backend")

    def compact_fn(evicted: list[ConversationTurn]) -> str:
        return "; ".join(t.content for t in evicted)

    sbm = SummaryBufferMemory(
        max_tokens=40,
        compact_fn=compact_fn,
        tokenizer=tokenizer,
    )
    manager2 = MemoryManager(conversation_memory=sbm)

    manager2.add_user_message("Hello!")
    manager2.add_assistant_message("Hi there! How can I help?")
    manager2.add_user_message("What is context engineering?")
    manager2.add_assistant_message("It is the practice of assembling context for LLMs.")

    print(f"  Type: {manager2.conversation_type}")
    items2 = manager2.get_context_items()
    for item in items2:
        is_summary = item.metadata.get("summary", False)
        role = item.metadata.get("role", "?")
        label = "[SUMMARY]" if is_summary else f"[{role:>9}]"
        print(f"  prio={item.priority}  score={item.score:.4f}  {label}  {item.content[:60]}")


# ===========================================================================
# 6. Recency Scorers -- Linear vs. Exponential
# ===========================================================================


def demo_recency_scorers() -> None:
    header(6, "Recency Scorers (Linear vs Exponential)")

    print("Recency scorers control how conversation turns are weighted by")
    print("their position.  More recent turns get higher scores.\n")

    messages = [
        ("user", "First message"),
        ("assistant", "First reply"),
        ("user", "Second message"),
        ("assistant", "Second reply"),
        ("user", "Third message"),
        ("assistant", "Third reply"),
        ("user", "Fourth message"),
        ("assistant", "Fourth reply"),
    ]

    linear_scorer = LinearRecencyScorer(min_score=0.3)
    exp_scorer = ExponentialRecencyScorer(decay_rate=2.0)

    mem_linear = SlidingWindowMemory(
        max_tokens=500,
        tokenizer=tokenizer,
        recency_scorer=linear_scorer,
    )
    mem_exp = SlidingWindowMemory(
        max_tokens=500,
        tokenizer=tokenizer,
        recency_scorer=exp_scorer,
    )

    for role, content in messages:
        mem_linear.add_turn(role, content)
        mem_exp.add_turn(role, content)

    linear_items = mem_linear.to_context_items()
    exp_items = mem_exp.to_context_items()

    print(f"  {'Position':<10} {'Turn':<25} {'Linear':>10} {'Exponential':>12}")
    print(f"  {'--------':<10} {'----':<25} {'------':>10} {'-----------':>12}")

    for i, (li, ei) in enumerate(zip(linear_items, exp_items, strict=True)):
        role = li.metadata.get("role", "?")
        label = f"[{role}] {li.content}"
        print(f"  {i:<10} {label:<25} {li.score:>10.4f} {ei.score:>12.4f}")

    print()
    print("Observation: ExponentialRecencyScorer gives much lower scores to")
    print("older turns, creating a steeper recency bias.  This is useful when")
    print("you want the most recent context to dominate scoring.")

    print()
    subheader("Score distribution comparison")

    linear_oldest = linear_items[0].score
    linear_newest = linear_items[-1].score
    exp_oldest = exp_items[0].score
    exp_newest = exp_items[-1].score

    linear_ratio = linear_newest / max(linear_oldest, 0.001)
    exp_ratio = exp_newest / max(exp_oldest, 0.001)
    print(
        f"  Linear:      oldest={linear_oldest:.4f},"
        f" newest={linear_newest:.4f}, ratio={linear_ratio:.2f}x"
    )
    print(
        f"  Exponential: oldest={exp_oldest:.4f},"
        f" newest={exp_newest:.4f}, ratio={exp_ratio:.2f}x"
    )


# ===========================================================================
# Main
# ===========================================================================


def main() -> None:
    print("=" * 60)
    print("  astro-context Memory Showcase")
    print("  Demonstrating all memory features")
    print("=" * 60)

    demo_sliding_window()
    demo_importance_eviction()
    demo_paired_eviction()
    demo_summary_buffer()
    demo_memory_manager()
    demo_recency_scorers()

    print()
    print("=" * 60)
    print("  All memory features demonstrated successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
