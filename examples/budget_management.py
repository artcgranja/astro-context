"""Example: Token Budget Management. Run with: python examples/budget_management.py

Demonstrates how to use TokenBudget and the factory functions
(default_chat_budget, default_rag_budget, default_agent_budget) to
control how tokens are allocated across different context sources.

Shows how overflow is handled when the total context exceeds the budget.
"""

from __future__ import annotations

from astro_context.memory.manager import MemoryManager
from astro_context.models.budget import BudgetAllocation, TokenBudget
from astro_context.models.budget_defaults import (
    default_agent_budget,
    default_chat_budget,
    default_rag_budget,
)
from astro_context.models.context import SourceType
from astro_context.pipeline.pipeline import ContextPipeline

# ---------------------------------------------------------------------------
# Simple whitespace tokenizer (avoids tiktoken dependency)
# ---------------------------------------------------------------------------


class WhitespaceTokenizer:
    """Minimal tokenizer for demonstration."""

    def count_tokens(self, text: str) -> int:
        if not text or not text.strip():
            return 0
        return len(text.split())

    def truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        words = text.split()
        return " ".join(words[:max_tokens])


# ---------------------------------------------------------------------------
# Example 1: Using preset budgets
# ---------------------------------------------------------------------------


def show_preset_budgets() -> None:
    """Display the three preset budget configurations."""
    print("=== Preset Token Budgets (for 8192 max tokens) ===")
    print()

    for name, factory in [
        ("Chat", default_chat_budget),
        ("RAG", default_rag_budget),
        ("Agent", default_agent_budget),
    ]:
        budget = factory(8192)
        print(f"--- {name} Budget ---")
        print(f"  Total tokens: {budget.total_tokens}")
        print(f"  Reserve tokens: {budget.reserve_tokens}")
        print(f"  Shared pool: {budget.shared_pool}")
        for alloc in budget.allocations:
            print(
                f"  {alloc.source.value}: max={alloc.max_tokens}, "
                f"priority={alloc.priority}, overflow={alloc.overflow_strategy}"
            )
        print()


# ---------------------------------------------------------------------------
# Example 2: Custom budget with pipeline
# ---------------------------------------------------------------------------


def run_custom_budget_pipeline() -> None:
    """Create a custom budget and run a pipeline with it."""
    tokenizer = WhitespaceTokenizer()

    # Create a custom budget that allocates tokens by source
    budget = TokenBudget(
        total_tokens=200,  # Intentionally small to trigger overflow
        allocations=[
            BudgetAllocation(
                source=SourceType.SYSTEM,
                max_tokens=50,
                priority=10,
            ),
            BudgetAllocation(
                source=SourceType.CONVERSATION,
                max_tokens=80,
                priority=7,
            ),
        ],
        reserve_tokens=20,  # Reserved for LLM response overhead
    )

    print("=== Custom Budget Pipeline ===")
    print(f"Total: {budget.total_tokens} tokens")
    print(f"Reserve: {budget.reserve_tokens} tokens")
    print(f"Shared pool: {budget.shared_pool} tokens")
    print()

    # Build pipeline with the budget
    memory = MemoryManager(conversation_tokens=200, tokenizer=tokenizer)
    memory.add_user_message("Tell me about context engineering and how it works")
    memory.add_assistant_message(
        "Context engineering is the practice of carefully assembling "
        "and managing information that goes into an LLM's context window"
    )
    memory.add_user_message("What about token budgets?")
    memory.add_assistant_message(
        "Token budgets allocate portions of the available context "
        "window to different sources like system prompts, conversation "
        "history, and retrieved documents. This ensures the most "
        "important information always fits."
    )
    memory.add_user_message("Can you explain overflow handling?")

    pipeline = (
        ContextPipeline(max_tokens=200, tokenizer=tokenizer, budget=budget)
        .with_memory(memory)
        .add_system_prompt("You are a helpful assistant.")
    )

    result = pipeline.build("How does overflow work?")

    print("--- Results ---")
    print(f"Items included: {result.diagnostics.get('items_included', 0)}")
    print(f"Items overflowed: {len(result.overflow_items)}")
    print(f"Token utilization: {result.window.utilization:.1%}")
    print()

    if result.overflow_items:
        print("Overflowed items (did not fit in budget):")
        for item in result.overflow_items:
            print(f"  [{item.source.value}] {item.content[:60]}...")
    else:
        print("No overflow -- all items fit within the budget.")

    print()
    print("Included items:")
    for item in result.window.items:
        print(
            f"  [{item.source.value}] (prio={item.priority}, "
            f"tokens={item.token_count}) {item.content[:60]}..."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    show_preset_budgets()
    print()
    run_custom_budget_pipeline()


if __name__ == "__main__":
    main()
