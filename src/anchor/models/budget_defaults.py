"""Factory functions for common TokenBudget configurations.

These provide sensible defaults for different application archetypes.
All percentages are approximate and can be customised by constructing
a ``TokenBudget`` directly.
"""

from __future__ import annotations

from .budget import BudgetAllocation, TokenBudget
from .context import SourceType


def default_chat_budget(max_tokens: int) -> TokenBudget:
    """Budget optimised for conversational applications.

    Allocation breakdown:
        - System: 10%
        - Memory (persistent facts): 10%
        - Conversation (session turns): 20%
        - Retrieval: 25%
        - Reserve (for LLM response): 15%
        - Shared pool (unallocated): 20%

    Parameters:
        max_tokens: Total token budget for the context window.

    Returns:
        A ``TokenBudget`` instance.
    """
    if max_tokens <= 0:
        msg = "max_tokens must be a positive integer"
        raise ValueError(msg)

    return TokenBudget(
        total_tokens=max_tokens,
        reserve_tokens=int(max_tokens * 0.15),
        allocations=[
            BudgetAllocation(
                source=SourceType.SYSTEM,
                max_tokens=int(max_tokens * 0.10),
                priority=10,
            ),
            BudgetAllocation(
                source=SourceType.MEMORY,
                max_tokens=int(max_tokens * 0.10),
                priority=8,
            ),
            BudgetAllocation(
                source=SourceType.CONVERSATION,
                max_tokens=int(max_tokens * 0.20),
                priority=7,
            ),
            BudgetAllocation(
                source=SourceType.RETRIEVAL,
                max_tokens=int(max_tokens * 0.25),
                priority=5,
            ),
        ],
    )


def default_rag_budget(max_tokens: int) -> TokenBudget:
    """Budget optimised for RAG-heavy applications.

    Allocation breakdown:
        - System: 10%
        - Memory (persistent facts): 5%
        - Conversation (session turns): 10%
        - Retrieval: 40%
        - Reserve (for LLM response): 15%
        - Shared pool (unallocated): 20%

    Parameters:
        max_tokens: Total token budget for the context window.

    Returns:
        A ``TokenBudget`` instance.
    """
    if max_tokens <= 0:
        msg = "max_tokens must be a positive integer"
        raise ValueError(msg)

    return TokenBudget(
        total_tokens=max_tokens,
        reserve_tokens=int(max_tokens * 0.15),
        allocations=[
            BudgetAllocation(
                source=SourceType.SYSTEM,
                max_tokens=int(max_tokens * 0.10),
                priority=10,
            ),
            BudgetAllocation(
                source=SourceType.MEMORY,
                max_tokens=int(max_tokens * 0.05),
                priority=8,
            ),
            BudgetAllocation(
                source=SourceType.CONVERSATION,
                max_tokens=int(max_tokens * 0.10),
                priority=7,
            ),
            BudgetAllocation(
                source=SourceType.RETRIEVAL,
                max_tokens=int(max_tokens * 0.40),
                priority=5,
            ),
        ],
    )


def default_agent_budget(max_tokens: int) -> TokenBudget:
    """Budget optimised for agentic applications.

    Allocation breakdown:
        - System: 15%
        - Memory (persistent facts): 10%
        - Conversation (session turns): 15%
        - Retrieval: 20%
        - Tool: 15%
        - Reserve (for LLM response): 15%
        - Shared pool (unallocated): 10%

    Parameters:
        max_tokens: Total token budget for the context window.

    Returns:
        A ``TokenBudget`` instance.
    """
    if max_tokens <= 0:
        msg = "max_tokens must be a positive integer"
        raise ValueError(msg)

    return TokenBudget(
        total_tokens=max_tokens,
        reserve_tokens=int(max_tokens * 0.15),
        allocations=[
            BudgetAllocation(
                source=SourceType.SYSTEM,
                max_tokens=int(max_tokens * 0.15),
                priority=10,
            ),
            BudgetAllocation(
                source=SourceType.MEMORY,
                max_tokens=int(max_tokens * 0.10),
                priority=8,
            ),
            BudgetAllocation(
                source=SourceType.CONVERSATION,
                max_tokens=int(max_tokens * 0.15),
                priority=7,
            ),
            BudgetAllocation(
                source=SourceType.RETRIEVAL,
                max_tokens=int(max_tokens * 0.20),
                priority=5,
            ),
            BudgetAllocation(
                source=SourceType.TOOL,
                max_tokens=int(max_tokens * 0.15),
                priority=6,
            ),
        ],
    )
