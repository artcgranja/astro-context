"""Memory CRUD tools for the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_context.agent.models import AgentTool
from astro_context.agent.tool_decorator import tool

if TYPE_CHECKING:
    from astro_context.memory.manager import MemoryManager


def memory_tools(memory: MemoryManager) -> list[AgentTool]:
    """Create tools for managing long-term facts (full CRUD).

    Returns four tools:
    - ``save_fact``: saves an important user fact to persistent memory
    - ``search_facts``: searches previously saved facts
    - ``update_fact``: updates an existing fact's content
    - ``delete_fact``: removes an outdated or incorrect fact
    """

    @tool(
        description=(
            "Save a NEW fact about the user. Only use when the user shares "
            "information that is NOT already in your saved facts. "
            "If a similar fact already exists, use update_fact instead."
        ),
    )
    def save_fact(fact: str) -> str:
        """Save a NEW fact about the user.

        Args:
            fact: The fact to save, as a clear statement.
        """
        entry = memory.add_fact(fact, tags=["auto"])
        # add_fact returns existing entry if content-hash matches (dedup)
        return f"Saved: {fact} (id: {entry.id[:8]})"

    @tool(
        description=(
            "Search previously saved facts about the user. "
            "Use before saving to check if a fact already exists. "
            "Returns fact IDs that can be used with update_fact or delete_fact."
        ),
    )
    def search_facts(query: str) -> str:
        """Search previously saved facts about the user.

        Args:
            query: Search query for finding relevant facts.
        """
        facts = memory.get_relevant_facts(query)
        if not facts:
            return "No relevant facts found."
        return "\n".join(f"- [{f.id[:8]}] {f.content}" for f in facts)

    @tool(
        description=(
            "Update an existing fact with new or corrected information. "
            "Use when a previously saved fact needs to be changed "
            "(e.g., user corrects their age, changes a preference). "
            "Requires the fact ID from search_facts."
        ),
    )
    def update_fact(fact_id: str, content: str) -> str:
        """Update an existing fact.

        Args:
            fact_id: The full ID of the fact to update (from search_facts).
            content: The new content for this fact.
        """
        updated = memory.update_fact(fact_id, content)
        if updated is None:
            return f"Fact not found with id starting with '{fact_id}'."
        return f"Updated fact {updated.id[:8]}: {updated.content}"

    @tool(
        description=(
            "Delete a saved fact that is no longer accurate or relevant. "
            "Requires the fact ID from search_facts."
        ),
    )
    def delete_fact(fact_id: str) -> str:
        """Delete a saved fact.

        Args:
            fact_id: The full ID of the fact to delete (from search_facts).
        """
        deleted = memory.delete_fact(fact_id)
        if not deleted:
            return f"Fact not found with id starting with '{fact_id}'."
        return f"Deleted fact {fact_id[:8]}."

    return [save_fact, search_facts, update_fact, delete_fact]
