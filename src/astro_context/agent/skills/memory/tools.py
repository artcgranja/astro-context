"""Memory CRUD tools for the agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_context.agent.models import AgentTool

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

    def save_fact(fact: str) -> str:
        entry = memory.add_fact(fact, tags=["auto"])
        # add_fact returns existing entry if content-hash matches (dedup)
        return f"Saved: {fact} (id: {entry.id[:8]})"

    def search_facts(query: str) -> str:
        facts = memory.get_relevant_facts(query)
        if not facts:
            return "No relevant facts found."
        return "\n".join(f"- [{f.id[:8]}] {f.content}" for f in facts)

    def update_fact(fact_id: str, content: str) -> str:
        updated = memory.update_fact(fact_id, content)
        if updated is None:
            return f"Fact not found with id starting with '{fact_id}'."
        return f"Updated fact {updated.id[:8]}: {updated.content}"

    def delete_fact(fact_id: str) -> str:
        deleted = memory.delete_fact(fact_id)
        if not deleted:
            return f"Fact not found with id starting with '{fact_id}'."
        return f"Deleted fact {fact_id[:8]}."

    return [
        AgentTool(
            name="save_fact",
            description=(
                "Save a NEW fact about the user. Only use when the user shares "
                "information that is NOT already in your saved facts. "
                "If a similar fact already exists, use update_fact instead."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fact": {
                        "type": "string",
                        "description": "The fact to save, as a clear statement.",
                    }
                },
                "required": ["fact"],
            },
            fn=save_fact,
        ),
        AgentTool(
            name="search_facts",
            description=(
                "Search previously saved facts about the user. "
                "Use before saving to check if a fact already exists. "
                "Returns fact IDs that can be used with update_fact or delete_fact."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant facts.",
                    }
                },
                "required": ["query"],
            },
            fn=search_facts,
        ),
        AgentTool(
            name="update_fact",
            description=(
                "Update an existing fact with new or corrected information. "
                "Use when a previously saved fact needs to be changed "
                "(e.g., user corrects their age, changes a preference). "
                "Requires the fact ID from search_facts."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": (
                            "The full ID of the fact to update (from search_facts)."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": "The new content for this fact.",
                    },
                },
                "required": ["fact_id", "content"],
            },
            fn=update_fact,
        ),
        AgentTool(
            name="delete_fact",
            description=(
                "Delete a saved fact that is no longer accurate or relevant. "
                "Requires the fact ID from search_facts."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "fact_id": {
                        "type": "string",
                        "description": (
                            "The full ID of the fact to delete (from search_facts)."
                        ),
                    },
                },
                "required": ["fact_id"],
            },
            fn=delete_fact,
        ),
    ]
