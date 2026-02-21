"""Memory skill factory."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_context.agent.skills.models import Skill

from .tools import memory_tools

if TYPE_CHECKING:
    from astro_context.memory.manager import MemoryManager


def memory_skill(memory: MemoryManager) -> Skill:
    """Create a Skill that wraps :func:`memory_tools`.

    Returns a skill with four tools: ``save_fact``, ``search_facts``,
    ``update_fact``, and ``delete_fact``.

    Parameters
    ----------
    memory:
        The :class:`MemoryManager` instance to operate on.
    """
    tools = memory_tools(memory)
    return Skill(
        name="memory",
        description="Save, search, update, and delete long-term user facts.",
        instructions=(
            "Memory skill provides CRUD operations for persistent user facts.\n"
            "- Use search_facts BEFORE saving to avoid duplicates.\n"
            "- Use update_fact when information changes (don't delete + re-save).\n"
            "- Use delete_fact only for facts that are no longer relevant."
        ),
        tools=tuple(tools),
        activation="always",
        tags=("core", "memory"),
    )
