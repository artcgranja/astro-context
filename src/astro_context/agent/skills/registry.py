"""Skill registry for progressive tool disclosure."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_context.agent.skills.models import Skill

if TYPE_CHECKING:
    from astro_context.agent.models import AgentTool


class SkillRegistry:
    """Registry that tracks registered skills and their activation state.

    Always-loaded skills are considered active from the moment they are
    registered.  On-demand skills must be explicitly activated via
    :meth:`activate` (typically by the ``activate_skill`` meta-tool).
    """

    __slots__ = ("_activated", "_skills")

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._activated: set[str] = set()

    # -- Mutation --

    def register(self, skill: Skill) -> None:
        """Register a skill.  Raises :class:`ValueError` on duplicate name."""
        if skill.name in self._skills:
            msg = f"Skill already registered: '{skill.name}'"
            raise ValueError(msg)
        self._skills[skill.name] = skill

    def activate(self, name: str) -> Skill:
        """Mark an on-demand skill as active.  Returns the skill.

        Raises :class:`KeyError` if the skill is not registered.
        """
        skill = self._skills.get(name)
        if skill is None:
            msg = f"Unknown skill: '{name}'"
            raise KeyError(msg)
        self._activated.add(name)
        return skill

    def deactivate(self, name: str) -> None:
        """Remove a skill from the active set."""
        self._activated.discard(name)

    def reset(self) -> None:
        """Clear all activation state (keeps registrations)."""
        self._activated.clear()

    # -- Queries --

    def get(self, name: str) -> Skill | None:
        """Look up a skill by name, or ``None`` if not found."""
        return self._skills.get(name)

    def is_active(self, name: str) -> bool:
        """Return ``True`` if the skill's tools should be available now.

        Always-loaded skills are always active.  On-demand skills are
        active only after an explicit :meth:`activate` call.
        """
        skill = self._skills.get(name)
        if skill is None:
            return False
        if skill.activation == "always":
            return True
        return name in self._activated

    def active_tools(self) -> list[AgentTool]:
        """Return all tools from currently-active skills.

        Raises :class:`ValueError` if two active skills provide tools
        with the same name.
        """
        tools: list[AgentTool] = []
        seen_names: set[str] = set()
        for name, skill in self._skills.items():
            if not self.is_active(name):
                continue
            for tool in skill.tools:
                if tool.name in seen_names:
                    msg = f"Duplicate tool name across active skills: '{tool.name}'"
                    raise ValueError(msg)
                seen_names.add(tool.name)
                tools.append(tool)
        return tools

    def on_demand_skills(self) -> list[Skill]:
        """Return skills that require activation."""
        return [s for s in self._skills.values() if s.activation == "on_demand"]

    def skill_discovery_prompt(self) -> str:
        """Build the Tier-1 discovery text for the system prompt.

        Returns an empty string when there are no on-demand skills.
        """
        on_demand = self.on_demand_skills()
        if not on_demand:
            return ""
        lines = ["Available skills (use activate_skill to enable):"]
        for skill in on_demand:
            active = " [active]" if self.is_active(skill.name) else ""
            lines.append(f"  - {skill.name}: {skill.description}{active}")
        return "\n".join(lines)
