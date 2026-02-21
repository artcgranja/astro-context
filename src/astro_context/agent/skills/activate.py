"""Meta-tool for on-demand skill activation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from astro_context.agent.models import AgentTool

if TYPE_CHECKING:
    from astro_context.agent.skills.registry import SkillRegistry


def _make_activate_skill_tool(registry: SkillRegistry) -> AgentTool:
    """Create the ``activate_skill`` meta-tool bound to *registry*.

    When called, this tool activates an on-demand skill and returns
    its instructions plus the names of newly available tools so the
    agent knows what it can call in subsequent rounds.
    """

    def activate_skill(skill_name: str) -> str:
        try:
            skill = registry.activate(skill_name)
        except KeyError:
            available = [s.name for s in registry.on_demand_skills()]
            return (
                f"Unknown skill: '{skill_name}'. "
                f"Available skills: {', '.join(available) or 'none'}"
            )

        tool_names = [t.name for t in skill.tools]
        parts = [f"Skill '{skill.name}' activated."]
        if skill.instructions:
            parts.append(f"\n{skill.instructions}")
        parts.append(f"\nNew tools available: {', '.join(tool_names)}")
        return "\n".join(parts)

    return AgentTool(
        name="activate_skill",
        description=(
            "Activate an on-demand skill to make its tools available. "
            "Call this with the skill name from the available skills list."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "skill_name": {
                    "type": "string",
                    "description": "Name of the skill to activate.",
                },
            },
            "required": ["skill_name"],
        },
        fn=activate_skill,
    )
