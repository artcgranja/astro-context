"""Skill data model for progressive tool disclosure."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:
    from astro_context.agent.models import AgentTool


@dataclass(frozen=True, slots=True)
class Skill:
    """A named group of tools with optional on-demand activation.

    Skills organise :class:`AgentTool` instances into discoverable units.
    An *always*-loaded skill has its tools available from the first API
    round.  An *on_demand* skill is advertised in a discovery prompt
    (Tier 1) and only fully loaded when the agent calls
    ``activate_skill`` (Tier 2/3).

    Parameters
    ----------
    name:
        Unique identifier for the skill (e.g. ``"memory"``).
    description:
        Short human-readable summary shown in the discovery prompt.
    instructions:
        Detailed usage guide injected when the skill is activated.
    tools:
        The :class:`AgentTool` instances this skill provides.
    activation:
        ``"always"`` means tools are loaded from round 1.
        ``"on_demand"`` means the agent must call ``activate_skill`` first.
    tags:
        Optional tags for filtering or grouping skills.
    """

    name: str
    description: str
    instructions: str = ""
    tools: tuple[AgentTool, ...] = ()
    activation: Literal["always", "on_demand"] = "always"
    tags: tuple[str, ...] = field(default_factory=tuple)
