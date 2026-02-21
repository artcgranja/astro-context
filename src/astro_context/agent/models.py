"""Base model for agent tools."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class AgentTool:
    """A tool that the Agent can use during conversation.

    Each tool has a name, description, JSON Schema for inputs,
    and a callable that executes the tool logic.
    """

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., str]

    def to_anthropic_schema(self) -> dict[str, Any]:
        """Convert to Anthropic tool definition format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }

    def validate_input(self, tool_input: dict[str, Any]) -> tuple[bool, str]:
        """Validate tool input against the JSON Schema.

        Checks required fields and basic type matching (string, number,
        integer, boolean).  Extra fields are allowed (lenient mode).

        Returns
        -------
        tuple[bool, str]
            ``(True, "")`` when valid, ``(False, error_message)`` otherwise.
        """
        properties: dict[str, Any] = self.input_schema.get("properties", {})
        required: list[str] = self.input_schema.get("required", [])

        # Check required fields
        for field in required:
            if field not in tool_input:
                return False, f"Missing required field: '{field}'"

        # Basic type checking for provided fields that have a schema entry
        _type_map: dict[str, type | tuple[type, ...]] = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
        }
        for key, value in tool_input.items():
            if key not in properties:
                continue  # extra fields are allowed
            expected_type_name = properties[key].get("type")
            if expected_type_name is None:
                continue
            expected = _type_map.get(expected_type_name)
            if expected is None:
                continue  # unknown schema type — skip
            if not isinstance(value, expected):
                return (
                    False,
                    f"Field '{key}' expected type '{expected_type_name}', "
                    f"got '{type(value).__name__}'",
                )

        return True, ""
