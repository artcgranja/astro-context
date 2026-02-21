"""Tool definitions and factory functions for the Agent."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from astro_context.memory.manager import MemoryManager
from astro_context.models.query import QueryBundle


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


def memory_tools(memory: MemoryManager) -> list[AgentTool]:
    """Create tools for saving and searching long-term facts.

    Returns two tools:
    - ``save_fact``: saves an important user fact to persistent memory
    - ``search_facts``: searches previously saved facts
    """

    def save_fact(fact: str) -> str:
        entry = memory.add_fact(fact, tags=["auto"])
        return f"Saved: {fact} (id: {entry.id[:8]})"

    def search_facts(query: str) -> str:
        facts = memory.get_relevant_facts(query)
        if not facts:
            return "No relevant facts found."
        return "\n".join(f"- {f.content}" for f in facts)

    return [
        AgentTool(
            name="save_fact",
            description=(
                "Save an important fact about the user. Use when the user shares "
                "personal details worth remembering: name, preferences, projects, etc."
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
                "Use when you need to recall something the user told you earlier."
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
    ]


def rag_tools(
    retriever: Any,
    embed_fn: Callable[[str], list[float]] | None = None,
) -> list[AgentTool]:
    """Create a ``search_docs`` tool for agentic RAG.

    The model decides when to search documentation, making this
    agentic RAG -- the model controls retrieval timing.

    Parameters
    ----------
    retriever:
        Any object with a ``retrieve(query, top_k)`` method.
    embed_fn:
        Optional embedding function.  If the retriever needs
        embeddings in the QueryBundle, provide this.
    """

    def search_docs(query: str) -> str:
        q = QueryBundle(query_str=query)
        if embed_fn is not None:
            q = q.model_copy(update={"embedding": embed_fn(query)})
        results = retriever.retrieve(q, top_k=5)
        if not results:
            return "No relevant documents found."
        parts: list[str] = []
        for item in results:
            section = item.metadata.get("section", "")
            prefix = f"[{section}] " if section else ""
            parts.append(f"{prefix}{item.content[:500]}")
        return "\n\n---\n\n".join(parts)

    return [
        AgentTool(
            name="search_docs",
            description=(
                "Search documentation for relevant information. Use when the user "
                "asks about features, APIs, concepts, or anything that might be in the docs."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for finding relevant documentation.",
                    }
                },
                "required": ["query"],
            },
            fn=search_docs,
        ),
    ]
