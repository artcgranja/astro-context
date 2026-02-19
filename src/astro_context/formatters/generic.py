"""Generic text formatter."""

from __future__ import annotations

from astro_context.models.context import ContextWindow, SourceType


class GenericTextFormatter:
    """Formats context as structured plain text with section headers."""

    @property
    def format_type(self) -> str:
        return "generic"

    def format(self, window: ContextWindow) -> str:
        sections: dict[str, list[str]] = {
            "SYSTEM": [],
            "MEMORY": [],
            "CONTEXT": [],
        }

        for item in window.items:
            if item.source == SourceType.SYSTEM:
                sections["SYSTEM"].append(item.content)
            elif item.source == SourceType.MEMORY:
                sections["MEMORY"].append(item.content)
            else:
                sections["CONTEXT"].append(item.content)

        parts: list[str] = []
        for section_name, items in sections.items():
            if items:
                parts.append(f"=== {section_name} ===")
                parts.append("\n".join(items))
                parts.append("")

        return "\n".join(parts).strip()
