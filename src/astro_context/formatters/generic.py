"""Generic text formatter."""

from __future__ import annotations

from astro_context.models.context import ContextWindow

from .utils import classify_window_items


class GenericTextFormatter:
    """Formats context as structured plain text with section headers.

    Security Note:
        Content from memory and retrieval items is inserted verbatim into
        the formatted output without sanitization.  If these items originate
        from untrusted sources (e.g. user-supplied documents, web scrapes),
        they may contain prompt injection payloads.  Callers should implement
        content validation or filtering *before* items enter the pipeline.
    """

    @property
    def format_type(self) -> str:
        return "generic"

    def format(self, window: ContextWindow) -> str:
        """Format the context window as plain text with ``=== SECTION ===`` headers.

        Produces up to three sections (SYSTEM, MEMORY, CONTEXT) separated by
        blank lines.  Empty sections are omitted.
        """
        classified = classify_window_items(window)

        sections: dict[str, list[str]] = {
            "SYSTEM": classified.system_parts,
            "MEMORY": [item.content for item in classified.memory_items],
            "CONTEXT": classified.context_parts,
        }

        parts: list[str] = []
        for section_name, items in sections.items():
            if items:
                parts.append(f"=== {section_name} ===")
                parts.append("\n".join(items))
                parts.append("")

        return "\n".join(parts).strip()
