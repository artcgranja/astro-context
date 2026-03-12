"""Tools for the brainstorm skill."""

from __future__ import annotations

import json
from pathlib import Path

from astro_context.agent.tool_decorator import tool


@tool
def save_brainstorm_result(title: str, summary: str, approaches: str) -> str:
    """Save a brainstorm result to a JSON file.

    Args:
        title: Title of the brainstorming session.
        summary: Summary of the chosen approach.
        approaches: Description of all approaches considered.
    """
    result = {"title": title, "summary": summary, "approaches": approaches}
    output_path = Path("brainstorm_output.json")
    output_path.write_text(json.dumps(result, indent=2))
    return f"Brainstorm saved to {output_path}"
