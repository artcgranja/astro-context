"""Auto-generate Pydantic models from function signatures and docstrings."""

from __future__ import annotations

import inspect
import re
from typing import Any, Union, get_args, get_origin

from pydantic import BaseModel, Field, create_model


def parse_docstring_args(fn: Any) -> dict[str, str]:
    """Parse Google-style ``Args:`` section from a function's docstring.

    Returns a mapping of parameter name to description string.
    Only the ``Args:`` block is parsed; other sections are ignored.

    Example docstring::

        def greet(name: str, loud: bool = False) -> str:
            \"\"\"Greet someone.

            Args:
                name: The person's name.
                loud: Whether to shout.
            \"\"\"
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return {}

    # Find the Args: block
    args_match = re.search(r"^Args?:\s*$", doc, re.MULTILINE)
    if not args_match:
        return {}

    # Extract everything after "Args:" until the next section header or end
    rest = doc[args_match.end() :]
    # A section header is a line that starts with a word followed by a colon
    # and is NOT indented (or less indented than args content)
    section_end = re.search(r"^\S", rest, re.MULTILINE)
    args_block = rest[: section_end.start()] if section_end else rest

    result: dict[str, str] = {}
    current_param: str | None = None
    current_desc_lines: list[str] = []

    for line in args_block.splitlines():
        # Match "    param_name: description" or "    param_name (type): description"
        param_match = re.match(r"^\s+(\w+)(?:\s*\([^)]*\))?\s*:\s*(.*)", line)
        if param_match:
            # Save previous param
            if current_param is not None:
                result[current_param] = " ".join(current_desc_lines).strip()
            current_param = param_match.group(1)
            desc = param_match.group(2).strip()
            current_desc_lines = [desc] if desc else []
        elif current_param is not None and line.strip():
            # Continuation line for current param
            current_desc_lines.append(line.strip())

    # Save last param
    if current_param is not None:
        result[current_param] = " ".join(current_desc_lines).strip()

    return result


def _get_first_doc_paragraph(fn: Any) -> str:
    """Extract the first paragraph from a function's docstring.

    Returns all lines up to the first blank line, joined with spaces.
    This captures multi-sentence tool descriptions like::

        \"\"\"Save a NEW fact about the user. Only use when the user shares
        information that is NOT already in your saved facts.
        If a similar fact already exists, use update_fact instead.\"\"\"
    """
    doc = inspect.getdoc(fn)
    if not doc:
        return ""
    lines: list[str] = []
    for line in doc.splitlines():
        stripped = line.strip()
        if not stripped:
            if lines:
                break
            continue
        # Stop at section headers like "Args:", "Returns:", "Parameters"
        if (
            stripped.endswith(":")
            and not stripped.startswith("-")
            and any(
                stripped.lower().startswith(s)
                for s in ("args", "arg", "returns", "raises", "parameters", "note")
            )
        ):
            break
        lines.append(stripped)
    return " ".join(lines)


def _is_optional(annotation: Any) -> tuple[bool, Any]:
    """Check if a type annotation is Optional[X] and return (True, X) or (False, annotation)."""
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        # Optional[X] is Union[X, None]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1 and len(args) == 2:
            return True, non_none[0]
    return False, annotation


def build_input_model(
    fn: Any,
    name: str | None = None,
) -> type[BaseModel]:
    """Build a Pydantic BaseModel from a function's type hints.

    Inspects the function's signature for parameter names, types, and
    defaults, then parses its Google-style docstring for per-parameter
    descriptions.  Returns a dynamically created Pydantic model whose
    ``.model_json_schema()`` produces a valid JSON Schema.

    Handles: ``str``, ``int``, ``float``, ``bool``, ``list[X]``,
    ``dict[str, X]``, ``Optional[X]``, ``Enum`` subclasses.

    Parameters
    ----------
    fn:
        The function to inspect.
    name:
        Model class name.  Defaults to ``{fn.__name__}_input``.
    """
    sig = inspect.signature(fn)
    hints = _get_type_hints_safe(fn)
    param_docs = parse_docstring_args(fn)
    model_name = name or f"{fn.__name__}_input"

    fields: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        annotation = hints.get(param_name, Any)

        # Skip 'return' and *args/**kwargs
        if param_name == "return":
            continue
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        description = param_docs.get(param_name, "")
        has_default = param.default is not inspect.Parameter.empty

        # Handle Optional[X]
        is_opt, _inner_type = _is_optional(annotation)

        if has_default:
            field_info = Field(default=param.default, description=description or None)
            fields[param_name] = (annotation, field_info)
        elif is_opt:
            field_info = Field(default=None, description=description or None)
            fields[param_name] = (annotation, field_info)
        else:
            field_info = Field(description=description or None)
            fields[param_name] = (annotation, field_info)

    model: type[BaseModel] = create_model(model_name, **fields)

    # Rebuild the model with the caller's type namespace so that
    # locally-defined types (e.g. Enum subclasses) can be resolved.
    ns: dict[str, Any] = {}
    for _pname, _param in sig.parameters.items():
        ann = hints.get(_pname)
        if ann is not None and isinstance(ann, type):
            ns[ann.__name__] = ann
    if ns:
        model.model_rebuild(_types_namespace=ns)

    return model


def _get_type_hints_safe(fn: Any) -> dict[str, Any]:
    """Get type hints with a fallback for edge cases.

    Uses ``get_type_hints`` but falls back to ``__annotations__``
    if that fails (e.g. forward references that can't be resolved).
    """
    try:
        from typing import get_type_hints

        return get_type_hints(fn)
    except Exception:
        return getattr(fn, "__annotations__", {})


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a Pydantic JSON Schema for use with LLM tool APIs.

    Removes Pydantic-specific keys like ``title`` from nested property
    definitions while preserving the structure LLM providers expect.
    """
    cleaned: dict[str, Any] = {"type": "object"}

    if "properties" in schema:
        props: dict[str, Any] = {}
        for key, prop in schema["properties"].items():
            clean_prop = dict(prop)
            clean_prop.pop("title", None)
            props[key] = clean_prop
        cleaned["properties"] = props

    if "required" in schema:
        cleaned["required"] = schema["required"]

    return cleaned
