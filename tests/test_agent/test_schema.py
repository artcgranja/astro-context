"""Tests for agent/schema.py — docstring parsing and input model generation."""

from __future__ import annotations

from enum import Enum
from typing import Any

import pytest
from pydantic import ValidationError

from astro_context.agent.schema import (
    build_input_model,
    clean_schema,
    parse_docstring_args,
)


class Color(Enum):
    """Module-level enum for testing (required for get_type_hints resolution)."""

    RED = "red"
    BLUE = "blue"


# ---------------------------------------------------------------------------
# parse_docstring_args
# ---------------------------------------------------------------------------


class TestParseDocstringArgs:
    def test_google_style(self) -> None:
        def fn(name: str, age: int) -> str:
            """Do something.

            Args:
                name: The person's name.
                age: The person's age.
            """

        result = parse_docstring_args(fn)
        assert result == {"name": "The person's name.", "age": "The person's age."}

    def test_no_docstring(self) -> None:
        def fn(x: str) -> str: ...

        assert parse_docstring_args(fn) == {}

    def test_no_args_section(self) -> None:
        def fn(x: str) -> str:
            """Just a summary."""

        assert parse_docstring_args(fn) == {}

    def test_multiline_description(self) -> None:
        def fn(query: str) -> str:
            """Search stuff.

            Args:
                query: The search query
                    that spans multiple lines.
            """

        result = parse_docstring_args(fn)
        assert result["query"] == "The search query that spans multiple lines."

    def test_multiple_sections(self) -> None:
        def fn(x: str) -> str:
            """Summary.

            Args:
                x: The input.

            Returns:
                A string.
            """

        result = parse_docstring_args(fn)
        assert result == {"x": "The input."}

    def test_param_with_type_annotation_in_doc(self) -> None:
        def fn(city: str) -> str:
            """Get weather.

            Args:
                city (str): The city name.
            """

        result = parse_docstring_args(fn)
        assert result == {"city": "The city name."}


# ---------------------------------------------------------------------------
# build_input_model
# ---------------------------------------------------------------------------


class TestBuildInputModel:
    def test_basic_types(self) -> None:
        def fn(name: str, count: int, active: bool) -> str:
            """Do work.

            Args:
                name: A name.
                count: A count.
                active: Is active.
            """

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert "name" in schema["properties"]
        assert "count" in schema["properties"]
        assert "active" in schema["properties"]
        assert set(schema["required"]) == {"name", "count", "active"}

    def test_default_values(self) -> None:
        def fn(query: str, limit: int = 10) -> str:
            """Search.

            Args:
                query: What to search.
                limit: Max results.
            """

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert schema["required"] == ["query"]
        assert schema["properties"]["limit"]["default"] == 10

    def test_optional_type(self) -> None:
        def fn(name: str, tag: str | None = None) -> str:
            """Do stuff."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        # tag should not be required
        assert "tag" not in schema.get("required", [])

    def test_list_type(self) -> None:
        def fn(items: list[str]) -> str:
            """Process items."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        prop = schema["properties"]["items"]
        assert prop["type"] == "array"

    def test_enum_type(self) -> None:
        def fn(color: Color) -> str:
            """Pick a color."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        # Pydantic generates enum values in the schema
        assert "color" in schema["properties"]

    def test_custom_model_name(self) -> None:
        def fn(x: str) -> str:
            """Hello."""

        model = build_input_model(fn, name="MyCustomModel")
        assert model.__name__ == "MyCustomModel"

    def test_default_model_name(self) -> None:
        def fn(x: str) -> str:
            """Hello."""

        model = build_input_model(fn)
        assert model.__name__ == "fn_input"

    def test_descriptions_from_docstring(self) -> None:
        def fn(city: str, units: str = "celsius") -> str:
            """Get weather.

            Args:
                city: The city name to look up.
                units: Temperature units.
            """

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert schema["properties"]["city"]["description"] == "The city name to look up."
        assert schema["properties"]["units"]["description"] == "Temperature units."

    def test_no_params(self) -> None:
        def fn() -> str:
            """Do nothing."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert schema.get("properties", {}) == {}

    def test_float_type(self) -> None:
        def fn(temperature: float) -> str:
            """Read temp."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert "temperature" in schema["properties"]

    def test_dict_type(self) -> None:
        def fn(metadata: dict[str, Any]) -> str:
            """Process metadata."""

        model = build_input_model(fn)
        schema = model.model_json_schema()
        assert "metadata" in schema["properties"]

    def test_validates_correctly(self) -> None:
        def fn(name: str, age: int) -> str:
            """Test fn."""

        model = build_input_model(fn)
        instance = model.model_validate({"name": "Alice", "age": 30})
        assert instance.name == "Alice"  # type: ignore[attr-defined]
        assert instance.age == 30  # type: ignore[attr-defined]

    def test_validation_fails_on_missing_required(self) -> None:
        def fn(name: str) -> str:
            """Test fn."""

        model = build_input_model(fn)
        with pytest.raises(ValidationError):
            model.model_validate({})

    def test_validation_fails_on_wrong_type(self) -> None:
        def fn(count: int) -> str:
            """Test fn."""

        model = build_input_model(fn)
        with pytest.raises(ValidationError):
            model.model_validate({"count": "not a number"})


# ---------------------------------------------------------------------------
# clean_schema
# ---------------------------------------------------------------------------


class TestCleanSchema:
    def test_removes_title_from_properties(self) -> None:
        raw = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "title": "Name", "description": "A name"},
            },
            "required": ["name"],
            "title": "MyModel",
        }
        cleaned = clean_schema(raw)
        assert "title" not in cleaned["properties"]["name"]
        assert cleaned["properties"]["name"]["description"] == "A name"
        assert cleaned["required"] == ["name"]

    def test_preserves_type_and_required(self) -> None:
        raw = {"type": "object", "properties": {}, "required": []}
        cleaned = clean_schema(raw)
        assert cleaned["type"] == "object"

    def test_no_properties(self) -> None:
        raw = {"type": "object"}
        cleaned = clean_schema(raw)
        assert cleaned == {"type": "object"}
