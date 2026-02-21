"""Tests for astro_context.models.streaming -- StreamDelta, StreamResult, StreamUsage."""

from __future__ import annotations

from astro_context.models.streaming import StreamDelta, StreamResult, StreamUsage

# ---------------------------------------------------------------------------
# StreamDelta
# ---------------------------------------------------------------------------


class TestStreamDelta:
    """StreamDelta model tests."""

    def test_default_construction_with_text(self) -> None:
        delta = StreamDelta(text="hello")
        assert delta.text == "hello"
        assert delta.index == 0

    def test_custom_index(self) -> None:
        delta = StreamDelta(text="world", index=3)
        assert delta.text == "world"
        assert delta.index == 3

    def test_empty_text(self) -> None:
        delta = StreamDelta(text="")
        assert delta.text == ""
        assert delta.index == 0

    def test_serialization_roundtrip(self) -> None:
        delta = StreamDelta(text="chunk", index=2)
        data = delta.model_dump()
        restored = StreamDelta.model_validate(data)
        assert restored.text == delta.text
        assert restored.index == delta.index

    def test_model_dump_keys(self) -> None:
        delta = StreamDelta(text="x")
        data = delta.model_dump()
        assert set(data.keys()) == {"text", "index"}


# ---------------------------------------------------------------------------
# StreamUsage
# ---------------------------------------------------------------------------


class TestStreamUsage:
    """StreamUsage model tests."""

    def test_default_construction(self) -> None:
        usage = StreamUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_creation_input_tokens == 0
        assert usage.cache_read_input_tokens == 0

    def test_custom_values(self) -> None:
        usage = StreamUsage(
            input_tokens=100,
            output_tokens=50,
            cache_creation_input_tokens=10,
            cache_read_input_tokens=20,
        )
        assert usage.input_tokens == 100
        assert usage.output_tokens == 50
        assert usage.cache_creation_input_tokens == 10
        assert usage.cache_read_input_tokens == 20

    def test_serialization_roundtrip(self) -> None:
        usage = StreamUsage(input_tokens=42, output_tokens=13)
        data = usage.model_dump()
        restored = StreamUsage.model_validate(data)
        assert restored.input_tokens == usage.input_tokens
        assert restored.output_tokens == usage.output_tokens
        assert restored.cache_creation_input_tokens == usage.cache_creation_input_tokens
        assert restored.cache_read_input_tokens == usage.cache_read_input_tokens

    def test_total_tokens_via_fields(self) -> None:
        """input_tokens + output_tokens gives the total usage."""
        usage = StreamUsage(input_tokens=100, output_tokens=50)
        total = usage.input_tokens + usage.output_tokens
        assert total == 150

    def test_model_dump_keys(self) -> None:
        usage = StreamUsage()
        data = usage.model_dump()
        expected_keys = {
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        }
        assert set(data.keys()) == expected_keys


# ---------------------------------------------------------------------------
# StreamResult
# ---------------------------------------------------------------------------


class TestStreamResult:
    """StreamResult model tests."""

    def test_default_construction(self) -> None:
        result = StreamResult()
        assert result.text == ""
        assert result.model == ""
        assert result.stop_reason == ""
        assert isinstance(result.usage, StreamUsage)
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_construction_with_text_and_model(self) -> None:
        result = StreamResult(text="Hello world", model="claude-3")
        assert result.text == "Hello world"
        assert result.model == "claude-3"

    def test_with_usage(self) -> None:
        usage = StreamUsage(input_tokens=200, output_tokens=100)
        result = StreamResult(text="response", usage=usage, stop_reason="end_turn")
        assert result.usage.input_tokens == 200
        assert result.usage.output_tokens == 100
        assert result.stop_reason == "end_turn"

    def test_without_usage_gets_default(self) -> None:
        result = StreamResult(text="no usage")
        assert result.usage.input_tokens == 0
        assert result.usage.output_tokens == 0

    def test_serialization_roundtrip(self) -> None:
        usage = StreamUsage(input_tokens=50, output_tokens=25, cache_read_input_tokens=10)
        result = StreamResult(
            text="full round trip",
            usage=usage,
            model="test-model",
            stop_reason="max_tokens",
        )
        data = result.model_dump()
        restored = StreamResult.model_validate(data)
        assert restored.text == result.text
        assert restored.model == result.model
        assert restored.stop_reason == result.stop_reason
        assert restored.usage.input_tokens == result.usage.input_tokens
        assert restored.usage.output_tokens == result.usage.output_tokens
        assert restored.usage.cache_read_input_tokens == result.usage.cache_read_input_tokens

    def test_model_dump_keys(self) -> None:
        result = StreamResult()
        data = result.model_dump()
        expected_keys = {"text", "usage", "model", "stop_reason"}
        assert set(data.keys()) == expected_keys

    def test_usage_is_nested_dict_in_dump(self) -> None:
        result = StreamResult(usage=StreamUsage(input_tokens=10))
        data = result.model_dump()
        assert isinstance(data["usage"], dict)
        assert data["usage"]["input_tokens"] == 10
