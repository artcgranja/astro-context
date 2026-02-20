"""Tests for astro_context.memory.extractor."""

from __future__ import annotations

from typing import Any

import pytest

from astro_context.memory.extractor import CallbackExtractor
from astro_context.models.memory import ConversationTurn, MemoryEntry, MemoryType


def _make_turns(n: int = 3) -> list[ConversationTurn]:
    """Create a list of ConversationTurn objects for testing."""
    turns: list[ConversationTurn] = []
    for i in range(n):
        role = "user" if i % 2 == 0 else "assistant"
        turns.append(
            ConversationTurn(
                role=role,  # type: ignore[arg-type]
                content=f"Turn {i} content",
                token_count=3,
            )
        )
    return turns


class TestCallbackExtractorBasic:
    """Basic extraction functionality."""

    def test_calls_user_function_with_turns(self) -> None:
        received_turns: list[list[ConversationTurn]] = []

        def capture_fn(turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            received_turns.append(turns)
            return [{"content": "extracted fact"}]

        extractor = CallbackExtractor(extract_fn=capture_fn)
        turns = _make_turns(2)
        extractor.extract(turns)

        assert len(received_turns) == 1
        assert received_turns[0] is turns

    def test_extracts_valid_memory_entries(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [
                {"content": "User prefers dark mode"},
                {"content": "User's name is Alice"},
            ]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())

        assert len(entries) == 2
        assert all(isinstance(e, MemoryEntry) for e in entries)
        assert entries[0].content == "User prefers dark mode"
        assert entries[1].content == "User's name is Alice"


class TestCallbackExtractorOptionalFields:
    """Handling optional fields in extraction dicts."""

    def test_handles_tags(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact", "tags": ["preference", "ui"]}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].tags == ["preference", "ui"]

    def test_handles_memory_type(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact", "memory_type": "episodic"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].memory_type == MemoryType.EPISODIC

    def test_handles_metadata(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact", "metadata": {"source": "chat"}}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].metadata == {"source": "chat"}

    def test_handles_relevance_score(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact", "relevance_score": 0.9}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].relevance_score == 0.9

    def test_handles_user_id_and_session_id(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [
                {
                    "content": "fact",
                    "user_id": "user-123",
                    "session_id": "sess-456",
                }
            ]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].user_id == "user-123"
        assert entries[0].session_id == "sess-456"


class TestCallbackExtractorDefaultType:
    """default_type is used when memory_type is not specified."""

    def test_uses_default_type_when_not_specified(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact"}]

        extractor = CallbackExtractor(
            extract_fn=extract_fn,
            default_type=MemoryType.PROCEDURAL,
        )
        entries = extractor.extract(_make_turns())
        assert entries[0].memory_type == MemoryType.PROCEDURAL

    def test_explicit_type_overrides_default(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact", "memory_type": "episodic"}]

        extractor = CallbackExtractor(
            extract_fn=extract_fn,
            default_type=MemoryType.PROCEDURAL,
        )
        entries = extractor.extract(_make_turns())
        assert entries[0].memory_type == MemoryType.EPISODIC

    def test_default_default_type_is_semantic(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "fact"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].memory_type == MemoryType.SEMANTIC


class TestCallbackExtractorEdgeCases:
    """Edge cases and error handling."""

    def test_empty_extraction_result(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return []

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries == []

    def test_missing_content_key_raises(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"tags": ["no-content"]}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        with pytest.raises(ValueError, match="content"):
            extractor.extract(_make_turns())

    def test_source_turns_populated_from_input_turns(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [{"content": "extracted"}]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        turns = _make_turns(2)
        entries = extractor.extract(turns)
        # source_turns should be populated from the input turn timestamps
        assert len(entries[0].source_turns) == 2
        # Each source_turn should be an ISO timestamp string
        for st in entries[0].source_turns:
            assert isinstance(st, str)

    def test_explicit_source_turns_preserved(self) -> None:
        def extract_fn(_turns: list[ConversationTurn]) -> list[dict[str, Any]]:
            return [
                {"content": "fact", "source_turns": ["2024-01-01T00:00:00+00:00"]}
            ]

        extractor = CallbackExtractor(extract_fn=extract_fn)
        entries = extractor.extract(_make_turns())
        assert entries[0].source_turns == ["2024-01-01T00:00:00+00:00"]
