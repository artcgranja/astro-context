"""Tests for extended MemoryEntry fields and MemoryType enum."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

from astro_context.models.memory import ConversationTurn, MemoryEntry, MemoryType

# ---------------------------------------------------------------------------
# MemoryType enum
# ---------------------------------------------------------------------------


class TestMemoryType:
    """MemoryType enum values and behaviour."""

    def test_semantic_value(self) -> None:
        assert MemoryType.SEMANTIC == "semantic"

    def test_episodic_value(self) -> None:
        assert MemoryType.EPISODIC == "episodic"

    def test_procedural_value(self) -> None:
        assert MemoryType.PROCEDURAL == "procedural"

    def test_conversation_value(self) -> None:
        assert MemoryType.CONVERSATION == "conversation"

    def test_all_values(self) -> None:
        expected = {"semantic", "episodic", "procedural", "conversation"}
        assert {m.value for m in MemoryType} == expected

    def test_is_str_enum(self) -> None:
        """MemoryType values are also plain strings."""
        assert isinstance(MemoryType.SEMANTIC, str)


# ---------------------------------------------------------------------------
# MemoryEntry defaults
# ---------------------------------------------------------------------------


class TestMemoryEntryDefaults:
    """MemoryEntry creation with default values for new fields."""

    def test_default_memory_type(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.memory_type == MemoryType.SEMANTIC

    def test_default_user_id(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.user_id is None

    def test_default_session_id(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.session_id is None

    def test_default_expires_at(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.expires_at is None

    def test_default_updated_at_is_set(self) -> None:
        before = datetime.now(UTC)
        entry = MemoryEntry(content="hello")
        after = datetime.now(UTC)
        assert before <= entry.updated_at <= after

    def test_default_source_turns(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.source_turns == []

    def test_default_links(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.links == []

    def test_default_content_hash_is_computed(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.content_hash != ""

    def test_default_access_count(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.access_count == 0

    def test_default_relevance_score(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.relevance_score == 0.5


# ---------------------------------------------------------------------------
# content_hash auto-computation
# ---------------------------------------------------------------------------


class TestContentHash:
    """content_hash is automatically derived from content."""

    def test_hash_matches_md5_of_content(self) -> None:
        content = "The quick brown fox"
        entry = MemoryEntry(content=content)
        expected = hashlib.md5(content.encode()).hexdigest()  # noqa: S324
        assert entry.content_hash == expected

    def test_hash_changes_when_content_changes(self) -> None:
        entry_a = MemoryEntry(content="alpha")
        entry_b = MemoryEntry(content="beta")
        assert entry_a.content_hash != entry_b.content_hash

    def test_same_content_same_hash(self) -> None:
        entry_a = MemoryEntry(content="identical")
        entry_b = MemoryEntry(content="identical")
        assert entry_a.content_hash == entry_b.content_hash

    def test_explicit_hash_is_preserved(self) -> None:
        """If caller provides content_hash, the validator does NOT overwrite it."""
        entry = MemoryEntry(content="hello", content_hash="custom-hash")
        assert entry.content_hash == "custom-hash"

    def test_empty_hash_string_triggers_computation(self) -> None:
        """An explicitly empty string triggers auto-computation."""
        entry = MemoryEntry(content="hello", content_hash="")
        assert entry.content_hash != ""
        expected = hashlib.md5(b"hello").hexdigest()  # noqa: S324
        assert entry.content_hash == expected


# ---------------------------------------------------------------------------
# is_expired property
# ---------------------------------------------------------------------------


class TestIsExpired:
    """is_expired property for time-dependent expiration checks."""

    def test_no_expires_at_means_not_expired(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.is_expired is False

    def test_future_expires_at_means_not_expired(self) -> None:
        future = datetime.now(UTC) + timedelta(hours=1)
        entry = MemoryEntry(content="hello", expires_at=future)
        assert entry.is_expired is False

    def test_past_expires_at_means_expired(self) -> None:
        past = datetime.now(UTC) - timedelta(hours=1)
        entry = MemoryEntry(content="hello", expires_at=past)
        assert entry.is_expired is True

    def test_far_future_not_expired(self) -> None:
        far_future = datetime.now(UTC) + timedelta(days=365)
        entry = MemoryEntry(content="hello", expires_at=far_future)
        assert entry.is_expired is False

    def test_far_past_is_expired(self) -> None:
        far_past = datetime.now(UTC) - timedelta(days=365)
        entry = MemoryEntry(content="hello", expires_at=far_past)
        assert entry.is_expired is True


# ---------------------------------------------------------------------------
# touch() method
# ---------------------------------------------------------------------------


class TestTouch:
    """touch() returns a copy with updated access_count and last_accessed."""

    def test_increments_access_count(self) -> None:
        entry = MemoryEntry(content="hello")
        assert entry.access_count == 0
        touched = entry.touch()
        assert touched.access_count == 1

    def test_increments_from_existing_count(self) -> None:
        entry = MemoryEntry(content="hello", access_count=5)
        touched = entry.touch()
        assert touched.access_count == 6

    def test_updates_last_accessed(self) -> None:
        old_time = datetime(2020, 1, 1, tzinfo=UTC)
        entry = MemoryEntry(content="hello", last_accessed=old_time)
        before = datetime.now(UTC)
        touched = entry.touch()
        after = datetime.now(UTC)
        assert before <= touched.last_accessed <= after

    def test_returns_new_object(self) -> None:
        """touch() returns a NEW object -- original is unchanged (immutability)."""
        entry = MemoryEntry(content="hello")
        touched = entry.touch()
        assert touched is not entry
        assert entry.access_count == 0
        assert touched.access_count == 1

    def test_preserves_other_fields(self) -> None:
        entry = MemoryEntry(
            content="hello",
            relevance_score=0.9,
            memory_type=MemoryType.EPISODIC,
            user_id="user-1",
            session_id="sess-1",
            tags=["important"],
        )
        touched = entry.touch()
        assert touched.content == entry.content
        assert touched.relevance_score == entry.relevance_score
        assert touched.memory_type == entry.memory_type
        assert touched.user_id == entry.user_id
        assert touched.session_id == entry.session_id
        assert touched.tags == entry.tags
        assert touched.id == entry.id

    def test_multiple_touches_increment(self) -> None:
        entry = MemoryEntry(content="hello")
        t1 = entry.touch()
        t2 = t1.touch()
        t3 = t2.touch()
        assert t3.access_count == 3


# ---------------------------------------------------------------------------
# Custom field values
# ---------------------------------------------------------------------------


class TestMemoryEntryCustomFields:
    """MemoryEntry with explicitly set user_id, session_id, etc."""

    def test_with_user_id(self) -> None:
        entry = MemoryEntry(content="hello", user_id="user-42")
        assert entry.user_id == "user-42"

    def test_with_session_id(self) -> None:
        entry = MemoryEntry(content="hello", session_id="sess-abc")
        assert entry.session_id == "sess-abc"

    def test_with_both_user_and_session(self) -> None:
        entry = MemoryEntry(content="hello", user_id="u1", session_id="s1")
        assert entry.user_id == "u1"
        assert entry.session_id == "s1"

    def test_with_source_turns(self) -> None:
        turns = ["turn-1", "turn-2", "turn-3"]
        entry = MemoryEntry(content="hello", source_turns=turns)
        assert entry.source_turns == turns

    def test_with_links(self) -> None:
        links = ["mem-a", "mem-b"]
        entry = MemoryEntry(content="hello", links=links)
        assert entry.links == links

    def test_with_memory_type_episodic(self) -> None:
        entry = MemoryEntry(content="hello", memory_type=MemoryType.EPISODIC)
        assert entry.memory_type == MemoryType.EPISODIC

    def test_with_memory_type_procedural(self) -> None:
        entry = MemoryEntry(content="hello", memory_type=MemoryType.PROCEDURAL)
        assert entry.memory_type == MemoryType.PROCEDURAL

    def test_with_memory_type_conversation(self) -> None:
        entry = MemoryEntry(content="hello", memory_type=MemoryType.CONVERSATION)
        assert entry.memory_type == MemoryType.CONVERSATION

    def test_with_expires_at(self) -> None:
        expires = datetime(2030, 12, 31, tzinfo=UTC)
        entry = MemoryEntry(content="hello", expires_at=expires)
        assert entry.expires_at == expires


# ---------------------------------------------------------------------------
# Serialization round-trip
# ---------------------------------------------------------------------------


class TestMemoryEntrySerialization:
    """model_dump / model_validate round-trip."""

    def test_round_trip(self) -> None:
        entry = MemoryEntry(
            content="important fact",
            relevance_score=0.8,
            memory_type=MemoryType.EPISODIC,
            user_id="user-1",
            session_id="sess-1",
            tags=["fact"],
            source_turns=["t1", "t2"],
            links=["link-a"],
            metadata={"key": "value"},
        )
        data = entry.model_dump()
        restored = MemoryEntry.model_validate(data)

        assert restored.content == entry.content
        assert restored.relevance_score == entry.relevance_score
        assert restored.memory_type == entry.memory_type
        assert restored.user_id == entry.user_id
        assert restored.session_id == entry.session_id
        assert restored.tags == entry.tags
        assert restored.source_turns == entry.source_turns
        assert restored.links == entry.links
        assert restored.metadata == entry.metadata
        assert restored.content_hash == entry.content_hash
        assert restored.id == entry.id

    def test_dump_contains_new_fields(self) -> None:
        entry = MemoryEntry(content="hello", user_id="u1")
        data = entry.model_dump()
        assert "memory_type" in data
        assert "user_id" in data
        assert "session_id" in data
        assert "expires_at" in data
        assert "updated_at" in data
        assert "content_hash" in data
        assert "source_turns" in data
        assert "links" in data

    def test_dump_mode_json(self) -> None:
        """model_dump(mode='json') serialises datetimes to strings."""
        entry = MemoryEntry(content="hello")
        data = entry.model_dump(mode="json")
        # Datetimes should be serialised as ISO strings in json mode
        assert isinstance(data["created_at"], str)
        assert isinstance(data["updated_at"], str)
        assert isinstance(data["last_accessed"], str)


# ---------------------------------------------------------------------------
# ConversationTurn basic checks
# ---------------------------------------------------------------------------


class TestConversationTurn:
    """Basic ConversationTurn model tests."""

    def test_defaults(self) -> None:
        turn = ConversationTurn(role="user", content="hi")
        assert turn.role == "user"
        assert turn.content == "hi"
        assert turn.token_count == 0
        assert turn.metadata == {}
        assert turn.timestamp is not None

    def test_all_roles(self) -> None:
        for role in ("user", "assistant", "system", "tool"):
            turn = ConversationTurn(role=role, content="x")
            assert turn.role == role
