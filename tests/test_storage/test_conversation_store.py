"""Tests for ConversationStore protocol and InMemory implementation."""

from anchor.models.memory import ConversationTurn, SummaryTier
from anchor.protocols.storage import ConversationStore
from anchor.storage._serialization import (
    conversation_turn_to_row,
    row_to_conversation_turn,
    summary_tier_to_row,
    row_to_summary_tier,
)
from anchor.storage.memory_store import InMemoryConversationStore


def _turn(role: str, content: str) -> ConversationTurn:
    return ConversationTurn(role=role, content=content)


def _tier(level: int, content: str, turn_count: int = 1) -> SummaryTier:
    return SummaryTier(level=level, content=content, token_count=10, source_turn_count=turn_count)


class TestConversationSerialization:
    def test_turn_round_trip(self):
        turn = _turn("user", "hello world")
        row = conversation_turn_to_row(turn)
        restored = row_to_conversation_turn(row)
        assert restored.role == turn.role
        assert restored.content == turn.content

    def test_tier_round_trip(self):
        tier = _tier(1, "summary text", turn_count=5)
        row = summary_tier_to_row(tier)
        restored = row_to_summary_tier(row)
        assert restored.level == tier.level
        assert restored.content == tier.content
        assert restored.source_turn_count == 5


class TestInMemoryConversationStoreProtocol:
    def test_satisfies_protocol(self):
        store = InMemoryConversationStore()
        assert isinstance(store, ConversationStore)


class TestInMemoryConversationStoreTurns:
    def test_append_and_load(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"
        assert turns[1].content == "hi"

    def test_load_with_limit(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_load_empty_session(self):
        store = InMemoryConversationStore()
        assert store.load_turns("nonexistent") == []

    def test_truncate_turns(self):
        store = InMemoryConversationStore()
        for i in range(10):
            store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        store.truncate_turns("sess-1", keep_last=3)
        turns = store.load_turns("sess-1")
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_sessions_are_isolated(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert len(store.load_turns("sess-1")) == 1
        assert len(store.load_turns("sess-2")) == 1


class TestInMemoryConversationStoreTiers:
    def test_save_and_load_tiers(self):
        store = InMemoryConversationStore()
        tiers = {1: _tier(1, "summary"), 2: None, 3: None}
        store.save_summary_tiers("sess-1", tiers)
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1] is not None
        assert loaded[1].content == "summary"
        assert loaded[2] is None

    def test_load_tiers_empty_session(self):
        store = InMemoryConversationStore()
        loaded = store.load_summary_tiers("nonexistent")
        assert loaded == {1: None, 2: None, 3: None}

    def test_save_tiers_overwrites(self):
        store = InMemoryConversationStore()
        store.save_summary_tiers("sess-1", {1: _tier(1, "old"), 2: None, 3: None})
        store.save_summary_tiers("sess-1", {1: _tier(1, "new"), 2: None, 3: None})
        loaded = store.load_summary_tiers("sess-1")
        assert loaded[1].content == "new"


class TestInMemoryConversationStoreSession:
    def test_list_sessions(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(store.list_sessions()) == ["sess-1", "sess-2"]

    def test_delete_session(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.save_summary_tiers("sess-1", {1: _tier(1, "summary"), 2: None, 3: None})
        assert store.delete_session("sess-1") is True
        assert store.load_turns("sess-1") == []
        assert store.delete_session("sess-1") is False

    def test_clear(self):
        store = InMemoryConversationStore()
        store.append_turn("sess-1", _turn("user", "hello"))
        store.append_turn("sess-2", _turn("user", "world"))
        store.clear()
        assert store.list_sessions() == []


import pytest
from anchor.storage.sqlite import SqliteConnectionManager


@pytest.fixture
def sqlite_conv_store():
    from anchor.storage.sqlite._conversation_store import SqliteConversationStore
    conn_mgr = SqliteConnectionManager(":memory:")
    return SqliteConversationStore(conn_mgr)


class TestSqliteConversationStore:
    def test_satisfies_protocol(self, sqlite_conv_store):
        assert isinstance(sqlite_conv_store, ConversationStore)

    def test_append_and_load(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = sqlite_conv_store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"

    def test_load_with_limit(self, sqlite_conv_store):
        for i in range(10):
            sqlite_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = sqlite_conv_store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_truncate_turns(self, sqlite_conv_store):
        for i in range(10):
            sqlite_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        sqlite_conv_store.truncate_turns("sess-1", keep_last=3)
        turns = sqlite_conv_store.load_turns("sess-1")
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    def test_save_and_load_tiers(self, sqlite_conv_store):
        tiers = {1: _tier(1, "summary", 5), 2: None, 3: None}
        sqlite_conv_store.save_summary_tiers("sess-1", tiers)
        loaded = sqlite_conv_store.load_summary_tiers("sess-1")
        assert loaded[1].content == "summary"
        assert loaded[1].source_turn_count == 5
        assert loaded[2] is None

    def test_delete_session(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        assert sqlite_conv_store.delete_session("sess-1") is True
        assert sqlite_conv_store.load_turns("sess-1") == []

    def test_list_sessions(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(sqlite_conv_store.list_sessions()) == ["sess-1", "sess-2"]

    def test_clear(self, sqlite_conv_store):
        sqlite_conv_store.append_turn("sess-1", _turn("user", "hello"))
        sqlite_conv_store.clear()
        assert sqlite_conv_store.list_sessions() == []


try:
    import asyncpg
    HAS_ASYNCPG = True
except ImportError:
    HAS_ASYNCPG = False

postgres_only = pytest.mark.skipif(not HAS_ASYNCPG, reason="asyncpg not installed")


@postgres_only
@pytest.mark.asyncio
class TestPostgresConversationStore:
    """Tests for PostgresConversationStore. Requires running PostgreSQL."""

    @pytest.fixture
    async def pg_conv_store(self):
        import os
        dsn = os.environ.get("ANCHOR_TEST_POSTGRES_DSN")
        if not dsn:
            pytest.skip("ANCHOR_TEST_POSTGRES_DSN not set")
        from anchor.storage.postgres._conversation_store import PostgresConversationStore
        from anchor.storage.postgres._connection import PostgresConnectionManager
        mgr = PostgresConnectionManager(dsn)
        store = PostgresConversationStore(mgr)
        yield store
        await store.clear()

    async def test_append_and_load(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        await pg_conv_store.append_turn("sess-1", _turn("assistant", "hi"))
        turns = await pg_conv_store.load_turns("sess-1")
        assert len(turns) == 2
        assert turns[0].content == "hello"

    async def test_load_with_limit(self, pg_conv_store):
        for i in range(10):
            await pg_conv_store.append_turn("sess-1", _turn("user", f"msg-{i}"))
        turns = await pg_conv_store.load_turns("sess-1", limit=3)
        assert len(turns) == 3
        assert turns[0].content == "msg-7"

    async def test_save_and_load_tiers(self, pg_conv_store):
        tiers = {1: _tier(1, "summary", 5), 2: None, 3: None}
        await pg_conv_store.save_summary_tiers("sess-1", tiers)
        loaded = await pg_conv_store.load_summary_tiers("sess-1")
        assert loaded[1].content == "summary"

    async def test_delete_session(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        assert await pg_conv_store.delete_session("sess-1") is True
        assert await pg_conv_store.load_turns("sess-1") == []

    async def test_list_sessions(self, pg_conv_store):
        await pg_conv_store.append_turn("sess-1", _turn("user", "hello"))
        await pg_conv_store.append_turn("sess-2", _turn("user", "world"))
        assert sorted(await pg_conv_store.list_sessions()) == ["sess-1", "sess-2"]
