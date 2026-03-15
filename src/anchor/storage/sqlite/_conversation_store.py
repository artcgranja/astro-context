"""SQLite-backed conversation store implementation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from anchor.models.memory import ConversationTurn, SummaryTier
from anchor.storage._serialization import (
    conversation_turn_to_row,
    row_to_conversation_turn,
    row_to_summary_tier,
    summary_tier_to_row,
)
from anchor.storage.sqlite._schema import ensure_tables, ensure_tables_async

if TYPE_CHECKING:
    from anchor.storage.sqlite._connection import SqliteConnectionManager

logger = logging.getLogger(__name__)


class SqliteConversationStore:
    """SQLite-backed conversation store. Implements ConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager
        ensure_tables(self._conn_manager.get_connection())

    def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        conn = self._conn_manager.get_connection()
        row = conversation_turn_to_row(turn)
        result = conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
            "FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        turn_index = result["next_idx"]
        conn.execute(
            "INSERT INTO conversation_turns "
            "(session_id, turn_index, role, content, token_count, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_index, row["role"], row["content"],
             row["token_count"], row["metadata_json"], row["created_at"]),
        )
        conn.commit()

    def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        conn = self._conn_manager.get_connection()
        if limit is not None:
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
            rows = list(reversed(rows))
        else:
            rows = conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index ASC",
                (session_id,),
            ).fetchall()
        return [row_to_conversation_turn(r) for r in rows]

    def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        for level, tier in tiers.items():
            if tier is not None:
                row = summary_tier_to_row(tier)
                conn.execute(
                    "INSERT INTO summary_tiers "
                    "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, row["tier_level"], row["content"], row["token_count"],
                     row["source_turn_count"], row["created_at"], row["updated_at"]),
                )
        conn.commit()

    def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT * FROM summary_tiers WHERE session_id = ?", (session_id,)
        ).fetchall()
        result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
        for row in rows:
            tier = row_to_summary_tier(row)
            result[tier.level] = tier
        return result

    def truncate_turns(self, session_id: str, keep_last: int) -> None:
        conn = self._conn_manager.get_connection()
        row = conn.execute(
            "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        ).fetchone()
        if row["max_idx"] is None:
            return
        cutoff = row["max_idx"] - keep_last + 1
        conn.execute(
            "DELETE FROM conversation_turns WHERE session_id = ? AND turn_index < ?",
            (session_id, cutoff),
        )
        conn.commit()

    def delete_session(self, session_id: str) -> bool:
        conn = self._conn_manager.get_connection()
        c1 = conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        c2 = conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        conn.commit()
        return (c1.rowcount + c2.rowcount) > 0

    def list_sessions(self) -> list[str]:
        conn = self._conn_manager.get_connection()
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM conversation_turns "
            "UNION SELECT DISTINCT session_id FROM summary_tiers"
        ).fetchall()
        return [r["session_id"] for r in rows]

    def clear(self) -> None:
        conn = self._conn_manager.get_connection()
        conn.execute("DELETE FROM conversation_turns")
        conn.execute("DELETE FROM summary_tiers")
        conn.commit()

    def __repr__(self) -> str:
        return "SqliteConversationStore()"


class AsyncSqliteConversationStore:
    """Async SQLite-backed conversation store. Implements AsyncConversationStore protocol."""

    __slots__ = ("_conn_manager",)

    def __init__(self, connection_manager: SqliteConnectionManager) -> None:
        self._conn_manager = connection_manager

    async def append_turn(self, session_id: str, turn: ConversationTurn) -> None:
        conn = await self._conn_manager.get_async_connection()
        row = conversation_turn_to_row(turn)
        cursor = await conn.execute(
            "SELECT COALESCE(MAX(turn_index), -1) + 1 AS next_idx "
            "FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        result = await cursor.fetchone()
        turn_index = result["next_idx"]
        await conn.execute(
            "INSERT INTO conversation_turns "
            "(session_id, turn_index, role, content, token_count, metadata_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (session_id, turn_index, row["role"], row["content"],
             row["token_count"], row["metadata_json"], row["created_at"]),
        )
        await conn.commit()

    async def load_turns(self, session_id: str, limit: int | None = None) -> list[ConversationTurn]:
        conn = await self._conn_manager.get_async_connection()
        if limit is not None:
            cursor = await conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index DESC LIMIT ?",
                (session_id, limit),
            )
            rows = list(reversed(await cursor.fetchall()))
        else:
            cursor = await conn.execute(
                "SELECT * FROM conversation_turns WHERE session_id = ? "
                "ORDER BY turn_index ASC",
                (session_id,),
            )
            rows = await cursor.fetchall()
        return [row_to_conversation_turn(r) for r in rows]

    async def save_summary_tiers(self, session_id: str, tiers: dict[int, SummaryTier | None]) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        for level, tier in tiers.items():
            if tier is not None:
                row = summary_tier_to_row(tier)
                await conn.execute(
                    "INSERT INTO summary_tiers "
                    "(session_id, tier_level, content, token_count, source_turn_count, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (session_id, row["tier_level"], row["content"], row["token_count"],
                     row["source_turn_count"], row["created_at"], row["updated_at"]),
                )
        await conn.commit()

    async def load_summary_tiers(self, session_id: str) -> dict[int, SummaryTier | None]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT * FROM summary_tiers WHERE session_id = ?", (session_id,)
        )
        rows = await cursor.fetchall()
        result: dict[int, SummaryTier | None] = {1: None, 2: None, 3: None}
        for row in rows:
            tier = row_to_summary_tier(row)
            result[tier.level] = tier
        return result

    async def truncate_turns(self, session_id: str, keep_last: int) -> None:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT MAX(turn_index) AS max_idx FROM conversation_turns WHERE session_id = ?",
            (session_id,),
        )
        row = await cursor.fetchone()
        if row["max_idx"] is None:
            return
        cutoff = row["max_idx"] - keep_last + 1
        await conn.execute(
            "DELETE FROM conversation_turns WHERE session_id = ? AND turn_index < ?",
            (session_id, cutoff),
        )
        await conn.commit()

    async def delete_session(self, session_id: str) -> bool:
        conn = await self._conn_manager.get_async_connection()
        c1 = await conn.execute("DELETE FROM conversation_turns WHERE session_id = ?", (session_id,))
        c2 = await conn.execute("DELETE FROM summary_tiers WHERE session_id = ?", (session_id,))
        await conn.commit()
        return (c1.rowcount + c2.rowcount) > 0

    async def list_sessions(self) -> list[str]:
        conn = await self._conn_manager.get_async_connection()
        cursor = await conn.execute(
            "SELECT DISTINCT session_id FROM conversation_turns "
            "UNION SELECT DISTINCT session_id FROM summary_tiers"
        )
        rows = await cursor.fetchall()
        return [r["session_id"] for r in rows]

    async def clear(self) -> None:
        conn = await self._conn_manager.get_async_connection()
        await conn.execute("DELETE FROM conversation_turns")
        await conn.execute("DELETE FROM summary_tiers")
        await conn.commit()

    def __repr__(self) -> str:
        return "AsyncSqliteConversationStore()"
