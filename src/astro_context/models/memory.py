"""Memory models for conversation history and persistent memory."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import Any, Literal, TypeAlias

from pydantic import BaseModel, Field

Role: TypeAlias = Literal["user", "assistant", "system", "tool"]


class ConversationTurn(BaseModel):
    """A single turn in a conversation (user or assistant message)."""

    role: Role
    content: str
    token_count: int = Field(default=0, ge=0)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(UTC))
    metadata: dict[str, Any] = Field(default_factory=dict)


class MemoryEntry(BaseModel):
    """A persistent memory entry with relevance tracking.

    Inspired by Mem0's memory entries with priority scoring.
    """

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    content: str
    relevance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    access_count: int = Field(default=0, ge=0)
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(UTC))
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
