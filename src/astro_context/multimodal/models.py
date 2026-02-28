"""Multi-modal content models for astro-context."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from astro_context.models.context import SourceType


class ModalityType(StrEnum):
    """The type of content modality."""

    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    CODE = "code"
    AUDIO = "audio"


class MultiModalContent(BaseModel):
    """Represents content that may include multiple modalities.

    Each instance captures a single modality (text, image, table, etc.)
    along with optional raw binary data and MIME type information.
    Content is immutable after creation.
    """

    model_config = ConfigDict(frozen=True)

    modality: ModalityType
    content: str
    raw_data: bytes | None = None
    mime_type: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class MultiModalItem(BaseModel):
    """A context item with multi-modal content.

    Groups one or more ``MultiModalContent`` pieces into a single
    retrievable unit, mirroring the structure of ``ContextItem`` but
    supporting heterogeneous content types.  Immutable after creation.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    contents: list[MultiModalContent]
    source: SourceType
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)
    token_count: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
