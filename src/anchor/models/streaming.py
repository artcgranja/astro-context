"""Streaming event models for LLM response tracking."""

from __future__ import annotations

from pydantic import BaseModel, Field


class StreamDelta(BaseModel):
    """A single text delta from a streaming LLM response."""

    text: str
    index: int = 0


class StreamUsage(BaseModel):
    """Token usage from a completed streaming response."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0


class StreamResult(BaseModel):
    """Accumulated result from a completed streaming response."""

    text: str = ""
    usage: StreamUsage = Field(default_factory=StreamUsage)
    model: str = ""
    stop_reason: str = ""
