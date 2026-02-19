"""Query models for passing context through the pipeline."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryBundle(BaseModel):
    """Encapsulates a query as it flows through the retrieval pipeline.

    Carries the original query text plus optional embedding and metadata
    through all pipeline stages.
    """

    query_str: str
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
