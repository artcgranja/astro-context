"""MCP Bridge — bidirectional Model Context Protocol integration.

Exposes anchor capabilities as MCP tools/resources/prompts and
consumes external MCP servers from anchor's Agent.
"""

from __future__ import annotations

from anchor.mcp.errors import (
    MCPConfigError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
)

__all__ = [
    "MCPConfigError",
    "MCPConnectionError",
    "MCPError",
    "MCPTimeoutError",
    "MCPToolError",
]
