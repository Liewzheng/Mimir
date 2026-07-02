"""Shared lifespan context type for the Mimir MCP server."""

from dataclasses import dataclass

from mimir.mcp.session import SessionManager


@dataclass
class ServerContext:
    """Lifespan-scoped context shared across MCP requests."""

    session: SessionManager
