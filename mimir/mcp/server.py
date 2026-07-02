"""MCP server entry point for Mimir memory integration."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from mcp.server.fastmcp import FastMCP

from mimir.mcp.context import ServerContext
from mimir.mcp.session import SessionManager
from mimir.mcp.tools import (
    checkpoint,
    consolidate,
    forget,
    recall,
    replace_memories,
    restore,
    status,
    store,
    summarize_memories,
)

logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(
    _server: FastMCP,
    backend: str,
    base_url: str,
    model: str,
) -> AsyncIterator[ServerContext]:
    """Initialize and persist the workspace session."""
    session = SessionManager(
        backend=backend,
        base_url=base_url,
        model=model,
    )
    try:
        logger.info(
            "Mimir MCP server started for workspace %s (%s)",
            session.workspace_hash,
            session.workspace_path,
        )
        yield ServerContext(session=session)
    finally:
        session.close()
        logger.info("Mimir MCP server stopped")


def create_server(
    backend: str = "llama-server",
    base_url: str = "http://127.0.0.1:11435",
    model: str = "all-MiniLM-L6-v2",
) -> FastMCP:
    """Create a configured Mimir MCP server instance."""

    @asynccontextmanager
    async def lifespan(server: FastMCP) -> AsyncIterator[ServerContext]:
        async with _lifespan(server, backend, base_url, model) as ctx:
            yield ctx

    mcp = FastMCP("Mimir", lifespan=lifespan)

    mcp.add_tool(store)
    mcp.add_tool(recall)
    mcp.add_tool(consolidate)
    mcp.add_tool(forget)
    mcp.add_tool(checkpoint)
    mcp.add_tool(restore)
    mcp.add_tool(status)
    mcp.add_tool(summarize_memories)
    mcp.add_tool(replace_memories)

    return mcp


def run_server(
    backend: str = "llama-server",
    base_url: str = "http://127.0.0.1:11435",
    model: str = "all-MiniLM-L6-v2",
) -> None:
    """Run the Mimir MCP server over stdio."""
    mcp = create_server(backend=backend, base_url=base_url, model=model)
    mcp.run(transport="stdio")
