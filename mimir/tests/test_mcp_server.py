"""Tests for Mimir MCP server wiring."""

import pytest
from mcp.server.fastmcp import FastMCP

from mimir.mcp.server import create_server


@pytest.fixture
def mcp() -> FastMCP:
    """Return an Mimir MCP server instance with the fake backend."""
    return create_server(backend="fake")


def test_create_server(mcp: FastMCP) -> None:
    assert mcp.name == "Mimir"


@pytest.mark.anyio
async def test_server_lists_tools(mcp: FastMCP) -> None:
    tools = await mcp.list_tools()
    names = {tool.name for tool in tools}
    assert names == {
        "store",
        "recall",
        "consolidate",
        "forget",
        "checkpoint",
        "restore",
        "status",
        "summarize_memories",
        "replace_memories",
    }
