"""Tests for Mimir MCP tool functions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest
from mcp.server.fastmcp import Context

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


@dataclass
class _FakeRequestContext:
    lifespan_context: ServerContext


@pytest.fixture
def session(tmp_path: Path) -> SessionManager:
    """Return a SessionManager backed by the fake embedding backend."""
    return SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=tmp_path / ".mimir",
    )


@pytest.fixture
def ctx(session: SessionManager) -> Context[Any, ServerContext, Any]:
    """Return an MCP Context wrapping the test session."""
    return Context[Any, ServerContext, Any](
        request_context=cast(Any, _FakeRequestContext(ServerContext(session=session)))
    )


def test_store(ctx: Context[Any, ServerContext, Any], session: SessionManager) -> None:
    result = json.loads(store("hello", ctx=ctx))
    assert result["stored"] is True
    assert len(session.adapter._memories) == 1


def test_recall(ctx: Context[Any, ServerContext, Any]) -> None:
    store("I like apples", ctx=ctx)
    result = json.loads(recall("I like apples", ctx=ctx))
    assert len(result["results"]) > 0


def test_consolidate(ctx: Context[Any, ServerContext, Any], session: SessionManager) -> None:
    store("hello", ctx=ctx)
    before = session.adapter._mimir.store.prototypes.clone()
    result = json.loads(consolidate(ctx=ctx))
    after = session.adapter._mimir.store.prototypes.clone()
    assert result["memories_reinforced"] == 1
    assert not (before == after).all()


def test_forget(ctx: Context[Any, ServerContext, Any]) -> None:
    store("hello", ctx=ctx)
    result = json.loads(forget(ctx=ctx))
    assert result["forgotten"] is True


def test_checkpoint_and_restore(
    ctx: Context[Any, ServerContext, Any], session: SessionManager
) -> None:
    store("hello", ctx=ctx)
    json.loads(checkpoint("v1", ctx=ctx))
    store("world", ctx=ctx)
    assert len(session.adapter._memories) == 2

    json.loads(restore("v1", ctx=ctx))
    assert len(session.adapter._memories) == 1


def test_status(ctx: Context[Any, ServerContext, Any]) -> None:
    store("hello", ctx=ctx)
    result = json.loads(status(ctx=ctx))
    assert result["memory_count"] == 1
    assert result["backend"] == "fake"


def test_summarize_memories(ctx: Context[Any, ServerContext, Any]) -> None:
    store("I like apples", ctx=ctx)
    store("I hate bananas", ctx=ctx)
    result = summarize_memories(ctx=ctx)
    assert "I like apples" in result
    assert "I hate bananas" in result
    assert "mimir_replace_memories" in result


def test_replace_memories_tool(ctx: Context[Any, ServerContext, Any]) -> None:
    store("old one", ctx=ctx)
    store("old two", ctx=ctx)
    result = json.loads(replace_memories(["summary A", "summary B"], ctx=ctx))
    assert result["replaced"] is True
    assert result["memory_count"] == 2
