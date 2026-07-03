"""MCP tool definitions for Mimir memory operations.

All tools operate on the workspace-scoped session created when the MCP server
started. Memories, checkpoints, and status are isolated per workspace under
``~/.mimir/workspaces/<hash>/``.
"""

from __future__ import annotations

import json
from typing import Any

from mcp.server.fastmcp import Context

from mimir.mcp.context import ServerContext
from mimir.mcp.session import SessionManager


def _session(ctx: Context[Any, ServerContext, Any]) -> SessionManager:
    """Extract the SessionManager from the lifespan context."""
    return ctx.request_context.lifespan_context.session


def store(text: str, importance: float = 1.0, *, ctx: Context[Any, ServerContext, Any]) -> str:
    """Store a fact in the workspace's long-term memory.

    The text is observed and learned (the Mimir prototype matrix is updated),
    so it can be retrieved later by `recall`. Use this for user preferences,
    project conventions, decisions, or any context that should persist across
    turns.

    When the server is configured for async storage, the embedding backend and
    learning happen on a background thread. In that case the response contains
    `{"stored": "pending"}` and the fact is queued for later processing.

    Args:
        text: The text to remember. Non-empty strings only.
        importance: Learning weight (default 1.0). Higher values emphasize the
            fact during consolidation.

    Returns:
        JSON with `stored` (True, "pending", or False), `text`, `memory_count`,
        `capacity_usage`, and optionally `pending_count` / `reason`.
    """
    result = _session(ctx).store(text, importance=importance)
    return json.dumps(result, ensure_ascii=False, indent=2)


def recall(
    query: str,
    top_k: int = 5,
    min_score: float = 0.0,
    *,
    ctx: Context[Any, ServerContext, Any],
) -> str:
    """Retrieve memories most relevant to the query.

    Results are ranked by a hybrid score that combines vector cosine similarity
    with BM25 keyword overlap, then reranks by lifecycle metadata (recency,
    importance, access patterns). If no memory reaches `min_score`, an empty
    result list is returned.

    Args:
        query: The query text. Empty queries return no results.
        top_k: Maximum number of memories to return (default 5).
        min_score: Minimum fused relevance score, 0.0 to 1.0 (default 0.0,
            no filtering). Raise this to exclude low-relevance matches.

    Returns:
        JSON with `query` and `results`. Each result has `text`, `score`, and
        `created_at`.
    """
    result = _session(ctx).recall(query, top_k=top_k, min_score=min_score)
    return json.dumps(result, ensure_ascii=False, indent=2)


def consolidate(*, ctx: Context[Any, ServerContext, Any]) -> str:
    """Reinforce all memories currently in the working buffer.

    This runs the learning pipeline over the buffered memories again, making
    them more stable in the prototype matrix. It is useful after a batch of
    `store` calls or at the end of a turn.
    """
    result = _session(ctx).consolidate()
    return json.dumps(result, ensure_ascii=False, indent=2)


def forget(*, ctx: Context[Any, ServerContext, Any]) -> str:
    """Clear the working memory and reset the prototype matrix for this workspace.

    This removes in-memory memories and resets learned prototypes. It does NOT
    delete named checkpoints created with `checkpoint`; use `restore` to roll
    back to one of those.
    """
    result = _session(ctx).forget()
    return json.dumps(result, ensure_ascii=False, indent=2)


def checkpoint(name: str, *, ctx: Context[Any, ServerContext, Any]) -> str:
    """Save the current memory state to a named checkpoint.

    Checkpoints are scoped to the current workspace. The name may only contain
    letters, digits, underscores, and hyphens.

    Args:
        name: Checkpoint identifier. Allowed: `a-zA-Z0-9_-`.

    Returns:
        JSON with `checkpoint`, `saved`, and `memory_count`.
    """
    result = _session(ctx).checkpoint(name)
    return json.dumps(result, ensure_ascii=False, indent=2)


def restore(name: str, *, ctx: Context[Any, ServerContext, Any]) -> str:
    """Restore a named checkpoint created with `checkpoint`.

    The current unsaved working memory is discarded. If the checkpoint does not
    exist, the response lists available checkpoint names.

    Args:
        name: The checkpoint to restore.

    Returns:
        JSON with `checkpoint`, `restored`, and `memory_count`, or an error.
    """
    result = _session(ctx).restore(name)
    return json.dumps(result, ensure_ascii=False, indent=2)


def status(*, ctx: Context[Any, ServerContext, Any]) -> str:
    """Return workspace memory statistics.

    Includes the workspace path and hash, embedding backend, memory count,
    prototype capacity/usage, current learning step, available checkpoints, and
    any load-time errors.
    """
    result = _session(ctx).status()
    return json.dumps(result, ensure_ascii=False, indent=2)


def summarize_memories(*, ctx: Context[Any, ServerContext, Any]) -> str:
    """Prepare the current working memory for LLM-based consolidation.

    Returns a numbered list of all buffered memories together with instructions
    asking the LLM to merge duplicates, drop temporary information, and rewrite
    the rest as concise factual statements. After reviewing, call
    `replace_memories` to write the cleaned list back.
    """
    result = _session(ctx).list_memories()
    lines: list[str] = [
        "请整理以下 Mimir 工作记忆：",
        "",
        "- 合并重复或高度相似的内容",
        "- 删除临时性、无长期价值的内容",
        "- 把用户偏好、规则、决策、重要事实保留为简洁陈述句",
        "- 输出结果请调用 mimir_replace_memories(memories=[...])",
        "",
        f"当前共有 {result['memory_count']} 条记忆：",
    ]
    for item in result["memories"]:
        lines.append(f"{item['index']}. {item['text']}")
    if not result["memories"]:
        lines.append("（暂无记忆）")
    return "\n".join(lines)


def replace_memories(memories: list[str], *, ctx: Context[Any, ServerContext, Any]) -> str:
    """Replace all working memories with a cleaned list.

    The prototype matrix is reset and retrained on the new list, and the result
    is persisted immediately. This is the write-back step after
    `summarize_memories`.

    Args:
        memories: List of concise factual strings. Empty or non-string items
            are ignored.

    Returns:
        JSON with `replaced` and `memory_count`.
    """
    result = _session(ctx).replace_memories(memories)
    return json.dumps(result, ensure_ascii=False, indent=2)
