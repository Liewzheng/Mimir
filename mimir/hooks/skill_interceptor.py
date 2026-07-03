"""PreToolUse hook for skill shortcut expansion.

This module intercepts an incoming tool call before it executes, checks whether
it matches a stored skill, and returns a possibly expanded tool input. It never
executes commands itself; expansion is purely textual and conservative.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

from mimir.mcp.session import _detect_workspace_path, _workspace_hash
from mimir.skills.expander import SkillExpander
from mimir.skills.store import SkillStore
from mimir.skills.validator import SafeCommandClassifier

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".mimir" / "workspaces"
_MAX_STDIN_BYTES = 10 * 1024 * 1024  # 10 MiB


class SkillInterceptorError(Exception):
    """Base error for the skill interceptor hook."""


def _workspace_dir(base_dir: Path, workspace_path: Path) -> Path:
    """Return the workspace directory for ``workspace_path``."""
    base = base_dir.resolve()
    target = (base / _workspace_hash(workspace_path)).resolve()
    if not target.is_relative_to(base):
        raise SkillInterceptorError(
            f"Workspace path {workspace_path!r} resolves outside the base directory"
        )
    return target


def _read_stdin() -> str:
    """Read stdin with a size limit to avoid memory exhaustion."""
    data = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
    if len(data) > _MAX_STDIN_BYTES:
        raise SkillInterceptorError(f"stdin exceeds {_MAX_STDIN_BYTES} bytes limit")
    return data.decode("utf-8", errors="replace")


def _extract_command(tool_input: dict[str, Any]) -> str:
    """Return the shell command from a tool input."""
    command = tool_input.get("command", "")
    return command if isinstance(command, str) else str(command)


def handle_pre_tool_use(
    payload: dict[str, Any],
    workspace_path: Path,
    base_dir: Path,
    min_confidence: float = 0.85,
) -> dict[str, Any]:
    """Check the incoming tool call for a skill shortcut and expand it."""
    ws_dir = _workspace_dir(base_dir, workspace_path)
    store = SkillStore(ws_dir / "skills.jsonl")
    expander = SkillExpander(
        store.load(),
        min_confidence=min_confidence,
        classifier=SafeCommandClassifier(),
    )

    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}

    expanded = expander.expand(tool_name, tool_input)

    if expanded is None:
        return {
            "tool_input": tool_input,
            "intercepted": False,
            "requires_approval": False,
            "reason": "no matching skill",
        }

    expanded_command = _extract_command(expanded)
    # The expander already refuses to expand into dangerous commands. The
    # approval gate below is for future protocol support where the agent CLI
    # can surface a prompt to the user.
    if not SafeCommandClassifier().is_safe("Shell", expanded_command):
        return {
            "tool_input": tool_input,
            "intercepted": False,
            "requires_approval": True,
            "reason": f"expanded command requires approval: {expanded_command}",
        }

    return {
        "tool_input": expanded,
        "intercepted": True,
        "requires_approval": False,
        "reason": "",
    }


def _error_response(reason: str) -> None:
    """Print a uniform error response."""
    print(
        json.dumps(
            {
                "tool_input": {},
                "intercepted": False,
                "requires_approval": False,
                "reason": reason,
            },
            ensure_ascii=False,
        )
    )


def main(argv: list[str] | None = None) -> int:
    """Entry point for the PreToolUse hook."""
    parser = argparse.ArgumentParser(description="Mimir skill interceptor hook for PreToolUse events.")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=None,
        help="Root directory for all workspaces",
    )
    parser.add_argument(
        "--workspace-path",
        type=Path,
        default=None,
        help="Override the detected workspace path",
    )
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.85,
        help="Minimum skill confidence to expand (default: %(default)s)",
    )
    parser.add_argument(
        "--format",
        choices=["json", "text"],
        default="json",
        help="Output format (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    try:
        workspace_path = _detect_workspace_path(args.workspace_path)
        base_dir = args.base_dir or _DEFAULT_BASE_DIR
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize workspace path")
        _error_response(str(exc))
        return 1

    try:
        payload_text = _read_stdin()
    except SkillInterceptorError as exc:
        logger.error("stdin error: %s", exc)
        _error_response(str(exc))
        return 1

    if not payload_text:
        logger.warning("No event payload on stdin")
        _error_response("empty stdin")
        return 1

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON on stdin: %s", exc)
        _error_response(f"invalid JSON: {exc}")
        return 1

    try:
        result = handle_pre_tool_use(payload, workspace_path, base_dir, args.min_confidence)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Skill interceptor failed")
        _error_response(str(exc))
        return 1

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
    else:
        if result["intercepted"]:
            print(f"[Mimir Skill Interceptor] expanded to: {result['tool_input'].get('command', '')}")
        else:
            print("[Mimir Skill Interceptor] no expansion")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
