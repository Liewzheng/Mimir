"""Observer hook for PostToolUse events.

This module receives tool-call events from agent CLI hooks, records them in the
skill tracker, and optionally triggers skill extraction. It is intentionally
observational: it does not block or intercept tool calls in the first phase.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mimir.infrastructure.redaction import Redactor
from mimir.mcp.session import _detect_workspace_path, _workspace_hash
from mimir.skills.extractor import Skeleton
from mimir.skills.injector import InjectorConfig, SkillInjector
from mimir.skills.store import Skill, SkillStore
from mimir.skills.tracker import CommandEvent, SkillTracker

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".mimir" / "workspaces"


def _extract_command(tool_name: str, tool_input: dict[str, Any]) -> str:
    """Extract the human-readable command string from a tool input."""
    if tool_name == "Shell":
        command = tool_input.get("command", "")
        return command if isinstance(command, str) else str(command)
    # For other tools, use a JSON representation.
    return json.dumps(tool_input, ensure_ascii=False, sort_keys=True)


def _workspace_dir(base_dir: Path, workspace_path: Path) -> Path:
    """Return the workspace directory for ``workspace_path``."""
    base = base_dir.resolve()
    target = (base / _workspace_hash(workspace_path)).resolve()
    if not target.is_relative_to(base):
        raise ValueError(f"Workspace path {workspace_path!r} resolves outside the base directory")
    return target


def _load_tracker(workspace_dir: Path) -> SkillTracker:
    """Load or create a tracker for the workspace."""
    tracker_path = workspace_dir / "skill_tracker_state.json"
    if tracker_path.exists():
        try:
            data = json.loads(tracker_path.read_text(encoding="utf-8"))
            tracker = SkillTracker()
            tracker.restore(data)
            return tracker
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load skill tracker state; starting fresh")
    return SkillTracker()


def _save_tracker(tracker: SkillTracker, workspace_dir: Path) -> None:
    """Persist the full tracker state across hook invocations."""
    tracker_path = workspace_dir / "skill_tracker_state.json"
    tracker_path.write_text(json.dumps(tracker.state(), indent=2), encoding="utf-8")


def _extract_skill(cluster_key: str, cluster: Any) -> Skill | None:
    """Convert a ready cluster into a skill record."""
    skeleton: Skeleton | None = getattr(cluster, "skeleton", None)
    if skeleton is None or not skeleton.template:
        return None
    # Use the cluster key as a base name; clean it for human readability.
    name = cluster_key.replace("Shell:", "").replace("Tool:", "").replace("<empty>", "empty")
    skill_type = "alias" if skeleton.variable_count == 0 else "workflow"
    skill_id = f"{name}_{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
    return Skill(
        id=skill_id,
        type=skill_type,  # type: ignore[arg-type]
        name=name,
        trigger_pattern=skeleton.template,
        template=skeleton.template if skill_type == "workflow" else None,
        expansion=skeleton.template if skill_type == "alias" else None,
        confidence=cluster.skeleton.fixed_ratio if cluster.skeleton else 0.0,
    )


def handle_post_tool_use(
    payload: dict[str, Any],
    workspace_path: Path,
    base_dir: Path,
    tracker: SkillTracker | None = None,
) -> dict[str, Any]:
    """Record a tool call and extract skills if any cluster is ready."""
    ws_dir = _workspace_dir(base_dir, workspace_path)
    tracker = tracker or _load_tracker(ws_dir)
    store = SkillStore(ws_dir / "skills.jsonl")

    redactor = Redactor()
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    command = redactor.redact(_extract_command(tool_name, tool_input))

    event = CommandEvent(tool_name=tool_name, command=command, context=payload)
    tracker.observe(event)

    extracted: list[Skill] = []
    for cluster in tracker.ready_clusters():
        skill = _extract_skill(cluster.key, cluster)
        if skill is not None:
            store.add(skill)
            extracted.append(skill)
            tracker.reset(cluster.key)

    _save_tracker(tracker, ws_dir)

    injector = SkillInjector(InjectorConfig(max_active=10, min_confidence=0.85))
    injection = injector.inject(store.load())

    return {
        "status": "ok",
        "recorded": True,
        "extracted": [s.to_dict() for s in extracted],
        "injection": injection["formatted"],
    }


def main(argv: list[str] | None = None) -> int:
    """Entry point for the PostToolUse hook."""
    parser = argparse.ArgumentParser(description="Mimir skill observer hook for PostToolUse events.")
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
        "--format",
        choices=["text", "json"],
        default="json",
        help="Output format (default: %(default)s)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    workspace_path = _detect_workspace_path(args.workspace_path)
    base_dir = args.base_dir or _DEFAULT_BASE_DIR

    payload_text = sys.stdin.read()
    if not payload_text:
        logger.warning("No event payload on stdin")
        if args.format == "json":
            print(json.dumps({"status": "error", "reason": "empty stdin"}, ensure_ascii=False))
        return 0

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON on stdin: %s", exc)
        if args.format == "json":
            print(
                json.dumps(
                    {"status": "error", "reason": f"invalid JSON: {exc}"},
                    ensure_ascii=False,
                )
            )
        return 0

    try:
        result = handle_post_tool_use(payload, workspace_path, base_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Skill observer failed")
        if args.format == "json":
            print(json.dumps({"status": "error", "reason": str(exc)}, ensure_ascii=False))
        return 0

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
