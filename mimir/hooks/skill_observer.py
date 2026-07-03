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
from pathlib import Path
from typing import Any

from mimir.infrastructure.persistence.atomic_write import atomic_write_json
from mimir.infrastructure.redaction import Redactor
from mimir.mcp.session import _detect_workspace_path, _workspace_hash
from mimir.skills.extractor import Skeleton
from mimir.skills.injector import InjectorConfig, SkillInjector
from mimir.skills.revisor import SkillRevisor
from mimir.skills.store import Skill, SkillStore
from mimir.skills.tracker import CommandEvent, SkillTracker
from mimir.skills.validator import SkillValidator, _extract_result, _match_template

logger = logging.getLogger(__name__)

_DEFAULT_BASE_DIR = Path.home() / ".mimir" / "workspaces"
_MAX_STDIN_BYTES = 10 * 1024 * 1024  # 10 MiB


class SkillObserverError(Exception):
    """Base error for the skill observer hook."""


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
        raise SkillObserverError(
            f"Workspace path {workspace_path!r} resolves outside the base directory"
        )
    return target


def _load_tracker(workspace_dir: Path) -> SkillTracker:
    """Load or create a tracker for the workspace."""
    tracker_path = workspace_dir / "skill_tracker_state.json"
    if tracker_path.exists():
        try:
            data = json.loads(tracker_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise SkillObserverError("corrupted tracker state: not a JSON object")
            tracker = SkillTracker()
            tracker.restore(data)
            return tracker
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load skill tracker state; starting fresh")
    return SkillTracker()


def _save_tracker(tracker: SkillTracker, workspace_dir: Path) -> None:
    """Persist the full tracker state across hook invocations."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    tracker_path = workspace_dir / "skill_tracker_state.json"
    atomic_write_json(tracker_path, tracker.state())


def _extract_skill(cluster_key: str, cluster: Any, skill_counter: int) -> Skill | None:
    """Convert a ready cluster into a skill record."""
    skeleton: Skeleton | None = getattr(cluster, "skeleton", None)
    if skeleton is None or not skeleton.template:
        return None
    # Use the cluster key as a base name; clean it for human readability.
    name = cluster_key.replace("Shell:", "").replace("Tool:", "").replace("<empty>", "empty")
    skill_type = "alias" if skeleton.variable_count == 0 else "workflow"
    skill_id = f"{name}_{skill_counter}"
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
    """Record a tool call, extract skills, and validate against active skills."""
    ws_dir = _workspace_dir(base_dir, workspace_path)
    tracker = tracker or _load_tracker(ws_dir)
    store = SkillStore(ws_dir / "skills.jsonl")
    injector = SkillInjector(InjectorConfig(max_active=10, min_confidence=0.85))
    validator = SkillValidator(store)
    revisor = SkillRevisor()

    redactor = Redactor()
    tool_name = payload.get("tool_name", "")
    tool_input = payload.get("tool_input") or {}
    command = redactor.redact(_extract_command(tool_name, tool_input))

    event = CommandEvent(tool_name=tool_name, command=command, context=payload)
    tracker.observe(event)

    extracted: list[Skill] = []
    skill_counter = 1
    for cluster in tracker.ready_clusters():
        skill = _extract_skill(cluster.key, cluster, skill_counter)
        skill_counter += 1
        if skill is not None:
            store.add(skill)
            extracted.append(skill)
            tracker.reset(cluster.key)

    # Post-hoc validation: compare the just-executed command with active skills.
    success = _extract_result(payload)
    if success is not None:
        validator.validate(tool_name, command, success)
        # If the command did not match an active skill, try to revise the
        # closest existing skill using the current cluster data.
        if not _command_matches_active_skill(store, command):
            _attempt_revision(store, tracker, revisor, command, tool_name)

    _save_tracker(tracker, ws_dir)

    injection = injector.inject(store.load())

    return {
        "status": "ok",
        "recorded": True,
        "extracted_count": len(extracted),
        "extracted": [s.to_dict() for s in extracted],
        "injection": injection["formatted"],
    }


def _command_matches_active_skill(store: SkillStore, command: str) -> bool:
    """Return True if any active skill template matches the command."""
    for skill in store.load():
        template = skill.template or skill.expansion
        if template and _match_template(template, command):
            return True
    return False


def _attempt_revision(
    store: SkillStore,
    tracker: SkillTracker,
    revisor: SkillRevisor,
    command: str,
    tool_name: str = "Shell",
) -> None:
    """Revise active skills whose cluster skeleton has drifted."""
    from mimir.skills.validator import SafeCommandClassifier

    classifier = SafeCommandClassifier()
    if not classifier.is_safe(tool_name, command):
        return

    event = CommandEvent(tool_name=tool_name, command=command)
    cluster_key = tracker._cluster_key(event)
    cluster = tracker._clusters.get(cluster_key)
    if cluster is None or cluster.skeleton is None:
        return

    for skill in list(store.load()):
        if skill.name != cluster_key.replace("Shell:", "").replace("Tool:", ""):
            continue
        revised = revisor.revise(skill, cluster)
        if revised is not None:
            store.deprecate(skill.id)
            store.add(revised)


def _cluster_key_for_command(command: str) -> str:
    """Compute the tracker cluster key for a shell command."""
    tokens = [t for t in command.split() if t]
    if not tokens:
        return "Shell:<empty>"
    first = tokens[0]
    if len(tokens) >= 2 and not tokens[1].startswith("-"):
        return f"Shell:{first} {tokens[1]}"
    return f"Shell:{first}"


def _read_stdin() -> str:
    """Read stdin with a size limit to avoid memory exhaustion."""
    data = sys.stdin.buffer.read(_MAX_STDIN_BYTES + 1)
    if len(data) > _MAX_STDIN_BYTES:
        raise SkillObserverError(f"stdin exceeds {_MAX_STDIN_BYTES} bytes limit")
    return data.decode("utf-8", errors="replace")


def _error_response(reason: str, format_name: str) -> None:
    """Print a uniform error response."""
    if format_name == "json":
        print(json.dumps({"status": "error", "reason": reason}, ensure_ascii=False))
    else:
        print(f"[Mimir Skill Observer] error: {reason}")


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

    try:
        workspace_path = _detect_workspace_path(args.workspace_path)
        base_dir = args.base_dir or _DEFAULT_BASE_DIR
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to initialize workspace path")
        _error_response(str(exc), args.format)
        return 1

    try:
        payload_text = _read_stdin()
    except SkillObserverError as exc:
        logger.error("stdin error: %s", exc)
        _error_response(str(exc), args.format)
        return 1

    if not payload_text:
        logger.warning("No event payload on stdin")
        _error_response("empty stdin", args.format)
        return 1

    try:
        payload = json.loads(payload_text)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON on stdin: %s", exc)
        _error_response(f"invalid JSON: {exc}", args.format)
        return 1

    try:
        result = handle_post_tool_use(payload, workspace_path, base_dir)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Skill observer failed")
        _error_response(str(exc), args.format)
        return 1

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(f"[Mimir Skill Observer] recorded event; extracted {result['extracted_count']} skill(s)")
        if result["injection"]:
            print(result["injection"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
