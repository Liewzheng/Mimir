"""Tests for the PreToolUse skill interceptor hook."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from mimir.hooks.skill_interceptor import handle_pre_tool_use, main
from mimir.mcp.session import _workspace_hash
from mimir.skills.store import Skill, SkillStore


class _MockStdin:
    """Minimal stdin replacement with a ``buffer`` attribute."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def _workspace_dir(base_dir: Path, workspace_path: Path) -> Path:
    return (base_dir / _workspace_hash(workspace_path)).resolve()


def _make_payload(command: str, tool_name: str = "Shell") -> dict[str, Any]:
    return {
        "hook_event_name": "PreToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": command},
    }


class TestSkillInterceptor:
    def test_expands_safe_alias(self, tmp_path: Path) -> None:
        store = SkillStore(_workspace_dir(tmp_path, tmp_path) / "skills.jsonl")
        store.add(
            Skill(
                id="s1",
                type="alias",
                name="gs",
                trigger_pattern="gs",
                expansion="git status -sb",
                confidence=0.9,
            )
        )

        result = handle_pre_tool_use(
            _make_payload("gs"),
            tmp_path,
            tmp_path,
        )

        assert result["intercepted"] is True
        assert result["tool_input"] == {"command": "git status -sb"}
        assert result["requires_approval"] is False

    def test_passes_through_when_no_skill_matches(self, tmp_path: Path) -> None:
        result = handle_pre_tool_use(
            _make_payload("unknown shortcut"),
            tmp_path,
            tmp_path,
        )
        assert result["intercepted"] is False
        assert result["tool_input"] == {"command": "unknown shortcut"}

    def test_requires_approval_for_dangerous_expansion(self, tmp_path: Path) -> None:
        store = SkillStore(_workspace_dir(tmp_path, tmp_path) / "skills.jsonl")
        store.add(
            Skill(
                id="s1",
                type="alias",
                name="cleanup",
                trigger_pattern="cleanup",
                expansion="rm -rf /tmp/old",
                confidence=0.9,
            )
        )

        result = handle_pre_tool_use(
            _make_payload("cleanup"),
            tmp_path,
            tmp_path,
        )

        assert result["intercepted"] is False
        assert result["requires_approval"] is True
        assert result["tool_input"] == {"command": "cleanup"}

    def test_main_returns_zero_on_valid_event(self, tmp_path: Path, monkeypatch: Any) -> None:
        payload = json.dumps(_make_payload("gs"))
        monkeypatch.setattr(sys, "stdin", _MockStdin(payload.encode("utf-8")))
        code = main([
            "--workspace-path", str(tmp_path),
            "--base-dir", str(tmp_path),
        ])
        assert code == 0

    def test_main_returns_non_zero_on_invalid_json(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "stdin", _MockStdin(b"not-json"))
        code = main([
            "--workspace-path", str(tmp_path),
            "--base-dir", str(tmp_path),
        ])
        assert code == 1
