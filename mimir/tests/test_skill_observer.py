"""Tests for the skill observer hook."""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

from mimir.hooks.skill_observer import handle_post_tool_use, main
from mimir.mcp.session import _workspace_hash
from mimir.skills.store import Skill, SkillStore
from mimir.skills.tracker import SkillTracker


class _MockStdin:
    """Minimal stdin replacement with a ``buffer`` attribute."""

    def __init__(self, data: bytes) -> None:
        self.buffer = io.BytesIO(data)


def _workspace_dir(base_dir: Path, workspace_path: Path) -> Path:
    return (base_dir / _workspace_hash(workspace_path)).resolve()


def _make_payload(command: str, tool_name: str = "Shell") -> dict[str, Any]:
    return {
        "hook_event_name": "PostToolUse",
        "tool_name": tool_name,
        "tool_input": {"command": command},
    }


class TestSkillObserver:
    def test_records_shell_command(self, tmp_path: Path) -> None:
        tracker = SkillTracker()
        result = handle_post_tool_use(
            _make_payload("git status -sb"),
            tmp_path,
            tmp_path,
            tracker=tracker,
        )
        assert result["status"] == "ok"
        assert result["extracted_count"] == 0
        assert len(tracker._buffer) == 1

    def test_extracts_skill_after_repetition(self, tmp_path: Path) -> None:
        tracker = SkillTracker()
        extracted_total = 0
        for i in range(10):
            result = handle_post_tool_use(
                _make_payload(f"adb -s DEV{i} shell reboot bootloader"),
                tmp_path,
                tmp_path,
                tracker=tracker,
            )
            extracted_total += result["extracted_count"]
        assert extracted_total >= 1
        store = SkillStore(_workspace_dir(tmp_path, tmp_path) / "skills.jsonl")
        assert len(store.load()) >= 1

    def test_persists_tracker_state(self, tmp_path: Path) -> None:
        handle_post_tool_use(
            _make_payload("git status -sb"),
            tmp_path,
            tmp_path,
        )
        tracker_path = _workspace_dir(tmp_path, tmp_path) / "skill_tracker_state.json"
        assert tracker_path.exists()
        data = json.loads(tracker_path.read_text(encoding="utf-8"))
        assert data["buffer"]

    def test_redacts_command_before_storage(self, tmp_path: Path) -> None:
        tracker = SkillTracker()
        handle_post_tool_use(
            _make_payload("curl -H 'Authorization: Bearer secret-token' example.com"),
            tmp_path,
            tmp_path,
            tracker=tracker,
        )
        event = tracker._buffer[0]
        assert "secret-token" not in event.command

    def test_main_returns_non_zero_on_invalid_json(self, tmp_path: Path, monkeypatch: Any) -> None:
        monkeypatch.setattr(sys, "stdin", _MockStdin(b"not-json"))
        code = main(["--workspace-path", str(tmp_path), "--base-dir", str(tmp_path)])
        assert code == 1

    def test_main_returns_zero_on_valid_event(self, tmp_path: Path, monkeypatch: Any) -> None:
        payload = json.dumps(_make_payload("git status"))
        monkeypatch.setattr(sys, "stdin", _MockStdin(payload.encode("utf-8")))
        code = main(["--workspace-path", str(tmp_path), "--base-dir", str(tmp_path)])
        assert code == 0

    def test_main_text_format_outputs_summary(self, tmp_path: Path, monkeypatch: Any, capsys: Any) -> None:
        payload = json.dumps(_make_payload("git status"))
        monkeypatch.setattr(sys, "stdin", _MockStdin(payload.encode("utf-8")))
        code = main([
            "--workspace-path", str(tmp_path),
            "--base-dir", str(tmp_path),
            "--format", "text",
        ])
        assert code == 0
        captured = capsys.readouterr()
        assert "Mimir Skill Observer" in captured.out


class TestSkillObserverValidation:
    def _make_payload(
        self,
        command: str,
        tool_name: str = "Shell",
        tool_result: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "hook_event_name": "PostToolUse",
            "tool_name": tool_name,
            "tool_input": {"command": command},
        }
        if tool_result is not None:
            payload["tool_result"] = tool_result
        return payload

    def test_validation_updates_usage_on_success(self, tmp_path: Path) -> None:
        tracker = SkillTracker()
        # Pre-populate an active skill.
        store = SkillStore(_workspace_dir(tmp_path, tmp_path) / "skills.jsonl")
        store.add(
            Skill(
                id="s1",
                type="alias",
                name="git status",
                trigger_pattern="git status -sb",
                expansion="git status -sb",
                confidence=0.9,
            )
        )

        result = handle_post_tool_use(
            self._make_payload("git status -sb", tool_result={"exit_code": 0}),
            tmp_path,
            tmp_path,
            tracker=tracker,
        )
        assert result["status"] == "ok"

        updated = store.get_by_id("s1")
        assert updated is not None
        assert updated.usage_count == 1

    def test_validation_deprecates_skill_after_failures(self, tmp_path: Path) -> None:
        tracker = SkillTracker()
        store = SkillStore(_workspace_dir(tmp_path, tmp_path) / "skills.jsonl")
        store.add(
            Skill(
                id="s1",
                type="alias",
                name="git status",
                trigger_pattern="git status -sb",
                expansion="git status -sb",
                confidence=0.9,
            )
        )

        for _ in range(5):
            result = handle_post_tool_use(
                self._make_payload(
                    "git status -sb",
                    tool_result={"exit_code": 1, "error": "not a repo"},
                ),
                tmp_path,
                tmp_path,
                tracker=tracker,
            )
            assert result["status"] == "ok"

        updated = store.get_by_id("s1")
        assert updated is not None
        assert updated.deprecated
