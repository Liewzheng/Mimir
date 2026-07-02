"""Tests for the Mimir agent CLI hook."""

import json
from pathlib import Path
from typing import Any

from mimir.hooks.mimir_turn import main
from mimir.mcp.session import SessionManager


def _run_hook(
    stdin_payload: dict[str, Any],
    extra_args: list[str] | None = None,
    base_dir: Path | None = None,
) -> tuple[int, str, str]:
    import sys
    from io import StringIO

    old_stdin = sys.stdin
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    try:
        sys.stdin = StringIO(json.dumps(stdin_payload))
        stdout_capture = StringIO()
        stderr_capture = StringIO()
        sys.stdout = stdout_capture
        sys.stderr = stderr_capture
        args = ["--backend", "fake"]
        if base_dir is not None:
            args += ["--base-dir", str(base_dir)]
        args += extra_args or []
        code = main(args)
        return code, stdout_capture.getvalue(), stderr_capture.getvalue()
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
        sys.stderr = old_stderr


def test_user_prompt_submit_returns_recall_results(tmp_path: Path) -> None:
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    text = "I love Python and fast sorting algorithms"
    session.store(text)
    session.close()

    payload = {
        "hook_event_name": "UserPromptSubmit",
        "hook_input": {"content": text},
    }
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0
    assert "[Mimir 记忆]" in stdout
    assert text in stdout


def test_user_prompt_submit_no_state_is_silent(tmp_path: Path) -> None:
    payload = {
        "hook_event_name": "UserPromptSubmit",
        "hook_input": {"content": "hello"},
    }
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=tmp_path / ".mimir",
        extra_args=["--workspace-path", str(tmp_path / "workspace")],
    )
    assert code == 0
    assert stdout == ""


def test_user_prompt_submit_kimi_code_payload(tmp_path: Path) -> None:
    """Kimi Code sends the prompt as a list of ContentPart objects."""
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    text = "I love Python and fast sorting algorithms"
    session.store(text)
    session.close()

    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": [{"type": "text", "text": text}],
    }
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0
    assert "[Mimir 记忆]" in stdout
    assert text in stdout


def test_user_prompt_submit_claude_code_payload(tmp_path: Path) -> None:
    """Claude Code sends the prompt as ``user_prompt``."""
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    text = "I love Python and fast sorting algorithms"
    session.store(text)
    session.close()

    payload = {
        "hook_event_name": "UserPromptSubmit",
        "user_prompt": text,
    }
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0
    assert "[Mimir 记忆]" in stdout
    assert text in stdout


def test_user_prompt_submit_codex_payload(tmp_path: Path) -> None:
    """Codex sends the prompt as a plain string."""
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    text = "I love Python and fast sorting algorithms"
    session.store(text)
    session.close()

    payload = {
        "hook_event_name": "UserPromptSubmit",
        "prompt": text,
    }
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0
    assert "[Mimir 记忆]" in stdout
    assert text in stdout


def test_stop_saves_exchange(tmp_path: Path) -> None:
    payload = {
        "hook_event_name": "Stop",
        "messages": [
            {"role": "user", "content": "记住我喜欢用中文"},
            {"role": "assistant", "content": "好的，我会用中文回复你。"},
        ],
    }
    base_dir = tmp_path / ".mimir"
    code, _stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(tmp_path / "workspace")],
    )
    assert code == 0

    # A subsequent recall should find the saved memory.
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    try:
        result = session.recall("中文", top_k=3)
        assert len(result["results"]) > 0
        texts = {r["text"] for r in result["results"]}
        assert any("中文" in t for t in texts)
    finally:
        session.close()


def test_stop_saves_last_assistant_message(tmp_path: Path) -> None:
    """Claude Code / Codex expose only ``last_assistant_message``."""
    payload = {
        "hook_event_name": "Stop",
        "last_assistant_message": "好的，我会用中文回复你。",
    }
    base_dir = tmp_path / ".mimir"
    code, _stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(tmp_path / "workspace")],
    )
    assert code == 0

    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    try:
        result = session.recall("中文", top_k=3)
        assert len(result["results"]) > 0
        texts = {r["text"] for r in result["results"]}
        assert any("中文" in t for t in texts)
    finally:
        session.close()


def test_stop_consolidates_existing_memories(tmp_path: Path) -> None:
    """Kimi Code Stop has no messages; hook should still consolidate existing memories."""
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    session.store("remember this important fact")
    session.close()

    payload = {
        "hook_event_name": "Stop",
        "stop_hook_active": False,
    }
    code, _stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0

    # Memory should still be present after consolidate-only Stop.
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    try:
        assert session.adapter.memory_count == 1
    finally:
        session.close()


def test_session_start_reports_summary(tmp_path: Path) -> None:
    base_dir = tmp_path / ".mimir"
    session = SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=base_dir,
    )
    session.store("hello world")
    session.close()

    payload = {"hook_event_name": "SessionStart", "reason": "startup"}
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=base_dir,
        extra_args=["--workspace-path", str(session.workspace_path)],
    )
    assert code == 0
    assert "已加载" in stdout
    assert "1 条记忆" in stdout


def test_session_start_no_state_reports_empty(tmp_path: Path) -> None:
    payload = {"hook_event_name": "SessionStart", "reason": "startup"}
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=tmp_path / ".mimir",
        extra_args=["--workspace-path", str(tmp_path / "workspace")],
    )
    assert code == 0
    assert "尚无记忆" in stdout


def test_unsupported_event_is_ignored(tmp_path: Path) -> None:
    payload = {"hook_event_name": "Notification", "type": "task.completed"}
    code, stdout, _stderr = _run_hook(
        payload,
        base_dir=tmp_path / ".mimir",
        extra_args=["--workspace-path", str(tmp_path / "workspace")],
    )
    assert code == 0
    assert stdout == ""


def test_invalid_json_is_ignored() -> None:
    import sys
    from io import StringIO

    old_stdin = sys.stdin
    old_stdout = sys.stdout
    try:
        sys.stdin = StringIO("not json")
        sys.stdout = StringIO()
        code = main(["--backend", "fake"])
        assert code == 0
    finally:
        sys.stdin = old_stdin
        sys.stdout = old_stdout
