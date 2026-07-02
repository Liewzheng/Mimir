"""Tests for the Mimir MCP session manager."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from mimir.infrastructure.redaction import Redactor
from mimir.mcp.session import SessionManager


@pytest.fixture
def session(tmp_path: Path) -> SessionManager:
    """Return a SessionManager using a temporary fake backend workspace."""
    return SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=tmp_path / ".mimir",
    )


def test_workspace_hash_is_stable(session: SessionManager) -> None:
    assert len(session.workspace_hash) == 16
    assert session.workspace_dir.exists()


def test_store_adds_memory(session: SessionManager) -> None:
    result = session.store("hello world")
    assert result["stored"] is True
    assert result["memory_count"] == 1
    assert session.adapter.memory_count == 1
    state = session.adapter.memories_state()
    assert state[0]["text"] == "hello world"


def test_recall_returns_results(session: SessionManager) -> None:
    session.store("I like apples")
    result = session.recall("I like apples", top_k=3)
    assert result["query"] == "I like apples"
    assert len(result["results"]) > 0


def test_recall_empty_memory(session: SessionManager) -> None:
    result = session.recall("anything")
    assert result["results"] == []


def test_consolidate_reinforces(session: SessionManager) -> None:
    session.store("hello")
    before = session.adapter._mimir.store.prototypes.clone()
    result = session.consolidate()
    after = session.adapter._mimir.store.prototypes.clone()
    assert result["memories_reinforced"] == 1
    assert not (before == after).all()


def test_forget_clears(session: SessionManager) -> None:
    session.store("hello")
    result = session.forget()
    assert result["cleared_memories"] == 1
    assert result["memory_count"] == 0
    assert session.adapter.memory_count == 0


def test_checkpoint_and_restore(session: SessionManager) -> None:
    session.store("hello")
    session.checkpoint("v1")

    session.store("world")
    assert session.adapter.memory_count == 2

    session.restore("v1")
    assert session.adapter.memory_count == 1
    assert session.adapter.memories_state()[0]["text"] == "hello"


def test_checkpoint_invalid_name(session: SessionManager) -> None:
    with pytest.raises(ValueError, match="letters, digits"):
        session.checkpoint("../etc")


def test_restore_missing_checkpoint(session: SessionManager) -> None:
    with pytest.raises(FileNotFoundError, match="Checkpoint 'missing' not found"):
        session.restore("missing")


def test_store_blocks_duplicates(session: SessionManager) -> None:
    session.store("I like Python")
    result = session.store("I like Python")
    assert result["stored"] is False
    assert result["reason"] == "duplicate"
    assert "similar_memory" in result
    assert session.adapter.memory_count == 1


def test_store_reports_contradiction_hint(session: SessionManager) -> None:
    session.store("I use Python for backends")
    result = session.store("I do not use Python for backends")
    assert result["stored"] is True
    assert "contradictions" in result
    assert len(result["contradictions"]) >= 1


def test_store_duplicate_can_be_disabled(session: SessionManager) -> None:
    session.config = replace(session.config, quality_gate_enabled=False)
    session.store("I like Python")
    result = session.store("I like Python")
    assert result["stored"] is True
    assert session.adapter.memory_count == 2


def test_store_contradiction_hints_can_be_disabled(session: SessionManager) -> None:
    session.config = replace(session.config, quality_gate_enabled=False)
    session.store("I use Python for backends")
    result = session.store("I do not use Python for backends")
    assert result["stored"] is True
    assert "contradictions" not in result


def test_store_rejects_empty_and_non_string_input(session: SessionManager) -> None:
    assert session.store("")["stored"] is False
    assert session.store(None)["stored"] is False  # type: ignore[arg-type]
    assert session.store(123)["stored"] is False  # type: ignore[arg-type]
    assert session.adapter.memory_count == 0


def test_store_redacts_secrets(session: SessionManager) -> None:
    raw = "my key is sk-1234567890abcdef"
    result = session.store(raw)
    assert result["stored"] is True
    assert "sk-1234567890abcdef" not in result["text"]
    assert "[REDACTED]" in result["text"]
    state = session.adapter.memories_state()
    assert state[0]["text"] == result["text"]


def test_project_context_discovery_ingests_files(session: SessionManager) -> None:
    session.workspace_path.mkdir(parents=True, exist_ok=True)
    agents_md = session.workspace_path / "AGENTS.md"
    agents_md.write_text("Use Python for scripts.", encoding="utf-8")

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    texts = [m["text"] for m in session2.adapter.memories_state()]
    assert any("Project Context" in t for t in texts)
    assert any("Use Python for scripts" in t for t in texts)


def test_store_custom_redaction_pattern(session: SessionManager) -> None:
    session.config = replace(
        session.config,
        redaction_patterns=[r"secret-\w+"],
    )
    session.redactor = Redactor(
        enabled=session.config.redaction_enabled,
        patterns=session.config.redaction_patterns,
    )
    result = session.store("my code is secret-abc123")
    assert result["stored"] is True
    assert "secret-abc123" not in result["text"]
    assert "[REDACTED]" in result["text"]


def test_status_reports(session: SessionManager) -> None:
    session.store("hello")
    session.checkpoint("v1")
    result = session.status()
    assert result["backend"] == "fake"
    assert result["memory_count"] == 1
    assert "v1" in result["checkpoints"]


def test_list_memories(session: SessionManager) -> None:
    session.store("I like apples")
    session.store("I hate bananas")
    result = session.list_memories()
    assert result["memory_count"] == 2
    assert result["memories"][0]["text"] == "I like apples"
    assert result["memories"][1]["text"] == "I hate bananas"


def test_replace_memories(session: SessionManager) -> None:
    session.store("old memory one")
    session.store("old memory two")
    result = session.replace_memories(["new summary A", "new summary B"])
    assert result["replaced"] is True
    assert result["memory_count"] == 2
    state = session.adapter.memories_state()
    assert [item["text"] for item in state] == ["new summary A", "new summary B"]


def test_replace_memories_persists_immediately(session: SessionManager) -> None:
    session.store("old memory")
    session.replace_memories(["persisted summary"])

    # Simulate a separate process reading from disk, like agent CLI hooks do.
    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    assert session2.adapter.memory_count == 1
    assert session2.adapter.memories_state()[0]["text"] == "persisted summary"


def test_close_persists_and_reloads(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    assert session2.adapter.memory_count == 1
    assert session2.adapter.memories_state()[0]["text"] == "hello"


def test_close_saves_memories_json(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    assert session.memories_path.exists()
    data = json.loads(session.memories_path.read_text(encoding="utf-8"))
    assert len(data) == 1
    assert data[0]["text"] == "hello"


def test_status_reports_load_errors(session: SessionManager) -> None:
    session.store("hello")
    session._load_errors.append("simulated load error")

    result = session.status()
    assert result["load_errors"] == ["simulated load error"]


def test_close_propagates_save_failure(session: SessionManager) -> None:
    session.store("hello")
    session.workspace_dir.chmod(0o555)
    try:
        with pytest.raises(OSError):
            session.close()
    finally:
        session.workspace_dir.chmod(0o755)


def test_load_corrupted_state_keeps_memories(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    # Corrupt the Mimir checkpoint so restore() fails.
    session.mimir_path.write_bytes(b"not a valid checkpoint")

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    # Memories are still recovered from memories.json even when the checkpoint
    # is corrupted, so nothing is silently lost.
    assert session2.adapter.memory_count == 1
    assert session2.adapter.memories_state()[0]["text"] == "hello"
    assert len(session2._load_errors) == 1
    assert session2.status()["load_errors"]


def test_load_without_mimir_checkpoint_keeps_memories(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    # Simulate a state where only the memories snapshot survives.
    session.mimir_path.unlink()

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    assert session2.adapter.memory_count == 1
    assert session2.adapter.memories_state()[0]["text"] == "hello"
    assert not session2._load_errors


def test_legacy_session_checkpoint_migration(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    # Simulate the pre-v0.2 layout where the primary checkpoint lived as
    # checkpoints/mimir.pt.
    legacy_path = session.checkpoints_dir / "mimir.pt"
    session.mimir_path.replace(legacy_path)

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    assert session2.adapter.memory_count == 1
    assert not session2._load_errors
    assert session.mimir_path.exists()
    assert not legacy_path.exists()


def test_load_corrupted_memories_starts_fresh(session: SessionManager) -> None:
    session.store("hello")
    session.close()

    # Corrupt the memories JSON so load_memories_state() fails.
    session.memories_path.write_text("not json", encoding="utf-8")

    session2 = SessionManager(
        backend="fake",
        workspace_path=session.workspace_path,
        base_dir=session.workspace_dir.parent,
    )
    assert session2.adapter.memory_count == 0
    assert any("memories" in err for err in session2._load_errors)


def test_checkpoint_round_trip_integrity(session: SessionManager) -> None:
    session.store("hello")
    session.store("world")
    session.checkpoint("v1")

    session.store("extra")
    assert session.adapter.memory_count == 3

    session.restore("v1")
    assert session.adapter.memory_count == 2
    texts = {m["text"] for m in session.adapter.memories_state()}
    assert texts == {"hello", "world"}


def test_checkpoint_name_path_traversal(session: SessionManager) -> None:
    with pytest.raises(ValueError, match="letters, digits"):
        session.checkpoint("../evil")
    with pytest.raises(ValueError, match="letters, digits"):
        session.checkpoint("name with spaces")


def test_restore_missing_checkpoint_lists_available(session: SessionManager) -> None:
    session.checkpoint("v1")
    with pytest.raises(FileNotFoundError, match="Checkpoint 'missing' not found"):
        session.restore("missing")


def test_checkpoint_restore_empty_session(session: SessionManager) -> None:
    session.checkpoint("empty")
    session.store("hello")
    session.restore("empty")
    assert session.adapter.memory_count == 0


def test_restore_without_memories_file_clears_buffer(session: SessionManager) -> None:
    session.store("hello")
    session.checkpoint("v1")

    # Remove the memories sidecar.
    memories_file = session.checkpoints_dir / "v1_memories.json"
    memories_file.unlink()

    session.store("world")
    session.restore("v1")
    assert session.adapter.memory_count == 0
