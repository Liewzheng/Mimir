"""Tests for `mimir setup` command."""

from pathlib import Path

import pytest

from mimir.setup import (
    ClaudeCodeSetup,
    CodexSetup,
    KimiCodeSetup,
    create_setup,
    list_supported_agents,
)


class TestSetupRegistry:
    def test_supported_agents(self) -> None:
        agents = list_supported_agents()
        assert "kimi-code" in agents
        assert "claude-code" in agents
        assert "codex" in agents

    def test_create_setup_unknown(self) -> None:
        with pytest.raises(ValueError):
            create_setup("unknown-agent")


class TestKimiCodeSetup:
    def test_install_creates_config(self, tmp_path: Path) -> None:
        setup = KimiCodeSetup(home_dir=tmp_path)
        path = setup.install()
        assert path.exists()
        assert "mimir.hooks.mimir_turn" in path.read_text()
        assert "UserPromptSubmit" in path.read_text()
        assert "Stop" in path.read_text()

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        setup = KimiCodeSetup(home_dir=tmp_path)
        setup.install()
        first_size = setup.config_path.stat().st_size
        setup.install()
        assert setup.config_path.stat().st_size == first_size

    def test_is_installed(self, tmp_path: Path) -> None:
        setup = KimiCodeSetup(home_dir=tmp_path)
        assert not setup.is_installed()
        setup.install()
        assert setup.is_installed()


class TestClaudeCodeSetup:
    def test_install_creates_settings(self, tmp_path: Path) -> None:
        setup = ClaudeCodeSetup(config_dir=tmp_path)
        path = setup.install()
        assert path.exists()
        data = __import__("json").loads(path.read_text())
        assert "Stop" in data
        assert "UserPromptSubmit" in data
        assert any("mimir_turn" in __import__("json").dumps(h) for h in data["Stop"])

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        setup = ClaudeCodeSetup(config_dir=tmp_path)
        setup.install()
        setup.install()
        data = __import__("json").loads(setup.settings_path.read_text())
        assert len(data["Stop"]) == 1


class TestCodexSetup:
    def test_install_creates_config(self, tmp_path: Path) -> None:
        setup = CodexSetup(home_dir=tmp_path)
        path = setup.install()
        assert path.exists()
        assert "mimir.hooks.mimir_turn" in path.read_text()

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        setup = CodexSetup(home_dir=tmp_path)
        setup.install()
        first_size = setup.config_path.stat().st_size
        setup.install()
        assert setup.config_path.stat().st_size == first_size
