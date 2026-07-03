"""Tests for `mimir setup` command."""

import json
from pathlib import Path

import pytest

from mimir.setup import (
    ClaudeCodeSetup,
    CodexSetup,
    KimiCodeSetup,
    OpenCodeSetup,
    create_setup,
    list_supported_agents,
)


class TestSetupRegistry:
    def test_supported_agents(self) -> None:
        agents = list_supported_agents()
        assert "kimi-code" in agents
        assert "claude-code" in agents
        assert "codex" in agents
        assert "opencode" in agents

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
        data = json.loads(path.read_text())
        assert "Stop" in data
        assert "UserPromptSubmit" in data
        assert any("mimir_turn" in json.dumps(h) for h in data["Stop"])

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        setup = ClaudeCodeSetup(config_dir=tmp_path)
        setup.install()
        setup.install()
        data = json.loads(setup.settings_path.read_text())
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


class TestOpenCodeSetup:
    def test_install_creates_config(self, tmp_path: Path) -> None:
        plugin_path = tmp_path / "mimir-opencode-plugin"
        plugin_path.mkdir()
        setup = OpenCodeSetup(config_dir=tmp_path, plugin_path=plugin_path)
        path = setup.install()
        assert path.exists()
        data = json.loads(path.read_text())
        assert "plugin" in data
        assert any(setup._is_mimir_plugin(entry) for entry in data["plugin"])

    def test_install_is_idempotent(self, tmp_path: Path) -> None:
        plugin_path = tmp_path / "mimir-opencode-plugin"
        plugin_path.mkdir()
        setup = OpenCodeSetup(config_dir=tmp_path, plugin_path=plugin_path)
        setup.install()
        first_size = setup.config_path.stat().st_size
        setup.install()
        assert setup.config_path.stat().st_size == first_size

    def test_is_installed(self, tmp_path: Path) -> None:
        plugin_path = tmp_path / "mimir-opencode-plugin"
        plugin_path.mkdir()
        setup = OpenCodeSetup(config_dir=tmp_path, plugin_path=plugin_path)
        assert not setup.is_installed()
        setup.install()
        assert setup.is_installed()

    def test_install_migrates_stale_plugins_key(self, tmp_path: Path) -> None:
        plugin_path = tmp_path / "mimir-opencode-plugin"
        plugin_path.mkdir()
        setup = OpenCodeSetup(config_dir=tmp_path, plugin_path=plugin_path)
        setup.config_path.write_text(
            json.dumps({"plugins": [{"package": str(plugin_path), "options": {}}]}),
            encoding="utf-8",
        )
        setup.install()
        data = json.loads(setup.config_path.read_text())
        assert "plugins" not in data
        assert "plugin" in data
        assert any(setup._is_mimir_plugin(entry) for entry in data["plugin"])
