"""Tests for project context discovery."""

from pathlib import Path

import pytest

from mimir.infrastructure.context_discovery import ContextDiscovery


@pytest.fixture
def discoverer() -> ContextDiscovery:
    return ContextDiscovery()


class TestContextDiscovery:
    def test_discovers_known_files(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("Use Python 3.10+")
        (tmp_path / "AGENTS.md").write_text("Always write tests")
        contexts = discoverer.discover(tmp_path)
        assert len(contexts) == 2
        names = {c.path.name for c in contexts}
        assert names == {"CLAUDE.md", "AGENTS.md"}

    def test_ignores_unknown_files(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        (tmp_path / "README.md").write_text("hello")
        assert discoverer.discover(tmp_path) == []

    def test_case_insensitive(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        (tmp_path / "claude.md").write_text("lowercase file")
        contexts = discoverer.discover(tmp_path)
        assert len(contexts) == 1
        assert contexts[0].content == "lowercase file"

    def test_skips_large_files(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        (tmp_path / "CLAUDE.md").write_text("x" * 200_000)
        assert discoverer.discover(tmp_path) == []

    def test_ignores_symlinks(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        real_file = tmp_path / "real.md"
        real_file.write_text("Use Python 3.10+")
        link = tmp_path / "CLAUDE.md"
        link.symlink_to(real_file)
        assert discoverer.discover(tmp_path) == []


    def test_format_memory(self, discoverer: ContextDiscovery, tmp_path: Path) -> None:
        path = tmp_path / "AGENTS.md"
        path.write_text("Write docs.")
        contexts = discoverer.discover(tmp_path)
        formatted = discoverer.format_memory(contexts[0])
        assert "[Project Context from AGENTS.md]" in formatted
        assert "Write docs." in formatted
