"""Tests for the command-line interface."""

from pathlib import Path
from unittest import mock

import pytest

from mimir.cli import _build_parser, main


@pytest.fixture
def fake_backend_args() -> list[str]:
    """Return CLI args that select the deterministic fake backend."""
    return ["--backend", "fake"]


def test_build_parser_requires_command() -> None:
    """The parser rejects invocation without a subcommand."""
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_cli_encode(fake_backend_args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """encode command prints the embedding shape."""
    argv = ["encode", *fake_backend_args, "hello", "world"]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "Shape:" in captured.out


def test_cli_learn(fake_backend_args: list[str], capsys: pytest.CaptureFixture[str]) -> None:
    """learn command prints a non-empty report."""
    argv = ["learn", *fake_backend_args, "test text"]
    assert main(argv) == 0
    captured = capsys.readouterr()
    assert "updated" in captured.out


def test_cli_save_and_load(
    fake_backend_args: list[str],
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    """save followed by load round-trips the checkpoint."""
    checkpoint = tmp_path / "checkpoint.pt"

    save_argv = ["save", *fake_backend_args, "--path", str(checkpoint)]
    assert main(save_argv) == 0
    assert "Saved to" in capsys.readouterr().out

    load_argv = ["load", *fake_backend_args, "--path", str(checkpoint)]
    assert main(load_argv) == 0
    assert "Loaded from" in capsys.readouterr().out


def test_cli_mcp_runs_server(fake_backend_args: list[str]) -> None:
    """mcp command delegates to the MCP server runner."""
    with mock.patch("mimir.mcp.server.run_server") as run_server_mock:
        argv = ["mcp", *fake_backend_args]
        assert main(argv) == 0
        run_server_mock.assert_called_once_with(
            backend="fake",
            base_url="http://127.0.0.1:11435",
            model="all-MiniLM-L6-v2",
        )


def test_cli_error_return_code(fake_backend_args: list[str]) -> None:
    """Unhandled errors return exit code 1."""
    with mock.patch("mimir.cli._create_mimir", side_effect=RuntimeError("boom")):
        assert main(["encode", *fake_backend_args, "text"]) == 1
