"""Tests for the torch checkpoint repository."""

from pathlib import Path

import pytest
import torch

from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)


def test_round_trip(tmp_path: Path) -> None:
    """Save and load restore the same tensors and metadata."""
    repo = TorchCheckpointRepository()
    path = tmp_path / "checkpoint.pt"
    prototypes = torch.randn(8, 4)
    metadata = torch.zeros(8, 4)

    repo.save(path, prototypes, metadata, step=5, extra_key="extra_value")
    state = repo.load(path)

    assert state["step"] == 5
    assert torch.equal(state["prototypes"], prototypes)  # type: ignore[arg-type]
    assert torch.equal(state["metadata"], metadata)  # type: ignore[arg-type]
    assert state["extra_key"] == "extra_value"


def test_missing_checkpoint(tmp_path: Path) -> None:
    """Loading a missing checkpoint raises FileNotFoundError."""
    repo = TorchCheckpointRepository()
    with pytest.raises(FileNotFoundError):
        repo.load(tmp_path / "missing.pt")


def test_missing_required_keys(tmp_path: Path) -> None:
    """A checkpoint without required keys raises ValueError."""
    repo = TorchCheckpointRepository()
    path = tmp_path / "bad.pt"
    torch.save({"version": "0.1.0"}, path)
    with pytest.raises(ValueError, match="missing keys"):
        repo.load(path)
