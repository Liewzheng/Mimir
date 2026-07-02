"""PyTorch-based checkpoint repository."""

import os
import tempfile
from contextlib import suppress
from pathlib import Path

import torch

from mimir.domain.repository.checkpoint_repository import CheckpointRepository


class TorchCheckpointRepository(CheckpointRepository):
    """Save and load prototype states using PyTorch serialization."""

    VERSION = "0.1.0"

    def save(
        self,
        path: str | Path,
        prototypes: torch.Tensor,
        metadata: torch.Tensor,
        step: int,
        **extras: object,
    ) -> None:
        """Persist state to disk atomically."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint: dict[str, object] = {
            "version": self.VERSION,
            "step": step,
            "prototypes": prototypes,
            "metadata": metadata,
        }
        checkpoint.update(extras)

        # Atomic write: save to a temp file in the same directory, then rename.
        with tempfile.NamedTemporaryFile(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as tmp_file:
            tmp_path = Path(tmp_file.name)
            try:
                torch.save(checkpoint, tmp_path)
                os.replace(tmp_path, path)
            except Exception:
                with suppress(FileNotFoundError):
                    os.unlink(tmp_path)
                raise

    def load(self, path: str | Path) -> dict[str, object]:
        """Load state from disk.

        Returns:
            Dict with keys: version, step, prototypes, metadata, and any
            extra fields previously saved.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        # Mimir checkpoints only contain tensors, primitives, and dicts of
        # tensors.  Using weights_only=True avoids arbitrary pickle execution.
        state: dict[str, object] = torch.load(path, map_location="cpu", weights_only=True)
        required = {"version", "step", "prototypes", "metadata"}
        missing = required - set(state.keys())
        if missing:
            raise ValueError(f"Checkpoint missing keys: {missing}")
        return state
