"""Checkpoint repository interface for Mimir state persistence."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class CheckpointRepository(ABC):
    """Persistence contract for Mimir checkpoints.

    A checkpoint contains the prototype matrix, metadata tensor, global step,
    and optional prediction-policy state.
    """

    @abstractmethod
    def save(
        self,
        path: str | Path,
        prototypes: torch.Tensor,
        metadata: torch.Tensor,
        step: int,
        **extras: object,
    ) -> None:
        """Persist the Mimir state to ``path``.

        Args:
            path: Destination file path.
            prototypes: Prototype matrix tensor.
            metadata: Prototype metadata tensor.
            step: Global training step.
            **extras: Optional additional fields to persist.
        """
        ...

    @abstractmethod
    def load(self, path: str | Path) -> dict[str, object]:
        """Load the Mimir state from ``path``.

        Args:
            path: Source file path.

        Returns:
            A dictionary containing at least ``prototypes``, ``metadata``,
            and ``step``.
        """
        ...
