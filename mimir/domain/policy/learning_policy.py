"""Learning policy interface."""

from abc import ABC, abstractmethod

import torch


class LearningPolicy(ABC):
    """Strategy for updating prototype vectors."""

    @abstractmethod
    def compute_delta(
        self,
        prototype: torch.Tensor,
        input_vector: torch.Tensor,
        access_count: int,
        learning_rate_base: float,
        learning_rate_decay: float,
    ) -> torch.Tensor:
        """Return the update delta for a single prototype."""
        ...
