"""Prediction policy interface."""

from abc import ABC, abstractmethod

import torch


class PredictionPolicy(ABC):
    """Strategy for predicting the next prototype and scoring surprise."""

    last_proto_id: int | None

    @abstractmethod
    def update(self, proto_id: int, step: int) -> None:
        """Observe a prototype id and update internal statistics."""

    @abstractmethod
    def predict_next(self, last_proto_id: int) -> int | None:
        """Return the most likely next prototype given the last one."""

    @abstractmethod
    def surprise_score(self, proto_id: int, last_proto_id: int | None = None) -> float:
        """Return how surprising the current prototype is.

        Higher values mean the transition was less expected.
        """

    @abstractmethod
    def reset(self) -> None:
        """Clear any per-session state (e.g. last observed prototype)."""

    @abstractmethod
    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable snapshot of learned statistics."""

    @abstractmethod
    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore learned statistics from a snapshot."""
