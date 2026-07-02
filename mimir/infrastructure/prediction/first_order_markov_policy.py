"""First-order Markov prediction policy."""

import torch

from mimir.domain.policy.prediction_policy import PredictionPolicy


class FirstOrderMarkovPredictionPolicy(PredictionPolicy):
    """Predict the next prototype from a first-order transition matrix.

    Maintains a count matrix ``transitions[last, next]`` and computes
    surprise as ``1 - P(next | last)``.  A small Laplace smoothing term
    prevents zero probabilities before the matrix has been observed much.
    """

    def __init__(self, num_prototypes: int, smoothing: float = 1.0) -> None:
        self.num_prototypes = num_prototypes
        self.smoothing = smoothing
        self.transitions = torch.zeros(num_prototypes, num_prototypes, dtype=torch.float32)
        self.row_counts = torch.zeros(num_prototypes, dtype=torch.float32)
        self.last_proto_id: int | None = None

    def reset(self) -> None:
        """Reset per-session state without clearing learned statistics."""
        self.last_proto_id = None

    def update(self, proto_id: int, step: int) -> None:
        """Record a transition from the last observed prototype."""
        if self.last_proto_id is not None:
            self.transitions[self.last_proto_id, proto_id] += 1.0
            self.row_counts[self.last_proto_id] += 1.0
        self.last_proto_id = proto_id

    def predict_next(self, last_proto_id: int) -> int | None:
        """Return the most likely next prototype.

        Returns ``None`` if the given prototype has never been observed.
        """
        if last_proto_id < 0 or last_proto_id >= self.num_prototypes:
            return None
        row = self.transitions[last_proto_id]
        if row.sum().item() == 0.0:
            return None
        return int(row.argmax().item())

    def surprise_score(self, proto_id: int, last_proto_id: int | None = None) -> float:
        """Return 1 - P(proto_id | last_proto_id).

        If no last prototype is provided, the instance's last observed id
        is used.  If the last prototype has never been seen, the score is
        1.0 (maximal surprise).
        """
        last = last_proto_id if last_proto_id is not None else self.last_proto_id
        if last is None:
            return 1.0
        if last < 0 or last >= self.num_prototypes:
            return 1.0

        total = self.row_counts[last].item()
        if total == 0.0:
            return 1.0

        count = self.transitions[last, proto_id].item()
        prob = (count + self.smoothing) / (total + self.smoothing * self.num_prototypes)
        return 1.0 - prob

    def state_dict(self) -> dict[str, torch.Tensor]:
        """Return a serializable snapshot of the learned counts."""
        return {
            "transitions": self.transitions.clone(),
            "row_counts": self.row_counts.clone(),
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        """Restore learned counts from a snapshot."""
        self.transitions = state["transitions"].clone()
        self.row_counts = state["row_counts"].clone()
