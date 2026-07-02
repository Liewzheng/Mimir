"""Main Mimir orchestrator."""

from pathlib import Path

import torch

from mimir.core.config import MimirConfig
from mimir.core.inference import InferencePipeline
from mimir.core.learning import LearningPipeline
from mimir.core.prototype_store import PrototypeStore
from mimir.domain.model.engine import EmbeddingEngine
from mimir.domain.policy.learning_policy import LearningPolicy
from mimir.domain.policy.prediction_policy import PredictionPolicy
from mimir.domain.repository.checkpoint_repository import CheckpointRepository
from mimir.domain.service.event_bus import EventBus


class Mimir:
    """Plastic embedding system that remembers."""

    def __init__(
        self,
        config: MimirConfig,
        engine: EmbeddingEngine,
        persistence: CheckpointRepository,
        learning_policy: LearningPolicy,
        event_bus: EventBus | None = None,
        prediction_policy: PredictionPolicy | None = None,
    ) -> None:
        self.config = config
        self.step = 0
        self.engine = engine
        self._learning_policy = learning_policy
        self.store = PrototypeStore(
            dim=self.engine.output_dim,
            config=config,
            policy=learning_policy,
        )
        self.inference = InferencePipeline(self.store, config)
        self.learning = LearningPipeline(self.store, config)
        self._persistence = persistence
        self._event_bus = event_bus
        self._prediction_policy = prediction_policy
        self._checkpoint_dir = (
            Path(self.config.checkpoint_dir).expanduser().resolve()
            if self.config.checkpoint_dir is not None
            else None
        )

    def _resolve_checkpoint_path(self, path: str | Path) -> Path:
        """Resolve a checkpoint path, optionally sandboxing it."""
        path = Path(path).expanduser()
        if self._checkpoint_dir is not None:
            # Sandboxed mode: checkpoints must live inside checkpoint_dir.
            if path.is_absolute():
                raise ValueError(
                    f"Checkpoint path must be relative to checkpoint_dir "
                    f"'{self._checkpoint_dir}', got absolute path '{path}'"
                )
            resolved = (self._checkpoint_dir / path).resolve()
            try:
                # Verify the resolved path is still under checkpoint_dir.
                resolved.relative_to(self._checkpoint_dir)
            except ValueError as exc:
                raise ValueError(
                    f"Checkpoint path '{path}' escapes checkpoint_dir '{self._checkpoint_dir}'"
                ) from exc
            return resolved
        return path.resolve()

    def encode(self, texts: str | list[str]) -> torch.Tensor:
        """Encode texts into plastic embeddings.

        This is a read-only operation: it does not modify the internal state.

        Args:
            texts: A single text or a list of texts. An empty list returns
                an empty tensor of shape ``(0, output_dim)``.

        Returns:
            Tensor of shape ``(num_texts, output_dim)``.
        """
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return torch.empty(0, self.engine.output_dim)
        base = self.engine.encode(texts)
        output = self.inference.encode(base)

        if self._event_bus is not None:
            weights = self.store.activation_weights(base)
            self._event_bus.publish(
                {
                    "type": "encode",
                    "texts": texts,
                    "base": base,
                    "output": output,
                    "prototype_weights": weights,
                    "step": self.step,
                }
            )

        return output

    def learn(
        self,
        texts: str | list[str],
        importance: float = 1.0,
    ) -> dict[str, object]:
        """Explicitly reinforce memory for the given texts."""
        if isinstance(texts, str):
            texts = [texts]
        if not texts:
            return {
                "updated": 0,
                "unique_updated": 0,
                "capacity_usage": self.store.capacity_usage(),
                "updated_ids": [],
            }
        # Run the slow-weight embedding model (e.g. sentence-transformer or API).
        base = self.engine.encode(texts)
        # Update prototypes and apply global forgetting.
        report = self.learning.learn(base, self.step, importance)

        if self._prediction_policy is not None:
            updated_ids = report.get("updated_ids", [])
            if isinstance(updated_ids, list):
                # Feed the observed prototype sequence to the transition model.
                for proto_id in updated_ids:
                    if isinstance(proto_id, int):
                        self._prediction_policy.update(proto_id, self.step)
                # Surprise is computed relative to the last updated prototype.
                if updated_ids and isinstance(updated_ids[-1], int):
                    last_id = updated_ids[-1]
                    report["surprise_score"] = self._prediction_policy.surprise_score(last_id)

        if self._event_bus is not None:
            self._event_bus.publish(
                {
                    "type": "learn",
                    "texts": texts,
                    "base": base,
                    "updated_ids": report.get("updated_ids", []),
                    "report": report,
                    "step": self.step,
                }
            )

        self.step += 1
        return report

    def predict_next(self, proto_id: int | None = None) -> int | None:
        """Predict the next prototype based on the prediction policy."""
        if self._prediction_policy is None:
            return None
        last = proto_id if proto_id is not None else self._prediction_policy.last_proto_id
        if last is None:
            return None
        return self._prediction_policy.predict_next(last)

    def reset(self) -> None:
        """Reset the Mimir to a fresh state.

        Re-initializes the prototype store and resets the global step counter.
        This does not affect the underlying embedding engine.
        """
        self.store = PrototypeStore(
            dim=self.engine.output_dim,
            config=self.config,
            policy=self._learning_policy,
        )
        self.inference = InferencePipeline(self.store, self.config)
        self.learning = LearningPipeline(self.store, self.config)
        self.step = 0
        if self._prediction_policy is not None:
            self._prediction_policy.reset()

    def save(self, path: str | Path) -> None:
        """Persist the Mimir state."""
        resolved = self._resolve_checkpoint_path(path)
        state = self.store.state_dict()
        # Persist the full prototype matrix plus optional prediction state.
        self._persistence.save(
            path=resolved,
            prototypes=state["prototypes"],
            metadata=state["metadata"],
            step=self.step,
            prediction_policy=(
                self._prediction_policy.state_dict()
                if self._prediction_policy is not None
                else None
            ),
        )

    def load(self, path: str | Path) -> None:
        """Restore the Mimir state."""
        resolved = self._resolve_checkpoint_path(path)
        state = self._persistence.load(resolved)

        # Validate checkpoint schema before mutating internal state.
        prototypes = state["prototypes"]
        metadata = state["metadata"]
        step = state["step"]

        if not isinstance(prototypes, torch.Tensor):
            raise ValueError("Checkpoint 'prototypes' must be a torch.Tensor")
        if not isinstance(metadata, torch.Tensor):
            raise ValueError("Checkpoint 'metadata' must be a torch.Tensor")
        if not isinstance(step, int):
            raise ValueError("Checkpoint 'step' must be an int")

        self.store.load_state_dict({"prototypes": prototypes, "metadata": metadata})
        self.step = step

        # Restore transition statistics if a prediction policy is attached.
        if self._prediction_policy is not None and "prediction_policy" in state:
            policy_state = state["prediction_policy"]
            if isinstance(policy_state, dict):
                self._prediction_policy.load_state_dict(policy_state)
