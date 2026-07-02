"""Application-layer factories for assembling Mimir components.

These factories keep ``mimir.core`` free of infrastructure dependencies by
constructing concrete engines, persistence, and learning policies here in the
application layer.
"""

from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model.engine import EmbeddingEngine
from mimir.infrastructure.embedding import create_engine
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)


def create_embedding_engine(
    backend: str = "llama-server",
    base_url: str = "http://127.0.0.1:11435",
    model: str = "all-MiniLM-L6-v2",
) -> EmbeddingEngine:
    """Create and return an embedding engine for the requested backend."""
    return create_engine(backend=backend, base_url=base_url, model=model)


def create_mimir(
    config: MimirConfig,
    engine: EmbeddingEngine | None = None,
    backend: str = "llama-server",
    base_url: str = "http://127.0.0.1:11435",
    model: str = "all-MiniLM-L6-v2",
) -> Mimir:
    """Assemble a fully configured Mimir instance.

    Args:
        config: Mimir configuration.
        engine: Optional pre-built embedding engine. If omitted, one is created
            from ``backend`` / ``base_url`` / ``model``.
        backend: Embedding backend when ``engine`` is not provided.
        base_url: URL for the llama-server backend.
        model: Model name for the sentence-transformer backend.

    Returns:
        An Mimir instance wired with concrete infrastructure implementations.
    """
    engine = engine or create_embedding_engine(
        backend=backend,
        base_url=base_url,
        model=model,
    )

    # Ensure the engine knows its output dimension before building the store.
    if engine.output_dim == 0:
        _ = engine.encode(["__mimir_probe__"])

    return Mimir(
        config=config,
        engine=engine,
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )


def create_prototype_store_policy() -> OjaLearningPolicy:
    """Return the default learning policy for PrototypeStore."""
    return OjaLearningPolicy()
