"""Tests for application-layer factories."""

import pytest

from mimir.application.factories import create_mimir
from mimir.core.config import MimirConfig
from mimir.domain.model.engine import EmbeddingEngine


class _FailingProbeEngine(EmbeddingEngine):
    """Engine with unknown output_dim whose probe encode call always fails."""

    def __init__(self) -> None:
        self.output_dim = 0

    def encode(self, texts: list[str], batch_size: int = 32) -> None:  # type: ignore[override]
        raise ConnectionError("embedding backend unreachable")


def test_create_mimir_probe_failure_raises_descriptive_error() -> None:
    engine = _FailingProbeEngine()
    config = MimirConfig(base_model="dummy", num_prototypes=4)
    with pytest.raises(RuntimeError, match="Embedding engine probe failed"):
        create_mimir(config, engine=engine)
