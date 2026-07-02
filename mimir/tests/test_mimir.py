"""Tests for the Mimir orchestrator."""

import tempfile
from pathlib import Path

import pytest
import torch

from mimir.application.events.event_bus import EventBus
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.domain.model.engine import EmbeddingEngine
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)
from mimir.infrastructure.prediction.first_order_markov_policy import (
    FirstOrderMarkovPredictionPolicy,
)


@pytest.fixture
def persistence() -> TorchCheckpointRepository:
    """Return a TorchCheckpointRepository for tests."""
    return TorchCheckpointRepository()


@pytest.fixture
def oja_learning_policy() -> OjaLearningPolicy:
    """Return the default Oja learning policy for tests."""
    return OjaLearningPolicy()


class FakeEngine(EmbeddingEngine):
    """Mock embedding engine for tests."""

    def __init__(self, dim: int = 8) -> None:
        self.output_dim = dim
        self._device = "cpu"

    def encode(self, texts: list[str], batch_size: int = 32) -> torch.Tensor:
        # Deterministic embeddings based on text hash.
        vectors = []
        for text in texts:
            torch.manual_seed(hash(text) % (2**31))
            vec = torch.randn(self.output_dim)
            vec = vec / torch.linalg.norm(vec)
            vectors.append(vec)
        return torch.stack(vectors)


@pytest.fixture
def mimir(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> Mimir:
    """Return an Mimir instance backed by the fake 8-dim engine."""
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    return Mimir(
        config,
        engine=engine,
        persistence=persistence,
        learning_policy=oja_learning_policy,
    )


def test_encode_shape(mimir: Mimir) -> None:
    emb = mimir.encode(["hello", "world"])
    assert emb.shape == (2, 8)


def test_encode_is_deterministic(mimir: Mimir) -> None:
    emb1 = mimir.encode("hello")
    emb2 = mimir.encode("hello")
    assert torch.allclose(emb1, emb2)


def test_learn_returns_report(mimir: Mimir) -> None:
    report = mimir.learn("hello")
    assert "updated" in report
    assert "capacity_usage" in report


def test_learn_changes_embedding(mimir: Mimir) -> None:
    before = mimir.encode("hello")
    mimir.learn("hello", importance=5.0)
    after = mimir.encode("hello")
    assert not torch.allclose(before, after, atol=1e-6)


def test_save_load_roundtrip(
    mimir: Mimir, persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    mimir.learn("hello")
    before = mimir.encode("hello")

    with tempfile.TemporaryDirectory() as tmpdir:
        config = MimirConfig(
            base_model="dummy",
            checkpoint_dir=Path(tmpdir),
        )
        # Create a new mimir sharing the same engine dimension and checkpoint_dir.
        scoped_mimir = Mimir(
            config,
            engine=FakeEngine(dim=8),
            persistence=persistence,
            learning_policy=oja_learning_policy,
        )
        scoped_mimir.store = mimir.store
        scoped_mimir.step = mimir.step

        scoped_mimir.save("checkpoint.pt")

        new_mimir = Mimir(
            config,
            engine=FakeEngine(dim=8),
            persistence=persistence,
            learning_policy=oja_learning_policy,
        )
        new_mimir.load("checkpoint.pt")
        after = new_mimir.encode("hello")

    assert torch.allclose(before, after, atol=1e-6)
    assert new_mimir.step == mimir.step


def test_save_rejects_absolute_path_when_sandboxed(
    mimir: Mimir,
    persistence: TorchCheckpointRepository,
    oja_learning_policy: OjaLearningPolicy,
) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        config = MimirConfig(
            base_model="dummy",
            checkpoint_dir=Path(tmpdir),
        )
        scoped_mimir = Mimir(
            config,
            engine=FakeEngine(dim=8),
            persistence=persistence,
            learning_policy=oja_learning_policy,
        )
        with pytest.raises(ValueError, match="must be relative"):
            scoped_mimir.save(Path(tmpdir) / "checkpoint.pt")


def test_reset_clears_learned_state(mimir: Mimir) -> None:
    mimir.learn("hello", importance=5.0)
    assert mimir.step > 0
    assert mimir.store.metadata[:, 1].sum().item() > 0  # access_count

    mimir.reset()

    assert mimir.step == 0
    assert mimir.store.metadata[:, 1].sum().item() == 0


def test_encode_empty_input(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    """Empty text list should return an empty tensor without crashing."""
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    mimir = Mimir(
        config, engine=engine, persistence=persistence, learning_policy=oja_learning_policy
    )

    result = mimir.encode([])
    assert result.shape == (0, 8)


def test_learn_empty_input(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    """Empty text list should return a safe report without crashing."""
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    mimir = Mimir(
        config, engine=engine, persistence=persistence, learning_policy=oja_learning_policy
    )

    report = mimir.learn([])
    assert report["updated"] == 0
    assert report["unique_updated"] == 0
    assert report["updated_ids"] == []
    assert isinstance(report["capacity_usage"], float)


def test_repeated_learning_does_not_explode(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    """Learning the same text many times should keep embeddings finite."""
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    mimir = Mimir(
        config, engine=engine, persistence=persistence, learning_policy=oja_learning_policy
    )

    for _ in range(100):
        mimir.learn("hello", importance=5.0)

    emb = mimir.encode("hello")
    assert torch.all(torch.isfinite(emb))


def test_encode_without_learning_is_stable(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    """Encoding the same text without learning should yield identical embeddings."""
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    mimir = Mimir(
        config, engine=engine, persistence=persistence, learning_policy=oja_learning_policy
    )

    emb1 = mimir.encode("hello")
    emb2 = mimir.encode("hello")
    assert torch.allclose(emb1, emb2, atol=1e-6)


def test_zero_learning_rate_no_embedding_shift(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    """When learning_rate_base is zero, learning should not change future encodings.

    We also disable decay to isolate the learning rule effect.
    """
    config = MimirConfig(
        base_model="dummy",
        learning_rate_base=0.0,
        forgetting_decay=1.0,
    )
    engine = FakeEngine(dim=8)
    mimir = Mimir(
        config, engine=engine, persistence=persistence, learning_policy=oja_learning_policy
    )

    before = mimir.encode("hello")
    mimir.learn("hello")
    after = mimir.encode("hello")

    assert torch.allclose(before, after, atol=1e-6)


def test_encode_emits_event(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    config = MimirConfig(base_model="dummy")
    engine = FakeEngine(dim=8)
    bus = EventBus()
    events: list[dict[str, object]] = []
    bus.subscribe(lambda event: events.append(event))
    mimir = Mimir(
        config,
        engine=engine,
        persistence=persistence,
        event_bus=bus,
        learning_policy=oja_learning_policy,
    )

    mimir.encode("hello")

    assert len(events) == 1
    assert events[0]["type"] == "encode"


def test_learn_with_prediction_policy(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    config = MimirConfig(base_model="dummy", num_prototypes=4)
    engine = FakeEngine(dim=8)
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    mimir = Mimir(
        config,
        engine=engine,
        persistence=persistence,
        prediction_policy=policy,
        learning_policy=oja_learning_policy,
    )

    for _ in range(5):
        mimir.learn("hello")
    report = mimir.learn("world")

    assert "surprise_score" in report
    assert isinstance(report["surprise_score"], float)


def test_predict_next(
    persistence: TorchCheckpointRepository, oja_learning_policy: OjaLearningPolicy
) -> None:
    config = MimirConfig(base_model="dummy", num_prototypes=4)
    engine = FakeEngine(dim=8)
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    mimir = Mimir(
        config,
        engine=engine,
        persistence=persistence,
        prediction_policy=policy,
        learning_policy=oja_learning_policy,
    )

    for _ in range(5):
        mimir.learn("hello")
        mimir.learn("world")

    assert mimir.predict_next() is not None
