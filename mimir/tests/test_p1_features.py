"""Tests for P1 features: top-k activation, prediction, event bus."""

from typing import Any

import pytest
import torch

from mimir.application.events.event_bus import EventBus
from mimir.core.config import MimirConfig
from mimir.core.prototype_store import PrototypeStore
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.prediction.first_order_markov_policy import (
    FirstOrderMarkovPredictionPolicy,
)


@pytest.fixture
def config() -> MimirConfig:
    """Return a small MimirConfig for fast deterministic P1 tests."""
    return MimirConfig(
        base_model="fake",
        num_prototypes=8,
        prototype_init_scale=0.01,
        residual_scale=0.3,
        temperature=1.0,
    )


def test_top_k_activation_is_sparse(config: MimirConfig) -> None:
    test_config = MimirConfig(**{**config.__dict__, "top_k": 2})
    store = PrototypeStore(dim=16, config=test_config, policy=OjaLearningPolicy())
    base = torch.randn(4, 16)
    weights = store.activation_weights(base)

    assert weights.shape == (4, 8)
    # Each row should have exactly top_k non-zero entries.
    nonzero_counts = (weights != 0).sum(dim=1)
    assert torch.all(nonzero_counts == 2)
    # Rows should sum to 1 (softmax normalization).
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_top_k_degrades_to_softmax(config: MimirConfig) -> None:
    store = PrototypeStore(dim=16, config=config, policy=OjaLearningPolicy())
    base = torch.randn(4, 16)
    weights = store.activation_weights(base)

    assert weights.shape == (4, 8)
    assert torch.all(weights != 0)
    assert torch.allclose(weights.sum(dim=1), torch.ones(4), atol=1e-5)


def test_top_k_lookup_respects_config(config: MimirConfig) -> None:
    test_config = MimirConfig(**{**config.__dict__, "top_k": 3, "residual_scale": 1.0})
    store = PrototypeStore(dim=16, config=test_config, policy=OjaLearningPolicy())
    base = torch.randn(2, 16)

    residual = store.lookup(base)
    assert residual.shape == (2, 16)


def test_markov_predicts_repeated_sequence() -> None:
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    sequence = [0, 1, 2, 0, 1, 2]
    for proto_id in sequence:
        policy.update(proto_id, step=0)

    assert policy.predict_next(0) == 1
    assert policy.predict_next(1) == 2


def test_markov_surprise_high_for_unexpected() -> None:
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    for _ in range(10):
        policy.update(0, step=0)
        policy.update(1, step=0)

    # Transition 0 -> 1 is expected, 0 -> 2 is unexpected.
    assert policy.surprise_score(1, last_proto_id=0) < 0.5
    assert policy.surprise_score(2, last_proto_id=0) > 0.5


def test_markov_state_dict_roundtrip() -> None:
    policy = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    policy.update(0, step=0)
    policy.update(1, step=0)

    state = policy.state_dict()
    policy2 = FirstOrderMarkovPredictionPolicy(num_prototypes=4)
    policy2.load_state_dict(state)

    assert policy2.predict_next(0) == 1


def test_event_bus_publish_subscribe() -> None:
    bus = EventBus()
    received: list[Any] = []

    def handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(handler)
    bus.publish({"type": "test"})

    assert len(received) == 1
    assert received[0]["type"] == "test"


def test_event_bus_unsubscribe() -> None:
    bus = EventBus()
    received: list[Any] = []

    def handler(event: Any) -> None:
        received.append(event)

    bus.subscribe(handler)
    bus.unsubscribe(handler)
    bus.publish({"type": "test"})

    assert len(received) == 0
