"""Tests for PrototypeStore."""

import pytest
import torch

from mimir.core.config import MimirConfig
from mimir.core.prototype_store import PrototypeStore
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy


@pytest.fixture
def store() -> PrototypeStore:
    """Return a PrototypeStore with 8-dimensional prototypes."""
    config = MimirConfig(base_model="dummy")
    return PrototypeStore(dim=8, config=config, policy=OjaLearningPolicy())


def test_lookup_shape(store: PrototypeStore) -> None:
    base = torch.randn(4, 8)
    residual = store.lookup(base)
    assert residual.shape == (4, 8)


def test_lookup_initially_near_zero(store: PrototypeStore) -> None:
    """With small init scale, residual should be small."""
    base = torch.randn(1, 8)
    base = base / torch.linalg.norm(base)
    residual = store.lookup(base)
    assert torch.linalg.norm(residual).item() < 0.1


def test_update_nearest_changes_prototype(store: PrototypeStore) -> None:
    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)

    before = store.prototypes.clone()
    updated = store.update_nearest(target, step=0)
    after = store.prototypes.clone()

    assert len(updated) == 1
    proto_id = updated[0]
    assert not torch.allclose(before[proto_id], after[proto_id], atol=1e-6)


def test_update_increases_access_count(store: PrototypeStore) -> None:
    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)

    store.update_nearest(target, step=0)
    updated = store.update_nearest(target, step=1)
    proto_id = updated[0]
    access_count = store.metadata[proto_id, 1].item()

    assert access_count == 2.0


def test_decay_reduces_strength(store: PrototypeStore) -> None:
    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)
    store.update_nearest(target, step=0)

    proto_id = 0
    before = store.metadata[proto_id, 0].item()
    store.decay(step=1)
    after = store.metadata[proto_id, 0].item()

    assert after < before or after == 1.0


def test_state_dict_roundtrip(store: PrototypeStore) -> None:
    state = store.state_dict()
    store.prototypes.normal_()
    store.load_state_dict(state)

    assert torch.allclose(store.prototypes, state["prototypes"])
    assert torch.allclose(store.metadata, state["metadata"])


def test_capacity_full_still_stable() -> None:
    """Learning more unique samples than prototypes should not crash or produce NaNs."""
    config = MimirConfig(base_model="dummy", num_prototypes=4)
    store = PrototypeStore(dim=8, config=config, policy=OjaLearningPolicy())

    for i in range(10):
        target = torch.randn(1, 8)
        target = target / torch.linalg.norm(target)
        store.update_nearest(target, step=i)

    base = torch.randn(2, 8)
    residual = store.lookup(base)
    assert torch.all(torch.isfinite(residual))
    assert residual.shape == (2, 8)


def test_zero_learning_rate_no_update() -> None:
    """When learning_rate_base is zero, prototypes should not change."""
    config = MimirConfig(base_model="dummy", learning_rate_base=0.0)
    store = PrototypeStore(dim=8, config=config, policy=OjaLearningPolicy())

    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)

    before = store.prototypes.clone()
    store.update_nearest(target, step=0)
    after = store.prototypes.clone()

    assert torch.allclose(before, after, atol=1e-6)


def test_decay_does_not_collapse_prototypes(store: PrototypeStore) -> None:
    """Repeated decay should keep prototypes finite and non-NaN."""
    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)
    store.update_nearest(target, step=0)

    for step in range(1, 200):
        store.decay(step)

    assert torch.all(torch.isfinite(store.prototypes))
    assert torch.all(torch.isfinite(store.metadata))


def test_created_step_not_overwritten_on_later_updates(store: PrototypeStore) -> None:
    """A prototype created at step 0 should keep its created_step after step 1."""
    target = torch.randn(1, 8)
    target = target / torch.linalg.norm(target)

    updated = store.update_nearest(target, step=0)
    proto_id = updated[0]
    created_at_step_0 = store.metadata[proto_id, store._CREATED_STEP_IDX].item()
    assert created_at_step_0 == 0.0

    store.update_nearest(target, step=1)
    created_at_step_1 = store.metadata[proto_id, store._CREATED_STEP_IDX].item()
    assert created_at_step_1 == 0.0
