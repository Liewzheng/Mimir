"""Tests for agent CLI memory adapter."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import numpy as np
import pytest
import torch

from mimir.adapters.agents import InMemoryAgentAdapter, Memory, Message
from mimir.core.config import MimirConfig
from mimir.core.mimir import Mimir
from mimir.infrastructure.learning.oja_learning_policy import OjaLearningPolicy
from mimir.infrastructure.persistence.torch_checkpoint_repository import (
    TorchCheckpointRepository,
)
from mimir.tests.fake_engine import FakeEngine


@pytest.fixture
def adapter() -> InMemoryAgentAdapter:
    """Return an InMemoryAgentAdapter backed by a fake embedding engine."""
    config = MimirConfig(base_model="dummy", num_prototypes=8)
    return InMemoryAgentAdapter(config=config, engine=FakeEngine(dim=8))


def test_observe_adds_memories(adapter: InMemoryAgentAdapter) -> None:
    messages = [
        Message(role="user", content="hello"),
        Message(role="assistant", content="hi there"),
    ]
    adapter.observe(messages)

    assert len(adapter._memories) == 2
    assert adapter._memories[0].text == "hello"
    assert isinstance(adapter._memories[0].embedding, list)


def test_observe_empty_list_does_nothing(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([])
    assert len(adapter._memories) == 0


def test_recall_returns_relevant_memories(adapter: InMemoryAgentAdapter) -> None:
    # Use deterministic theme vectors so fruit and code are separable.
    engine = cast(FakeEngine, adapter._mimir.engine)
    engine.set_theme("fruit", ["I like apples"])
    engine.set_theme("code", ["Python is great"])

    adapter.observe(
        [
            Message(role="user", content="I like apples"),
            Message(role="user", content="Python is great"),
        ]
    )

    results = adapter.recall("I like apples", top_k=1)
    assert len(results) == 1
    assert "apple" in results[0].text.lower()
    # Multiplicative lifecycle boost can push the score above 1.0.
    assert 0.0 <= results[0].score <= 1.0 + 0.3 + 1e-6


def test_recall_respects_min_score(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([Message(role="user", content="completely unrelated")])

    results = adapter.recall("completely unrelated", top_k=5, min_score=1.01)
    assert len(results) == 0


def test_recall_empty_buffer(adapter: InMemoryAgentAdapter) -> None:
    assert adapter.recall("anything") == []


def test_consolidate_reinforces_memories(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([Message(role="user", content="hello")])

    before = adapter._mimir.store.prototypes.clone()
    adapter.consolidate()
    after = adapter._mimir.store.prototypes.clone()

    assert not torch.allclose(before, after, atol=1e-6)
    assert len(adapter._memories) == 1


def test_learn_on_observe_updates_prototypes(adapter: InMemoryAgentAdapter) -> None:
    adapter._learn_on_observe = True
    before = adapter._mimir.store.prototypes.clone()
    adapter.observe([Message(role="user", content="hello")])
    after = adapter._mimir.store.prototypes.clone()

    assert not torch.allclose(before, after, atol=1e-6)


def test_capacity_limit(adapter: InMemoryAgentAdapter) -> None:
    adapter._max_memories = 3
    for i in range(5):
        adapter.observe([Message(role="user", content=f"msg {i}")])

    assert len(adapter._memories) == 3
    assert adapter._memories[0].text == "msg 2"


def test_checkpoint_restore_roundtrip(tmp_path: Path) -> None:
    adapter = InMemoryAgentAdapter(
        config=MimirConfig(base_model="dummy"),
        engine=FakeEngine(dim=8),
        checkpoint_dir=tmp_path,
    )
    adapter.observe([Message(role="user", content="hello")])
    before = adapter._mimir.encode("hello")

    checkpoint = "agent.pt"
    adapter.checkpoint(checkpoint)

    adapter.observe([Message(role="user", content="world")])
    adapter.restore(checkpoint)
    after = adapter._mimir.encode("hello")

    assert torch.allclose(before, after, atol=1e-6)
    assert len(adapter._memories) == 0


def test_checkpoint_sandbox_rejects_escape(adapter: InMemoryAgentAdapter, tmp_path: Path) -> None:
    adapter = InMemoryAgentAdapter(
        config=MimirConfig(base_model="dummy"),
        engine=FakeEngine(dim=8),
        checkpoint_dir=tmp_path / "sandbox",
    )
    with pytest.raises(ValueError, match="escapes checkpoint_dir"):
        adapter.checkpoint("../etc/passwd")


def test_checkpoint_rejects_absolute_path(adapter: InMemoryAgentAdapter, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be relative"):
        adapter.checkpoint(str(tmp_path / "agent.pt"))


def test_restore_missing_checkpoint_raises(adapter: InMemoryAgentAdapter) -> None:
    with pytest.raises(FileNotFoundError):
        adapter.restore("does_not_exist.pt")


def test_reset_clears_memories(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([Message(role="user", content="hello")])
    adapter.reset()

    assert len(adapter._memories) == 0
    assert adapter._mimir.step == 0


def test_max_text_length_enforced(adapter: InMemoryAgentAdapter) -> None:
    adapter._max_text_length = 10
    with pytest.raises(ValueError, match="exceeds max_text_length"):
        adapter.observe([Message(role="user", content="this is too long")])

    with pytest.raises(ValueError, match="exceeds max_text_length"):
        adapter.recall("this is too long")


def test_memory_dataclass_fields() -> None:
    mem = Memory(
        text="test",
        embedding=[0.1, 0.2, 0.3],
        score=0.5,
        created_at=datetime.now(timezone.utc),
    )
    assert mem.text == "test"
    assert mem.score == 0.5


def test_adapter_requires_config_or_mimir() -> None:
    with pytest.raises(ValueError, match="Must provide either config or mimir"):
        InMemoryAgentAdapter()


def test_adapter_rejects_both_config_and_mimir() -> None:
    config = MimirConfig(base_model="dummy")
    mimir = Mimir(
        config,
        engine=FakeEngine(dim=8),
        persistence=TorchCheckpointRepository(),
        learning_policy=OjaLearningPolicy(),
    )
    with pytest.raises(ValueError, match="Cannot provide both"):
        InMemoryAgentAdapter(config=config, mimir=mimir)


def test_memories_state_round_trip(adapter: InMemoryAgentAdapter) -> None:
    msg = Message(role="user", content="hello")
    adapter.observe([msg])

    state = adapter.memories_state()
    assert len(state) == 1
    assert state[0]["text"] == "hello"
    assert isinstance(state[0]["embedding"], list)
    assert state[0]["source"] is not None
    assert state[0]["source"]["role"] == "user"

    adapter.load_memories_state(state)
    assert len(adapter._memories) == 1
    assert adapter._memories[0].text == "hello"
    assert isinstance(adapter._memories[0].embedding, list)


def test_memories_state_empty_round_trip(adapter: InMemoryAgentAdapter) -> None:
    state = adapter.memories_state()
    assert state == []
    adapter.load_memories_state(state)
    assert adapter._memories == []


def test_memory_count(adapter: InMemoryAgentAdapter) -> None:
    assert adapter.memory_count == 0
    adapter.observe([Message(role="user", content="hello")])
    assert adapter.memory_count == 1


def test_clear_memories(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([Message(role="user", content="hello")])
    assert adapter.memory_count == 1
    adapter.clear_memories()
    assert adapter.memory_count == 0


def test_memories_state_with_torch_embedding(adapter: InMemoryAgentAdapter) -> None:
    adapter._memories.append(
        Memory(
            text="torch",
            embedding=torch.tensor([0.4, 0.5, 0.6]).tolist(),
            score=0.5,
            created_at=datetime.now(timezone.utc),
        )
    )
    state = adapter.memories_state()
    assert isinstance(state[0]["embedding"], list)
    assert [round(x, 4) for x in state[0]["embedding"]] == [0.4, 0.5, 0.6]


def test_memories_state_with_numpy_embedding(adapter: InMemoryAgentAdapter) -> None:
    adapter._memories.append(
        Memory(
            text="numpy",
            embedding=np.array([0.1, 0.2, 0.3]).tolist(),
            score=0.5,
            created_at=datetime.now(timezone.utc),
        )
    )
    state = adapter.memories_state()
    assert state[0]["embedding"] == [0.1, 0.2, 0.3]
    assert isinstance(state[0]["embedding"], list)


def test_memories_state_json_serializable(adapter: InMemoryAgentAdapter) -> None:
    adapter.observe([Message(role="user", content="hello")])
    state = adapter.memories_state()
    json_text = json.dumps(state)
    loaded = json.loads(json_text)
    adapter.load_memories_state(loaded)
    assert adapter.memory_count == 1
    assert adapter._memories[0].text == "hello"
