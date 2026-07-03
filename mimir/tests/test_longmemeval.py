"""Tests for the LongMemEval harness and Ollama backend integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mimir.adapters.agents import InMemoryAgentAdapter
from mimir.application.factories import create_mimir
from mimir.benchmarks.longmemeval import LongMemEvalDataset, LongMemEvalHarness
from mimir.benchmarks.longmemeval_metrics import Metrics, aggregate_metrics
from mimir.core.config import MimirConfig
from mimir.infrastructure.embedding.engine_factory import create_engine
from mimir.infrastructure.embedding.fake_engine import FakeEngine
from mimir.infrastructure.embedding.ollama_engine import OllamaEmbeddingEngine


def test_create_engine_ollama() -> None:
    engine = create_engine(backend="ollama", model="test-model")
    assert isinstance(engine, OllamaEmbeddingEngine)


def _make_fake_adapter(num_prototypes: int = 8) -> InMemoryAgentAdapter:
    engine = FakeEngine(dim=16)
    config = MimirConfig(
        base_model="eval",
        num_prototypes=num_prototypes,
        learning_rate_base=0.1,
        learning_rate_decay=0.1,
        filter_enabled=False,
        quality_gate_enabled=False,
    )
    mimir = create_mimir(config, engine=engine)
    return InMemoryAgentAdapter(
        mimir=mimir,
        learn_on_observe=True,
        max_memories=10_000,
        max_text_length=10_000,
    )


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    data = [
        {
            "question_id": "q1",
            "question_type": "single-session-user",
            "question": "What is the user's favorite color?",
            "question_date": "2024-01-02",
            "answer": "Blue",
            "answer_session_ids": ["s2"],
            "haystack_dates": ["2024-01-01", "2024-01-02"],
            "haystack_session_ids": ["s1", "s2"],
            "haystack_sessions": [
                [
                    {"role": "user", "content": "The weather is nice today."},
                    {"role": "assistant", "content": "Yes, it is."},
                ],
                [
                    {"role": "user", "content": "My favorite color is blue."},
                    {"role": "assistant", "content": "Blue is a great color."},
                ],
            ],
        }
    ]
    path = tmp_path / "longmemeval_synthetic.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestLongMemEvalDataset:
    def test_loads_samples(self, synthetic_dataset: Path) -> None:
        dataset = LongMemEvalDataset(synthetic_dataset)
        assert len(dataset) == 1
        sample = dataset.samples[0]
        assert sample.question_id == "q1"
        assert sample.answer_session_ids == ["s2"]
        assert len(sample.haystack_sessions) == 2

    def test_limit_truncates(self, synthetic_dataset: Path) -> None:
        dataset = LongMemEvalDataset(synthetic_dataset, limit=0)
        assert len(dataset) == 0


class TestLongMemEvalHarness:
    def test_recall_finds_gold_session(self, synthetic_dataset: Path) -> None:
        engine = FakeEngine(dim=16)
        # Give the question and the gold memory a shared theme so retrieval works.
        engine.set_theme(
            "color",
            [
                "What is the user's favorite color?",
                "My favorite color is blue.",
            ],
        )
        config = MimirConfig(
            base_model="eval",
            num_prototypes=8,
            learning_rate_base=0.1,
            learning_rate_decay=0.1,
            filter_enabled=False,
            quality_gate_enabled=False,
        )
        mimir = create_mimir(config, engine=engine)
        adapter = InMemoryAgentAdapter(
            mimir=mimir,
            learn_on_observe=True,
            max_memories=10_000,
            max_text_length=10_000,
        )
        harness = LongMemEvalHarness(adapter, reader=None, top_k=5)
        dataset = LongMemEvalDataset(synthetic_dataset)

        sample = dataset.samples[0]
        adapter.reset()
        harness.ingest(sample)
        result = harness.evaluate(sample)

        assert result["recall_hit"] is True
        assert "s2" in result["retrieved_session_ids"]


class TestMetrics:
    def test_aggregate(self) -> None:
        metrics = Metrics()
        metrics.add("single-session-user", recall_hit=True, qa_correct=True)
        metrics.add("single-session-user", recall_hit=True, qa_correct=False)
        metrics.add("knowledge-update", recall_hit=False, qa_correct=False)

        report = aggregate_metrics(metrics)
        assert round(report["overall"]["recall@k"], 4) == round(2 / 3, 4)
        assert round(report["overall"]["qa_accuracy"], 4) == round(1 / 3, 4)
        assert report["by_type"]["single-session-user"]["recall@k"] == 1.0
        assert report["by_type"]["knowledge-update"]["recall@k"] == 0.0
