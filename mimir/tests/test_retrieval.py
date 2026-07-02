"""Tests for hybrid retrieval and lifecycle scoring."""

from datetime import datetime, timedelta, timezone

import pytest
import torch

from mimir.adapters.agents import InMemoryAgentAdapter
from mimir.application.factories import create_embedding_engine
from mimir.core.config import MimirConfig
from mimir.domain.model import Memory, Message
from mimir.infrastructure.lifecycle.metadata import MemoryMetadata
from mimir.infrastructure.lifecycle.scorer import (
    LifecycleScorer,
    LifecycleWeights,
    deduplicate_memories,
    ensure_lifecycle_metadata,
)
from mimir.infrastructure.retrieval.bm25_scorer import BM25Scorer
from mimir.infrastructure.retrieval.protocol import RankFusion, WeightedFusion
from mimir.infrastructure.retrieval.vector_scorer import VectorScorer


def _memory(text: str, embedding: list[float] | None = None) -> Memory:
    return Memory(
        text=text,
        embedding=embedding or [0.0] * 8,
        score=0.0,
        created_at=datetime.now(timezone.utc),
    )


def _make_adapter() -> InMemoryAgentAdapter:
    engine = create_embedding_engine(backend="fake")
    config = MimirConfig(base_model="fake", num_prototypes=8, top_k=4)
    return InMemoryAgentAdapter(config=config, engine=engine)


class TestBM25Scorer:
    def test_exact_match_scores_highest(self) -> None:
        memories = [
            _memory("Python programming language"),
            _memory("Java programming language"),
            _memory("Ruby gems and bundles"),
        ]
        scorer = BM25Scorer()
        scores = scorer.score("Python programming", memories)
        # Memories with no token overlap are omitted from the BM25 result.
        assert len(scores) == 2
        assert scores[0] > scores[1]

    def test_no_match_returns_empty(self) -> None:
        memories = [_memory("completely unrelated text")]
        scorer = BM25Scorer()
        assert scorer.score("xyz missing", memories) == {}

    def test_cjk_tokenization(self) -> None:
        memories = [
            _memory("我喜欢 Python"),
            _memory("今天天气很好"),
        ]
        scorer = BM25Scorer()
        scores = scorer.score("喜欢 Python", memories)
        assert scores[0] > scores.get(1, -1.0)


class TestVectorScorer:
    def test_cosine_similarity_range(self) -> None:
        query = torch.tensor([1.0, 0.0, 0.0])
        memories = [
            _memory("same direction", [1.0, 0.0, 0.0]),
            _memory("orthogonal", [0.0, 1.0, 0.0]),
            _memory("opposite", [-1.0, 0.0, 0.0]),
        ]
        scorer = VectorScorer(query_embedding=query)
        scores = scorer.score("ignored", memories)
        assert scores[0] == pytest.approx(1.0)
        assert scores[1] == pytest.approx(0.0, abs=1e-6)
        assert scores[2] == pytest.approx(-1.0)

    def test_empty_memories(self) -> None:
        scorer = VectorScorer(query_embedding=torch.randn(8))
        assert scorer.score("query", []) == {}


class TestFusionStrategies:
    def test_rank_fusion_prefers_top_of_both(self) -> None:
        rankings = [
            {0: 1.0, 1: 0.5, 2: 0.1},
            {0: 0.8, 2: 0.9, 1: 0.3},
        ]
        fused = RankFusion().fuse(rankings)
        # Both rank memory 0 first, so it should win.
        assert fused[0] > fused[1]
        assert fused[0] > fused[2]

    def test_weighted_fusion_respects_weights(self) -> None:
        rankings = [
            {0: 1.0, 1: 0.0},
            {0: 0.0, 1: 1.0},
        ]
        fused = WeightedFusion(weights=[0.5, 0.5]).fuse(rankings)
        assert fused[0] == pytest.approx(0.5)
        assert fused[1] == pytest.approx(0.5)

    def test_weighted_fusion_mismatched_count_raises(self) -> None:
        with pytest.raises(ValueError):
            WeightedFusion(weights=[0.5, 0.5]).fuse([{0: 1.0}])


class TestLifecycleScorer:
    def test_recency_decay(self) -> None:
        now = datetime.now(timezone.utc)
        old = _memory("old")
        old.created_at = now - timedelta(days=30)
        new = _memory("new")
        new.created_at = now

        scorer = LifecycleScorer(recency_half_life_days=30.0, now=now)
        scores = scorer.score([old, new])
        # New memory has recency 1.0 + importance 1.0 = 2.0.
        # Old memory has recency 0.5 + importance 1.0 = 1.5.
        assert scores[1] > scores[0]
        assert scores[1] == pytest.approx(
            LifecycleWeights().recency
            + LifecycleWeights().importance
        )
        assert scores[0] == pytest.approx(scores[1] * 0.75, rel=1e-3)

    def test_importance_boost(self) -> None:
        now = datetime.now(timezone.utc)
        low = _memory("low")
        low.metadata["lifecycle"] = MemoryMetadata(
            created_at=now, importance=1.0
        )
        high = _memory("high")
        high.metadata["lifecycle"] = MemoryMetadata(
            created_at=now, importance=5.0
        )

        scorer = LifecycleScorer(now=now)
        scores = scorer.score([low, high])
        assert scores[1] > scores[0]

    def test_stale_penalty(self) -> None:
        now = datetime.now(timezone.utc)
        fresh = _memory("fresh")
        fresh.metadata["lifecycle"] = MemoryMetadata(created_at=now)
        stale = _memory("stale")
        stale.metadata["lifecycle"] = MemoryMetadata(
            created_at=now, stale=True
        )

        scorer = LifecycleScorer(now=now)
        scores = scorer.score([fresh, stale])
        assert scores[0] > 0
        assert scores[1] == pytest.approx(
            scores[0] * (1 - LifecycleWeights().staleness_penalty)
        )

    def test_mark_stale_only_old_unimportant(self) -> None:
        now = datetime.now(timezone.utc)
        old_unimportant = _memory("old")
        old_unimportant.metadata["lifecycle"] = MemoryMetadata(
            created_at=now - timedelta(days=100),
            importance=0.5,
        )
        old_important = _memory("important")
        old_important.metadata["lifecycle"] = MemoryMetadata(
            created_at=now - timedelta(days=100),
            importance=10.0,
        )
        fresh = _memory("fresh")
        fresh.metadata["lifecycle"] = MemoryMetadata(
            created_at=now - timedelta(days=1),
            importance=0.5,
        )

        scorer = LifecycleScorer(now=now)
        stale_indices = scorer.mark_stale(
            [old_unimportant, old_important, fresh],
            max_age_days=90,
            min_access_count=5,
            min_importance=1.0,
        )
        assert stale_indices == [0]


class TestLifecycleUtilities:
    def test_deduplicate_memories_keeps_most_recent(self) -> None:
        now = datetime.now(timezone.utc)
        older = _memory("dup")
        older.created_at = now
        newer = _memory("dup")
        newer.created_at = now + timedelta(seconds=1)
        unique = _memory("unique")
        result = deduplicate_memories([older, newer, unique])
        assert len(result) == 2
        assert result[0] is newer
        assert result[1] is unique

    def test_ensure_lifecycle_metadata_creates_default(self) -> None:
        memory = _memory("text")
        assert "lifecycle" not in memory.metadata
        meta = ensure_lifecycle_metadata(memory)
        assert memory.metadata["lifecycle"] is meta
        assert meta.created_at == memory.created_at


class TestAdapterHybridRecall:
    def test_recall_returns_normalized_scores(self) -> None:
        adapter = _make_adapter()
        adapter.observe([Message(role="user", content="Python fast sorting")])
        adapter.learn(["Python fast sorting"])

        results = adapter.recall("Python fast sorting", top_k=5)
        assert len(results) == 1
        assert results[0].text == "Python fast sorting"
        assert 0.0 <= results[0].score <= 1.0

    def test_recall_bm25_disabled_falls_back_to_vector(self) -> None:
        adapter = _make_adapter()
        adapter.observe([Message(role="user", content="Python fast sorting")])
        adapter.learn(["Python fast sorting"])

        results = adapter.recall(
            "Python fast sorting", top_k=5, use_bm25=False, use_lifecycle=False
        )
        assert len(results) == 1
        assert results[0].score > 0.99

    def test_recall_min_score_filters_results(self) -> None:
        adapter = _make_adapter()
        adapter.observe([Message(role="user", content="Python fast sorting")])
        adapter.observe([Message(role="user", content="unrelated gardening")])
        adapter.learn(["Python fast sorting", "unrelated gardening"])

        results = adapter.recall("Python fast sorting", top_k=5, min_score=0.5)
        texts = {m.text for m in results}
        assert "Python fast sorting" in texts
        assert "unrelated gardening" not in texts

    def test_recall_updates_access_metadata(self) -> None:
        adapter = _make_adapter()
        adapter.observe([Message(role="user", content="Python fast sorting")])
        adapter.learn(["Python fast sorting"])

        adapter.recall("Python fast sorting")
        meta = adapter._memories[0].metadata["lifecycle"]
        assert meta.access_count == 1
