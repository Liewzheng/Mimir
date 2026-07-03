"""LongMemEval harness for Mimir.

Run with:
    python -m mimir.benchmarks.longmemeval \
        --backend ollama \
        --model dengcao/Qwen3-Embedding-8B:Q5_K_M \
        --reader-model qwen3.6:35b-a3b \
        --dataset eval_data/longmemeval/longmemeval_s_cleaned.json \
        --limit 50
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mimir.adapters.agents import InMemoryAgentAdapter
from mimir.application.factories import create_embedding_engine, create_mimir
from mimir.core.config import MimirConfig
from mimir.domain.model import Message
from mimir.infrastructure.llm.ollama_reader import OllamaReader

from .longmemeval_metrics import Metrics, aggregate_metrics

logger = logging.getLogger(__name__)


def _configure_logging() -> None:
    """Send plain log messages to stdout so benchmark output remains pipeable."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(logging.INFO)


@dataclass(frozen=True)
class LongMemEvalSample:
    """A single LongMemEval question with its haystack."""

    question_id: str
    question_type: str
    question: str
    answer: str
    answer_session_ids: list[str]
    haystack_session_ids: list[str]
    haystack_sessions: list[list[dict[str, str]]]


class LongMemEvalDataset:
    """Load and iterate over a LongMemEval JSON file."""

    def __init__(self, path: str | Path, limit: int | None = None) -> None:
        self.path = Path(path)
        self.limit = limit
        self.samples = self._load()

    def _load(self) -> list[LongMemEvalSample]:
        try:
            with self.path.open(encoding="utf-8") as f:
                raw: list[dict[str, Any]] = json.load(f)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse dataset %s: %s", self.path, exc)
            raise SystemExit(1) from exc
        except OSError as exc:
            logger.error("Failed to read dataset %s: %s", self.path, exc)
            raise SystemExit(1) from exc
        if self.limit is not None:
            raw = raw[: self.limit]
        return [
            LongMemEvalSample(
                question_id=item["question_id"],
                question_type=item["question_type"],
                question=item["question"],
                answer=item["answer"],
                answer_session_ids=item["answer_session_ids"],
                haystack_session_ids=item["haystack_session_ids"],
                haystack_sessions=item["haystack_sessions"],
            )
            for item in raw
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[LongMemEvalSample]:
        return iter(self.samples)


class LongMemEvalHarness:
    """Feed LongMemEval samples into Mimir and measure retrieval + QA."""

    def __init__(
        self,
        adapter: InMemoryAgentAdapter,
        reader: OllamaReader | None,
        top_k: int = 10,
        skip_consolidate: bool = False,
    ) -> None:
        self.adapter = adapter
        self.reader = reader
        self.top_k = top_k
        self.skip_consolidate = skip_consolidate

    def ingest(self, sample: LongMemEvalSample) -> None:
        """Observe all turns in the haystack, tagging each memory with its session id."""
        for session_id, session in zip(
            sample.haystack_session_ids, sample.haystack_sessions, strict=True
        ):
            messages = [
                Message(
                    role=turn["role"],
                    content=turn["content"],
                    metadata={"session_id": session_id, "question_id": sample.question_id},
                )
                for turn in session
            ]
            self.adapter.observe(messages)
        if not self.skip_consolidate:
            self.adapter.consolidate()

    def evaluate(self, sample: LongMemEvalSample) -> dict[str, Any]:
        """Recall memories for the question and optionally generate + judge an answer."""
        memories = self.adapter.recall(sample.question, top_k=self.top_k)
        retrieved_session_ids = {
            memory.metadata.get("session_id")
            for memory in memories
            if memory.metadata.get("session_id")
        }
        gold_ids = set(sample.answer_session_ids)
        recall_hit = bool(retrieved_session_ids & gold_ids)

        context = "\n\n".join(memory.text for memory in memories)
        prediction = ""
        judged_correct = False
        if self.reader is not None:
            prediction = self.reader.answer(context, sample.question)
            judged_correct = self.reader.judge(sample.question, sample.answer, prediction)

        return {
            "question_id": sample.question_id,
            "question_type": sample.question_type,
            "recall_hit": recall_hit,
            "retrieved_session_ids": sorted(retrieved_session_ids),
            "gold_session_ids": sorted(gold_ids),
            "prediction": prediction,
            "judged_correct": judged_correct,
        }

    def run(self, dataset: LongMemEvalDataset) -> tuple[list[dict[str, Any]], Metrics]:
        """Run the full evaluation and return per-sample results + aggregated metrics."""
        metrics = Metrics()
        results: list[dict[str, Any]] = []
        for idx, sample in enumerate(dataset):
            logger.info(
                "[%d/%d] %s %s",
                idx + 1,
                len(dataset),
                sample.question_id,
                sample.question_type,
            )
            self.adapter.reset()
            self.ingest(sample)
            result = self.evaluate(sample)
            results.append(result)
            metrics.add(
                question_type=sample.question_type,
                recall_hit=result["recall_hit"],
                qa_correct=result["judged_correct"],
            )
        return results, metrics


def _build_adapter(
    backend: str,
    model: str,
    base_url: str,
    num_prototypes: int = 1024,
) -> InMemoryAgentAdapter:
    """Build an InMemoryAgentAdapter for the requested embedding backend."""
    engine = create_embedding_engine(backend=backend, model=model, base_url=base_url)
    base_model = model if backend in ("sentence-transformer", "ollama") else base_url
    config = MimirConfig(
        base_model=base_model,
        num_prototypes=num_prototypes,
        learning_rate_base=0.01,
        learning_rate_decay=0.1,
        top_k=16,
        filter_enabled=False,
        quality_gate_enabled=False,
        async_store_enabled=False,
    )
    mimir = create_mimir(config, engine=engine)
    return InMemoryAgentAdapter(
        mimir=mimir,
        learn_on_observe=False,
        max_memories=100_000,
        max_text_length=100_000,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m mimir.benchmarks.longmemeval",
        description="Run Mimir against the LongMemEval benchmark.",
    )
    parser.add_argument(
        "--backend",
        choices=["llama-server", "sentence-transformer", "ollama", "fake"],
        default="ollama",
        help="Embedding backend to use",
    )
    parser.add_argument(
        "--base-url",
        default="http://localhost:11434",
        help="Base URL for the embedding backend (Ollama or llama-server)",
    )
    parser.add_argument(
        "--model",
        default="dengcao/Qwen3-Embedding-8B:Q5_K_M",
        help="Embedding model name",
    )
    parser.add_argument(
        "--reader-model",
        default="qwen3.6:35b-a3b",
        help="Ollama chat model used for answer generation and judging",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to longmemeval_s_cleaned.json",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of memories to retrieve for each question",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only evaluate the first N samples",
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=1024,
        help="Number of Mimir prototypes",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip LLM reader/judge; only compute Recall@K",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip the Mimir consolidate step after ingesting each sample",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON results",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    _configure_logging()
    parser = _build_parser()
    args = parser.parse_args(argv)

    if not args.dataset.exists():
        logger.error("Dataset not found: %s", args.dataset)
        return 1

    reader = None if args.skip_qa else OllamaReader(args.reader_model)
    adapter = _build_adapter(
        backend=args.backend,
        model=args.model,
        base_url=args.base_url,
        num_prototypes=args.num_prototypes,
    )
    harness = LongMemEvalHarness(
        adapter,
        reader=reader,
        top_k=args.top_k,
        skip_consolidate=args.skip_consolidate,
    )
    dataset = LongMemEvalDataset(args.dataset, limit=args.limit)

    try:
        results, metrics = harness.run(dataset)
    finally:
        if reader is not None:
            reader.close()

    report = aggregate_metrics(metrics)
    report["per_sample"] = results
    logger.info(json.dumps(report, indent=2, ensure_ascii=False))

    if args.output is not None:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.info("Results written to %s", args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
