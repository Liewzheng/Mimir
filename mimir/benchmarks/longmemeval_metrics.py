"""Metrics aggregation for LongMemEval harness."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Metrics:
    """Accumulate per-sample results and compute aggregated scores."""

    recall_hits: int = 0
    qa_correct: int = 0
    total: int = 0
    by_type: dict[str, dict[str, int]] = field(default_factory=dict)

    def add(self, question_type: str, recall_hit: bool, qa_correct: bool) -> None:
        self.total += 1
        if recall_hit:
            self.recall_hits += 1
        if qa_correct:
            self.qa_correct += 1

        bucket = self.by_type.setdefault(question_type, {"recall_hits": 0, "qa_correct": 0, "total": 0})
        bucket["total"] += 1
        if recall_hit:
            bucket["recall_hits"] += 1
        if qa_correct:
            bucket["qa_correct"] += 1


def aggregate_metrics(metrics: Metrics) -> dict[str, Any]:
    """Return a JSON-serializable report from accumulated metrics."""
    overall = {
        "recall@k": round(metrics.recall_hits / max(metrics.total, 1), 4),
        "qa_accuracy": round(metrics.qa_correct / max(metrics.total, 1), 4),
        "total": metrics.total,
    }
    by_type = {}
    for q_type, bucket in metrics.by_type.items():
        by_type[q_type] = {
            "recall@k": round(bucket["recall_hits"] / max(bucket["total"], 1), 4),
            "qa_accuracy": round(bucket["qa_correct"] / max(bucket["total"], 1), 4),
            "total": bucket["total"],
        }
    return {"overall": overall, "by_type": by_type}
