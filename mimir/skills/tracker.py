"""Track repeated command patterns and compute a frustration score."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mimir.skills.extractor import Skeleton, extract_skeleton


@dataclass
class CommandEvent:
    """A single tool-call event relevant to skill tracking."""

    tool_name: str
    command: str
    context: dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillTrackerConfig:
    """Configuration for the skill tracker."""

    window_size: int = 50
    min_repetitions: int = 5
    frustration_threshold: float = 50.0
    min_fixed_ratio: float = 0.6


@dataclass
class PatternCluster:
    """A cluster of similar command events and its derived skeleton."""

    key: str
    events: list[CommandEvent] = field(default_factory=list)
    skeleton: Skeleton | None = None

    def add(self, event: CommandEvent) -> None:
        self.events.append(event)

    @property
    def repeat_count(self) -> int:
        return len(self.events)

    def compute_frustration(self, min_repetitions: int) -> float:
        if self.repeat_count < min_repetitions or self.skeleton is None:
            return 0.0
        return max(0, self.repeat_count - min_repetitions) * len(self.skeleton.fixed_part) * self.skeleton.fixed_ratio


class SkillTracker:
    """Observe tool calls, detect repetition, and trigger skill extraction."""

    def __init__(self, config: SkillTrackerConfig | None = None) -> None:
        self.config = config or SkillTrackerConfig()
        self._buffer: list[CommandEvent] = []
        self._clusters: dict[str, PatternCluster] = {}

    def observe(self, event: CommandEvent) -> None:
        """Record a tool-call event and update clusters."""
        self._buffer.append(event)
        if len(self._buffer) > self.config.window_size:
            self._buffer.pop(0)

        key = self._cluster_key(event)
        cluster = self._clusters.get(key)
        if cluster is None:
            cluster = PatternCluster(key=key)
            self._clusters[key] = cluster
        cluster.add(event)
        cluster.skeleton = extract_skeleton([e.command for e in cluster.events])

    def _cluster_key(self, event: CommandEvent) -> str:
        """Group shell commands by the first token; other tools by tool name."""
        if event.tool_name == "Shell" and event.command:
            first = event.command.split()[0] if event.command.split() else ""
            return f"Shell:{first}"
        return f"Tool:{event.tool_name}"

    def ready_clusters(self) -> list[PatternCluster]:
        """Return clusters whose frustration score exceeds the threshold."""
        ready: list[PatternCluster] = []
        for cluster in self._clusters.values():
            if cluster.skeleton is None:
                continue
            if cluster.skeleton.fixed_ratio < self.config.min_fixed_ratio:
                continue
            frustration = cluster.compute_frustration(self.config.min_repetitions)
            if frustration >= self.config.frustration_threshold:
                ready.append(cluster)
        return sorted(ready, key=lambda c: c.compute_frustration(self.config.min_repetitions), reverse=True)

    def reset(self, key: str | None = None) -> None:
        """Reset a cluster (after extracting a skill) or all clusters."""
        if key:
            self._clusters.pop(key, None)
        else:
            self._clusters.clear()
            self._buffer.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return a serializable snapshot for debugging."""
        return {
            "buffer_size": len(self._buffer),
            "cluster_count": len(self._clusters),
            "clusters": {
                key: {
                    "repeat_count": cluster.repeat_count,
                    "skeleton": cluster.skeleton.template if cluster.skeleton else None,
                    "fixed_ratio": cluster.skeleton.fixed_ratio if cluster.skeleton else 0.0,
                    "frustration": cluster.compute_frustration(self.config.min_repetitions),
                }
                for key, cluster in self._clusters.items()
            },
        }
