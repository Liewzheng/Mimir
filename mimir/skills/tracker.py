"""Track repeated command patterns and compute a frustration score."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from mimir.skills.extractor import Skeleton, extract_skeleton


@dataclass
class CommandEvent:
    """A single tool-call event relevant to skill tracking."""

    tool_name: str
    command: str
    context: dict[str, Any] = field(default_factory=dict)
    event_id: int = field(default=0, compare=False)

    def to_dict(self) -> dict[str, Any]:
        return {"tool_name": self.tool_name, "command": self.command, "context": self.context}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CommandEvent:
        return cls(
            tool_name=data["tool_name"],
            command=data["command"],
            context=data.get("context", {}),
        )


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

    def remove(self, event_id: int) -> None:
        self.events = [e for e in self.events if e.event_id != event_id]

    @property
    def repeat_count(self) -> int:
        return len(self.events)

    def compute_frustration(self, min_repetitions: int) -> float:
        if self.repeat_count < min_repetitions or self.skeleton is None:
            return 0.0
        return max(0, self.repeat_count - min_repetitions) * len(self.skeleton.fixed_part) * self.skeleton.fixed_ratio

    def update_skeleton(self) -> None:
        self.skeleton = extract_skeleton([e.command for e in self.events])


class SkillTracker:
    """Observe tool calls, detect repetition, and trigger skill extraction."""

    def __init__(self, config: SkillTrackerConfig | None = None) -> None:
        self.config = config or SkillTrackerConfig()
        self._buffer: list[CommandEvent] = []
        self._clusters: dict[str, PatternCluster] = {}
        self._next_event_id = 0

    def observe(self, event: CommandEvent) -> None:
        """Record a tool-call event and update clusters."""
        self._next_event_id += 1
        event.event_id = self._next_event_id
        self._buffer.append(event)

        # Prune oldest events if the window is exceeded.
        while len(self._buffer) > self.config.window_size:
            old = self._buffer.pop(0)
            self._prune_event(old)

        key = self._cluster_key(event)
        cluster = self._clusters.get(key)
        if cluster is None:
            cluster = PatternCluster(key=key)
            self._clusters[key] = cluster
        cluster.add(event)
        cluster.update_skeleton()

    def _prune_event(self, event: CommandEvent) -> None:
        """Remove an old event from its cluster and clean up empty clusters."""
        key = self._cluster_key(event)
        cluster = self._clusters.get(key)
        if cluster is None:
            return
        cluster.remove(event.event_id)
        if cluster.repeat_count == 0:
            del self._clusters[key]
        else:
            cluster.update_skeleton()

    def _cluster_key(self, event: CommandEvent) -> str:
        """Group shell commands by the first meaningful token(s).

        For shell commands, we use the first non-empty token plus the second token
        if it exists and is not a short option. This reduces the chance of mixing
        unrelated commands like ``adb shell`` and ``adb logcat`` in the same
        cluster.
        """
        if event.tool_name == "Shell":
            tokens = [t for t in event.command.split() if t]
            if not tokens:
                return "Shell:<empty>"
            first = tokens[0]
            if len(tokens) >= 2 and not tokens[1].startswith("-"):
                return f"Shell:{first} {tokens[1]}"
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
            self._next_event_id = 0

    def state(self) -> dict[str, Any]:
        """Return a fully serializable state for persistence across hook calls."""
        return {
            "config": asdict(self.config),
            "next_event_id": self._next_event_id,
            "buffer": [e.to_dict() for e in self._buffer],
            "clusters": {
                key: {
                    "events": [e.to_dict() for e in cluster.events],
                }
                for key, cluster in self._clusters.items()
            },
        }

    def restore(self, data: dict[str, Any]) -> None:
        """Restore tracker state from a previously saved snapshot."""
        self.config = SkillTrackerConfig(**data.get("config", {}))
        self._next_event_id = data.get("next_event_id", 0)
        self._buffer = [CommandEvent.from_dict(e) for e in data.get("buffer", [])]
        self._clusters = {
            key: PatternCluster(
                key=key,
                events=[CommandEvent.from_dict(e) for e in cluster_data.get("events", [])],
            )
            for key, cluster_data in data.get("clusters", {}).items()
        }
        for cluster in self._clusters.values():
            cluster.update_skeleton()

    def snapshot(self) -> dict[str, Any]:
        """Return a summary snapshot for debugging."""
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
