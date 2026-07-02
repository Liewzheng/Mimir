"""Event bus protocol for Mimir runtime events."""

from typing import Any, Protocol


class EventBus(Protocol):
    """Publish/subscribe event bus for inference and learning events."""

    def publish(self, event: Any) -> None:
        """Dispatch an event to all subscribed handlers."""
        ...
