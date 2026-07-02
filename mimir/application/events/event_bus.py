"""Simple in-memory event bus."""

from collections.abc import Callable
from typing import Any

EventHandler = Callable[[Any], None]


class EventBus:
    """Publish/subscribe event bus for decoupling inference and learning."""

    def __init__(self) -> None:
        self._handlers: list[EventHandler] = []

    def subscribe(self, handler: EventHandler) -> None:
        """Register a handler that will be called for every published event."""
        self._handlers.append(handler)

    def publish(self, event: Any) -> None:
        """Dispatch an event to all subscribed handlers."""
        for handler in self._handlers:
            handler(event)

    def unsubscribe(self, handler: EventHandler) -> None:
        """Remove a previously registered handler."""
        if handler in self._handlers:
            self._handlers.remove(handler)


class EncodeEvent:
    """Emitted after an encode() call."""

    def __init__(
        self,
        texts: list[str],
        base: object,
        output: object,
        prototype_weights: object,
        step: int,
    ) -> None:
        self.texts = texts
        self.base = base
        self.output = output
        self.prototype_weights = prototype_weights
        self.step = step


class LearnEvent:
    """Emitted after a learn() call."""

    def __init__(
        self,
        texts: list[str],
        base: object,
        updated_ids: list[int],
        report: dict[str, object],
        step: int,
    ) -> None:
        self.texts = texts
        self.base = base
        self.updated_ids = updated_ids
        self.report = report
        self.step = step
