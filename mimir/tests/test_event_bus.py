"""Tests for the in-memory event bus."""

from mimir.application.events.event_bus import EncodeEvent, EventBus, LearnEvent


def test_event_bus_publishes_to_subscribers() -> None:
    """Published events reach all subscribed handlers."""
    bus = EventBus()
    received: list[object] = []
    bus.subscribe(received.append)
    bus.publish("hello")
    assert received == ["hello"]


def test_event_bus_unsubscribe() -> None:
    """Unsubscribed handlers no longer receive events."""
    bus = EventBus()
    received: list[object] = []
    bus.subscribe(received.append)
    bus.unsubscribe(received.append)
    bus.publish("hello")
    assert received == []


def test_encode_event_attributes() -> None:
    """EncodeEvent stores the expected attributes."""
    import torch

    event = EncodeEvent(
        texts=["hello"],
        base=torch.randn(1, 8),
        output=torch.randn(1, 8),
        prototype_weights=torch.randn(1, 8),
        step=7,
    )
    assert event.texts == ["hello"]
    assert event.step == 7


def test_learn_event_attributes() -> None:
    """LearnEvent stores the expected attributes."""
    import torch

    report: dict[str, object] = {"updated": 1}
    event = LearnEvent(
        texts=["hello"],
        base=torch.randn(1, 8),
        updated_ids=[3],
        report=report,
        step=2,
    )
    assert event.updated_ids == [3]
    assert event.report is report
