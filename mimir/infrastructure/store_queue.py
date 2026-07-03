"""Asynchronous embedding queue for Mimir store operations.

The queue decouples the MCP tool response from the embedding backend and
learning pipeline. Callers enqueue already-filtered, redacted texts; a single
worker thread consumes items and runs the processor callback. On shutdown the
remaining items can be flushed so that no memory is silently lost.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


Processor = Callable[[str, float], dict[str, Any]]


@dataclass(frozen=True)
class StoreItem:
    """A single item waiting to be embedded and learned."""

    text: str
    importance: float = 1.0


class MemoryStoreQueue:
    """Thread-safe queue that processes store items in the background.

    Args:
        processor: Callback that receives a safe text and importance, and
            returns a report dict. The callback is responsible for duplicate
            checks, observe/learn, and persistence.
        max_size: Maximum number of pending items. ``0`` means unlimited.
    """

    def __init__(
        self,
        processor: Processor,
        *,
        max_size: int = 0,
    ) -> None:
        self._processor = processor
        self._queue: queue.Queue[StoreItem | None] = queue.Queue(maxsize=max_size)
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker, daemon=True)
        self._worker_thread.start()

    def _worker(self) -> None:
        """Consume items until a stop sentinel is received."""
        while not self._stop_event.is_set():
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is None:
                # Stop sentinel.
                self._queue.task_done()
                return
            try:
                self._processor(item.text, item.importance)
            except Exception:  # noqa: BLE001
                logger.exception("Failed to process queued memory: %r", item.text[:80])
            finally:
                self._queue.task_done()

        # Drain remaining items without processing when stop is requested.
        try:
            while True:
                item = self._queue.get_nowait()
                self._queue.task_done()
                if item is None:
                    return
        except queue.Empty:
            return

    def put(self, text: str, importance: float = 1.0) -> None:
        """Enqueue a safe text for background processing.

        Raises:
            queue.Full: If the queue has reached ``max_size`` and the item cannot
            be accepted without blocking.
        """
        self._queue.put_nowait(StoreItem(text=text, importance=importance))

    def flush(self, timeout: float | None = None) -> bool:
        """Wait until all queued items are processed.

        Returns ``True`` if the queue emptied within the timeout, ``False``
        otherwise. A ``None`` timeout blocks until the queue is empty.
        """
        if timeout is None:
            self._queue.join()
            return True
        deadline = time.monotonic() + timeout
        while self._queue.unfinished_tasks > 0 and time.monotonic() < deadline:
            time.sleep(0.05)
        return self._queue.unfinished_tasks == 0

    def stop(self, timeout: float | None = None) -> bool:
        """Signal the worker to stop and wait for it to finish.

        Returns ``True`` if the worker stopped cleanly, ``False`` if it is still
        alive after the timeout.
        """
        self._stop_event.set()
        self._queue.put(None)  # wake the worker if it is sleeping
        self._worker_thread.join(timeout=timeout)
        return not self._worker_thread.is_alive()

    @property
    def pending_count(self) -> int:
        """Return the approximate number of items waiting to be processed."""
        return self._queue.unfinished_tasks

    @property
    def is_alive(self) -> bool:
        """Return whether the worker thread is still running."""
        return self._worker_thread.is_alive()
