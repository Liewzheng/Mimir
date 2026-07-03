"""Tests for the asynchronous store queue integration."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest

from mimir.infrastructure.store_queue import MemoryStoreQueue
from mimir.mcp.session import SessionManager


@pytest.fixture
def async_session(tmp_path: Path) -> SessionManager:
    """Return a SessionManager with async store enabled."""
    return SessionManager(
        backend="fake",
        workspace_path=tmp_path / "workspace",
        base_dir=tmp_path / ".mimir",
        async_store_enabled=True,
        async_store_flush_timeout=5.0,
    )


class TestMemoryStoreQueue:
    def test_processes_items_in_background(self) -> None:
        processed: list[tuple[str, float]] = []

        def processor(text: str, importance: float) -> dict[str, object]:
            processed.append((text, importance))
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=100)
        queue.put("hello", importance=2.0)
        queue.flush(timeout=2.0)
        queue.stop(timeout=2.0)
        assert ("hello", 2.0) in processed

    def test_flush_timeout_returns_false_when_busy(self) -> None:
        def processor(text: str, importance: float) -> dict[str, object]:
            time.sleep(1.0)
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=100)
        queue.put("slow item")
        # The worker is now sleeping; flush should time out.
        flushed = queue.flush(timeout=0.1)
        assert flushed is False
        queue.stop(timeout=3.0)

    def test_failed_items_are_logged_not_raised(self) -> None:
        def processor(text: str, importance: float) -> dict[str, object]:
            raise RuntimeError("boom")

        queue = MemoryStoreQueue(processor=processor, max_size=100)
        queue.put("will fail")
        # Should not raise; flush returns normally.
        queue.flush(timeout=2.0)
        queue.stop(timeout=2.0)

    def test_put_nowait_raises_when_full(self) -> None:
        import queue as stdlib_queue

        def processor(text: str, importance: float) -> dict[str, object]:
            time.sleep(1.0)
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=1)
        queue.put("first item")
        # The first item is in-flight; unfinished_tasks == 1 == max_size.
        with pytest.raises(stdlib_queue.Full):
            queue.put("second item")
        queue.stop(timeout=3.0)

    def test_unlimited_queue_size_accepts_many_items(self) -> None:
        processed: list[str] = []

        def processor(text: str, importance: float) -> dict[str, object]:
            processed.append(text)
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=0)
        for i in range(50):
            queue.put(f"item {i}")
        queue.flush(timeout=5.0)
        queue.stop(timeout=5.0)
        assert len(processed) == 50

    def test_stop_returns_false_when_worker_busy(self) -> None:
        def processor(text: str, importance: float) -> dict[str, object]:
            time.sleep(2.0)
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=100)
        queue.put("slow item")
        stopped = queue.stop(timeout=0.1)
        assert stopped is False
        # Give the worker time to finish after the test so resources are released.
        queue.stop(timeout=3.0)

    def test_stop_returns_true_when_idle(self) -> None:
        def processor(text: str, importance: float) -> dict[str, object]:
            return {"stored": True}

        queue = MemoryStoreQueue(processor=processor, max_size=100)
        queue.put("item")
        queue.flush(timeout=2.0)
        stopped = queue.stop(timeout=2.0)
        assert stopped is True


class TestAsyncSessionManager:
    def test_store_returns_pending(self, async_session: SessionManager) -> None:
        result = async_session.store("I like Python")
        assert result["stored"] == "pending"
        assert result["text"] == "I like Python"
        assert result["pending_count"] >= 1

    def test_store_learns_after_flush(self, async_session: SessionManager) -> None:
        assert async_session._store_queue is not None
        async_session.store("I like Python")
        async_session._store_queue.flush(timeout=2.0)
        assert async_session.adapter.memory_count == 1

    def test_close_flushes_pending_items(self, async_session: SessionManager) -> None:
        async_session.store("I like Python")
        async_session.close()
        assert async_session.adapter.memory_count == 1

    def test_async_duplicate_check(self, async_session: SessionManager) -> None:
        assert async_session._store_queue is not None
        async_session.store("I like Python")
        async_session._store_queue.flush(timeout=2.0)
        async_session.store("I like Python")
        async_session._store_queue.flush(timeout=2.0)
        assert async_session.adapter.memory_count == 1

    def test_store_returns_queue_full_when_full(self, async_session: SessionManager) -> None:
        # Create a fresh queue with max_size=1 and slow processor so the first item
        # stays in-flight and the second store() returns queue_full instead of pending.
        from mimir.infrastructure.store_queue import MemoryStoreQueue

        def slow_processor(text: str, importance: float) -> dict[str, object]:
            time.sleep(2.0)
            return {"stored": True}

        async_session._store_queue = MemoryStoreQueue(
            processor=slow_processor,
            max_size=1,
        )
        async_session.store("first item")
        result = async_session.store("second item")
        assert result["stored"] is False
        assert result["reason"] == "queue_full"

    def test_close_logs_warning_when_flush_times_out(self, async_session: SessionManager) -> None:
        from mimir.infrastructure.store_queue import MemoryStoreQueue

        assert async_session._store_queue is not None

        def slow_processor(text: str, importance: float) -> dict[str, object]:
            time.sleep(2.0)
            return {"stored": True}

        async_session._store_queue = MemoryStoreQueue(
            processor=slow_processor,
            max_size=100,
        )
        async_session.config = replace(
            async_session.config,
            async_store_flush_timeout=0.0,
        )
        async_session.store("I like Python")
        async_session.close()
        # With flush timeout 0 and a slow processor, the item should not have been learned.
        assert async_session.adapter.memory_count == 0

    def test_store_rejects_invalid_importance(self, async_session: SessionManager) -> None:
        result = async_session.store("I like Python", importance=-1.0)
        assert result["stored"] is False
        assert result["reason"] == "invalid_importance"

    def test_async_processor_failure_is_logged_not_raised(self, async_session: SessionManager) -> None:
        from mimir.infrastructure.store_queue import MemoryStoreQueue

        assert async_session._store_queue is not None

        def failing_processor(text: str, importance: float) -> dict[str, object]:
            raise RuntimeError("async boom")

        async_session._store_queue = MemoryStoreQueue(
            processor=failing_processor,
            max_size=100,
        )
        async_session.store("I like Python")
        # Should not raise; the queue keeps working.
        async_session._store_queue.flush(timeout=2.0)
        async_session._store_queue.stop(timeout=2.0)

    def test_status_reports_async_queue_fields(self, async_session: SessionManager) -> None:
        async_session.store("I like Python")
        status = async_session.status()
        assert status["async_store"]["enabled"] is True
        assert status["async_store"]["pending_count"] >= 1
        assert isinstance(status["async_store"]["worker_alive"], bool)

    def test_concurrent_store_and_recall(self, async_session: SessionManager) -> None:
        import threading

        errors: list[Exception] = []

        def store_loop() -> None:
            for i in range(20):
                try:
                    async_session.store(f"memory {i}")
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(exc)

        def recall_loop() -> None:
            for _ in range(20):
                try:
                    async_session.recall("memory")
                except Exception as exc:  # pragma: no cover - defensive
                    errors.append(exc)

        t1 = threading.Thread(target=store_loop)
        t2 = threading.Thread(target=recall_loop)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)
        assert not errors, f"Concurrent errors: {errors}"
        async_session.close()


class TestSyncSessionManagerStillWorks:
    def test_sync_store_unchanged(self, tmp_path: Path) -> None:
        session = SessionManager(
            backend="fake",
            workspace_path=tmp_path / "workspace",
            base_dir=tmp_path / ".mimir",
        )
        result = session.store("I like Python")
        assert result["stored"] is True
        assert session.adapter.memory_count == 1


class TestConfigValidation:
    def test_rejects_negative_async_queue_size(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="async_store_queue_size must be non-negative"):
            SessionManager(
                backend="fake",
                workspace_path=tmp_path / "workspace",
                base_dir=tmp_path / ".mimir",
                async_store_enabled=True,
                async_store_queue_size=-1,
            )

    def test_rejects_negative_flush_timeout(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="async_store_flush_timeout must be non-negative"):
            SessionManager(
                backend="fake",
                workspace_path=tmp_path / "workspace",
                base_dir=tmp_path / ".mimir",
                async_store_enabled=True,
                async_store_flush_timeout=-1.0,
            )

    def test_allows_unlimited_queue_size(self, tmp_path: Path) -> None:
        session = SessionManager(
            backend="fake",
            workspace_path=tmp_path / "workspace",
            base_dir=tmp_path / ".mimir",
            async_store_enabled=True,
            async_store_queue_size=0,
        )
        assert session.config.async_store_queue_size == 0
        session.close()
