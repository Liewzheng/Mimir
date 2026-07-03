"""Workspace-scoped session management for the Mimir MCP server."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import pickle
import queue
import re
import subprocess
import tempfile
import threading
from collections import deque
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any, Literal, TypeVar, cast

from mimir.adapters.agents import InMemoryAgentAdapter
from mimir.application.factories import create_embedding_engine
from mimir.core.config import MimirConfig
from mimir.domain.model import Message
from mimir.infrastructure.context_discovery import ContextDiscovery
from mimir.infrastructure.filtering import FilterConfig, FilterEngine
from mimir.infrastructure.quality_gate import QualityGate, QualityResult
from mimir.infrastructure.redaction import Redactor
from mimir.infrastructure.store_queue import MemoryStoreQueue

logger = logging.getLogger(__name__)


T = TypeVar("T", bound=Callable[..., Any])


def _locked(method: T) -> T:
    """Decorator that acquires the adapter lock around a SessionManager method."""

    @wraps(method)
    def wrapper(self: SessionManager, *args: Any, **kwargs: Any) -> Any:
        with self._adapter_lock:
            return method(self, *args, **kwargs)

    return cast(T, wrapper)


_CHECKPOINT_NAME_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")
_MAX_LOAD_ERRORS = 32


def _atomic_write(path: Path, data: str) -> None:
    """Write ``data`` to ``path`` atomically via a temp file + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
        mode="w",
        encoding="utf-8",
    ) as tmp_file:
        tmp_path = Path(tmp_file.name)
        try:
            tmp_file.write(data)
            tmp_file.flush()
            os.replace(tmp_path, path)
        except OSError:
            tmp_path.unlink(missing_ok=True)
            raise


def _memory_key(item: dict[str, Any]) -> str:
    """Return a stable deduplication key for a memory entry.

    A content hash is used instead of (text, created_at) so that memories with
    the same text but different timestamps are deduplicated, while avoiding
    collisions from timestamp precision differences.
    """
    text = item.get("text", "")
    created_at = item.get("created_at", "")
    return hashlib.sha256(f"{text}::{created_at}".encode()).hexdigest()


def _merge_memories_state(
    on_disk: list[dict[str, Any]],
    in_memory: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge on-disk memories (e.g. written by agent CLI hooks) with in-memory state.

    Hook writes happen in a separate process, so the long-running MCP server's
    in-memory adapter does not see them until we reload and merge on shutdown.

    On-disk items are processed first so that hook updates are not overwritten
    by stale in-memory state.
    """
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for item in on_disk + in_memory:
        if not isinstance(item, dict):
            continue
        key = _memory_key(item)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _detect_workspace_path(cwd: str | Path | None = None) -> Path:
    """Return the git repository root, or the resolved cwd if not inside git."""
    cwd = Path(cwd or os.getcwd()).resolve()
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=False,
            timeout=5.0,
        )
        if result.returncode == 0:
            return Path(result.stdout.strip()).resolve()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return cwd


def _workspace_hash(path: Path) -> str:
    """Return a short deterministic hash for a workspace path."""
    digest = hashlib.sha256(str(path).encode("utf-8")).hexdigest()
    return digest[:16]


def _validate_checkpoint_name(name: str) -> None:
    """Reject checkpoint names that could escape the checkpoints directory."""
    if not _CHECKPOINT_NAME_PATTERN.match(name):
        raise ValueError(
            f"Checkpoint name must only contain letters, digits, underscores, "
            f"and hyphens; got {name!r}"
        )


class SessionManager:
    """Manages a single workspace's Mimir session and persistence.

    Each workspace is isolated under ``~/.mimir/workspaces/<hash>/``.
    Session state includes both the Mimir prototype matrix and the working
    memory buffer so that recall returns the expected results after restore.

    The primary session checkpoint is stored as ``checkpoints/session`` inside
    the workspace directory (without a ``.pt`` extension so it is not listed as
    a user checkpoint).  Named checkpoints created via :meth:`checkpoint` live
    alongside it as ``checkpoints/<name>.pt``.

    .. note::
        :meth:`close` propagates save failures as exceptions so that callers
        can detect silent data loss.  MCP server lifespans should handle or log
        these errors.
    """

    def __init__(
        self,
        backend: str = "llama-server",
        base_url: str = "http://127.0.0.1:11435",
        model: str = "all-MiniLM-L6-v2",
        workspace_path: str | Path | None = None,
        base_dir: str | Path | None = None,
        num_prototypes: int = 64,
        top_k: int = 4,
        *,
        async_store_enabled: bool = False,
        async_store_queue_size: int = 1000,
        async_store_flush_timeout: float = 5.0,
    ) -> None:
        """Initialize the session manager.

        Args:
            backend: ``llama-server``, ``sentence-transformer``, or ``fake``.
            base_url: URL for the llama-server backend.
            model: Model name for the sentence-transformer backend.
            workspace_path: Project directory used to derive workspace identity.
            base_dir: Root directory for all workspaces. Defaults to
                ``~/.mimir/workspaces``.
            num_prototypes: Number of prototypes in the Mimir store.
            top_k: Number of prototypes to activate during inference.
            async_store_enabled: Whether to defer embedding/learning to a
                background worker for non-blocking ``store()`` calls.
            async_store_queue_size: Max pending items for the async queue.
            async_store_flush_timeout: Seconds to wait for queue flush on close.
        """
        self.backend = backend
        self.base_url = base_url
        self.model = model
        self.workspace_path = _detect_workspace_path(workspace_path)
        self.workspace_hash = _workspace_hash(self.workspace_path)

        base_dir = Path(base_dir).expanduser().resolve() if base_dir else None
        self.workspace_dir = (
            base_dir or Path.home() / ".mimir" / "workspaces"
        ) / self.workspace_hash
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.memories_path = self.workspace_dir / "memories.json"
        self.checkpoints_dir = self.workspace_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        # The primary Mimir session checkpoint is kept as a named checkpoint
        # so it respects the adapter's checkpoint sandboxing.  It intentionally
        # has no .pt extension so it is not listed as a user checkpoint.
        self.session_checkpoint_name = "session"
        self.mimir_path = self.checkpoints_dir / self.session_checkpoint_name

        engine = create_embedding_engine(backend=backend, base_url=base_url, model=model)
        config = MimirConfig(
            base_model=model if backend == "sentence-transformer" else base_url,
            num_prototypes=num_prototypes,
            top_k=top_k,
            async_store_enabled=async_store_enabled,
            async_store_queue_size=async_store_queue_size,
            async_store_flush_timeout=async_store_flush_timeout,
        )
        self.config = config
        self.adapter = InMemoryAgentAdapter(
            config=config,
            engine=engine,
            checkpoint_dir=self.checkpoints_dir,
            learn_on_observe=False,
        )
        self.filter_engine = FilterEngine(
            FilterConfig(
                enabled=config.filter_enabled,
                min_store_length=config.filter_min_store_length,
                min_hook_length=config.filter_min_hook_length,
                min_hook_importance=config.filter_min_hook_importance,
                small_talk_ratio_threshold=config.filter_small_talk_ratio_threshold,
                user_resource_dir=config.filter_user_resource_dir,
            )
        )
        self.redactor = Redactor(
            enabled=config.redaction_enabled,
            patterns=config.redaction_patterns,
        )
        self.quality_gate = QualityGate(
            duplicate_threshold=config.quality_gate_duplicate_threshold,
            contradiction_threshold=config.quality_gate_contradiction_threshold,
        )
        self.context_discovery = ContextDiscovery()

        self._store_queue: MemoryStoreQueue | None = None
        if config.async_store_enabled:
            self._store_queue = MemoryStoreQueue(
                processor=self._async_process,
                max_size=config.async_store_queue_size,
            )

        # Protect adapter state when the async worker and MCP request handlers
        # may access it concurrently.
        self._adapter_lock = threading.RLock()

        self._load_errors: deque[str] = deque(maxlen=_MAX_LOAD_ERRORS)

        self._load()
        self._load_project_context()

    def _record_load_error(self, message: str) -> None:
        """Append a load error; old entries are dropped once the buffer is full."""
        self._load_errors.append(message)

    def _load(self) -> None:
        """Load Mimir state and working memory if they exist."""
        self._migrate_legacy_session_checkpoint()
        if self.mimir_path.exists():
            try:
                self.adapter.restore(self.session_checkpoint_name)
                logger.info("Loaded Mimir state from %s", self.mimir_path)
            except (FileNotFoundError, ValueError, RuntimeError, pickle.UnpicklingError):
                self.adapter.reset()
                self._record_load_error(f"Failed to load Mimir state from {self.mimir_path}")
                logger.exception("Failed to load Mimir state; will still try memories")

        if self.memories_path.exists():
            try:
                data = json.loads(self.memories_path.read_text(encoding="utf-8"))
                self.adapter.load_memories_state(data)
                logger.info(
                    "Loaded %d memories from %s",
                    len(data),
                    self.memories_path,
                )
            except (json.JSONDecodeError, ValueError, TypeError):
                self._record_load_error(f"Failed to load memories from {self.memories_path}")
                logger.exception("Failed to load memories; starting fresh")

    def _load_project_context(self) -> None:
        """Ingest agent instruction files from the workspace as high-importance memories.

        Files such as ``AGENTS.md``, ``CLAUDE.md``, and ``.cursorrules`` are
        loaded once per session. Duplicate content is ignored because the adapter
        deduplicates memories by content hash.
        """
        if not self.config.project_context_enabled:
            return

        contexts = self.context_discovery.discover(self.workspace_path)
        if not contexts:
            return

        existing_texts = {m["text"] for m in self.adapter.memories_state()}
        texts: list[str] = []
        for context in contexts:
            safe_text = self._prepare_text(
                self.context_discovery.format_memory(context), source="mcp"
            )
            if safe_text is not None and safe_text not in existing_texts:
                texts.append(safe_text)

        if not texts:
            return

        try:
            self._observe_and_learn(
                texts, importance=self.config.project_context_importance
            )
            logger.info(
                "Loaded %d project context file(s) from %s",
                len(texts),
                self.workspace_path,
            )
        except Exception:  # noqa: BLE001
            logger.exception("Failed to load project context")

    def _prepare_text(self, text: str, source: Literal["mcp", "hook"]) -> str | None:
        """Return a filtered and redacted version of *text*, or ``None`` if it should be dropped."""
        result = self.filter_engine.should_store(text, source=source)
        if not result.store:
            return None
        return self.redactor.redact(text)

    def _duplicate_check(self, safe_text: str) -> QualityResult | None:
        """Return a duplicate result if *safe_text* is too close to an existing memory.

        Returns ``None`` when the gate is disabled, there are no existing memories,
        or the embedding backend fails. The latter is logged but should not block
        storage, since the duplicate check is a quality improvement, not a hard
        invariant.
        """
        if not self.config.quality_gate_enabled:
            return None

        existing = self.adapter.memories_state()
        if not existing:
            return None

        try:
            candidate_embedding = self.adapter.encode([safe_text])[0]
        except Exception:  # noqa: BLE001
            logger.exception("Failed to encode candidate for duplicate check")
            return None

        existing_texts: list[str] = []
        existing_embeddings: list[Any] = []
        for item in existing:
            embedding = item.get("embedding")
            if embedding is None:
                continue
            existing_texts.append(item.get("text", ""))
            existing_embeddings.append(embedding)
        if not existing_embeddings:
            return None
        return self.quality_gate.check_duplicate(
            safe_text,
            candidate_embedding,
            existing_texts,
            existing_embeddings,
        )

    def _observe_and_learn(
        self, safe_texts: list[str], importance: float = 1.0
    ) -> dict[str, Any]:
        """Observe and learn from a batch of already-filtered, redacted texts."""
        if not safe_texts:
            return {"learned": False, "memory_count": self.adapter.memory_count}
        self.adapter.observe(
            [Message(role="user", content=text) for text in safe_texts]
        )
        return self.adapter.learn(safe_texts, importance=importance)

    def _async_process(self, safe_text: str, importance: float) -> dict[str, Any]:
        """Background processor for the async store queue.

        Runs the same quality gate and learning pipeline as the synchronous
        :meth:`store`, but without returning the result to the original caller.
        Contradiction hints are not surfaced because the caller has already
        received the pending response. The adapter lock is held while the
        adapter state is read or mutated.
        """
        with self._adapter_lock:
            try:
                dup = self._duplicate_check(safe_text)
                if dup is not None and not dup.ok:
                    logger.debug("Async store dropped duplicate: %s", safe_text[:80])
                    return {
                        "stored": False,
                        "reason": dup.reason,
                        "memory_count": self.adapter.memory_count,
                    }

                report = self._observe_and_learn([safe_text], importance=importance)
                self._save()
                return {
                    "stored": True,
                    "memory_count": self.adapter.memory_count,
                    "capacity_usage": report.get("capacity_usage", 0.0),
                }
            except Exception:
                logger.exception("Async store processor failed for text: %s", safe_text[:80])
                return {"stored": False, "reason": "processor_error", "memory_count": self.adapter.memory_count}

    def _contradiction_hints(self) -> list[dict[str, Any]]:
        """Return a list of contradiction hints for the current working memory."""
        if not self.config.quality_gate_enabled:
            return []
        all_texts = [m["text"] for m in self.adapter.memories_state()]
        return [
            {"index_a": i, "index_b": j, "hint": hint}
            for i, j, hint in self.quality_gate.find_contradictions(all_texts)
        ]

    def _migrate_legacy_session_checkpoint(self) -> None:
        """Move pre-v0.2 session checkpoints to the new location.

        Earlier versions saved the primary session checkpoint as
        ``checkpoints/mimir.pt`` and looked for it at the wrong path, so it
        was effectively never reloaded.  If such a file exists, rename it to
        ``checkpoints/session`` so the next ``_load()`` call picks it up.
        """
        legacy_path = self.checkpoints_dir / "mimir.pt"
        if legacy_path.exists() and not self.mimir_path.exists():
            try:
                os.replace(legacy_path, self.mimir_path)
                logger.info("Migrated legacy checkpoint from %s", legacy_path)
            except OSError:
                self._record_load_error(f"Failed to migrate legacy checkpoint from {legacy_path}")
                logger.exception("Failed to migrate legacy checkpoint")

    @_locked
    def _save(self) -> None:
        """Persist the current Mimir state and working memory.

        Before writing we reload ``memories.json`` from disk and merge it with
        the in-memory state. Agent CLI hooks run in separate processes and may
        have written new memories (e.g. from a Stop event) after the MCP server
        started, so a straight overwrite would discard those updates.
        """
        self.adapter.checkpoint(self.session_checkpoint_name)

        in_memory = self.adapter.memories_state()
        on_disk: list[dict[str, Any]] = []
        if self.memories_path.exists():
            try:
                on_disk = json.loads(self.memories_path.read_text(encoding="utf-8"))
                if not isinstance(on_disk, list):
                    on_disk = []
            except (json.JSONDecodeError, OSError):
                logger.exception("Failed to reload memories.json before save")

        merged = _merge_memories_state(on_disk, in_memory)
        self.adapter.load_memories_state(merged)
        _atomic_write(
            self.memories_path,
            json.dumps(merged, indent=2),
        )

    def close(self) -> None:
        """Persist session state on shutdown.

        If an async store queue is active, remaining items are flushed before
        the worker is stopped. Any item that cannot be processed within the
        configured timeout is logged as a warning.

        Raises:
            OSError: If the session state cannot be written to disk.  Callers
                should handle this to avoid silent data loss.
        """
        if self._store_queue is not None:
            timeout = self.config.async_store_flush_timeout
            flushed = self._store_queue.flush(timeout=timeout)
            if not flushed:
                logger.warning(
                    "Async store queue did not flush within %s seconds; "
                    "some pending memories may be lost",
                    timeout,
                )
            stopped = self._store_queue.stop(timeout=timeout)
            if not stopped:
                logger.warning("Async store worker did not stop within %s seconds", timeout)

        self._save()
        logger.info("Saved session state to %s", self.workspace_dir)

    @_locked
    def _sync_store_core(self, safe_text: str, importance: float) -> dict[str, Any]:
        """Run the synchronous duplicate check, learn, and persist pipeline."""
        dup = self._duplicate_check(safe_text)
        if dup is not None and not dup.ok:
            similar_memory = dup.similar_memory or ""
            if self.redactor.enabled:
                similar_memory = self.redactor.redact(similar_memory)
            return {
                "stored": False,
                "text": safe_text,
                "memory_count": self.adapter.memory_count,
                "reason": dup.reason,
                "similar_memory": similar_memory,
            }

        report = self._observe_and_learn([safe_text], importance=importance)
        contradiction_hints = self._contradiction_hints()

        self._save()
        response: dict[str, Any] = {
            "stored": True,
            "text": safe_text,
            "memory_count": self.adapter.memory_count,
            "capacity_usage": report.get("capacity_usage", 0.0),
        }
        if contradiction_hints:
            response["contradictions"] = contradiction_hints
        return response

    def _async_enqueue_store(self, safe_text: str, importance: float) -> dict[str, Any]:
        """Enqueue the text for the async store worker without holding the adapter lock."""
        assert self._store_queue is not None
        try:
            self._store_queue.put(safe_text, importance=importance)
        except queue.Full:
            return {
                "stored": False,
                "text": safe_text,
                "memory_count": self.adapter.memory_count,
                "reason": "queue_full",
            }
        return {
            "stored": "pending",
            "text": safe_text,
            "memory_count": self.adapter.memory_count,
            "pending_count": self._store_queue.pending_count,
        }

    def store(self, text: str, importance: float = 1.0) -> dict[str, Any]:
        """Store a text in memory and learn from it.

        Non-empty strings only; empty text is a no-op. After learning, the
        session checkpoint and memories sidecar are persisted so that agent CLI
        hooks can recall the new memory immediately.

        The response ``text`` field is the redacted form, so callers never
        receive secrets back.

        If ``MimirConfig.async_store_enabled`` is True, the expensive embedding
        and learning steps are deferred to a background worker. In that case the
        response contains ``{"stored": "pending"}`` until the worker eventually
        processes the item.

        Returns:
            A dictionary with at least these keys:

            - ``stored`` (bool or "pending"): whether the text was learned, or
              "pending" when async processing has been queued.
            - ``text`` (str): the redacted text that was (or would have been)
              stored.
            - ``memory_count`` (int): current number of memories after the call.
            - ``reason`` (str, optional): reason for rejection (filter/duplicate).
            - ``similar_memory`` (str, optional): the closest existing memory when
              a duplicate was blocked.
            - ``capacity_usage`` (float, optional): current prototype usage when
              storage succeeded.
            - ``contradictions`` (list, optional): contradiction hints produced
              after storage.
            - ``pending_count`` (int, optional): number of items queued when
              ``stored`` is "pending".
        """
        if not text or not isinstance(text, str):
            return {"stored": False, "text": text, "memory_count": self.adapter.memory_count}

        safe_text = self._prepare_text(text, source="mcp")
        if safe_text is None:
            filter_result = self.filter_engine.should_store(text, source="mcp")
            return {
                "stored": False,
                "text": self.redactor.redact(text),
                "memory_count": self.adapter.memory_count,
                "reason": filter_result.reason,
            }

        if not isinstance(importance, (int, float)) or importance < 0:
            return {
                "stored": False,
                "text": safe_text,
                "memory_count": self.adapter.memory_count,
                "reason": "invalid_importance",
            }

        if self._store_queue is not None:
            return self._async_enqueue_store(safe_text, importance=importance)

        return self._sync_store_core(safe_text, importance=importance)

    @_locked
    def recall(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> dict[str, Any]:
        """Return relevant memories for the query."""
        memories = self.adapter.recall(query, top_k=top_k, min_score=min_score)
        return {
            "query": query,
            "results": [
                {
                    "text": m.text,
                    "score": round(m.score, 6),
                    "created_at": m.created_at.isoformat(),
                }
                for m in memories
            ],
        }

    @_locked
    def consolidate(self) -> dict[str, Any]:
        """Reinforce all memories in the working buffer."""
        before = self.adapter.memory_count
        self.adapter.consolidate()
        return {
            "consolidated": True,
            "memories_reinforced": before,
            "memory_count": self.adapter.memory_count,
        }

    @_locked
    def list_memories(self) -> dict[str, Any]:
        """Return all working-memory texts as a numbered list."""
        memories = [
            {"index": i + 1, "text": item.get("text", "")}
            for i, item in enumerate(self.adapter.memories_state())
        ]
        return {
            "memory_count": len(memories),
            "memories": memories,
        }

    @_locked
    def replace_memories(self, memories: list[str]) -> dict[str, Any]:
        """Replace all working memories with a new list of texts.

        The adapter state is reset and the prototype matrix is retrained on the
        new memories so that recall remains consistent. The new state is
        persisted immediately so that agent CLI hooks running in separate
        processes do not reload the old memories from disk.
        """
        self.adapter.reset()
        for text in memories:
            if not text or not isinstance(text, str):
                continue
            self.adapter.observe([Message(role="user", content=text)])
            self.adapter.learn([text])

        self.adapter.checkpoint(self.session_checkpoint_name)
        _atomic_write(
            self.memories_path,
            json.dumps(self.adapter.memories_state(), indent=2),
        )
        return {
            "replaced": True,
            "memory_count": self.adapter.memory_count,
        }

    @_locked
    def forget(self) -> dict[str, Any]:
        """Reset the session state."""
        count = self.adapter.memory_count
        self.adapter.reset()
        return {
            "forgotten": True,
            "cleared_memories": count,
            "memory_count": 0,
        }

    def _checkpoint_paths(self, name: str) -> tuple[Path, Path]:
        """Return the Mimir checkpoint and memories sidecar paths for ``name``."""
        return (
            self.checkpoints_dir / f"{name}.pt",
            self.checkpoints_dir / f"{name}_memories.json",
        )

    @_locked
    def checkpoint(self, name: str) -> dict[str, Any]:
        """Save a named checkpoint including Mimir state and memories."""
        _validate_checkpoint_name(name)
        mimir_checkpoint, memories_checkpoint = self._checkpoint_paths(name)

        self.adapter.checkpoint(mimir_checkpoint.name)
        _atomic_write(
            memories_checkpoint,
            json.dumps(self.adapter.memories_state(), indent=2),
        )
        return {
            "checkpoint": name,
            "saved": True,
            "memory_count": self.adapter.memory_count,
        }

    @_locked
    def restore(self, name: str) -> dict[str, Any]:
        """Restore a named checkpoint."""
        _validate_checkpoint_name(name)
        mimir_checkpoint, memories_checkpoint = self._checkpoint_paths(name)

        if not mimir_checkpoint.exists():
            available = [p.stem for p in self.checkpoints_dir.glob("*.pt")]
            raise FileNotFoundError(f"Checkpoint '{name}' not found. Available: {available}")

        self.adapter.restore(mimir_checkpoint.name)

        if memories_checkpoint.exists():
            data = json.loads(memories_checkpoint.read_text(encoding="utf-8"))
            self.adapter.load_memories_state(data)
        else:
            self.adapter.clear_memories()

        return {
            "checkpoint": name,
            "restored": True,
            "memory_count": self.adapter.memory_count,
        }

    @_locked
    def status(self) -> dict[str, Any]:
        """Return session statistics.

        When async store is enabled, the response includes an ``async_store``
        dictionary with ``enabled`` (True), ``pending_count``, and ``worker_alive``.
        """
        available = sorted(p.stem for p in self.checkpoints_dir.glob("*.pt"))
        result: dict[str, Any] = {
            "workspace": str(self.workspace_path),
            "workspace_hash": self.workspace_hash,
            "backend": self.backend,
            "memory_count": self.adapter.memory_count,
            "prototype_capacity": self.adapter.prototype_capacity,
            "capacity_usage": self.adapter.capacity_usage,
            "step": self.adapter.step,
            "checkpoints": available,
            "load_errors": list(self._load_errors),
        }
        if self._store_queue is not None:
            result["async_store"] = {
                "enabled": True,
                "pending_count": self._store_queue.pending_count,
                "worker_alive": self._store_queue.is_alive,
            }
        return result
