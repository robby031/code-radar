from __future__ import annotations

import asyncio
import contextlib
import os
import time
from pathlib import Path
from typing import Any, Awaitable, Callable, Literal, cast

from code_radar.codebase import async_sync_files, async_sync_workspace
from code_radar.constants import SKIP_DIRS
from code_radar.logging import get_logger
from code_radar.state import HashCache, SyncProgress

try:
    from watchdog.events import FileSystemEventHandler  # type: ignore[import-not-found]
    from watchdog.observers import Observer  # type: ignore[import-not-found]

    WATCHDOG_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    class FileSystemEventHandler:  # type: ignore[no-redef]
        pass

    Observer = None  # type: ignore[assignment]
    WATCHDOG_AVAILABLE = False


log = get_logger(__name__)

EventType = Literal["created", "modified", "deleted"]


class CodebaseFileHandler(FileSystemEventHandler):  # type: ignore[misc]
    """Collect filesystem events from watchdog thread to asyncio queue safely."""

    def __init__(
        self,
        event_queue: asyncio.Queue[dict[str, str | float]],
        loop: asyncio.AbstractEventLoop,
        root_path: Path,
        mark_overflow: Callable[[], None],
    ):
        self.event_queue = event_queue
        self.loop = loop
        self.root_path = root_path
        self.mark_overflow = mark_overflow
        self.skip_dirs = SKIP_DIRS

    def _should_ignore(self, rel_path: Path) -> bool:
        parts = set(rel_path.parts)
        return bool(parts & self.skip_dirs)

    def _enqueue_from_loop(self, payload: dict[str, str | float]) -> None:
        try:
            self.event_queue.put_nowait(payload)
        except asyncio.QueueFull:
            self.mark_overflow()

    def _schedule_path_event(self, event_type: EventType, raw_path: str) -> None:
        try:
            absolute = Path(raw_path).resolve(strict=False)
            rel = absolute.relative_to(self.root_path)
        except Exception:
            return

        if self._should_ignore(rel):
            return

        payload = {
            "type": event_type,
            "path": rel.as_posix(),
            "timestamp": time.time(),
        }

        # callback runs in watchdog thread; queue mutation must happen in loop thread.
        try:
            self.loop.call_soon_threadsafe(self._enqueue_from_loop, payload)
        except RuntimeError:
            # Event loop already closed during shutdown.
            return

    def _schedule_event(self, event: object) -> None:
        if getattr(event, "is_directory", False):
            return

        raw_event_type = str(getattr(event, "event_type", "modified"))
        if raw_event_type == "created":
            event_type: EventType = "created"
        elif raw_event_type == "deleted":
            event_type = "deleted"
        else:
            event_type = "modified"

        src_path = getattr(event, "src_path", "")
        if not src_path:
            return

        self._schedule_path_event(event_type, src_path)

    def on_modified(self, event: object) -> None:
        self._schedule_event(event)

    def on_created(self, event: object) -> None:
        self._schedule_event(event)

    def on_deleted(self, event: object) -> None:
        self._schedule_event(event)

    def on_moved(self, event: object) -> None:
        src_path = getattr(event, "src_path", "")
        dest_path = getattr(event, "dest_path", "")
        if src_path:
            self._schedule_path_event("deleted", src_path)
        if dest_path:
            self._schedule_path_event("created", dest_path)


class FileWatcher:
    """Debounced event-driven auto-sync watcher.

    Protection layers:
    1) Event collection: watchdog thread -> bounded asyncio queue (thread-safe)
    2) Debounce + coalesce: delay, deduplicate by filepath, keep latest event
    3) Throttled worker: micro-batches + cooperative yields + cancellation
    4) State management: HashCache (SQLite KV) for real delta checks
    """

    def __init__(
        self,
        root_path: str,
        engine,
        store,
        hash_cache: HashCache,
        progress: SyncProgress | None = None,
        debounce_seconds: float = 2.5,
        micro_batch_size: int = 10,
        yield_seconds: float = 0.05,
        queue_maxsize: int = 10000,
        throttle_seconds: float = 0.05,
        sync_lock: asyncio.Lock | None = None,
    ):
        self.root_path = Path(root_path).resolve()
        self.engine = engine
        self.store = store
        self.hash_cache = hash_cache
        self.progress = progress

        self.debounce_seconds = max(0.1, debounce_seconds)
        self.micro_batch_size = max(1, micro_batch_size)
        self.yield_seconds = max(0.0, yield_seconds)
        self.queue_maxsize = max(100, queue_maxsize)
        self.throttle_seconds = max(0.0, throttle_seconds)
        self.sync_lock = sync_lock

        self.event_queue: asyncio.Queue[dict[str, str | float]] = asyncio.Queue(
            maxsize=self.queue_maxsize
        )
        self.pending_events: dict[str, EventType] = {}

        self.observer: object | None = None
        self.event_task: asyncio.Task | None = None
        self.debounce_task: asyncio.Task | None = None
        self.sync_task: asyncio.Task | None = None
        self.is_running = False

        self._queue_overflowed = False
        self._priority_adjusted = False

        # Metrics
        self.events_received = 0
        self.events_coalesced = 0
        self.events_dropped = 0
        self.sync_runs = 0
        self.last_sync_time = 0.0
        self.last_sync_error = ""

    def _mark_overflow(self) -> None:
        self._queue_overflowed = True
        self.events_dropped += 1

    async def start(self) -> bool:
        """Start watcher. Returns False if watchdog dependency is unavailable."""
        if self.is_running:
            return True

        if not WATCHDOG_AVAILABLE or Observer is None:
            log.warning(
                "watchdog is not installed; event-driven auto-sync disabled. "
                "Install dependency: pip install watchdog"
            )
            return False

        loop = asyncio.get_running_loop()
        handler = CodebaseFileHandler(
            event_queue=self.event_queue,
            loop=loop,
            root_path=self.root_path,
            mark_overflow=self._mark_overflow,
        )

        observer = Observer()  # type: ignore[operator]
        observer.schedule(cast(Any, handler), str(self.root_path), recursive=True)
        observer.start()

        self.observer = observer
        self.is_running = True
        self.event_task = asyncio.create_task(self._event_processor())

        log.info(
            "File watcher started  |  root=%s  |  debounce=%.2fs  |  micro_batch=%d",
            self.root_path,
            self.debounce_seconds,
            self.micro_batch_size,
        )
        return True

    async def stop(self) -> None:
        """Stop watcher gracefully."""
        self.is_running = False

        if self.observer is not None:
            stop_fn = getattr(self.observer, "stop", None)
            if callable(stop_fn):
                stop_fn()
            join_fn = getattr(self.observer, "join", None)
            if callable(join_fn):
                join_fn(timeout=2.0)
            self.observer = None

        for task in (self.event_task, self.debounce_task, self.sync_task):
            if task is not None and not task.done():
                task.cancel()

        for task in (self.event_task, self.debounce_task, self.sync_task):
            if task is not None:
                with contextlib.suppress(asyncio.CancelledError):
                    await task

        self.event_task = None
        self.debounce_task = None
        self.sync_task = None

        log.info("File watcher stopped  |  root=%s", self.root_path)

    async def _event_processor(self) -> None:
        while self.is_running:
            try:
                event = await self.event_queue.get()
            except asyncio.CancelledError:
                break
            except Exception as exc:
                log.warning("Event processor failure  |  error=%s", exc)
                await asyncio.sleep(0.05)
                continue

            raw_event_type = str(event.get("type", "modified"))
            rel_path = str(event.get("path", "")).strip()
            if not rel_path:
                continue

            if raw_event_type == "created":
                event_type: EventType = "created"
            elif raw_event_type == "deleted":
                event_type = "deleted"
            else:
                event_type = "modified"

            self.events_received += 1
            self.pending_events[rel_path] = event_type  # latest event wins per path
            self.events_coalesced = len(self.pending_events)

            if self.debounce_task is not None and not self.debounce_task.done():
                self.debounce_task.cancel()

            self.debounce_task = asyncio.create_task(self._debounced_flush())

    async def _debounced_flush(self) -> None:
        try:
            await asyncio.sleep(self.debounce_seconds)

            if not self.pending_events and not self._queue_overflowed:
                return

            snapshot = dict(self.pending_events)
            self.pending_events.clear()

            overflowed = self._queue_overflowed
            self._queue_overflowed = False

            # Cancellable worker: if sync lama masih jalan, cancel dan pakai state terbaru.
            if self.sync_task is not None and not self.sync_task.done():
                self.sync_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await self.sync_task

            self.sync_task = asyncio.create_task(
                self._sync_snapshot(snapshot, overflowed=overflowed)
            )

        except asyncio.CancelledError:
            pass

    async def _sync_snapshot(self, events: dict[str, EventType], overflowed: bool) -> None:
        try:
            self._lower_priority_once()
            self.sync_runs += 1

            if overflowed:
                log.warning(
                    "Event queue overflow detected; running full async sync as recovery"
                )
                await self._run_with_lock(self._full_resync)
                self.last_sync_time = time.time()
                self.last_sync_error = ""
                return

            deleted_paths = sorted(
                path for path, event_type in events.items() if event_type == "deleted"
            )
            touched_paths = sorted(
                path for path, event_type in events.items() if event_type in {"created", "modified"}
            )

            if self.progress is not None:
                self.progress.status = "processing"
                self.progress.directory = str(self.root_path)
                self.progress.start_time = time.perf_counter()
                self.progress.changed_files = len(touched_paths)
                self.progress.deleted_files = len(deleted_paths)

            # Process deletes first in micro-batches
            for i in range(0, len(deleted_paths), self.micro_batch_size):
                batch = deleted_paths[i : i + self.micro_batch_size]
                await self._run_with_lock(
                    lambda b=batch: async_sync_files(
                        directory=str(self.root_path),
                        engine=self.engine,
                        store=self.store,
                        hash_cache=self.hash_cache,
                        filepaths=[],
                        deleted_paths=b,
                        batch_size=getattr(self.engine, "default_batch_size", 8),
                        throttle_sec=0.0,
                        progress=None,
                    )
                )
                if self.yield_seconds > 0:
                    await asyncio.sleep(self.yield_seconds)

            # Process changed/created files in micro-batches
            for i in range(0, len(touched_paths), self.micro_batch_size):
                batch = touched_paths[i : i + self.micro_batch_size]
                await self._run_with_lock(
                    lambda b=batch: async_sync_files(
                        directory=str(self.root_path),
                        engine=self.engine,
                        store=self.store,
                        hash_cache=self.hash_cache,
                        filepaths=b,
                        deleted_paths=[],
                        batch_size=getattr(self.engine, "default_batch_size", 8),
                        throttle_sec=self.throttle_seconds,
                        progress=None,
                    )
                )
                if self.yield_seconds > 0:
                    await asyncio.sleep(self.yield_seconds)

            if self.progress is not None:
                self.progress.status = "done"
                self.progress.elapsed = time.perf_counter() - self.progress.start_time

            self.last_sync_time = time.time()
            self.last_sync_error = ""

        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.last_sync_error = str(exc)
            if self.progress is not None:
                self.progress.status = "error"
                self.progress.error = str(exc)
            log.error("Watcher sync failed  |  error=%s", exc)

    async def _run_with_lock(self, coro_factory: Callable[[], Awaitable]):
        if self.sync_lock is None:
            return await coro_factory()
        async with self.sync_lock:
            return await coro_factory()

    async def _full_resync(self):
        return await async_sync_workspace(
            directory=str(self.root_path),
            engine=self.engine,
            store=self.store,
            hash_cache=self.hash_cache,
            progress=self.progress,
            batch_size=getattr(self.engine, "default_batch_size", 8),
            throttle_sec=self.throttle_seconds,
        )

    def _lower_priority_once(self) -> None:
        if self._priority_adjusted:
            return

        if os.name == "posix":
            try:
                os.nice(10)
                self._priority_adjusted = True
                log.info("Watcher priority lowered with os.nice(10)")
            except OSError:
                pass

    def get_status(self) -> dict:
        return {
            "is_running": self.is_running,
            "events_received": self.events_received,
            "events_coalesced": self.events_coalesced,
            "events_dropped": self.events_dropped,
            "sync_runs": self.sync_runs,
            "queue_size": self.event_queue.qsize(),
            "last_sync_time": self.last_sync_time,
            "last_sync_error": self.last_sync_error,
            "sync_in_progress": self.sync_task is not None and not self.sync_task.done(),
        }
