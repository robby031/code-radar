"""Background sync runner + auto-trigger logic.

Depends on ``state`` for engine/store/hash-cache references and on
``codebase`` for the actual sync logic.
"""

import asyncio
import time

from code_radar.codebase import async_sync_workspace
from code_radar.logging import get_logger

from . import state

log = get_logger(__name__)


async def _ensure_file_watcher_started() -> None:
    """Start debounced event-driven watcher once (if enabled)."""
    if not state.env_bool("CODE_EVENT_DRIVEN_SYNC", True):
        return

    watcher = state._file_watcher
    if watcher is not None and watcher.is_running:
        return

    engine = state._engine
    store = state._store
    hash_cache = state._hash_cache
    root = state._root or "."

    if engine is None or store is None or hash_cache is None:
        return

    try:
        from code_radar.watcher import FileWatcher

        new_watcher = FileWatcher(
            root_path=root,
            engine=engine,
            store=store,
            hash_cache=hash_cache,
            progress=state._sync_progress,
            debounce_seconds=state.env_float("CODE_SYNC_DEBOUNCE_SEC", 2.5, minimum=0.1),
            micro_batch_size=state.env_int("CODE_SYNC_MICRO_BATCH_SIZE", 10),
            yield_seconds=state.env_float("CODE_SYNC_YIELD_SEC", 0.05, minimum=0.0),
            queue_maxsize=state.env_int("CODE_SYNC_EVENT_QUEUE_SIZE", 10000),
            throttle_seconds=state.env_float("CODE_SYNC_THROTTLE_SEC", 0.05, minimum=0.0),
            sync_lock=state.get_sync_guard(),
        )
        started = await new_watcher.start()
        if started:
            state._file_watcher = new_watcher
    except Exception as exc:
        log.warning("Failed starting file watcher  |  error=%s", exc)


async def run_sync_background(directory: str) -> None:
    """Run ``async_sync_workspace`` as a background ``asyncio.Task``.

    Updates ``state._sync_progress`` throughout.  Errors are captured
    rather than propagated so the server stays up.
    """
    engine = state._engine
    store = state._store
    hash_cache = state._hash_cache
    if engine is None or store is None or hash_cache is None:
        state._sync_progress.status = "error"
        state._sync_progress.error = "engine/store not configured"
        return

    # Ensure engine is loaded
    ok, err = state.ensure_engine_ready()
    if not ok:
        state._sync_progress.status = "error"
        state._sync_progress.error = err or "engine load failed"
        return

    throttle = state.env_float("CODE_SYNC_THROTTLE_SEC", 0.05)
    default_batch = getattr(engine, "default_batch_size", 8)

    progress = state._sync_progress
    progress.status = "scanning"
    progress.directory = directory
    progress.total_files = 0
    progress.scanned_files = 0
    progress.changed_files = 0
    progress.added_chunks = 0
    progress.deleted_files = 0
    progress.error = ""
    # FIX: status endpoints should have a live clock even while the task waits
    # for the shared sync lock or model readiness work.
    progress.start_time = time.perf_counter()
    progress.elapsed = 0.0

    sync_guard = state.get_sync_guard()

    try:
        async with sync_guard:
            await async_sync_workspace(
                directory=directory,
                engine=engine,
                store=store,
                hash_cache=hash_cache,
                progress=progress,
                batch_size=default_batch,
                throttle_sec=throttle,
            )
    except asyncio.CancelledError:
        state._sync_progress.status = "idle"
        log.info("Sync cancelled")
    except Exception as exc:
        state._sync_progress.status = "error"
        state._sync_progress.error = str(exc)
        log.error("Sync failed  |  error=%s", exc)


async def trigger_auto_sync() -> None:
    """Start initial full sync once, then keep event-driven watcher running."""
    if state._engine is None or state._store is None:
        return

    # Start watcher early so subsequent file changes are captured continuously.
    await _ensure_file_watcher_started()

    with state._initial_sync_lock:
        if state._initial_sync_done:
            return
        state._initial_sync_done = True

    root = state._root or "."
    log.info("Initial auto-sync triggered  |  root=%s", root)

    state._sync_task = asyncio.create_task(run_sync_background(root))
