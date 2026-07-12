"""Global server state, configuration, and env helpers.

All modules in this package import from here to access shared singleton
objects (engine, store, reranker, hash cache, sync state, etc.).
"""

import atexit
import asyncio
import contextlib
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

from code_radar.chroma import ChromaStore
from code_radar.envvars import get_env
from code_radar.engine import EmbeddingEngine
from code_radar.logging import get_logger
from code_radar.reranker import Reranker
from code_radar.sparse_index import SparseChunkIndex
from code_radar.state import HashCache, SyncProgress
from code_radar.workspace import derive_workspace_id

if TYPE_CHECKING:
    from code_radar.watcher import FileWatcher


log = get_logger(__name__)

# Core engine / store / reranker
_engine: EmbeddingEngine | None = None
_store: ChromaStore | None = None
_root: str | None = None
_workspace_id: str = "default"
_workspace_source: str = "unknown"

_reranker: Reranker | None = None
_reranker_ready: bool = False
_reranker_disabled: bool = False
_rerank_executor: ThreadPoolExecutor | None = None
_rerank_call_lock = threading.Lock()

# Sparse/BM25 index (SQLite FTS5)
_sparse_index: SparseChunkIndex | None = None
_sparse_bootstrap_lock = threading.Lock()

_engine_ready: bool = False
_engine_load_failed: bool = False
_engine_load_lock = threading.Lock()

# Non-blocking sync state
_hash_cache: HashCache | None = None
_sync_progress: SyncProgress = SyncProgress()
_sync_task: asyncio.Task | None = None
_initial_sync_done: bool = False
_initial_sync_lock = threading.Lock()
_sync_guard: asyncio.Lock | None = None

# Event-driven watcher state
_file_watcher: "FileWatcher | None" = None

# Shutdown hooks state
_shutdown_hooks_registered: bool = False
_shutdown_lock = threading.Lock()
_shutdown_started: bool = False


# Env helpers
def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = get_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid int for %s=%r, using default=%d", name, raw, default)
        return default
    return max(minimum, value)


def env_bool(name: str, default: bool) -> bool:
    raw = get_env(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name: str, default: float, minimum: float = 0.0) -> float:
    raw = get_env(name)
    if raw is None:
        return max(minimum, default)
    try:
        value = float(raw)
    except ValueError:
        log.warning("Invalid float for %s=%r, using default=%.4f", name, raw, default)
        return max(minimum, default)
    return max(minimum, value)


def get_sync_guard() -> asyncio.Lock:
    """Shared async lock for all sync paths (manual + watcher)."""
    global _sync_guard
    if _sync_guard is None:
        _sync_guard = asyncio.Lock()
    return _sync_guard


async def shutdown_runtime(timeout_sec: float = 5.0) -> None:
    """Gracefully stop watcher/tasks and release server resources."""
    global _file_watcher, _sync_task, _hash_cache, _sparse_index, _rerank_executor

    log.info("Shutting down runtime ...")

    watcher = _file_watcher
    _file_watcher = None
    if watcher is not None:
        try:
            await asyncio.wait_for(watcher.stop(), timeout=timeout_sec)
        except TimeoutError:
            log.warning("Timed out stopping file watcher")
        except Exception as exc:
            log.warning("Failed stopping file watcher  |  error=%s", exc)

    task = _sync_task
    _sync_task = None
    if task is not None and not task.done():
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await asyncio.wait_for(task, timeout=timeout_sec)

    if _rerank_executor is not None:
        try:
            _rerank_executor.shutdown(wait=False, cancel_futures=True)
        except Exception as exc:
            log.warning("Failed shutting down rerank executor  |  error=%s", exc)
        _rerank_executor = None

    if _sparse_index is not None:
        try:
            _sparse_index.close()
        except Exception as exc:
            log.warning("Failed closing sparse index  |  error=%s", exc)
        _sparse_index = None

    if _hash_cache is not None:
        try:
            _hash_cache.close()
        except Exception as exc:
            log.warning("Failed closing hash cache  |  error=%s", exc)
        _hash_cache = None

    _sync_progress.status = "idle"


def shutdown_runtime_sync(timeout_sec: float = 5.0) -> None:
    """Sync wrapper for graceful shutdown, safe to call multiple times."""
    global _shutdown_started

    with _shutdown_lock:
        if _shutdown_started:
            return
        _shutdown_started = True

    try:
        asyncio.run(shutdown_runtime(timeout_sec=timeout_sec))
    except RuntimeError:
        # Already inside a running event loop: schedule cleanup best-effort.
        try:
            loop = asyncio.get_running_loop()
            loop.create_task(shutdown_runtime(timeout_sec=timeout_sec))
        except Exception as exc:
            log.warning("Failed scheduling async shutdown  |  error=%s", exc)
    except Exception as exc:
        log.warning("Shutdown error  |  error=%s", exc)


def register_shutdown_hooks() -> None:
    """Register process-exit cleanup hook once."""
    global _shutdown_hooks_registered

    with _shutdown_lock:
        if _shutdown_hooks_registered:
            return
        _shutdown_hooks_registered = True

    atexit.register(shutdown_runtime_sync)


# Configure
def configure(
    engine: EmbeddingEngine,
    store: ChromaStore,
    root: str,
    auto_sync: bool = True,
    workspace_id: str | None = None,
    workspace_source: str = "runtime",
):
    """Configure global state and optionally prepare for auto-sync."""
    global _engine, _store, _root, _workspace_id, _workspace_source
    global _reranker, _reranker_ready, _reranker_disabled
    global _sparse_index
    global _engine_ready, _engine_load_failed
    global _hash_cache, _sync_progress, _initial_sync_done, _sync_guard
    global _file_watcher, _shutdown_started

    _engine = engine
    _store = store
    _root = root
    _workspace_source = workspace_source
    explicit_workspace_id = workspace_id or get_env("CODE_WORKSPACE_ID")
    _workspace_id = derive_workspace_id(root, explicit_workspace_id)

    if hasattr(store, "set_workspace"):
        store.set_workspace(_workspace_id)

    _engine_ready = getattr(engine, "model", None) is not None
    _engine_load_failed = False

    # Reset reranker state on configure
    _reranker = None
    _reranker_ready = False
    _reranker_disabled = False

    # Initialize reranker from config (lazy — no model loading yet)
    from code_radar.config import get_reranker_enabled, resolve_reranker_model_id

    if not get_reranker_enabled():
        _reranker_disabled = True
        log.info("Reranker disabled via config")
    else:
        try:
            reranker_model = resolve_reranker_model_id()
            _reranker = Reranker(model_name=reranker_model)
            log.info("Reranker configured (lazy load)  |  model=%s", reranker_model)
        except Exception as exc:
            _reranker_disabled = True
            log.warning("Reranker init failed, disabled  |  error=%s", exc)

    # Local hash cache (SQLite-backed, replaces ChromaDB metadata scan)
    _hash_cache = HashCache(store.path, workspace_id=_workspace_id)
    _sync_progress = SyncProgress()
    _sync_guard = None
    _file_watcher = None
    _shutdown_started = False

    # Sparse lexical index (SQLite FTS5)
    try:
        _sparse_index = SparseChunkIndex(store.path, workspace_id=_workspace_id)
        log.info("Sparse index ready  |  db=%s", _sparse_index.db_path)
    except Exception as exc:
        _sparse_index = None
        log.warning("Sparse index init failed, BM25 disabled  |  error=%s", exc)

    log.info(
        "MCP server configured  |  root=%s  |  source=%s  |  workspace_id=%s",
        root,
        _workspace_source,
        _workspace_id,
    )

    # Auto-sync
    if auto_sync and env_bool("CODE_AUTO_SYNC", True):
        log.info("Auto-sync will trigger on first tool call")
        with _initial_sync_lock:
            _initial_sync_done = False
    else:
        with _initial_sync_lock:
            _initial_sync_done = True


# Lazy-load helpers
def ensure_engine_ready() -> tuple[bool, str | None]:
    """Lazy-load embedding engine if not already loaded."""
    global _engine_ready, _engine_load_failed, _engine

    if _engine_ready:
        return True, None
    if _engine_load_failed:
        return False, "previous load attempt failed"
    if _engine is None:
        return False, "engine not configured"

    with _engine_load_lock:
        if _engine_ready:
            return True, None
        if _engine_load_failed:
            return False, "previous load attempt failed"

        try:
            log.info("Lazy-loading embedding engine ...")
            assert _engine is not None
            _engine.load()
            _engine_ready = True
            log.info("Engine loaded  |  model=%s", _engine.model_name)
            return True, None
        except Exception as exc:
            _engine_load_failed = True
            log.error("Engine load failed  |  error=%s", exc)
            return False, str(exc)


def ensure_reranker_ready() -> bool:
    """Lazy-load reranker model if configured."""
    global _reranker_ready, _reranker, _reranker_disabled

    if _reranker_ready:
        return True
    if _reranker_disabled:
        return False
    if _reranker is None:
        return False

    try:
        log.info("Lazy-loading reranker ...")
        _reranker.load()
        _reranker_ready = True
        log.info("Reranker loaded  |  model=%s", _reranker.model_name)
        return True
    except Exception as exc:
        _reranker_disabled = True
        log.warning("Reranker load failed, disabled  |  error=%s", exc)
        return False
