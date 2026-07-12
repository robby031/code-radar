"""Local hash cache + sync state management.

Replaces heavy ChromaDB metadata queries with a local KV cache
(SQLite + in-memory).  Provides sync progress tracking for
non-blocking operation.
"""

import threading
import time
from dataclasses import dataclass

from code_radar.hash_kv import HashKVStore
from code_radar.logging import get_logger

log = get_logger(__name__)


# SyncProgress - immutable-ish snapshot of a running / finished sync
@dataclass
class SyncProgress:
    """Progress snapshot for a running or completed sync operation."""

    status: str = "idle"  # idle / scanning / deleting / processing / done / error
    directory: str = ""
    total_files: int = 0
    scanned_files: int = 0
    changed_files: int = 0
    added_chunks: int = 0
    deleted_files: int = 0
    error: str = ""
    start_time: float = 0.0
    elapsed: float = 0.0

    # helpers
    def is_running(self) -> bool:
        return self.status in ("scanning", "deleting", "processing")

    def is_done(self) -> bool:
        return self.status in ("done", "error")

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict for MCP tool responses."""
        # FIX: while a sync is running, expose live elapsed time instead of the
        # last completed value. This prevents a stuck-looking 0.0 clock.
        elapsed = self.elapsed
        if self.is_running() and self.start_time > 0:
            elapsed = time.perf_counter() - self.start_time

        return {
            "status": self.status,
            "directory": self.directory,
            "total_files": self.total_files,
            "scanned_files": self.scanned_files,
            "changed_files": self.changed_files,
            "added_chunks": self.added_chunks,
            "deleted_files": self.deleted_files,
            "error": self.error,
            "elapsed_seconds": round(elapsed, 2),
            "is_running": self.is_running(),
            "is_done": self.is_done(),
        }


# HashCache – in-memory cache backed by SQLite KV store
class HashCache:
    """In-memory cache of file -> hash mappings, backed by SQLite.

    Populated lazily on first access.  Writes go through to both SQLite
    and the in-memory dict so subsequent reads are instantaneous.

    This replaces the previous approach of scanning all ChromaDB
    metadata just to learn which files have changed.
    """

    def __init__(self, chroma_path: str, workspace_id: str = "default") -> None:
        self.workspace_id = workspace_id
        self._kv = HashKVStore(chroma_path, workspace_id=workspace_id)
        self._cache: dict[str, str] | None = None
        self._lock = threading.Lock()

    # public API
    def get_all(self) -> dict[str, str]:
        """Return cached mapping, populating from SQLite if needed."""
        with self._lock:
            if self._cache is None:
                t0 = time.perf_counter()
                self._cache = self._kv.get_all()
                elapsed = time.perf_counter() - t0
                log.debug(
                    "HashCache populated  |  entries=%d  |  elapsed=%.2fs",
                    len(self._cache),
                    elapsed,
                )
            return dict(self._cache)

    def put_batch(self, items: list[tuple[str, str]]) -> None:
        """Store hashes in SQLite + in-memory cache."""
        if not items:
            return
        self._kv.put_batch(items)
        with self._lock:
            if self._cache is not None:
                for fp, fh in items:
                    self._cache[fp] = fh

    def delete_batch(self, filepaths: list[str]) -> None:
        """Remove entries from SQLite + in-memory cache."""
        if not filepaths:
            return
        self._kv.delete_batch(filepaths)
        with self._lock:
            if self._cache is not None:
                for fp in filepaths:
                    self._cache.pop(fp, None)

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        self._kv.close()
