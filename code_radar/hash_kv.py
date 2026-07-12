"""SQLite-backed KV store for filepath -> file_hash mapping.

Stored alongside ChromaDB so that deleting the ChromaDB folder also removes
this store, preventing state drift between hashes and vector data.
"""

import sqlite3
import threading
from pathlib import Path
from typing import Optional

from code_radar.logging import get_logger
from code_radar.workspace import build_scoped_db_filename, sanitize_workspace_id

log = get_logger(__name__)


class HashKVStore:
    """Thread-safe KV store for filepath -> file_hash, backed by SQLite.

    The database file lives inside the same directory as ChromaDB so both
    are naturally wiped together when the user does a full re-index.
    """

    def __init__(self, chroma_path: str, workspace_id: str = "default") -> None:
        self.workspace_id = sanitize_workspace_id(workspace_id)
        filename = build_scoped_db_filename("code_file_hashes", self.workspace_id)
        self.db_path = str(Path(chroma_path) / filename)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._ensure_db()

    # Internal helpers
    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()
        conn.execute(
            """CREATE TABLE IF NOT EXISTS file_hashes (
                filepath TEXT PRIMARY KEY,
                file_hash TEXT NOT NULL
            )"""
        )
        conn.commit()

    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.execute("PRAGMA journal_mode = WAL")
            self._conn.execute("PRAGMA synchronous = NORMAL")
        return self._conn

    # Public API
    def get_all(self) -> dict[str, str]:
        """Return all (filepath, file_hash) pairs as a dict."""
        with self._lock:
            conn = self._connect()
            cursor = conn.execute("SELECT filepath, file_hash FROM file_hashes")
            return {row[0]: row[1] for row in cursor}

    def put(self, filepath: str, file_hash: str) -> None:
        """Insert or replace a single file hash."""
        with self._lock:
            conn = self._connect()
            conn.execute(
                "INSERT OR REPLACE INTO file_hashes (filepath, file_hash) VALUES (?, ?)",
                (filepath, file_hash),
            )
            conn.commit()

    def put_batch(self, items: list[tuple[str, str]]) -> None:
        """Insert or replace multiple (filepath, file_hash) pairs."""
        if not items:
            return
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "INSERT OR REPLACE INTO file_hashes (filepath, file_hash) VALUES (?, ?)",
                items,
            )
            conn.commit()

    def delete(self, filepath: str) -> None:
        """Remove a single filepath from the store."""
        with self._lock:
            conn = self._connect()
            conn.execute("DELETE FROM file_hashes WHERE filepath = ?", (filepath,))
            conn.commit()

    def delete_batch(self, filepaths: list[str]) -> None:
        """Remove multiple filepaths from the store."""
        if not filepaths:
            return
        with self._lock:
            conn = self._connect()
            conn.executemany(
                "DELETE FROM file_hashes WHERE filepath = ?",
                [(fp,) for fp in filepaths],
            )
            conn.commit()

    def close(self) -> None:
        """Close the underlying SQLite connection."""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
