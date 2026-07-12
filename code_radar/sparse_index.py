"""SQLite FTS5 sparse index for chunk-level lexical search.

This index is stored next to ChromaDB data and keyed by ``chunk_id`` so it
stays aligned with vector chunks. It is intentionally lightweight and only
depends on stdlib ``sqlite3``.
"""

import re
import sqlite3
import threading
from pathlib import Path
from typing import Any, Sequence

from code_radar.logging import get_logger
from code_radar.workspace import build_scoped_db_filename, sanitize_workspace_id

log = get_logger(__name__)

_CODE_QUERY_SYNONYMS: dict[str, tuple[str, ...]] = {
    "function": ("func", "def", "method"),
    "functions": ("func", "def", "method"),
    "method": ("func", "def", "function"),
    "methods": ("func", "def", "function"),
}

_CODE_INTENT_TOKENS: frozenset[str] = frozenset(
    {
        "class",
        "classes",
        "def",
        "func",
        "function",
        "functions",
        "method",
        "methods",
        "type",
        "types",
    }
)


class SparseChunkIndex:
    """Thread-safe chunk index backed by SQLite FTS5 (BM25)."""

    def __init__(self, chroma_path: str, workspace_id: str = "default") -> None:
        self.workspace_id = sanitize_workspace_id(workspace_id)
        filename = build_scoped_db_filename("code_sparse_index", self.workspace_id)
        self.db_path = str(Path(chroma_path) / filename)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        self._ensure_db()

    # Internal helpers
    def _connect(self) -> sqlite3.Connection:
        if self._conn is None:
            conn = sqlite3.connect(self.db_path, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("PRAGMA synchronous = NORMAL")
            conn.execute("PRAGMA temp_store = MEMORY")
            self._conn = conn
        return self._conn

    def _ensure_db(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = self._connect()

        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    filepath TEXT NOT NULL,
                    chunk_type TEXT NOT NULL,
                    start_line INTEGER NOT NULL,
                    end_line INTEGER NOT NULL,
                    language TEXT,
                    content TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_sparse_chunks_filepath ON chunks(filepath)"
            )

            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts
                USING fts5(
                    content,
                    content='chunks',
                    content_rowid='rowid',
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )

            # Keep FTS in sync with the content table.
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ai AFTER INSERT ON chunks BEGIN
                    INSERT INTO chunks_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_ad AFTER DELETE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END
                """
            )
            conn.execute(
                """
                CREATE TRIGGER IF NOT EXISTS chunks_au AFTER UPDATE ON chunks BEGIN
                    INSERT INTO chunks_fts(chunks_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                    INSERT INTO chunks_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END
                """
            )
            conn.commit()
        except sqlite3.OperationalError as exc:
            raise RuntimeError(
                f"Failed to initialize SQLite FTS5 index at {self.db_path}: {exc}"
            ) from exc

    @staticmethod
    def _tokenize_query(query: str) -> list[str]:
        """Tokenize query into FTS-safe tokens (with camelCase splitting)."""
        if not query:
            return []

        base = re.findall(r"[A-Za-z_][A-Za-z0-9_]*|\d+", query)
        out: list[str] = []
        seen: set[str] = set()

        def add_token(value: str) -> None:
            norm = value.lower()
            if len(norm) <= 1 or norm in seen:
                return
            seen.add(norm)
            out.append(norm)

        for tok in base:
            norm = tok.lower()
            add_token(norm)

            for synonym in _CODE_QUERY_SYNONYMS.get(norm, ()):
                add_token(synonym)

            if len(norm) > 3 and norm.endswith("s"):
                add_token(norm[:-1])
            elif len(norm) > 2 and norm not in _CODE_INTENT_TOKENS:
                add_token(f"{norm}s")

            parts = re.findall(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+", tok)
            for p in parts:
                p_norm = p.lower()
                add_token(p_norm)

        return out[:24]

    @staticmethod
    def _path_recall_terms(tokens: list[str]) -> list[str]:
        """Terms worth matching against file paths and raw content."""
        return [t for t in tokens if t not in _CODE_INTENT_TOKENS][:8]

    @staticmethod
    def _filepath_predicate(filepath: str | None) -> tuple[str, list[str]]:
        if not filepath:
            return "", []

        clean = filepath.replace("\\", "/").strip("/")
        if not clean:
            return "", []

        # Filename-only filter: match the exact basename anywhere in the index.
        if "/" not in clean:
            return "(c.filepath = ? OR c.filepath LIKE ?)", [clean, f"%/{clean}"]

        # Directory filter: match children under the directory, including when
        # a user-provided workspace prefix was stripped earlier by the resolver.
        if Path(clean).suffix == "":
            child = clean.rstrip("/") + "/%"
            suffix_child = "%/" + child
            return (
                "(c.filepath LIKE ? OR c.filepath LIKE ?)",
                [child, suffix_child],
            )

        # Full file path filter must stay strict. Do not fall back to basename,
        # otherwise files with the same name under sibling providers leak in.
        return "(c.filepath = ? OR c.filepath LIKE ?)", [clean, f"%/{clean}"]

    @classmethod
    def _build_match_expr(cls, query: str) -> str:
        tokens = cls._tokenize_query(query)
        if not tokens:
            return ""
        # OR for wide recall in phase-1; BM25 handles lexical ranking.
        return " OR ".join(f'"{t}"' for t in tokens)

    # Public API
    def count_chunks(self) -> int:
        with self._lock:
            conn = self._connect()
            row = conn.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()
            return int(row["c"]) if row is not None else 0

    def upsert_chunks(self, chunks: Sequence[dict[str, Any]]) -> None:
        if not chunks:
            return

        rows: list[tuple[Any, ...]] = []
        for c in chunks:
            meta = c.get("metadata") or {}
            rows.append(
                (
                    c.get("id", ""),
                    c.get("filepath", ""),
                    c.get("chunk_type", "code_block"),
                    int(c.get("start_line", 1)),
                    int(c.get("end_line", 1)),
                    str(meta.get("language", "")),
                    c.get("content", "") or "",
                )
            )

        with self._lock:
            conn = self._connect()
            conn.executemany(
                """
                INSERT INTO chunks (
                    chunk_id, filepath, chunk_type, start_line, end_line, language, content
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    filepath=excluded.filepath,
                    chunk_type=excluded.chunk_type,
                    start_line=excluded.start_line,
                    end_line=excluded.end_line,
                    language=excluded.language,
                    content=excluded.content
                """,
                rows,
            )
            conn.commit()

    def delete_by_files(self, filepaths: Sequence[str]) -> None:
        paths = [p for p in filepaths if p]
        if not paths:
            return

        with self._lock:
            conn = self._connect()
            batch_size = 500
            for i in range(0, len(paths), batch_size):
                batch = paths[i : i + batch_size]
                placeholders = ",".join("?" for _ in batch)
                conn.execute(f"DELETE FROM chunks WHERE filepath IN ({placeholders})", batch)
            conn.commit()

    def search(
        self,
        query: str,
        n: int = 30,
        filepath: str | None = None,
    ) -> list[dict[str, Any]]:
        tokens = self._tokenize_query(query)
        match_expr = " OR ".join(f'"{t}"' for t in tokens)
        if not tokens:
            return []

        limit = max(1, int(n))
        path_terms = self._path_recall_terms(tokens)
        filepath_filter = filepath.replace("\\", "/").strip("/") if filepath else None
        filepath_sql, filepath_params = self._filepath_predicate(filepath_filter)
        rows: list[sqlite3.Row] = []

        with self._lock:
            conn = self._connect()

            if match_expr and filepath_filter:
                rows.extend(
                    conn.execute(
                        f"""
                        SELECT
                            c.chunk_id,
                            c.filepath,
                            c.chunk_type,
                            c.start_line,
                            c.end_line,
                            c.language,
                            c.content,
                            bm25(chunks_fts) AS bm25_score
                        FROM chunks_fts
                        JOIN chunks c ON c.rowid = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                          AND {filepath_sql}
                        ORDER BY bm25_score ASC
                        LIMIT ?
                        """,
                        [match_expr] + filepath_params + [limit],
                    ).fetchall()
                )
            elif match_expr:
                rows.extend(
                    conn.execute(
                        """
                        SELECT
                            c.chunk_id,
                            c.filepath,
                            c.chunk_type,
                            c.start_line,
                            c.end_line,
                            c.language,
                            c.content,
                            bm25(chunks_fts) AS bm25_score
                        FROM chunks_fts
                        JOIN chunks c ON c.rowid = chunks_fts.rowid
                        WHERE chunks_fts MATCH ?
                        ORDER BY bm25_score ASC
                        LIMIT ?
                        """,
                        (match_expr, limit),
                    ).fetchall()
                )

            if path_terms:
                score_parts: list[str] = []
                where_parts: list[str] = []
                params: list[Any] = []
                where_params: list[Any] = []

                for term in path_terms:
                    pattern = f"%{term}%"
                    score_parts.append(
                        (
                            "CASE WHEN lower(c.filepath) LIKE ? THEN 3 ELSE 0 END + "
                            "CASE WHEN lower(c.content) LIKE ? THEN 1 ELSE 0 END"
                        )
                    )
                    params.extend([pattern, pattern])
                    where_parts.append(
                        "(lower(c.filepath) LIKE ? OR lower(c.content) LIKE ?)"
                    )
                    where_params.extend([pattern, pattern])

                lexical_score = " + ".join(score_parts)
                where_sql = " OR ".join(where_parts)
                lexical_filepath_sql = ""
                if filepath_filter:
                    lexical_filepath_sql = f" AND {filepath_sql}"
                    where_params.extend(filepath_params)

                rows.extend(
                    conn.execute(
                        f"""
                        SELECT
                            c.chunk_id,
                            c.filepath,
                            c.chunk_type,
                            c.start_line,
                            c.end_line,
                            c.language,
                            c.content,
                            (-1000.0 - ({lexical_score})) AS bm25_score
                        FROM chunks c
                        WHERE ({where_sql}){lexical_filepath_sql}
                        ORDER BY ({lexical_score}) DESC, length(c.content) ASC
                        LIMIT ?
                        """,
                        params + where_params + params + [limit],
                    ).fetchall()
                )

        merged: dict[str, dict[str, Any]] = {}
        for row in rows:
            chunk_id = str(row["chunk_id"])
            bm25_score = float(row["bm25_score"])
            existing = merged.get(chunk_id)
            if existing is not None:
                if bm25_score < float(existing.get("bm25_score", 0.0)):
                    existing["bm25_score"] = bm25_score
                continue

            merged[chunk_id] = {
                "id": chunk_id,
                "document": str(row["content"]),
                "bm25_score": bm25_score,
                "metadata": {
                    "filepath": str(row["filepath"]),
                    "chunk_type": str(row["chunk_type"]),
                    "start_line": int(row["start_line"]),
                    "end_line": int(row["end_line"]),
                    "language": str(row["language"] or ""),
                },
            }

        out = list(merged.values())
        out.sort(key=lambda item: float(item.get("bm25_score", 0.0)))
        return out[:limit]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
