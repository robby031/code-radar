import threading
import time
from typing import Any, Sequence, cast

import chromadb
from chromadb.config import Settings

from code_radar.envvars import get_env
from code_radar.logging import get_logger
from code_radar.workspace import build_collection_name, sanitize_workspace_id

log = get_logger(__name__)


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = get_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid int for %s=%r, using default=%d", name, raw, default)
        return default
    return max(minimum, value)


class ChromaStore:
    def __init__(
        self,
        path: str = "./chroma_data",
        *,
        lazy_connect: bool = False,
        workspace_id: str = "default",
        collection_base: str = "code_radars",
    ):
        self.path = path
        self.client: Any | None = None
        self.collection: Any | None = None
        self._ready = False
        self._connect_lock = threading.Lock()

        self.collection_base = sanitize_workspace_id(collection_base, default="code_radars")
        self.workspace_id = sanitize_workspace_id(workspace_id)
        self.collection_name = build_collection_name(self.collection_base, self.workspace_id)

        self.upsert_batch_size = _env_int("CODE_CHROMA_UPSERT_BATCH_SIZE", 500)

        if lazy_connect:
            log.info(
                "ChromaDB lazy connect enabled  |  path=%s  |  collection=%s",
                path,
                self.collection_name,
            )
        else:
            self._ensure_ready()

    def _ensure_ready(self) -> None:
        if self._ready:
            return

        with self._connect_lock:
            if self._ready:
                return

            log.info("Connecting to ChromaDB  |  path=%s", self.path)
            client = chromadb.PersistentClient(
                path=self.path,
                settings=Settings(anonymized_telemetry=False),
            )
            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )

            self.client = client
            self.collection = collection
            self._ready = True

            count = collection.count()
            log.info(
                "ChromaDB ready  |  collection=%s  |  workspace_id=%s  |  existing_chunks=%d  |  hnsw_space=cosine",
                self.collection_name,
                self.workspace_id,
                count,
            )

    def _collection(self) -> Any:
        self._ensure_ready()
        collection = self.collection
        if collection is None:
            raise RuntimeError("Chroma collection is not initialized")
        return collection

    def set_workspace(self, workspace_id: str) -> None:
        """Switch active workspace namespace (collection) for this store."""
        ws = sanitize_workspace_id(workspace_id)
        if ws == self.workspace_id:
            return

        with self._connect_lock:
            self.workspace_id = ws
            self.collection_name = build_collection_name(self.collection_base, self.workspace_id)
            # Force re-bind collection on next access.
            self.collection = None
            self._ready = False

        log.info(
            "Chroma workspace switched  |  workspace_id=%s  |  collection=%s",
            self.workspace_id,
            self.collection_name,
        )

    def upsert(
        self,
        ids: Sequence[str],
        texts: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        metadatas: Sequence[dict[str, Any]] | None = None,
    ) -> None:
        """Upsert embeddings into the collection with bounded batches."""
        if not ids:
            return

        if len(ids) != len(texts) or len(ids) != len(embeddings):
            raise ValueError(
                "ids/texts/embeddings length mismatch: "
                f"{len(ids)}/{len(texts)}/{len(embeddings)}"
            )

        if metadatas is not None and len(ids) != len(metadatas):
            raise ValueError(
                "ids/metadatas length mismatch: "
                f"{len(ids)}/{len(metadatas)}"
            )

        collection = self._collection()

        t0 = time.perf_counter()
        total = len(ids)

        for i in range(0, total, self.upsert_batch_size):
            end = min(i + self.upsert_batch_size, total)

            batch_ids = list(ids[i:end])
            batch_texts = list(texts[i:end])
            batch_embeddings = list(embeddings[i:end])
            batch_metadatas = list(metadatas[i:end]) if metadatas is not None else None

            log.debug(
                "Upsert batch  |  batch=[%d..%d)  |  size=%d",
                i,
                end,
                len(batch_ids),
            )

            collection.upsert(
                ids=batch_ids,
                documents=batch_texts,
                embeddings=batch_embeddings,
                metadatas=cast(Any, batch_metadatas),
            )

        elapsed = time.perf_counter() - t0
        log.info("Upsert complete  |  total_chunks=%d  |  elapsed=%.2fs", total, elapsed)

    def query(self, embedding: list[float], n: int = 10, where: dict[str, Any] | None = None) -> Any:
        """Query the collection with optional metadata filter."""
        collection = self._collection()

        kwargs: dict[str, Any] = {
            "query_embeddings": [embedding],
            "n_results": max(1, n),
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        t0 = time.perf_counter()
        result = collection.query(**kwargs)
        elapsed = time.perf_counter() - t0

        n_results = len(result.get("ids", [[]])[0])
        log.debug(
            "Query returned %d results in %.2fms  |  n_requested=%d  |  has_filter=%s",
            n_results,
            elapsed * 1000,
            n,
            "yes" if where else "no",
        )
        return result

    def delete_by_file(self, filepath: str) -> None:
        collection = self._collection()
        log.info("Deleting chunks for file  |  filepath=%s", filepath)
        collection.delete(where={"filepath": filepath})

    def delete_by_files(self, filepaths: Sequence[str]) -> None:
        paths = [p for p in filepaths if p]
        if not paths:
            return
        collection = self._collection()
        collection.delete(where=cast(Any, {"filepath": {"$in": paths}}))

    def get_all_metadatas(self) -> Any:
        collection = self._collection()
        total = collection.count()
        if total <= 0:
            return {"ids": [], "metadatas": []}
        return collection.get(limit=total, include=["metadatas"])

    def get_all_chunks(self) -> Any:
        """Return all chunks with documents + metadata for index bootstrap."""
        collection = self._collection()
        total = collection.count()
        if total <= 0:
            return {"ids": [], "documents": [], "metadatas": []}
        return collection.get(limit=total, include=["documents", "metadatas"])

    def get_ids_by_file(self, filepath: str) -> list[str]:
        """Helper to get IDs of a file if you need to delete by ID."""
        collection = self._collection()
        res = collection.get(where={"filepath": filepath}, include=[])
        ids = res.get("ids", [])
        log.debug("get_ids_by_file  |  filepath=%s  |  ids_found=%d", filepath, len(ids))
        return ids

    def count(self) -> int:
        collection = self._collection()
        c = collection.count()
        log.debug("ChromaDB count  |  total_chunks=%d", c)
        return c
