import asyncio
import hashlib
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from code_radar.chunker import chunk_file
from code_radar.envvars import get_env
from code_radar.chroma import ChromaStore
from code_radar.engine import EmbeddingEngine
from code_radar.hash_kv import HashKVStore
from code_radar.logging import get_logger
from code_radar.reader import CodeReader
from code_radar.sparse_index import SparseChunkIndex
from code_radar.state import HashCache, SyncProgress

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


def _file_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _hash_file_stream(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _delete_paths_with_retry(
    store: ChromaStore,
    paths: list[str],
    max_retries: int,
    base_backoff_sec: float,
) -> None:
    if not paths:
        return

    for attempt in range(max_retries + 1):
        try:
            store.delete_by_files(paths)
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed deleting stale chunks after {max_retries + 1} attempts: {exc}"
                ) from exc
            sleep_s = base_backoff_sec * (2**attempt)
            log.warning(
                "Delete retry  |  attempt=%d/%d  |  paths=%d  |  sleep=%.2fs  |  error=%s",
                attempt + 1,
                max_retries + 1,
                len(paths),
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)


def _upsert_chunks_with_retry(
    store: ChromaStore,
    chunks: list[dict[str, Any]],
    embeddings: list[list[float]],
    max_retries: int,
    base_backoff_sec: float,
) -> None:
    ids = [c["id"] for c in chunks]
    texts = [c["content"] for c in chunks]
    metadatas = [
        {
            "filepath": c["filepath"],
            "chunk_type": c["chunk_type"],
            "start_line": c["start_line"],
            "end_line": c["end_line"],
            **c["metadata"],
        }
        for c in chunks
    ]

    for attempt in range(max_retries + 1):
        try:
            store.upsert(ids=ids, texts=texts, embeddings=embeddings, metadatas=metadatas)
            return
        except Exception as exc:
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Failed upsert after {max_retries + 1} attempts: {exc}"
                ) from exc
            sleep_s = base_backoff_sec * (2**attempt)
            log.warning(
                "Upsert retry  |  attempt=%d/%d  |  chunks=%d  |  sleep=%.2fs  |  error=%s",
                attempt + 1,
                max_retries + 1,
                len(chunks),
                sleep_s,
                exc,
            )
            time.sleep(sleep_s)


def _sparse_delete_safe(sparse_index: SparseChunkIndex | None, paths: list[str]) -> None:
    if sparse_index is None or not paths:
        return
    try:
        sparse_index.delete_by_files(paths)
    except Exception as exc:
        log.warning("Sparse delete failed  |  files=%d  |  error=%s", len(paths), exc)


def _sparse_upsert_safe(
    sparse_index: SparseChunkIndex | None,
    chunks: list[dict[str, Any]],
) -> None:
    if sparse_index is None or not chunks:
        return
    try:
        sparse_index.upsert_chunks(chunks)
    except Exception as exc:
        log.warning("Sparse upsert failed  |  chunks=%d  |  error=%s", len(chunks), exc)


def _flush_pending_chunks(
    pending_chunks: list[dict[str, Any]],
    engine: EmbeddingEngine,
    store: ChromaStore,
    embed_batch_size: int,
    upsert_max_retries: int,
    upsert_backoff_sec: float,
    sparse_index: SparseChunkIndex | None = None,
) -> int:
    if not pending_chunks:
        return 0

    texts = [c["content"] for c in pending_chunks]
    embeddings = engine.embed_texts(texts, batch_size=embed_batch_size)

    _upsert_chunks_with_retry(
        store=store,
        chunks=pending_chunks,
        embeddings=embeddings,
        max_retries=upsert_max_retries,
        base_backoff_sec=upsert_backoff_sec,
    )

    # Keep sparse/BM25 index aligned with vector store.
    _sparse_upsert_safe(sparse_index, pending_chunks)

    n = len(pending_chunks)
    pending_chunks.clear()
    return n


def sync_workspace(
    directory: str,
    engine: EmbeddingEngine,
    store: ChromaStore,
    batch_size: int = 8,
) -> tuple[int, int, int]:
    """Sync workspace ke DB. Hanya proses file yang berubah.

    Returns:
        tuple[int, int, int]: (chunks_added, files_unchanged, files_removed_or_updated)
    """
    if engine.tokenizer is None:
        raise RuntimeError("call engine.load() before sync_workspace()")

    # Operational params (predictable memory + backpressure)
    embed_batch_size = max(1, batch_size)
    chunk_buffer_size = _env_int("CODE_SYNC_CHUNK_BUFFER_SIZE", 512)
    delete_batch_size = _env_int("CODE_SYNC_DELETE_BATCH_SIZE", 1000)
    upsert_max_retries = _env_int("CODE_SYNC_UPSERT_MAX_RETRIES", 3)
    delete_max_retries = _env_int("CODE_SYNC_DELETE_MAX_RETRIES", 3)
    upsert_backoff_sec = max(0.05, float(get_env("CODE_SYNC_UPSERT_BACKOFF_SEC", "0.25") or "0.25"))
    delete_backoff_sec = max(0.05, float(get_env("CODE_SYNC_DELETE_BACKOFF_SEC", "0.25") or "0.25"))

    workspace_id = str(getattr(store, "workspace_id", "default"))

    kv_store = HashKVStore(store.path, workspace_id=workspace_id)
    existing_hashes = kv_store.get_all()
    reader = CodeReader(directory)

    sparse_index: SparseChunkIndex | None = None
    try:
        sparse_index = SparseChunkIndex(store.path, workspace_id=workspace_id)
    except Exception as exc:
        log.warning("Sparse index unavailable during sync  |  error=%s", exc)

    log.info(
        (
            "Scanning workspace  |  root=%s  |  embed_batch=%d  "
            "|  chunk_buffer=%d"
        ),
        directory,
        embed_batch_size,
        chunk_buffer_size,
    )

    # Phase 1: detect file deltas (low-memory hash-only scan)
    scan_t0 = time.perf_counter()
    current_files: set[str] = set()
    to_process: list[tuple[str, str]] = []  # (rel_path, file_hash)

    for fp in reader.files():
        rel = str(fp.relative_to(reader.root))
        current_files.add(rel)

        try:
            fh = _hash_file_stream(fp)
        except Exception:
            # Fallback to text read (e.g. transient file issues)
            try:
                content = fp.read_text(encoding="utf-8", errors="replace")
                fh = _file_hash(content)
            except Exception:
                continue

        if existing_hashes.get(rel) == fh:
            continue

        to_process.append((rel, fh))

    scan_elapsed = time.perf_counter() - scan_t0

    changed_paths = {rel for rel, _ in to_process}
    deleted_from_disk = set(existing_hashes.keys()) - current_files
    paths_to_delete = changed_paths | deleted_from_disk

    # Phase 2: delete stale chunks (changed + deleted)
    if paths_to_delete:
        t_del = time.perf_counter()
        path_list = list(paths_to_delete)

        for i in range(0, len(path_list), delete_batch_size):
            batch = path_list[i : i + delete_batch_size]
            _delete_paths_with_retry(
                store=store,
                paths=batch,
                max_retries=delete_max_retries,
                base_backoff_sec=delete_backoff_sec,
            )
            _sparse_delete_safe(sparse_index, batch)

        log.info(
            "Deleted stale chunks  |  files=%d  |  elapsed=%.2fs",
            len(paths_to_delete),
            time.perf_counter() - t_del,
        )

        # Also remove from KV store to prevent state drift
        kv_store.delete_batch(list(paths_to_delete))

    # Phase 3: chunk + embed + upsert in bounded buffers
    pending_chunks: list[dict[str, Any]] = []
    pending_hashes: list[tuple[str, str]] = []
    added_chunks = 0

    proc_t0 = time.perf_counter()
    for rel, fh in to_process:
        full_path = reader.root / rel
        try:
            chunks = chunk_file(full_path, reader.root, engine.tokenizer)
        except Exception as exc:
            log.warning("chunk_file failed  |  file=%s  |  error=%s", rel, exc)
            continue

        if not chunks:
            continue

        pending_chunks.extend(chunks)
        pending_hashes.append((rel, fh))

        if len(pending_chunks) >= chunk_buffer_size:
            added_chunks += _flush_pending_chunks(
                pending_chunks=pending_chunks,
                engine=engine,
                store=store,
                embed_batch_size=embed_batch_size,
                upsert_max_retries=upsert_max_retries,
                upsert_backoff_sec=upsert_backoff_sec,
                sparse_index=sparse_index,
            )
            kv_store.put_batch(pending_hashes)
            pending_hashes.clear()

    # flush remainder
    if pending_chunks:
        added_chunks += _flush_pending_chunks(
            pending_chunks=pending_chunks,
            engine=engine,
            store=store,
            embed_batch_size=embed_batch_size,
            upsert_max_retries=upsert_max_retries,
            upsert_backoff_sec=upsert_backoff_sec,
            sparse_index=sparse_index,
        )
        kv_store.put_batch(pending_hashes)
        pending_hashes.clear()

    kv_store.close()
    if sparse_index is not None:
        sparse_index.close()

    proc_elapsed = time.perf_counter() - proc_t0

    unchanged = len(current_files) - len(to_process)

    log.info(
        (
            "Sync summary  |  scanned=%d files in %.2fs  |  changed=%d  |  unchanged=%d  "
            "|  removed_or_updated=%d  |  chunks_added=%d  |  process_elapsed=%.2fs"
        ),
        len(current_files),
        scan_elapsed,
        len(to_process),
        unchanged,
        len(paths_to_delete),
        added_chunks,
        proc_elapsed,
    )

    return (added_chunks, unchanged, len(paths_to_delete))


# Async + throttled variant (asyncio)
def _chunk_file_safe(path: Path, root: Path, tokenizer: Any) -> list[dict[str, Any]]:
    """Wrapper around chunk_file that returns [] on failure instead of raising."""
    try:
        chunks = chunk_file(path, root, tokenizer)
        return chunks or []
    except Exception:
        return []


async def async_sync_workspace(
    directory: str,
    engine: EmbeddingEngine,
    store: ChromaStore,
    hash_cache: HashCache,
    progress: Optional[SyncProgress] = None,
    batch_size: int = 8,
    throttle_sec: float = 0.05,
) -> tuple[int, int, int]:
    """Async sync with CPU throttling and progress reporting.

    Each CPU-bound operation (hashing, chunking, embedding) runs via
    ``loop.run_in_executor`` so the event loop stays responsive.  A small
    ``asyncio.sleep(throttle_sec)`` is inserted between file processing
    steps to prevent 100 % CPU usage.

    Returns:
        tuple[int, int, int]: (chunks_added, files_unchanged, files_removed_or_updated)
    """
    if engine.tokenizer is None:
        raise RuntimeError("call engine.load() before async_sync_workspace()")

    # Operational params
    loop = asyncio.get_running_loop()
    embed_batch_size = max(1, batch_size)
    chunk_buffer_size = _env_int("CODE_SYNC_CHUNK_BUFFER_SIZE", 512)
    delete_batch_size = _env_int("CODE_SYNC_DELETE_BATCH_SIZE", 1000)
    upsert_max_retries = _env_int("CODE_SYNC_UPSERT_MAX_RETRIES", 3)
    delete_max_retries = _env_int("CODE_SYNC_DELETE_MAX_RETRIES", 3)
    upsert_backoff_sec = max(0.05, float(get_env("CODE_SYNC_UPSERT_BACKOFF_SEC", "0.25") or "0.25"))
    delete_backoff_sec = max(0.05, float(get_env("CODE_SYNC_DELETE_BACKOFF_SEC", "0.25") or "0.25"))

    # Existing hashes from local cache (fast, no ChromaDB scan)
    existing_hashes = hash_cache.get_all()
    reader = CodeReader(directory)
    workspace_id = str(getattr(store, "workspace_id", "default"))

    sparse_index: SparseChunkIndex | None = None
    try:
        sparse_index = SparseChunkIndex(store.path, workspace_id=workspace_id)
    except Exception as exc:
        log.warning("Sparse index unavailable during async sync  |  error=%s", exc)

    # Progress setup
    if progress is not None:
        progress.status = "scanning"
        progress.directory = directory
        progress.start_time = time.perf_counter()

    log.info(
        (
            "Async sync  |  root=%s  |  embed_batch=%d  "
            "|  chunk_buffer=%d  |  throttle=%.3fs"
        ),
        directory,
        embed_batch_size,
        chunk_buffer_size,
        throttle_sec,
    )

    # Phase 1: scan workspace - hash every file, compare with cache
    scan_t0 = time.perf_counter()
    current_files: set[str] = set()
    to_process: list[tuple[str, str]] = []

    for fp in reader.files():
        rel = str(fp.relative_to(reader.root))
        current_files.add(rel)

        # Hash file in executor so the event loop can breathe
        try:
            fh = await loop.run_in_executor(None, _hash_file_stream, fp)
        except Exception:
            try:
                content = await loop.run_in_executor(
                    None,
                    lambda p=fp: p.read_text(encoding="utf-8", errors="replace"),
                )
                fh = _file_hash(content)
            except Exception:
                continue

        if existing_hashes.get(rel) != fh:
            to_process.append((rel, fh))

        # Throttle: yield control briefly after every file
        if throttle_sec > 0:
            await asyncio.sleep(throttle_sec)

        if progress is not None:
            progress.scanned_files = len(current_files)
            progress.total_files = progress.scanned_files
            progress.changed_files = len(to_process)

    scan_elapsed = time.perf_counter() - scan_t0

    changed_paths = {rel for rel, _ in to_process}
    deleted_from_disk = set(existing_hashes.keys()) - current_files
    paths_to_delete = changed_paths | deleted_from_disk

    # Phase 2: delete stale chunks (changed + deleted from disk)
    if paths_to_delete:
        if progress is not None:
            progress.status = "deleting"

        t_del = time.perf_counter()
        path_list = list(paths_to_delete)

        for i in range(0, len(path_list), delete_batch_size):
            batch = path_list[i : i + delete_batch_size]
            await loop.run_in_executor(
                None,
                _delete_paths_with_retry,
                store,
                batch,
                delete_max_retries,
                delete_backoff_sec,
            )
            await loop.run_in_executor(None, _sparse_delete_safe, sparse_index, batch)

        # Remove from hash cache too
        hash_cache.delete_batch(list(paths_to_delete))

        if progress is not None:
            progress.deleted_files = len(paths_to_delete)

        log.info(
            "Deleted stale chunks  |  files=%d  |  elapsed=%.2fs",
            len(paths_to_delete),
            time.perf_counter() - t_del,
        )

    # Phase 3: chunk + embed + upsert in bounded buffers
    if progress is not None:
        progress.status = "processing"

    pending_chunks: list[dict[str, Any]] = []
    pending_hashes: list[tuple[str, str]] = []
    added_chunks = 0

    proc_t0 = time.perf_counter()

    def _hash_and_embed(
        chunks_inner: list[dict[str, Any]],
    ) -> tuple[list[list[float]], int]:
        texts = [c["content"] for c in chunks_inner]
        embs = engine.embed_texts(texts, batch_size=embed_batch_size)
        return embs, len(chunks_inner)

    for rel, fh in to_process:
        full_path = reader.root / rel

        # Chunk file in executor
        chunks = await loop.run_in_executor(
            None,
            _chunk_file_safe,
            full_path,
            reader.root,
            engine.tokenizer,
        )
        if not chunks:
            continue

        pending_chunks.extend(chunks)
        pending_hashes.append((rel, fh))

        # Throttle after every file
        if throttle_sec > 0:
            await asyncio.sleep(throttle_sec)

        if progress is not None:
            progress.changed_files = len(pending_hashes)

        # Flush when buffer is full
        if len(pending_chunks) >= chunk_buffer_size:
            embs, n = await loop.run_in_executor(None, _hash_and_embed, pending_chunks)
            await loop.run_in_executor(
                None,
                _upsert_chunks_with_retry,
                store,
                pending_chunks,
                embs,
                upsert_max_retries,
                upsert_backoff_sec,
            )
            await loop.run_in_executor(None, _sparse_upsert_safe, sparse_index, pending_chunks)
            added_chunks += n
            if progress is not None:
                progress.added_chunks = added_chunks
            hash_cache.put_batch(pending_hashes)
            pending_chunks.clear()
            pending_hashes.clear()

            # Throttle after flush
            if throttle_sec > 0:
                await asyncio.sleep(throttle_sec * 2)

    # Final flush
    if pending_chunks:
        embs, n = await loop.run_in_executor(None, _hash_and_embed, pending_chunks)
        await loop.run_in_executor(
            None,
            _upsert_chunks_with_retry,
            store,
            pending_chunks,
            embs,
            upsert_max_retries,
            upsert_backoff_sec,
        )
        await loop.run_in_executor(None, _sparse_upsert_safe, sparse_index, pending_chunks)
        added_chunks += n
        if progress is not None:
            progress.added_chunks = added_chunks
        hash_cache.put_batch(pending_hashes)
        pending_hashes.clear()

    proc_elapsed = time.perf_counter() - proc_t0
    unchanged = len(current_files) - len(to_process)

    # Wrap up
    if progress is not None:
        progress.status = "done"
        progress.added_chunks = added_chunks
        progress.total_files = len(current_files)
        progress.changed_files = len(to_process)
        progress.elapsed = time.perf_counter() - progress.start_time

    log.info(
        (
            "Async sync done  |  scanned=%d files in %.2fs  |  changed=%d  |  unchanged=%d  "
            "|  removed_or_updated=%d  |  chunks_added=%d  |  process_elapsed=%.2fs"
        ),
        len(current_files),
        scan_elapsed,
        len(to_process),
        unchanged,
        len(paths_to_delete),
        added_chunks,
        proc_elapsed,
    )

    if sparse_index is not None:
        sparse_index.close()

    return (added_chunks, unchanged, len(paths_to_delete))


async def async_sync_files(
    directory: str,
    engine: EmbeddingEngine,
    store: ChromaStore,
    hash_cache: HashCache,
    filepaths: Iterable[str],
    deleted_paths: Iterable[str],
    progress: Optional[SyncProgress] = None,
    batch_size: int = 8,
    throttle_sec: float = 0.05,
) -> tuple[int, int, int]:
    """Incremental async sync for a known set of file paths.

    Designed for event-driven watchers:
    - ``filepaths``: created/modified files to re-hash and (if changed) re-embed
    - ``deleted_paths``: files removed/moved away; stale chunks are deleted directly

    Returns:
        tuple[int, int, int]: (chunks_added, files_unchanged, files_removed_or_updated)
    """
    if engine.tokenizer is None:
        raise RuntimeError("call engine.load() before async_sync_files()")

    root = Path(directory).resolve()
    loop = asyncio.get_running_loop()

    embed_batch_size = max(1, batch_size)
    chunk_buffer_size = _env_int("CODE_SYNC_CHUNK_BUFFER_SIZE", 512)
    delete_batch_size = _env_int("CODE_SYNC_DELETE_BATCH_SIZE", 1000)
    upsert_max_retries = _env_int("CODE_SYNC_UPSERT_MAX_RETRIES", 3)
    delete_max_retries = _env_int("CODE_SYNC_DELETE_MAX_RETRIES", 3)
    upsert_backoff_sec = max(0.05, float(get_env("CODE_SYNC_UPSERT_BACKOFF_SEC", "0.25") or "0.25"))
    delete_backoff_sec = max(0.05, float(get_env("CODE_SYNC_DELETE_BACKOFF_SEC", "0.25") or "0.25"))

    def _normalise_rel(path: str) -> str | None:
        try:
            p = Path(path)
            rel = p if not p.is_absolute() else p.resolve(strict=False).relative_to(root)
            parts = [part for part in rel.parts if part not in {"", "."}]
            if not parts or any(part == ".." for part in parts):
                return None
            return Path(*parts).as_posix()
        except Exception:
            return None

    requested_files: set[str] = set()
    for path in filepaths:
        rel = _normalise_rel(path)
        if rel:
            requested_files.add(rel)

    explicit_deletes: set[str] = set()
    for path in deleted_paths:
        rel = _normalise_rel(path)
        if rel:
            explicit_deletes.add(rel)

    existing_hashes = hash_cache.get_all()
    workspace_id = str(getattr(store, "workspace_id", "default"))

    if progress is not None:
        progress.status = "scanning"
        progress.directory = directory
        progress.start_time = time.perf_counter()
        progress.total_files = len(requested_files)
        progress.scanned_files = 0
        progress.changed_files = 0

    to_process: list[tuple[str, str]] = []
    unchanged = 0
    paths_to_delete = set(explicit_deletes)

    for rel in sorted(requested_files):
        full_path = root / rel

        if not full_path.exists() or not full_path.is_file():
            paths_to_delete.add(rel)
            if progress is not None:
                progress.scanned_files += 1
            continue

        try:
            fh = await loop.run_in_executor(None, _hash_file_stream, full_path)
        except Exception:
            try:
                content = await loop.run_in_executor(
                    None,
                    lambda p=full_path: p.read_text(encoding="utf-8", errors="replace"),
                )
                fh = _file_hash(content)
            except Exception:
                if progress is not None:
                    progress.scanned_files += 1
                continue

        if existing_hashes.get(rel) == fh:
            unchanged += 1
        else:
            to_process.append((rel, fh))
            paths_to_delete.add(rel)

        if progress is not None:
            progress.scanned_files += 1
            progress.changed_files = len(to_process)

        if throttle_sec > 0:
            await asyncio.sleep(throttle_sec)

    sparse_index: SparseChunkIndex | None = None
    try:
        sparse_index = SparseChunkIndex(store.path, workspace_id=workspace_id)
    except Exception as exc:
        log.warning("Sparse index unavailable during incremental sync  |  error=%s", exc)

    try:
        if paths_to_delete:
            if progress is not None:
                progress.status = "deleting"

            path_list = list(paths_to_delete)
            for i in range(0, len(path_list), delete_batch_size):
                batch = path_list[i : i + delete_batch_size]
                await loop.run_in_executor(
                    None,
                    _delete_paths_with_retry,
                    store,
                    batch,
                    delete_max_retries,
                    delete_backoff_sec,
                )
                await loop.run_in_executor(None, _sparse_delete_safe, sparse_index, batch)

            hash_cache.delete_batch(path_list)
            if progress is not None:
                progress.deleted_files = len(path_list)

        if progress is not None:
            progress.status = "processing"

        pending_chunks: list[dict[str, Any]] = []
        pending_hashes: list[tuple[str, str]] = []
        added_chunks = 0

        def _embed(chunks_inner: list[dict[str, Any]]) -> tuple[list[list[float]], int]:
            texts = [c["content"] for c in chunks_inner]
            embs = engine.embed_texts(texts, batch_size=embed_batch_size)
            return embs, len(chunks_inner)

        for rel, fh in to_process:
            full_path = root / rel
            chunks = await loop.run_in_executor(
                None,
                _chunk_file_safe,
                full_path,
                root,
                engine.tokenizer,
            )
            if not chunks:
                continue

            pending_chunks.extend(chunks)
            pending_hashes.append((rel, fh))

            if len(pending_chunks) >= chunk_buffer_size:
                embs, n = await loop.run_in_executor(None, _embed, pending_chunks)
                await loop.run_in_executor(
                    None,
                    _upsert_chunks_with_retry,
                    store,
                    pending_chunks,
                    embs,
                    upsert_max_retries,
                    upsert_backoff_sec,
                )
                await loop.run_in_executor(None, _sparse_upsert_safe, sparse_index, pending_chunks)
                hash_cache.put_batch(pending_hashes)
                pending_chunks.clear()
                pending_hashes.clear()
                added_chunks += n

            if throttle_sec > 0:
                await asyncio.sleep(throttle_sec)

        if pending_chunks:
            embs, n = await loop.run_in_executor(None, _embed, pending_chunks)
            await loop.run_in_executor(
                None,
                _upsert_chunks_with_retry,
                store,
                pending_chunks,
                embs,
                upsert_max_retries,
                upsert_backoff_sec,
            )
            await loop.run_in_executor(None, _sparse_upsert_safe, sparse_index, pending_chunks)
            hash_cache.put_batch(pending_hashes)
            pending_hashes.clear()
            added_chunks += n

        if progress is not None:
            progress.status = "done"
            progress.added_chunks = added_chunks
            progress.changed_files = len(to_process)
            progress.elapsed = time.perf_counter() - progress.start_time

        return (added_chunks, unchanged, len(paths_to_delete))

    finally:
        if sparse_index is not None:
            sparse_index.close()
