"""MCP tool definitions.

Each ``@mcp.tool()`` decorated function is automatically registered when
this module is imported (see ``app.py``).
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any

from mcp.server.fastmcp import Context

from code_radar.logging import get_logger
from code_radar.workspace import ensure_safe_workspace_root, resolve_sync_directory

from . import state
from . import sync_runner as sync
from . import search_utils as utils
from .app import mcp

log = get_logger(__name__)


def _kick_auto_sync_background() -> None:
    """Best-effort trigger for initial sync/watcher from sync tool handlers."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    loop.create_task(sync.trigger_auto_sync())


# sync_workspace_tool - non-blocking background sync
@mcp.tool()
async def sync_workspace_tool(directory: str | None = None, ctx: Context | None = None) -> str:
    """
    Sync workspace files secara non-blocking.
    Memproses chunk, embed, dan simpan vector ke database.

    Args:
        directory: Optional path ke workspace directory.
                   Default: root yang telah dikonfigurasi.

    Returns:
        Pesan status yang mengkonfirmasi sync dimulai.
        Gunakan `get_sync_status` untuk memonitor progress.
    """
    if state._engine is None or state._store is None:
        log.warning("sync_workspace_tool called but engine/store not configured")
        return "Error: engine/store not configured."

    # If sync is already running, report that
    if state._sync_task is not None and not state._sync_task.done():
        status = state._sync_progress
        if status.is_running():
            return (
                f"Sync is already in progress for {status.directory}.\n"
                f"  Status: {status.status}\n"
                f"  Files scanned: {status.scanned_files}/{status.total_files}\n"
                f"  Chunks added: {status.added_chunks}\n"
                f"  Use `get_sync_status` for full progress."
            )
        # If previous task errored, allow re-trigger
        if status.status == "error":
            state._sync_task = None

    configured_root = Path(state._root or ".").resolve()

    root_path = resolve_sync_directory(configured_root, directory)

    # Prevent syncing outside configured workspace root.
    if configured_root not in (root_path, *root_path.parents):
        return (
            "Error: directory must be inside configured workspace root. "
            f"configured_root={configured_root} requested={root_path}"
        )

    try:
        ensure_safe_workspace_root(str(root_path), source="sync_workspace_tool")
    except ValueError as exc:
        return f"Error: {exc}"

    root = str(root_path)
    log.info("Tool called: sync_workspace  (non-blocking)  |  directory=%s", root)

    state._sync_task = asyncio.create_task(sync.run_sync_background(root))

    return (
        f"Sync started for {root} (non-blocking).\n"
        f"  Use `get_sync_status` to check progress."
    )


# get_sync_status - poll progress of background sync
@mcp.tool()
async def get_sync_status(ctx: Context | None = None) -> str:
    """
    Returns current sync progress as JSON.

    Fields: status, directory, workspace_id, total_files, scanned_files,
    changed_files, added_chunks, deleted_files, error, elapsed_seconds,
    is_running, is_done.
    """
    # Auto-trigger initial sync on first interaction
    await sync.trigger_auto_sync()

    payload = state._sync_progress.to_dict()
    payload["workspace_id"] = state._workspace_id
    if state._file_watcher is not None:
        payload["watcher"] = state._file_watcher.get_status()

    return json.dumps(payload, indent=2)


# semantic_search - search over the codebase
@mcp.tool()
def semantic_search(
    query: str,
    n: int = 5,
    filepath: str | None = None,
    language: str | None = None,
    ctx: Context | None = None,
) -> str:
    """
    Semantic search over the codebase.
    Kembalikan potongan kode yang relevan dengan query.

    Args:
        query: Query pencarian dalam bahasa natural.
        n: Jumlah hasil maksimal (default: 5).
        filepath: Filter hanya file tertentu (optional).
        language: Filter berdasarkan bahasa pemrograman (optional).
                  Contoh: "py", "go", "js", "ts", "java", "rs", "rb", "c",
                  "cpp", "csharp", "php", "swift", "kt", "scala".

    Returns:
        Teks berisi potongan kode dengan relevansi score dan posisi baris.
    """
    _kick_auto_sync_background()

    if state._engine is None or state._store is None:
        return "Error: engine/store not configured."

    ok, err = state.ensure_engine_ready()
    if not ok:
        return f"Error: failed to load embedding engine: {err}"

    log.info(
        "Semantic search  |  query=%r  |  n=%d  |  filepath=%s  |  language=%s",
        query,
        n,
        filepath or "all",
        language or "all",
    )
    filepath_filters = utils.resolve_filepath_filters(filepath)

    t0 = time.perf_counter()
    query_emb = state._engine.embed_text(query)
    elapsed_embed = time.perf_counter() - t0
    log.debug("Query embedding  |  elapsed=%.2fs", elapsed_embed)

    # Determine retrieval count
    retrieve_k = utils.resolve_retrieve_k(n)
    log.debug("Resolved retrieve_k=%d  |  requested_n=%d", retrieve_k, n)

    # Metadata filter
    where = utils.build_filepath_where(filepath_filters, language)

    # Retrieve from ChromaDB
    t1 = time.perf_counter()
    where_filter = dict(where) if where else None
    raw = state._store.query(query_emb, n=retrieve_k, where=where_filter)
    elapsed_retrieve = time.perf_counter() - t1

    ids: list[str] = raw.get("ids", [[]])[0]
    distances: list[float] = raw.get("distances", [[]])[0]
    documents: list[str] = raw.get("documents", [[]])[0]
    metadatas: list[dict[str, Any]] = raw.get("metadatas", [[]])[0]

    if not ids:
        warning = utils.query_quality_warning(query)
        if warning is not None:
            return f"No results found.\n{warning}"
        if filepath:
            return f"No results found. (filepath candidates={filepath_filters!r})"
        return "No results found."

    # Adaptive re-routing
    rerank_before = False
    if not state._reranker_disabled and state._reranker is not None:
        if utils.should_rerank_adaptive(retrieve_k, len(ids), query):
            rerank_before = True

    if rerank_before:
        return utils.rerank_and_format(ids, documents, metadatas, distances, query, n)

    # Direct (no reranker)
    lines: list[str] = []
    for idx, (doc_id, dist, doc, meta) in enumerate(
        zip(ids, distances, documents, metadatas)
    ):
        if idx >= n:
            break

        score = utils.distance_to_score(dist)
        fp = meta.get("filepath", "?") if meta else "?"
        ct = meta.get("chunk_type", "?") if meta else "?"
        sl = meta.get("start_line", "?") if meta else "?"
        el = meta.get("end_line", "?") if meta else "?"

        lines.append(f"[{idx + 1}] score={score:.4f}  |  {fp}:{sl}-{el}  ({ct})")
        lines.append("```" + utils.guess_lang(fp))
        lines.append(doc.rstrip("\n"))
        lines.append("```")
        lines.append("")

    elapsed_total = time.perf_counter() - t0

    header = (
        f"Found {min(n, len(ids))} results in {elapsed_total:.2f}s"
        f"  (embed={elapsed_embed:.2f}s, retrieve={elapsed_retrieve:.2f}s)\n"
    )
    warning = utils.query_quality_warning(query)
    if warning is not None:
        header += f"{warning}\n"
    return header + "\n".join(lines)


# smart_search - hybrid search (BM25 + Dense + Regex filter + Reranker)
@mcp.tool()
def smart_search(
    query: str,
    n: int = 5,
    filepath: str | None = None,
    regex: str | None = None,
    ctx: Context | None = None,
) -> str:
    """Ultimate Hybrid Search: Recall -> Filter -> Rerank.

    Phase 1: Wide Recall
      - Dense retrieval (Chroma vector)
      - Sparse retrieval (SQLite FTS5 BM25)
      - Reciprocal Rank Fusion (RRF)

    Phase 2: Pattern Filter
      - Optional regex post-filter on fused candidates only

    Phase 3: Precision Rerank
      - Cross-encoder reranker over filtered candidates
    """
    _kick_auto_sync_background()

    if state._engine is None or state._store is None:
        return "Error: engine/store not configured."

    base_query = (query or "").strip()
    if not base_query:
        return "Error: query cannot be empty."

    ok, err = state.ensure_engine_ready()
    if not ok:
        return f"Error: failed to load embedding engine: {err}"

    query_for_recall, regex_pattern, regex_source = utils.extract_query_and_regex(
        base_query,
        regex,
    )
    if not query_for_recall.strip():
        query_for_recall = base_query

    where: dict[str, Any] | None = None
    filepath_filters = utils.resolve_filepath_filters(filepath)
    where = utils.build_filepath_where(filepath_filters)

    dense_k = utils.resolve_smart_dense_k()
    sparse_k = utils.resolve_smart_sparse_k()
    recall_k = utils.resolve_smart_recall_k()
    rrf_k = utils.resolve_smart_rrf_k()

    log.info(
        (
            "Smart search  |  query=%r  |  n=%d  |  filepath=%s  "
            "|  dense_k=%d  |  sparse_k=%d  |  recall_k=%d"
        ),
        query_for_recall,
        n,
        filepath or "all",
        dense_k,
        sparse_k,
        recall_k,
    )

    t0 = time.perf_counter()

    # PHASE-1A: dense recall
    t_dense = time.perf_counter()
    assert state._engine is not None
    query_emb = state._engine.embed_text(query_for_recall)
    raw_dense = state._store.query(query_emb, n=dense_k, where=where)
    dense_candidates = utils.build_dense_candidates(raw_dense)
    elapsed_dense = time.perf_counter() - t_dense

    # PHASE-1B: sparse recall (BM25)
    t_sparse = time.perf_counter()
    sparse_candidates: list[dict[str, Any]] = []
    sparse_status = "disabled"
    if state._sparse_index is not None:
        sparse_ok, sparse_err = utils.ensure_sparse_index_bootstrapped()
        if sparse_ok:
            sparse_status = "ok"
            sparse_seen: set[str] = set()
            sparse_paths = filepath_filters or [None]
            for sparse_path in sparse_paths:
                for cand in state._sparse_index.search(
                    query=query_for_recall,
                    n=sparse_k,
                    filepath=sparse_path,
                ):
                    cand_id = str(cand.get("id", ""))
                    if cand_id in sparse_seen:
                        continue
                    sparse_seen.add(cand_id)
                    sparse_candidates.append(cand)
        else:
            sparse_status = f"error: {sparse_err}"
    elapsed_sparse = time.perf_counter() - t_sparse

    # PHASE-1C: RRF fusion
    fused = utils.reciprocal_rank_fuse(
        dense=dense_candidates,
        sparse=sparse_candidates,
        top_k=recall_k,
        rrf_k=max(1, rrf_k),
    )

    if not fused:
        warning = utils.query_quality_warning(query_for_recall, len(sparse_candidates))
        extra = f"\n{warning}" if warning else ""
        if filepath:
            extra += f"\n(filepath candidates={filepath_filters!r})"
        return "No results found." + extra

    # PHASE-2: regex post-filter on fused candidates only
    regex_case_sensitive = utils.resolve_smart_regex_case_sensitive()
    regex_compiled, regex_err = utils.compile_optional_regex(
        regex_pattern,
        case_sensitive=regex_case_sensitive,
    )
    if regex_err is not None:
        return f"Error: invalid regex pattern {regex_pattern!r}: {regex_err}"

    pre_filter_count = len(fused)
    if regex_compiled is not None:
        fused = [
            c
            for c in fused
            if regex_compiled.search(str(c.get("document", ""))) is not None
        ]

    if not fused:
        return "No results found after regex post-filter."

    # PHASE-3: precision rerank
    rerank_used = False
    rerank_error: str | None = None
    elapsed_rerank = 0.0
    rerank_docs_count = 0

    reranker = state._reranker
    if reranker is not None and not state._reranker_disabled and state.ensure_reranker_ready():
        rerank_limit = utils.resolve_reranker_candidate_limit(len(fused), requested_n=n)
        rerank_candidates = fused[:rerank_limit]
        docs = [str(c.get("document", "")) for c in rerank_candidates]
        rerank_docs_count = len(docs)
        if docs:
            t_rerank = time.perf_counter()
            rerank_batch = utils.resolve_reranker_batch_size(len(docs))
            rerank_timeout_s = utils.resolve_reranker_timeout_ms(len(docs)) / 1000.0
            try:
                scores = utils.run_rerank_with_timeout(
                    reranker=reranker,
                    query=query_for_recall,
                    docs=docs,
                    batch_size=rerank_batch,
                    timeout_s=rerank_timeout_s,
                )
                for cand, score in zip(rerank_candidates, scores):
                    cand["rerank_score"] = float(score)
                rerank_candidates.sort(
                    key=lambda c: float(c.get("rerank_score", 0.0)),
                    reverse=True,
                )
                fused = rerank_candidates + fused[rerank_limit:]
                rerank_used = True
            except Exception as exc:
                rerank_error = str(exc)
                log.warning("Smart search rerank failed, fallback to hybrid ranking  |  error=%s", exc)
            finally:
                elapsed_rerank = time.perf_counter() - t_rerank

    if not rerank_used:
        fused.sort(
            key=lambda c: (
                float(c.get("hybrid_score", 0.0)),
                float(c.get("dense_score") or 0.0),
            ),
            reverse=True,
        )

    top_n = max(1, n)
    selected = fused[:top_n]
    elapsed_total = time.perf_counter() - t0

    lines: list[str] = []
    for rank, cand in enumerate(selected, start=1):
        meta = cand.get("metadata") or {}
        fp = meta.get("filepath", "?")
        ct = meta.get("chunk_type", "?")
        sl = meta.get("start_line", "?")
        el = meta.get("end_line", "?")

        hybrid_score = float(cand.get("hybrid_score", 0.0))
        dense_score = cand.get("dense_score")
        rerank_score = cand.get("rerank_score")

        score_part = f"hybrid={hybrid_score:.5f}"
        if isinstance(dense_score, (int, float)):
            score_part += f"  dense={float(dense_score):.4f}"
        if isinstance(rerank_score, (int, float)):
            score_part = f"rerank={float(rerank_score):.4f}  " + score_part

        lines.append(
            f"[{rank}] {score_part}  |  {fp}:{sl}-{el}  ({ct})  [source={cand.get('source', '?')}]"
        )
        lines.append("```" + utils.guess_lang(str(fp)))
        lines.append(str(cand.get("document", "")).rstrip("\n"))
        lines.append("```")
        lines.append("")

    regex_line = "regex_filter=off"
    if regex_compiled is not None and regex_pattern is not None:
        regex_line = (
            f"regex_filter=on ({regex_source or 'inline'}: {regex_pattern!r}) "
            f"kept={len(fused)}/{pre_filter_count}"
        )

    rerank_line = "rerank=off"
    if rerank_used:
        rerank_line = f"rerank=on ({elapsed_rerank:.2f}s, docs={rerank_docs_count})"
    elif rerank_error:
        rerank_line = f"rerank=fallback ({rerank_error})"

    header = (
        f"Found {len(selected)} results in {elapsed_total:.2f}s\n"
        f"phase1: dense={len(dense_candidates)} sparse={len(sparse_candidates)} "
        f"fused={pre_filter_count} (dense={elapsed_dense:.2f}s, sparse={elapsed_sparse:.2f}s, sparse_status={sparse_status})\n"
        f"phase2: {regex_line}\n"
        f"phase3: {rerank_line}\n"
    )
    warning = utils.query_quality_warning(query_for_recall, len(sparse_candidates))
    if warning is not None:
        header += f"{warning}\n"
    return header + "\n".join(lines)


# read_full_file - read a file from workspace with line numbers
@mcp.tool()
def read_full_file(filepath: str, ctx: Context | None = None) -> str:
    """
    Baca isi file dari workspace dengan line numbers.

    Args:
        filepath: Relative path dari file yang akan dibaca.

    Returns:
        Isi file dengan format line numbers.
    """

    if state._root is None:
        return "Error: root not configured."

    root_path = Path(state._root).resolve()
    requested = Path(filepath)

    # Path traversal protection
    try:
        full_path = (root_path / requested).resolve(strict=False)
    except RuntimeError:
        return "Error: Access denied."

    if not str(full_path).startswith(str(root_path)):
        return "Error: Access denied."

    if not full_path.is_file():
        return f"Error: File not found: {filepath}"

    try:
        content = full_path.read_text(encoding="utf-8", errors="replace")
    except Exception as exc:
        return f"Error: Cannot read file: {exc}"

    lines = content.rstrip("\n").split("\n")
    num_width = len(str(len(lines)))
    numbered = "\n".join(
        f"{i + 1:{num_width}d} | {line}" for i, line in enumerate(lines)
    )

    return f"File: {filepath}\n\n{numbered}"
