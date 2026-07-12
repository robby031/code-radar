"""Query analysis, reranker helpers, and result formatting.

All functions depend on ``state`` for engine/store/reranker globals.
"""

import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from pathlib import Path
from typing import Any

from code_radar.logging import get_logger
from code_radar.reranker import RerankTimeoutError, Reranker

from . import state

log = get_logger(__name__)

_PATH_ANCHORS: tuple[str, ...] = (
    "src",
    "internal",
    "pkg",
    "cmd",
    "app",
    "apps",
    "config",
    "configs",
    "lib",
    "services",
    "service",
)

# Stopwords / code keywords
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "need", "dare", "ought",
        "used", "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "out", "off", "over", "under", "again",
        "further", "then", "once", "here", "there", "when", "where", "why",
        "how", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "nor", "not", "only", "own", "same",
        "so", "than", "too", "very", "just", "because", "but", "and", "or",
        "if", "while", "that", "this", "it", "its", "what", "which", "who",
        "whom", "whose",
    }
)

_CODE_KEYWORDS: frozenset[str] = frozenset(
    {
        "def", "class", "import", "from", "return", "if", "else", "elif",
        "for", "while", "try", "except", "finally", "with", "as", "yield",
        "lambda", "pass", "break", "continue", "raise", "assert", "del",
        "global", "nonlocal", "async", "await", "function", "const", "let",
        "var", "interface", "type", "extends", "implements", "new", "this",
        "super", "export", "module", "require", "console", "print",
        "function", "void", "int", "string", "bool", "null", "undefined",
        "true", "false", "struct", "enum", "trait", "impl", "fn", "pub",
        "self", "mut", "let", "match", "Some", "None", "Result", "Ok",
        "Err", "package", "nil", "defer", "go", "chan", "select", "case",
        "switch", "default", "goto", "fallthrough",
    }
)


# Query analysis
def is_complex_query(query: str) -> bool:
    """Heuristic: queries longer than 8 content words or with code symbols are 'complex'."""
    clean = re.sub(r"[^a-z0-9_ \n]", " ", query.lower())
    words = [w for w in clean.split() if w not in _STOPWORDS and len(w) > 1]
    return len(words) >= 8 or bool(re.search(r"[{}();=<>+\-*/%&|^~!@#]", query))


def normalize_filepath_filter(filepath: str | None) -> str | None:
    """Normalize user-provided filepath to indexed workspace-relative form."""
    if filepath is None:
        return None

    raw = filepath.strip()
    if not raw:
        return None

    normalized = raw.replace("\\", "/")
    root = state._root

    try:
        path = Path(raw).expanduser()
        if path.is_absolute() and root:
            root_path = Path(root).expanduser().resolve()
            resolved = path.resolve()
            try:
                return resolved.relative_to(root_path).as_posix()
            except ValueError:
                return normalized.lstrip("./")
    except OSError:
        pass

    if root:
        root_posix = Path(root).expanduser().resolve().as_posix()
        if normalized == root_posix:
            return ""
        prefix = root_posix.rstrip("/") + "/"
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix):]

    while normalized.startswith("./"):
        normalized = normalized[2:]
    return normalized.strip("/")


def _filepath_variants(filepath: str) -> list[str]:
    normalized = filepath.replace("\\", "/").strip("/")
    parts = [part for part in normalized.split("/") if part]
    variants: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        clean = value.replace("\\", "/").strip("/")
        if clean and clean not in seen:
            seen.add(clean)
            variants.append(clean)

    add(normalized)

    for idx, part in enumerate(parts):
        if part in _PATH_ANCHORS:
            add("/".join(parts[idx:]))

    if parts:
        add(parts[-1])

    return variants


def resolve_filepath_filters(filepath: str | None) -> list[str]:
    """Resolve a user filepath filter to indexed metadata filepath values."""
    normalized = normalize_filepath_filter(filepath)
    if not normalized:
        return []

    store = state._store
    if store is None:
        return [normalized]

    variants = _filepath_variants(normalized)

    try:
        raw = store.get_all_metadatas()
    except Exception as exc:
        log.debug("Failed resolving filepath filter from metadata  |  error=%s", exc)
        return [normalized]

    metadatas = raw.get("metadatas", []) if isinstance(raw, dict) else []
    full_variants = [v for v in variants if "/" in v]
    basename_variants = [v for v in variants if "/" not in v]

    exact_matches: list[str] = []
    directory_matches: list[str] = []
    basename_matches: list[str] = []
    seen_exact: set[str] = set()
    seen_directory: set[str] = set()
    seen_basename: set[str] = set()

    for meta in metadatas:
        if not isinstance(meta, dict):
            continue

        fp_raw = meta.get("filepath")
        if not isinstance(fp_raw, str) or not fp_raw:
            continue

        fp = fp_raw.replace("\\", "/").strip("/")
        fp_name = Path(fp).name

        if any(
            fp == variant
            or fp.endswith(f"/{variant}")
            or variant.endswith(f"/{fp}")
            for variant in full_variants
        ):
            if fp not in seen_exact:
                seen_exact.add(fp)
                exact_matches.append(fp)
            continue

        if any(
            Path(variant).suffix == "" and fp.startswith(f"{variant.rstrip('/')}/")
            for variant in full_variants
        ):
            if fp not in seen_directory:
                seen_directory.add(fp)
                directory_matches.append(fp)
            continue

        if not full_variants and any(fp_name == variant for variant in basename_variants):
            if fp not in seen_basename:
                seen_basename.add(fp)
                basename_matches.append(fp)

    if exact_matches:
        return exact_matches
    if directory_matches:
        return directory_matches
    if basename_matches:
        return basename_matches
    return variants[:1]


def build_filepath_where(
    filepath_values: list[str],
    language: str | None = None,
) -> dict[str, Any] | None:
    where_parts: list[dict[str, Any]] = []

    if filepath_values:
        if len(filepath_values) == 1:
            where_parts.append({"filepath": filepath_values[0]})
        else:
            where_parts.append({"filepath": {"$in": filepath_values}})

    if language:
        where_parts.append({"language": language})

    if not where_parts:
        return None
    if len(where_parts) == 1:
        return where_parts[0]
    return {"$and": where_parts}


def query_quality_warning(query: str, sparse_count: int | None = None) -> str | None:
    """Return a user-facing warning for low-signal queries."""
    clean = re.sub(r"[^a-z0-9_ \n]", " ", query.lower())
    words = [w for w in clean.split() if len(w) > 1]
    content_words = [w for w in words if w not in _STOPWORDS]

    if not content_words:
        return "warning: query is too short or only contains stopwords; lexical recall may be empty."
    if len(content_words) == 1 and sparse_count == 0:
        return "warning: sparse recall returned 0 hits for a very short query; add a symbol, filename, or second keyword."
    return None


def _legacy_should_skip_rerank_for_query(query: str) -> bool:
    """Skip reranker on trivial queries (< 4 content words, no code symbols)."""
    if not query:
        return True

    # Short query? skip
    if len(query.strip()) < 8:
        return True

    clean = re.sub(r"[^a-z0-9_ \n]", " ", query.lower())
    words = [w for w in clean.split() if w not in _STOPWORDS and len(w) > 1]

    if len(words) < 4:
        return True

    if not any(c in query for c in "{}();=<>+*-/&|^!@#%"):
        if not any(kw in query.lower() for kw in _CODE_KEYWORDS):
            if len(words) <= 6:
                return True

    return False


def should_rerank_adaptive(retrieve_k: int, n_results: int, query: str) -> bool:
    """Decide whether to rerank based on result count + query complexity."""
    # Not enough candidates to bother
    if n_results <= min(3, retrieve_k):
        return False

    query = query.strip()
    if not query:
        return False

    # Short/trivial queries don't need reranking
    if _legacy_should_skip_rerank_for_query(query):
        return False

    # Load reranker only when needed
    if not state.ensure_reranker_ready():
        return False

    return True


# Resolve parameters
def resolve_retrieve_k(n: int) -> int:
    """Determine how many candidates to retrieve before possible reranking."""
    retrieve_k = state.env_int("CODE_RETRIEVE_K", 0)
    if retrieve_k > 0:
        return retrieve_k
    # Adaptive default: 3x the requested n, clamped to [10, 12]
    return max(10, min(n * 3, 12))


def resolve_reranker_batch_size(n: int) -> int:
    """Pick a batch size for the cross-encoder."""
    configured = state.env_int("CODE_RERANKER_BATCH_SIZE", 0)
    if configured > 0:
        return configured
    if n <= 5:
        return 1
    if n <= 12:
        return 2
    return 4


def resolve_reranker_timeout_ms(n: int) -> int:
    """Reranker timeout per candidate (ms)."""
    configured = state.env_int("CODE_RERANKER_TIMEOUT_MS", 0)
    if configured > 0:
        return max(500, n * configured)
    return max(500, n * 250)


def resolve_reranker_candidate_limit(candidate_count: int, requested_n: int | None = None) -> int:
    """Bound candidate count before expensive cross-encoder inference."""
    max_candidates = state.env_int("CODE_RERANKER_MAX_CANDIDATES", 30)
    default_candidates = state.env_int("CODE_RERANKER_DEFAULT_CANDIDATES", 12)
    multiplier = state.env_int("CODE_RERANKER_CANDIDATE_MULTIPLIER", 3)

    limit = min(max(1, candidate_count), max_candidates)
    if default_candidates > 0:
        limit = min(limit, default_candidates)

    if requested_n is not None:
        requested = max(1, int(requested_n))
        limit = min(limit, max(requested, requested * multiplier))

    return max(1, limit)


def resolve_reranker_worker_timeout_s(timeout_s: float) -> float:
    """Hard worker wait budget with grace for MLX cleanup at batch boundaries."""
    configured = state.env_int("CODE_RERANKER_WORKER_TIMEOUT_MS", 0)
    if configured > 0:
        return max(timeout_s, configured / 1000.0)
    return timeout_s + max(1.0, min(10.0, timeout_s * 0.5))


def resolve_smart_dense_k() -> int:
    return state.env_int("CODE_SMART_SEARCH_DENSE_K", 30)


def resolve_smart_sparse_k() -> int:
    return state.env_int("CODE_SMART_SEARCH_SPARSE_K", 30)


def resolve_smart_recall_k() -> int:
    return state.env_int("CODE_SMART_SEARCH_RECALL_K", 30)


def resolve_smart_rrf_k() -> int:
    return state.env_int("CODE_SMART_SEARCH_RRF_K", 60)


def resolve_smart_regex_case_sensitive() -> bool:
    return state.env_bool("CODE_SMART_SEARCH_REGEX_CASE_SENSITIVE", False)


def ensure_sparse_index_bootstrapped() -> tuple[bool, str | None]:
    """Ensure BM25 index is populated; bootstrap from ChromaDB if still empty."""
    sparse = state._sparse_index
    store = state._store

    if sparse is None:
        return False, "sparse index not configured"
    if store is None:
        return False, "store not configured"

    try:
        if sparse.count_chunks() > 0:
            return True, None
    except Exception as exc:
        return False, f"sparse index count failed: {exc}"

    try:
        total_vectors = store.count()
    except Exception as exc:
        return False, f"vector count failed: {exc}"

    if total_vectors <= 0:
        return True, None

    with state._sparse_bootstrap_lock:
        try:
            if sparse.count_chunks() > 0:
                return True, None

            log.info(
                "Bootstrapping sparse index from ChromaDB  |  vectors=%d",
                total_vectors,
            )
            raw = store.get_all_chunks()
            ids: list[str] = raw.get("ids", [])
            docs: list[str] = raw.get("documents", [])
            metas_raw = raw.get("metadatas", [])
            metas: list[dict[str, Any]] = [m or {} for m in metas_raw]

            if not ids or not docs:
                return True, None

            if len(metas) < len(ids):
                metas.extend({} for _ in range(len(ids) - len(metas)))

            rows: list[dict[str, Any]] = []
            for doc_id, doc, meta in zip(ids, docs, metas):
                m = meta or {}
                rows.append(
                    {
                        "id": doc_id,
                        "filepath": m.get("filepath", "?"),
                        "content": doc,
                        "chunk_type": m.get("chunk_type", "code_block"),
                        "start_line": int(m.get("start_line", 1)),
                        "end_line": int(m.get("end_line", 1)),
                        "metadata": {"language": m.get("language", "")},
                    }
                )

            sparse.upsert_chunks(rows)
            log.info("Sparse bootstrap complete  |  chunks=%d", len(rows))
            return True, None
        except Exception as exc:
            return False, f"sparse bootstrap failed: {exc}"


def extract_query_and_regex(
    query: str,
    regex: str | None,
) -> tuple[str, str | None, str | None]:
    """Split user query into semantic query + optional regex pattern.

    Returns:
        (query_for_recall, regex_pattern, regex_source)
    """
    q = query.strip()
    explicit = (regex or "").strip()
    if explicit:
        return (q, explicit, "argument")

    patterns = [
        r"(?is)(?:^|\s)(?:regex|re)\s*[:=]\s*`([^`]+)`",
        r"(?is)(?:^|\s)(?:regex|re)\s*[:=]\s*" + '"([^"]+)"',
        r"(?is)(?:^|\s)(?:regex|re)\s*[:=]\s*'([^']+)'",
        r"(?is)(?:^|\s)(?:regex|re)\s*[:=]\s*(\S+)",
    ]

    for pat in patterns:
        m = re.search(pat, q)
        if m is None:
            continue

        rx = m.group(1).strip()
        cleaned = (q[: m.start()] + " " + q[m.end() :]).strip()
        if not cleaned:
            cleaned = q
        return (cleaned, rx, "inline")

    # Support pure /.../ query as regex-only mode.
    if len(q) >= 3 and q.startswith("/") and q.endswith("/"):
        return (q[1:-1], q[1:-1], "inline")

    return (q, None, None)


def compile_optional_regex(
    pattern: str | None,
    *,
    case_sensitive: bool,
) -> tuple[re.Pattern[str] | None, str | None]:
    if pattern is None or not pattern.strip():
        return None, None

    flags = 0 if case_sensitive else re.IGNORECASE
    try:
        return re.compile(pattern, flags), None
    except re.error as exc:
        return None, str(exc)


def build_dense_candidates(raw: dict[str, Any]) -> list[dict[str, Any]]:
    ids: list[str] = raw.get("ids", [[]])[0]
    distances: list[float] = raw.get("distances", [[]])[0]
    documents: list[str] = raw.get("documents", [[]])[0]
    metadatas: list[dict[str, Any]] = raw.get("metadatas", [[]])[0]

    out: list[dict[str, Any]] = []
    for doc_id, dist, doc, meta in zip(ids, distances, documents, metadatas):
        out.append(
            {
                "id": doc_id,
                "document": doc,
                "metadata": meta or {},
                "distance": float(dist),
                "dense_score": distance_to_score(float(dist)),
            }
        )
    return out


def reciprocal_rank_fuse(
    dense: list[dict[str, Any]],
    sparse: list[dict[str, Any]],
    *,
    top_k: int,
    rrf_k: int,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}

    for rank, item in enumerate(dense, start=1):
        doc_id = item["id"]
        cand = merged.get(doc_id)
        if cand is None:
            cand = {
                "id": doc_id,
                "document": item.get("document", ""),
                "metadata": item.get("metadata", {}) or {},
                "dense_rank": None,
                "sparse_rank": None,
                "distance": None,
                "dense_score": None,
                "bm25_score": None,
            }
            merged[doc_id] = cand

        cand["dense_rank"] = rank
        cand["distance"] = item.get("distance")
        cand["dense_score"] = item.get("dense_score")
        if not cand.get("document"):
            cand["document"] = item.get("document", "")
        if not cand.get("metadata"):
            cand["metadata"] = item.get("metadata", {}) or {}

    for rank, item in enumerate(sparse, start=1):
        doc_id = item["id"]
        cand = merged.get(doc_id)
        if cand is None:
            cand = {
                "id": doc_id,
                "document": item.get("document", ""),
                "metadata": item.get("metadata", {}) or {},
                "dense_rank": None,
                "sparse_rank": None,
                "distance": None,
                "dense_score": None,
                "bm25_score": None,
            }
            merged[doc_id] = cand

        cand["sparse_rank"] = rank
        cand["bm25_score"] = item.get("bm25_score")
        if not cand.get("document"):
            cand["document"] = item.get("document", "")
        if not cand.get("metadata"):
            cand["metadata"] = item.get("metadata", {}) or {}

    fused: list[dict[str, Any]] = []
    for cand in merged.values():
        score = 0.0
        d_rank = cand.get("dense_rank")
        s_rank = cand.get("sparse_rank")
        if isinstance(d_rank, int) and d_rank > 0:
            score += 1.0 / (rrf_k + d_rank)
        if isinstance(s_rank, int) and s_rank > 0:
            score += 1.0 / (rrf_k + s_rank)

        cand["hybrid_score"] = score
        cand["source"] = (
            "both"
            if d_rank and s_rank
            else ("dense" if d_rank else ("sparse" if s_rank else "unknown"))
        )
        fused.append(cand)

    fused.sort(
        key=lambda c: (
            float(c.get("hybrid_score", 0.0)),
            float(c.get("dense_score") or 0.0),
        ),
        reverse=True,
    )

    return fused[: max(1, top_k)]


# Reranker execution
def _get_rerank_executor() -> ThreadPoolExecutor:
    """Return (or create) a shared thread-pool for reranker calls."""
    if state._rerank_executor is None:
        state._rerank_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="rerank",
        )
    return state._rerank_executor


def _locked_rerank(
    reranker: Reranker,
    query: str,
    docs: list[str],
    batch_size: int,
    timeout_s: float,
) -> list[float]:
    """Serialize reranker calls to avoid GPU OOM."""
    acquired = state._rerank_call_lock.acquire(timeout=max(0.001, timeout_s))
    if not acquired:
        raise RerankTimeoutError(
            f"Reranker busy after {timeout_s:.1f}s for {len(docs)} docs "
            f"(batch_size={batch_size})"
        )

    try:
        return reranker.rerank(
            query,
            docs,
            batch_size=batch_size,
            timeout_sec=timeout_s,
        )
    finally:
        state._rerank_call_lock.release()


def run_rerank_with_timeout(
    reranker: Reranker,
    query: str,
    docs: list[str],
    batch_size: int,
    timeout_s: float,
) -> list[float]:
    """Run cross-encoder reranking in a thread with a timeout."""
    executor = _get_rerank_executor()
    future = executor.submit(_locked_rerank, reranker, query, docs, batch_size, timeout_s)
    worker_timeout_s = resolve_reranker_worker_timeout_s(timeout_s)
    try:
        return future.result(timeout=worker_timeout_s)
    except FuturesTimeoutError:
        future.cancel()
        raise RerankTimeoutError(
            f"Reranker timed out after {worker_timeout_s:.1f}s for {len(docs)} docs "
            f"(batch_size={batch_size})"
        )


# Formatting helpers
def distance_to_score(dist: float) -> float:
    """Convert ChromaDB cosine distance to a similarity score in [0, 1]."""
    return max(0.0, min(1.0, 1.0 - dist / 2.0))


def guess_lang(filepath: str) -> str:
    ext = Path(filepath).suffix.lower()
    return {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".jsx": "jsx",
        ".go": "go",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
        ".rb": "ruby",
        ".rs": "rust",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".php": "php",
        ".sh": "bash",
        ".bash": "bash",
        ".zsh": "bash",
        ".fish": "fish",
        ".ps1": "powershell",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".scss": "scss",
        ".sass": "sass",
        ".less": "less",
        ".vue": "vue",
        ".svelte": "svelte",
        ".json": "json",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".xml": "xml",
        ".md": "markdown",
        ".rst": "rst",
        ".csv": "csv",
        ".dockerfile": "dockerfile",
        ".makefile": "makefile",
    }.get(ext, "")


def rerank_and_format(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
    query: str,
    n: int,
) -> str:
    """Run cross-encoder reranking and format results."""
    if not documents:
        return "No results found."

    t_rerank = time.perf_counter()

    reranker = state._reranker
    if reranker is None:
        return "No reranker available."

    rerank_limit = resolve_reranker_candidate_limit(len(documents), requested_n=n)
    rerank_ids = ids[:rerank_limit]
    rerank_documents = documents[:rerank_limit]
    rerank_metadatas = metadatas[:rerank_limit]
    rerank_distances = distances[:rerank_limit]

    rerank_batch = resolve_reranker_batch_size(len(rerank_documents))
    rerank_timeout_s = resolve_reranker_timeout_ms(len(rerank_documents)) / 1000.0

    try:
        rerank_scores = run_rerank_with_timeout(
            reranker=reranker,
            query=query,
            docs=rerank_documents,
            batch_size=rerank_batch,
            timeout_s=rerank_timeout_s,
        )
    except RerankTimeoutError as exc:
        log.warning("Reranker timeout, falling back to direct scoring  |  error=%s", exc)
        return format_direct_results(ids, documents, metadatas, distances, n)

    elapsed_rerank = time.perf_counter() - t_rerank

    # Sort by reranker score descending
    indexed = list(
        enumerate(zip(rerank_ids, rerank_documents, rerank_metadatas, rerank_scores))
    )
    indexed.sort(key=lambda x: x[1][3], reverse=True)

    lines: list[str] = []
    for rank, (orig_idx, (doc_id, doc, meta, score)) in enumerate(indexed[:n]):
        fp = meta.get("filepath", "?") if meta else "?"
        ct = meta.get("chunk_type", "?") if meta else "?"
        sl = meta.get("start_line", "?") if meta else "?"
        el = meta.get("end_line", "?") if meta else "?"

        orig_dist = rerank_distances[orig_idx]
        orig_score = distance_to_score(orig_dist)

        lines.append(
            f"[{rank + 1}] score={score:.4f}  (orig={orig_score:.4f})  |  "
            f"{fp}:{sl}-{el}  ({ct})"
        )
        lines.append("```" + guess_lang(fp))
        lines.append(doc.rstrip("\n"))
        lines.append("```")
        lines.append("")

    header = (
        f"Found {min(n, len(indexed))} results (reranked in {elapsed_rerank:.2f}s)\n"
    )
    return header + "\n".join(lines)


def format_direct_results(
    ids: list[str],
    documents: list[str],
    metadatas: list[dict[str, Any]],
    distances: list[float],
    n: int,
) -> str:
    """Format results without reranking."""
    lines: list[str] = []
    for idx, (doc_id, dist, doc, meta) in enumerate(
        zip(ids, distances, documents, metadatas)
    ):
        if idx >= n:
            break
        score = distance_to_score(dist)
        fp = meta.get("filepath", "?") if meta else "?"
        ct = meta.get("chunk_type", "?") if meta else "?"
        sl = meta.get("start_line", "?") if meta else "?"
        el = meta.get("end_line", "?") if meta else "?"

        lines.append(f"[{idx + 1}] score={score:.4f}  |  {fp}:{sl}-{el}  ({ct})")
        lines.append("```" + guess_lang(fp))
        lines.append(doc.rstrip("\n"))
        lines.append("```")
        lines.append("")

    return "\n".join(lines) if lines else "No results found."
