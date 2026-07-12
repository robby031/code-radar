import gc
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import fmean
from typing import Iterator

import mlx.core as mx
import numpy as np

from code_radar.chunker import estimate_tokens
from code_radar.envvars import get_env
from code_radar.engine import EmbeddingEngine
from code_radar.logging import get_logger
from code_radar.reader import CodeReader
from code_radar.reranker import Reranker

log = get_logger(__name__)


@dataclass(frozen=True)
class BenchmarkTarget:
    label: str
    model_id: str
    tuning_name: str | None
    tuning_env: dict[str, int | float]


@dataclass(frozen=True)
class CollabBenchmarkTarget:
    label: str
    embedding_model_id: str
    reranker_model_id: str
    tuning_name: str | None
    tuning_env: dict[str, int | float]


@dataclass
class BenchmarkResult:
    target: BenchmarkTarget
    success: bool
    error: str | None

    batch_size: int
    max_batch_tokens: int
    max_tokens_per_text: int
    repeats: int
    sample_count: int
    total_tokens: int
    embedding_dim: int

    load_sec: float
    run_times_sec: list[float]
    avg_run_sec: float
    best_run_sec: float
    worst_run_sec: float

    texts_per_sec: float
    tokens_per_sec: float

    active_mem_mb: float | None
    peak_mem_mb: float | None


@dataclass
class RerankerBenchmarkResult:
    target: BenchmarkTarget
    success: bool
    error: str | None

    batch_size: int
    n_queries: int
    n_docs: int
    repeats: int

    load_sec: float
    run_times_sec: list[float]
    avg_run_sec: float
    best_run_sec: float
    worst_run_sec: float

    docs_per_sec: float

    active_mem_mb: float | None
    peak_mem_mb: float | None


@dataclass
class CollabBenchmarkResult:
    target: CollabBenchmarkTarget
    success: bool
    error: str | None

    corpus_size: int
    query_count: int
    eval_query_count: int
    retrieve_k: int
    top_n: int
    repeats: int

    embedding_batch_size: int
    reranker_batch_size: int

    embed_load_sec: float
    reranker_load_sec: float
    setup_sec: float

    run_times_sec: list[float]
    avg_run_sec: float
    best_run_sec: float
    worst_run_sec: float

    searches_per_sec: float
    rerank_docs_per_sec: float

    # Retrieval quality (before rerank)
    mrr_at_5_retrieval: float
    recall_at_12_retrieval: float
    ndcg_at_12_retrieval: float

    # Retrieval quality after rerank
    mrr_at_5_rerank: float
    recall_at_12_rerank: float
    ndcg_at_12_rerank: float

    active_mem_mb: float | None
    peak_mem_mb: float | None


def _safe_mb(getter_name: str) -> float | None:
    fn = getattr(mx, getter_name, None)
    if fn is None:
        return None
    try:
        value = float(fn())
        return value / 1e6
    except Exception:
        return None


@contextmanager
def _temporary_env(overrides: dict[str, int | float]) -> Iterator[None]:
    keys = list(overrides.keys())
    old = {k: os.environ.get(k) for k in keys}

    for k, v in overrides.items():
        os.environ[k] = str(v)

    try:
        yield
    finally:
        for k in keys:
            prev = old[k]
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev


def _snippets_from_text(text: str, max_chars: int) -> list[str]:
    s = text.strip()
    if not s:
        return []

    if len(s) <= max_chars:
        return [s]

    # deterministic slices from head/mid/tail to emulate realistic code contexts
    points = [0, max(0, len(s) // 2 - max_chars // 2), max(0, len(s) - max_chars)]
    out: list[str] = []
    seen: set[str] = set()

    for start in points:
        chunk = s[start : start + max_chars].strip()
        if chunk and chunk not in seen:
            out.append(chunk)
            seen.add(chunk)

    return out


def collect_benchmark_texts(
    workspace: str,
    target_samples: int,
    max_chars: int,
) -> list[str]:
    reader = CodeReader(workspace)
    snippets: list[str] = []

    for fp in reader.files():
        try:
            text = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for s in _snippets_from_text(text, max_chars=max_chars):
            snippets.append(s)
            if len(snippets) >= target_samples:
                break

        if len(snippets) >= target_samples:
            break

    if snippets:
        return snippets

    # fallback minimal corpus so benchmark still runs outside code repos
    fallback = [
        "def hello(name: str) -> str:\n    return f'hello {name}'",
        "class Config:\n    def __init__(self, path: str) -> None:\n        self.path = path",
        "SELECT id, name FROM users WHERE active = 1 ORDER BY created_at DESC",
        "function debounce(fn, ms) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }",
    ]
    return fallback[: max(1, min(target_samples, len(fallback)))]


# Default query-doc pairs for reranker benchmark (language-agnostic code snippets)
_RERANKER_BENCH_QUERIES_DOCS: list[tuple[str, str]] = [
    (
        "function to validate user authentication token",
        "def validate_jwt(token, secret):\n    try:\n        payload = jwt.decode(token, secret, algorithms=['HS256'])\n        return payload\n    except jwt.ExpiredSignatureError:\n        return None",
    ),
    (
        "function to validate user authentication token",
        "const fs = require('fs');\nfunction readConfig(path) {\n  const data = fs.readFileSync(path, 'utf8');\n  return JSON.parse(data);\n}",
    ),
    (
        "database query for active users",
        "SELECT id, name, email FROM users WHERE active = 1 ORDER BY created_at DESC LIMIT ?",
    ),
    (
        "database query for active users",
        "def get_user(id):\n    return db.execute('SELECT * FROM users WHERE id = ?', (id,)).fetchone()",
    ),
    (
        "error handling middleware",
        "app.use((err, req, res, next) => {\n  console.error(err.stack);\n  res.status(500).json({ error: 'Something broke!' });\n});",
    ),
    (
        "error handling middleware",
        "class AppError extends Error {\n  constructor(message, statusCode) {\n    super(message);\n    this.statusCode = statusCode;\n  }\n}",
    ),
    (
        "sort algorithm implementation",
        "function quickSort(arr) {\n  if (arr.length <= 1) return arr;\n  const pivot = arr[0];\n  const left = []; const right = [];\n  for (let i = 1; i < arr.length; i++) {\n    arr[i] < pivot ? left.push(arr[i]) : right.push(arr[i]);\n  }\n  return [...quickSort(left), pivot, ...quickSort(right)];\n}",
    ),
    (
        "sort algorithm implementation",
        "def bubble_sort(arr):\n    n = len(arr)\n    for i in range(n):\n        for j in range(0, n - i - 1):\n            if arr[j] > arr[j + 1]:\n                arr[j], arr[j + 1] = arr[j + 1], arr[j]\n    return arr",
    ),
]


def _seed_qrels() -> dict[str, dict[str, float]]:
    """Create graded relevance labels from built-in benchmark pairs."""
    grouped: dict[str, list[str]] = {}
    for q, d in _RERANKER_BENCH_QUERIES_DOCS:
        grouped.setdefault(q, []).append(d)

    out: dict[str, dict[str, float]] = {}
    for q, docs in grouped.items():
        rel_map: dict[str, float] = {}
        # First doc for a query is usually the strongest match in the seed list.
        if docs:
            rel_map[docs[0]] = 3.0
        if len(docs) >= 2:
            rel_map[docs[1]] = 2.0
        for extra in docs[2:]:
            rel_map[extra] = max(rel_map.get(extra, 0.0), 1.0)
        out[q] = rel_map

    return out


def collect_benchmark_queries(target_count: int) -> list[str]:
    unique_queries: list[str] = []
    seen: set[str] = set()

    for q, _ in _RERANKER_BENCH_QUERIES_DOCS:
        if q not in seen:
            seen.add(q)
            unique_queries.append(q)

    if target_count <= 0:
        return unique_queries

    if not unique_queries:
        return []

    if len(unique_queries) >= target_count:
        return unique_queries[:target_count]

    # Repeat seeded queries to keep evaluation grounded in known qrels.
    out: list[str] = []
    i = 0
    while len(out) < target_count:
        out.append(unique_queries[i % len(unique_queries)])
        i += 1

    return out


def _recall_at_k(ranked_idx: list[int], relevant: dict[int, float], k: int) -> float:
    if not relevant:
        return 0.0
    kk = max(1, min(k, len(ranked_idx)))
    top = ranked_idx[:kk]
    hits = sum(1 for idx in top if idx in relevant)
    return hits / float(len(relevant))


def _mrr_at_k(ranked_idx: list[int], relevant: dict[int, float], k: int) -> float:
    kk = max(1, min(k, len(ranked_idx)))
    for rank, idx in enumerate(ranked_idx[:kk], start=1):
        if idx in relevant:
            return 1.0 / float(rank)
    return 0.0


def _dcg_from_rels(rels: list[float]) -> float:
    if not rels:
        return 0.0
    total = 0.0
    for i, rel in enumerate(rels, start=1):
        gain = (2.0**rel) - 1.0
        total += gain / float(np.log2(i + 1))
    return total


def _ndcg_at_k(ranked_idx: list[int], relevant: dict[int, float], k: int) -> float:
    if not relevant:
        return 0.0

    kk = max(1, min(k, len(ranked_idx)))
    rels = [float(relevant.get(idx, 0.0)) for idx in ranked_idx[:kk]]
    dcg = _dcg_from_rels(rels)

    ideal_rels = sorted((float(v) for v in relevant.values()), reverse=True)[:kk]
    idcg = _dcg_from_rels(ideal_rels)
    if idcg <= 1e-12:
        return 0.0

    return dcg / idcg


def run_embedding_benchmark(
    target: BenchmarkTarget,
    texts: list[str],
    repeats: int,
    batch_size_override: int | None,
) -> BenchmarkResult:
    if not texts:
        return BenchmarkResult(
            target=target,
            success=False,
            error="No benchmark texts available.",
            batch_size=0,
            max_batch_tokens=0,
            max_tokens_per_text=0,
            repeats=repeats,
            sample_count=0,
            total_tokens=0,
            embedding_dim=0,
            load_sec=0.0,
            run_times_sec=[],
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            texts_per_sec=0.0,
            tokens_per_sec=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    run_times: list[float] = []
    embedding_dim = 0
    load_sec = 0.0
    batch_size = 0
    max_batch_tokens = 0
    max_tokens_per_text = 0

    total_tokens = sum(estimate_tokens(t) for t in texts)

    try:
        with _temporary_env(target.tuning_env):
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                try:
                    mx.reset_peak_memory()  # type: ignore[attr-defined]
                except Exception:
                    pass

            t0 = time.perf_counter()
            engine = EmbeddingEngine(model_name=target.model_id)
            engine.load()
            load_sec = time.perf_counter() - t0

            batch_size = max(1, batch_size_override or engine.default_batch_size)
            max_batch_tokens = int(engine.max_batch_tokens)
            max_tokens_per_text = int(engine.max_tokens_per_text)

            warmup_n = min(len(texts), max(4, min(32, batch_size)))
            _ = engine.embed_texts(texts[:warmup_n], batch_size=batch_size)

            for _i in range(repeats):
                t_run = time.perf_counter()
                vectors = engine.embed_texts(texts, batch_size=batch_size)
                dt = time.perf_counter() - t_run
                run_times.append(dt)

                if vectors:
                    embedding_dim = len(vectors[0])

            active_mb = _safe_mb("get_active_memory")
            peak_mb = _safe_mb("get_peak_memory")

            # cleanup
            del engine
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

    except Exception as exc:
        return BenchmarkResult(
            target=target,
            success=False,
            error=str(exc),
            batch_size=batch_size,
            max_batch_tokens=max_batch_tokens,
            max_tokens_per_text=max_tokens_per_text,
            repeats=repeats,
            sample_count=len(texts),
            total_tokens=total_tokens,
            embedding_dim=embedding_dim,
            load_sec=load_sec,
            run_times_sec=run_times,
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            texts_per_sec=0.0,
            tokens_per_sec=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    avg_run = fmean(run_times) if run_times else 0.0
    best_run = min(run_times) if run_times else 0.0
    worst_run = max(run_times) if run_times else 0.0

    texts_per_sec = (len(texts) / avg_run) if avg_run > 0 else 0.0
    tokens_per_sec = (total_tokens / avg_run) if avg_run > 0 else 0.0

    return BenchmarkResult(
        target=target,
        success=True,
        error=None,
        batch_size=batch_size,
        max_batch_tokens=max_batch_tokens,
        max_tokens_per_text=max_tokens_per_text,
        repeats=repeats,
        sample_count=len(texts),
        total_tokens=total_tokens,
        embedding_dim=embedding_dim,
        load_sec=load_sec,
        run_times_sec=run_times,
        avg_run_sec=avg_run,
        best_run_sec=best_run,
        worst_run_sec=worst_run,
        texts_per_sec=texts_per_sec,
        tokens_per_sec=tokens_per_sec,
        active_mem_mb=active_mb,
        peak_mem_mb=peak_mb,
    )


def _resolve_reranker_batch_size(default_value: int = 4) -> int:
    raw = get_env("CODE_RERANKER_BATCH_SIZE")
    if raw is None:
        return max(1, default_value)
    try:
        return max(1, int(raw))
    except ValueError:
        return max(1, default_value)


def run_reranker_benchmark(
    target: BenchmarkTarget,
    repeats: int,
    batch_size_override: int | None = None,
    queries: list[str] | None = None,
    documents: list[str] | None = None,
) -> RerankerBenchmarkResult:
    """Benchmark a reranker model: load time + scoring throughput."""
    run_times: list[float] = []
    load_sec = 0.0

    if queries is None:
        queries = [q for q, _ in _RERANKER_BENCH_QUERIES_DOCS]
    if documents is None:
        documents = [d for _, d in _RERANKER_BENCH_QUERIES_DOCS]

    queries = [q for q in queries if q.strip()]
    documents = [d for d in documents if d.strip()]

    if not queries or not documents:
        return RerankerBenchmarkResult(
            target=target,
            success=False,
            error="Queries/documents for reranker benchmark are empty.",
            batch_size=0,
            n_queries=len(queries),
            n_docs=len(documents),
            repeats=repeats,
            load_sec=0.0,
            run_times_sec=[],
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            docs_per_sec=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    batch_size = 4

    try:
        with _temporary_env(target.tuning_env):
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                try:
                    mx.reset_peak_memory()
                except Exception:
                    pass

            t0 = time.perf_counter()
            reranker = Reranker(model_name=target.model_id)
            reranker.load()
            load_sec = time.perf_counter() - t0

            batch_size = max(1, batch_size_override or _resolve_reranker_batch_size(default_value=4))

            # Warmup
            warm_docs = documents[: min(len(documents), max(2, min(batch_size, 8)))]
            _ = reranker.rerank(queries[0], warm_docs, batch_size=max(1, min(batch_size, 4)))

            # Measured runs: score each query against all docs
            for _i in range(repeats):
                t_run = time.perf_counter()
                for q in queries:
                    _ = reranker.rerank(q, documents, batch_size=batch_size)
                dt = time.perf_counter() - t_run
                run_times.append(dt)

            active_mb = _safe_mb("get_active_memory")
            peak_mb = _safe_mb("get_peak_memory")

            # cleanup
            reranker.unload()
            del reranker
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

    except Exception as exc:
        return RerankerBenchmarkResult(
            target=target,
            success=False,
            error=str(exc),
            batch_size=batch_size,
            n_queries=len(queries),
            n_docs=len(documents),
            repeats=repeats,
            load_sec=load_sec,
            run_times_sec=run_times,
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            docs_per_sec=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    avg_run = fmean(run_times) if run_times else 0.0
    best_run = min(run_times) if run_times else 0.0
    worst_run = max(run_times) if run_times else 0.0

    total_score_ops = len(queries) * len(documents)
    docs_per_sec = (total_score_ops / avg_run) if avg_run > 0 else 0.0

    return RerankerBenchmarkResult(
        target=target,
        success=True,
        error=None,
        batch_size=batch_size,
        n_queries=len(queries),
        n_docs=len(documents),
        repeats=repeats,
        load_sec=load_sec,
        run_times_sec=run_times,
        avg_run_sec=avg_run,
        best_run_sec=best_run,
        worst_run_sec=worst_run,
        docs_per_sec=docs_per_sec,
        active_mem_mb=active_mb,
        peak_mem_mb=peak_mb,
    )


def _normalize_vector(v: np.ndarray) -> np.ndarray:
    denom = float(np.linalg.norm(v))
    if denom <= 1e-12:
        return v
    return v / denom


def _top_k_indices(scores: np.ndarray, k: int) -> np.ndarray:
    k = max(1, min(k, scores.shape[0]))
    if k == scores.shape[0]:
        return np.argsort(scores)[::-1]

    idx = np.argpartition(scores, -k)[-k:]
    return idx[np.argsort(scores[idx])[::-1]]


def run_collab_benchmark(
    target: CollabBenchmarkTarget,
    texts: list[str],
    queries: list[str],
    repeats: int,
    retrieve_k: int = 12,
    top_n: int = 5,
    embedding_batch_size_override: int | None = None,
    reranker_batch_size_override: int | None = None,
) -> CollabBenchmarkResult:
    """Benchmark request-time collaboration: query embedding + vector retrieval + reranking.

    Also reports quality metrics using seeded graded relevance labels:
    - MRR@5
    - Recall@12
    - NDCG@12

    Metrics are reported for retrieval-only and after reranking.
    """
    texts = [t for t in texts if t.strip()]
    queries = [q for q in queries if q.strip()]

    retrieve_k = max(1, min(int(retrieve_k), 50))
    top_n = max(1, min(int(top_n), retrieve_k))

    seed_qrels = _seed_qrels()
    eval_seed_queries = [q for q in queries if q in seed_qrels]

    # Ensure graded-relevance docs exist inside corpus so metrics are meaningful.
    seed_docs: list[str] = []
    for q in eval_seed_queries:
        seed_docs.extend(seed_qrels[q].keys())

    corpus_texts = list(dict.fromkeys([*texts, *seed_docs]))

    if not corpus_texts or not queries:
        return CollabBenchmarkResult(
            target=target,
            success=False,
            error="Corpus texts and queries must be non-empty.",
            corpus_size=len(corpus_texts),
            query_count=len(queries),
            eval_query_count=0,
            retrieve_k=retrieve_k,
            top_n=top_n,
            repeats=repeats,
            embedding_batch_size=0,
            reranker_batch_size=0,
            embed_load_sec=0.0,
            reranker_load_sec=0.0,
            setup_sec=0.0,
            run_times_sec=[],
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            searches_per_sec=0.0,
            rerank_docs_per_sec=0.0,
            mrr_at_5_retrieval=0.0,
            recall_at_12_retrieval=0.0,
            ndcg_at_12_retrieval=0.0,
            mrr_at_5_rerank=0.0,
            recall_at_12_rerank=0.0,
            ndcg_at_12_rerank=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    run_times: list[float] = []
    embed_load_sec = 0.0
    reranker_load_sec = 0.0
    setup_sec = 0.0
    embedding_batch_size = 0
    reranker_batch_size = 0

    # Quality metric accumulators
    mrr_ret_vals: list[float] = []
    recall_ret_vals: list[float] = []
    ndcg_ret_vals: list[float] = []
    mrr_rr_vals: list[float] = []
    recall_rr_vals: list[float] = []
    ndcg_rr_vals: list[float] = []

    eval_query_count = 0

    try:
        with _temporary_env(target.tuning_env):
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
            if hasattr(mx, "reset_peak_memory"):
                try:
                    mx.reset_peak_memory()
                except Exception:
                    pass

            # Load models
            t_load = time.perf_counter()
            engine = EmbeddingEngine(model_name=target.embedding_model_id)
            engine.load()
            embed_load_sec = time.perf_counter() - t_load

            t_rr = time.perf_counter()
            reranker = Reranker(model_name=target.reranker_model_id)
            reranker.load()
            reranker_load_sec = time.perf_counter() - t_rr

            embedding_batch_size = max(1, embedding_batch_size_override or engine.default_batch_size)
            reranker_batch_size = max(1, reranker_batch_size_override or _resolve_reranker_batch_size(default_value=4))

            # Offline-index analogue: embed corpus once (not counted in measured loop)
            t_setup = time.perf_counter()
            corpus_vectors = engine.embed_texts(corpus_texts, batch_size=embedding_batch_size)
            corpus_mat = np.asarray(corpus_vectors, dtype=np.float32)
            if corpus_mat.ndim != 2 or corpus_mat.shape[0] != len(corpus_texts):
                raise RuntimeError("Invalid corpus embedding shape during collab benchmark.")

            norms = np.linalg.norm(corpus_mat, axis=1, keepdims=True)
            norms = np.maximum(norms, 1e-12)
            corpus_mat = corpus_mat / norms
            setup_sec = time.perf_counter() - t_setup

            # Prepare qrels in index space
            doc_to_idx = {doc: i for i, doc in enumerate(corpus_texts)}
            qrels_idx: dict[str, dict[int, float]] = {}
            for q in eval_seed_queries:
                rel_idx: dict[int, float] = {}
                for doc, rel in seed_qrels[q].items():
                    idx = doc_to_idx.get(doc)
                    if idx is not None:
                        rel_idx[int(idx)] = float(rel)
                if rel_idx:
                    qrels_idx[q] = rel_idx

            eval_query_count = sum(1 for q in queries if q in qrels_idx)

            # Warmup one full search
            q0 = queries[0]
            q0_vec = np.asarray(engine.embed_text(q0), dtype=np.float32)
            q0_vec = _normalize_vector(q0_vec)
            sim0 = corpus_mat @ q0_vec
            idx0 = _top_k_indices(sim0, retrieve_k)
            warm_candidates = [corpus_texts[int(i)] for i in idx0]
            _ = reranker.rerank(q0, warm_candidates, batch_size=max(1, min(reranker_batch_size, 4)))

            # Measured request-time loop
            for _i in range(repeats):
                t_run = time.perf_counter()

                for q in queries:
                    q_vec = np.asarray(engine.embed_text(q), dtype=np.float32)
                    q_vec = _normalize_vector(q_vec)
                    sim = corpus_mat @ q_vec
                    top_idx_arr = _top_k_indices(sim, retrieve_k)
                    top_idx = [int(i) for i in top_idx_arr.tolist()]
                    candidates = [corpus_texts[i] for i in top_idx]

                    rr_scores = reranker.rerank(q, candidates, batch_size=reranker_batch_size)

                    reranked_idx = top_idx
                    # Keep full behavior realistic: produce final top-n list
                    if rr_scores:
                        order = sorted(range(len(rr_scores)), key=lambda j: rr_scores[j], reverse=True)
                        reranked_idx = [top_idx[j] for j in order]
                        paired = list(zip(candidates, rr_scores))
                        paired.sort(key=lambda x: x[1], reverse=True)
                        _ = paired[:top_n]

                    rel = qrels_idx.get(q)
                    if rel is not None:
                        mrr_ret_vals.append(_mrr_at_k(top_idx, rel, 5))
                        recall_ret_vals.append(_recall_at_k(top_idx, rel, 12))
                        ndcg_ret_vals.append(_ndcg_at_k(top_idx, rel, 12))

                        mrr_rr_vals.append(_mrr_at_k(reranked_idx, rel, 5))
                        recall_rr_vals.append(_recall_at_k(reranked_idx, rel, 12))
                        ndcg_rr_vals.append(_ndcg_at_k(reranked_idx, rel, 12))

                dt = time.perf_counter() - t_run
                run_times.append(dt)

            active_mb = _safe_mb("get_active_memory")
            peak_mb = _safe_mb("get_peak_memory")

            # cleanup
            reranker.unload()
            del reranker
            del engine
            gc.collect()
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()

    except Exception as exc:
        return CollabBenchmarkResult(
            target=target,
            success=False,
            error=str(exc),
            corpus_size=len(corpus_texts),
            query_count=len(queries),
            eval_query_count=eval_query_count,
            retrieve_k=retrieve_k,
            top_n=top_n,
            repeats=repeats,
            embedding_batch_size=embedding_batch_size,
            reranker_batch_size=reranker_batch_size,
            embed_load_sec=embed_load_sec,
            reranker_load_sec=reranker_load_sec,
            setup_sec=setup_sec,
            run_times_sec=run_times,
            avg_run_sec=0.0,
            best_run_sec=0.0,
            worst_run_sec=0.0,
            searches_per_sec=0.0,
            rerank_docs_per_sec=0.0,
            mrr_at_5_retrieval=0.0,
            recall_at_12_retrieval=0.0,
            ndcg_at_12_retrieval=0.0,
            mrr_at_5_rerank=0.0,
            recall_at_12_rerank=0.0,
            ndcg_at_12_rerank=0.0,
            active_mem_mb=None,
            peak_mem_mb=None,
        )

    avg_run = fmean(run_times) if run_times else 0.0
    best_run = min(run_times) if run_times else 0.0
    worst_run = max(run_times) if run_times else 0.0

    searches_per_repeat = len(queries)
    effective_retrieve_k = min(retrieve_k, len(corpus_texts))
    rerank_docs_per_repeat = len(queries) * effective_retrieve_k

    searches_per_sec = (searches_per_repeat / avg_run) if avg_run > 0 else 0.0
    rerank_docs_per_sec = (rerank_docs_per_repeat / avg_run) if avg_run > 0 else 0.0

    mrr_at_5_retrieval = fmean(mrr_ret_vals) if mrr_ret_vals else 0.0
    recall_at_12_retrieval = fmean(recall_ret_vals) if recall_ret_vals else 0.0
    ndcg_at_12_retrieval = fmean(ndcg_ret_vals) if ndcg_ret_vals else 0.0

    mrr_at_5_rerank = fmean(mrr_rr_vals) if mrr_rr_vals else 0.0
    recall_at_12_rerank = fmean(recall_rr_vals) if recall_rr_vals else 0.0
    ndcg_at_12_rerank = fmean(ndcg_rr_vals) if ndcg_rr_vals else 0.0

    return CollabBenchmarkResult(
        target=target,
        success=True,
        error=None,
        corpus_size=len(corpus_texts),
        query_count=len(queries),
        eval_query_count=eval_query_count,
        retrieve_k=retrieve_k,
        top_n=top_n,
        repeats=repeats,
        embedding_batch_size=embedding_batch_size,
        reranker_batch_size=reranker_batch_size,
        embed_load_sec=embed_load_sec,
        reranker_load_sec=reranker_load_sec,
        setup_sec=setup_sec,
        run_times_sec=run_times,
        avg_run_sec=avg_run,
        best_run_sec=best_run,
        worst_run_sec=worst_run,
        searches_per_sec=searches_per_sec,
        rerank_docs_per_sec=rerank_docs_per_sec,
        mrr_at_5_retrieval=mrr_at_5_retrieval,
        recall_at_12_retrieval=recall_at_12_retrieval,
        ndcg_at_12_retrieval=ndcg_at_12_retrieval,
        mrr_at_5_rerank=mrr_at_5_rerank,
        recall_at_12_rerank=recall_at_12_rerank,
        ndcg_at_12_rerank=ndcg_at_12_rerank,
        active_mem_mb=active_mb,
        peak_mem_mb=peak_mb,
    )
