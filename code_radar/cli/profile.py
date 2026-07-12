"""Profile and benchmark CLI commands: ls, current, set, benchmark."""

import argparse
import sys

from code_radar.envvars import get_env

from code_radar.benchmark import (
    BenchmarkTarget,
    CollabBenchmarkTarget,
    collect_benchmark_queries,
    collect_benchmark_texts,
    run_collab_benchmark,
    run_embedding_benchmark,
    run_reranker_benchmark,
)
from code_radar.cli.helpers import (
    env_int,
    print_size,
    is_cached,
    cached_size,
)
from code_radar.config import (
    CONFIG_FILE,
    load_config,
    resolve_model_id,
    resolve_reranker_model_id,
    save_config,
)
from code_radar.models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    DEFAULT_RERANKER_MODEL_KEY,
    RERANKER_REGISTRY,
    get_model_config,
    get_reranker_model_config,
)
from code_radar.profiles import (
    DEFAULT_PROFILE_KEY,
    get_profile_config,
    list_profiles,
)
from code_radar.workspace import resolve_workspace_root


def _print_active_runtime_profile_details() -> None:
    """Single source of truth for active runtime configuration."""
    cfg = load_config()
    profile_key = cfg.get("profile", DEFAULT_PROFILE_KEY)

    try:
        profile = get_profile_config(profile_key)
    except ValueError:
        profile = get_profile_config(DEFAULT_PROFILE_KEY)
        profile_key = profile.key

    print(f"Active profile: {profile.key}")
    print(f"  Name:         {profile.name}")
    print(f"  Description:  {profile.description}")
    print()

    # Active embedding model
    model_key = cfg.get("model_key", DEFAULT_MODEL_KEY)
    if model_key in MODEL_REGISTRY:
        m = get_model_config(model_key)
        m_id = m.id
        print("Embedding model:")
        print(f"  Key:          {model_key}")
        print(f"  Name:         {m.name}")
        print(f"  HF ID:        {m_id}")
        print(f"  RAM:          {m.ram_gb:.1f} GB")
        print(f"  Speed:        {m.speed_tier}")
        print(f"  Accuracy:     {m.accuracy_tier}")
        print(f"  Multimodal:   {'Yes' if m.is_multimodal else 'No'}")
    else:
        m_id = resolve_model_id(model_key)
        print("Embedding model:")
        print(f"  Key:          {model_key}")
        print(f"  HF ID:        {m_id}")

    m_cached = is_cached(m_id)
    print(f"  Cached:       {'✓' if m_cached else '✗ (auto-download on load)'}")
    if m_cached:
        print_size("  Cache size", cached_size(m_id))
    print()

    # Active reranker model
    rr_key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)
    if rr_key in RERANKER_REGISTRY:
        rr = get_reranker_model_config(rr_key)
        rr_id = rr.id
        print("Reranker model:")
        print(f"  Key:          {rr_key}")
        print(f"  Name:         {rr.name}")
        print(f"  HF ID:        {rr_id}")
        print(f"  RAM:          {rr.ram_gb:.1f} GB")
        print(f"  Speed:        {rr.speed_tier}")
        print(f"  Accuracy:     {rr.accuracy_tier}")
        print(f"  Default BS:   {rr.default_batch_size}")
    else:
        rr_id = resolve_reranker_model_id(rr_key)
        print("Reranker model:")
        print(f"  Key:          {rr_key}")
        print(f"  HF ID:        {rr_id}")

    rr_cached = is_cached(rr_id)
    print(f"  Enabled:      {cfg.get('reranker_enabled', True)}")
    print(f"  Cached:       {'✓' if rr_cached else '✗ (auto-download on load)'}")
    if rr_cached:
        print_size("  Cache size", cached_size(rr_id))
    print()

    print("Tuning (default profile values):")
    for env_key, env_val in profile.tuning.items():
        override = get_env(env_key)
        if override is None:
            print(f"  {env_key}={env_val}")
        else:
            marker = "(override)" if override != str(env_val) else "(from env)"
            print(f"  {env_key}={override}  {marker}")

    print(f"\nConfig file:    {CONFIG_FILE}")


def cmd_profile_ls(_args: argparse.Namespace) -> None:
    """List available performance profiles."""
    cfg = load_config()
    active_profile = cfg.get("profile", DEFAULT_PROFILE_KEY)
    active_model_key = cfg.get("model_key", DEFAULT_MODEL_KEY)
    active_rr_key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)

    print("Available profiles:\n")

    for entry in list_profiles():
        key = entry["key"]
        model_key = entry["model_key"]
        rr_key = entry["reranker_model_key"] or "-"
        marker = "->" if key == active_profile else "  "

        model_cfg = MODEL_REGISTRY.get(model_key)
        model_id = model_cfg.id if model_cfg else model_key
        model_cached = is_cached(model_id)
        model_active = "[model active]" if model_key == active_model_key else ""

        # Reranker cache status
        if rr_key != "-":
            rr_cfg = RERANKER_REGISTRY.get(rr_key)
            if rr_cfg is not None:
                rr_cached = is_cached(rr_cfg.id)
                rr_status = "[reranker cached ✓]" if rr_cached else "[reranker not cached]"
            else:
                rr_status = ""
            rr_active = "[reranker active]" if rr_key == active_rr_key else ""
        else:
            rr_status = ""
            rr_active = ""

        print(f"{marker} {key:10s}  {entry['name']}")
        print(f"   {entry['description']}")
        embed_line = (
            f"   embed: {model_key}  "
            f"{'[cached ✓]' if model_cached else '[not cached]'}  {model_active}"
        )
        print(embed_line.rstrip())
        if rr_key != "-":
            rr_line = (
                f"   reranker: {rr_key}  "
                f"{rr_status}  {rr_active}"
            )
            print(rr_line.rstrip())
        print()


def cmd_profile_current(_args: argparse.Namespace) -> None:
    """Show unified active runtime profile + embedding/reranker details."""
    _print_active_runtime_profile_details()


# Benchmark resolution helpers

def _resolve_embedding_spec(name: str) -> tuple[str, str]:
    raw = name.strip()
    key = raw.lower()

    if key in MODEL_REGISTRY:
        return f"model:{key}", MODEL_REGISTRY[key].id

    if "/" in raw:
        return f"hf:{raw}", raw

    raise ValueError(
        "Unknown embedding model target. Use model key (e.g. qwen3-4b-4bit) "
        "or full HF ID."
    )


def _resolve_reranker_spec(name: str) -> tuple[str, str]:
    raw = name.strip()
    key = raw.lower()

    if key in RERANKER_REGISTRY:
        cfg_rr = get_reranker_model_config(key)
        return f"reranker:{key}", cfg_rr.id

    if "/" in raw:
        return f"hf:{raw}", raw

    raise ValueError(
        "Unknown reranker model target. Use reranker key (e.g. reranker-0.6b-4bit) "
        "or full HF ID."
    )


def _resolve_embedding_benchmark_target(name: str) -> BenchmarkTarget:
    raw = name.strip()
    key = raw.lower()

    try:
        prof = get_profile_config(key)
        model_id = MODEL_REGISTRY[prof.model_key].id
        return BenchmarkTarget(
            label=f"profile:{prof.key}",
            model_id=model_id,
            tuning_name=prof.key,
            tuning_env=dict(prof.tuning),
        )
    except ValueError:
        pass

    if key in RERANKER_REGISTRY:
        raise ValueError(
            "Target is a reranker key. Gunakan --mode reranker untuk benchmark reranker."
        )

    label, model_id = _resolve_embedding_spec(raw)
    return BenchmarkTarget(
        label=label,
        model_id=model_id,
        tuning_name=None,
        tuning_env={},
    )


def _resolve_reranker_benchmark_target(name: str) -> BenchmarkTarget:
    raw = name.strip()
    key = raw.lower()

    try:
        prof = get_profile_config(key)
        if not prof.reranker_model_key:
            raise ValueError(f"Profile '{prof.key}' tidak punya reranker_model_key.")
        rr_cfg = get_reranker_model_config(prof.reranker_model_key)
        return BenchmarkTarget(
            label=f"profile:{prof.key}",
            model_id=rr_cfg.id,
            tuning_name=prof.key,
            tuning_env=dict(prof.tuning),
        )
    except ValueError:
        pass

    if key in MODEL_REGISTRY:
        raise ValueError(
            "Target is an embedding model key. Gunakan --mode embedding untuk benchmark embedding."
        )

    label, model_id = _resolve_reranker_spec(raw)
    return BenchmarkTarget(
        label=label,
        model_id=model_id,
        tuning_name=None,
        tuning_env={},
    )


def _resolve_collab_benchmark_target(name: str) -> CollabBenchmarkTarget:
    raw = name.strip()
    key = raw.lower()

    try:
        prof = get_profile_config(key)
        if not prof.reranker_model_key:
            raise ValueError(f"Profile '{prof.key}' tidak punya reranker_model_key.")

        embed_id = MODEL_REGISTRY[prof.model_key].id
        rr_id = get_reranker_model_config(prof.reranker_model_key).id

        return CollabBenchmarkTarget(
            label=f"profile:{prof.key}",
            embedding_model_id=embed_id,
            reranker_model_id=rr_id,
            tuning_name=prof.key,
            tuning_env=dict(prof.tuning),
        )
    except ValueError:
        pass

    if "+" in raw:
        left, right = raw.split("+", 1)
        _, embed_id = _resolve_embedding_spec(left)
        _, rr_id = _resolve_reranker_spec(right)

        return CollabBenchmarkTarget(
            label=f"pair:{left.strip()}+{right.strip()}",
            embedding_model_id=embed_id,
            reranker_model_id=rr_id,
            tuning_name=None,
            tuning_env={},
        )

    raise ValueError(
        "Unknown collab target. Use profile key (fast/accurate/precise) or explicit pair "
        "<embedding_model>+<reranker_model>."
    )


# Benchmark result printers

def _print_benchmark_result(result) -> None:
    t = result.target
    print(f"\n== Benchmark: {t.label} ==")
    print(f"Model ID:      {t.model_id}")
    print(f"Tuning:        {t.tuning_name or '-'}")

    if not result.success:
        print("Status:        FAILED")
        print(f"Error:         {result.error}")
        return

    print("Status:        OK")
    print(f"Samples:       {result.sample_count}")
    print(f"Est. tokens:   {result.total_tokens}")
    print(f"Batch size:    {result.batch_size}")
    print(f"Max batch tok: {result.max_batch_tokens}")
    print(f"Max tok/text:  {result.max_tokens_per_text}")
    print(f"Repeats:       {result.repeats}")
    print(f"Embedding dim: {result.embedding_dim}")
    print(f"Load time:     {result.load_sec:.2f}s")
    print(
        f"Run time:      avg={result.avg_run_sec:.2f}s  "
        f"best={result.best_run_sec:.2f}s  worst={result.worst_run_sec:.2f}s"
    )
    print(f"Throughput:    {result.texts_per_sec:.1f} texts/s  |  {result.tokens_per_sec:.0f} tok/s")

    if result.active_mem_mb is not None:
        print(f"MLX active:    {result.active_mem_mb:.1f} MB")
    if result.peak_mem_mb is not None:
        print(f"MLX peak:      {result.peak_mem_mb:.1f} MB")


def _print_reranker_benchmark_result(result) -> None:
    t = result.target
    print(f"\n== Reranker Benchmark: {t.label} ==")
    print(f"Model ID:      {t.model_id}")
    print(f"Tuning:        {t.tuning_name or '-'}")

    if not result.success:
        print("Status:        FAILED")
        print(f"Error:         {result.error}")
        return

    print("Status:        OK")
    print(f"Batch size:    {result.batch_size}")
    print(f"Queries:       {result.n_queries}")
    print(f"Docs/query:    {result.n_docs}")
    print(f"Repeats:       {result.repeats}")
    print(f"Load time:     {result.load_sec:.2f}s")
    print(
        f"Run time:      avg={result.avg_run_sec:.2f}s  "
        f"best={result.best_run_sec:.2f}s  worst={result.worst_run_sec:.2f}s"
    )
    print(f"Throughput:    {result.docs_per_sec:.1f} docs/s")

    if result.active_mem_mb is not None:
        print(f"MLX active:    {result.active_mem_mb:.1f} MB")
    if result.peak_mem_mb is not None:
        print(f"MLX peak:      {result.peak_mem_mb:.1f} MB")


def _print_collab_benchmark_result(result) -> None:
    t = result.target
    print(f"\n== Collaboration Benchmark: {t.label} ==")
    print(f"Embedding ID:  {t.embedding_model_id}")
    print(f"Reranker ID:   {t.reranker_model_id}")
    print(f"Tuning:        {t.tuning_name or '-'}")

    if not result.success:
        print("Status:        FAILED")
        print(f"Error:         {result.error}")
        return

    print("Status:        OK")
    print(f"Corpus size:   {result.corpus_size}")
    print(f"Queries:       {result.query_count}")
    print(f"Eval queries:  {result.eval_query_count}")
    print(f"retrieve_k:    {result.retrieve_k}")
    print(f"top_n:         {result.top_n}")
    print(f"Repeats:       {result.repeats}")
    print(f"Embed batch:   {result.embedding_batch_size}")
    print(f"Rerank batch:  {result.reranker_batch_size}")
    print(f"Load time:     embed={result.embed_load_sec:.2f}s  reranker={result.reranker_load_sec:.2f}s")
    print(f"Setup time:    {result.setup_sec:.2f}s (corpus embedding, not in measured loop)")
    print(
        f"Run time:      avg={result.avg_run_sec:.2f}s  "
        f"best={result.best_run_sec:.2f}s  worst={result.worst_run_sec:.2f}s"
    )
    print(
        f"Throughput:    {result.searches_per_sec:.2f} searches/s  "
        f"|  {result.rerank_docs_per_sec:.1f} rerank-docs/s"
    )

    print("Quality (seeded qrels):")
    print(
        f"  Retrieval:   MRR@5={result.mrr_at_5_retrieval:.4f}  "
        f"Recall@12={result.recall_at_12_retrieval:.4f}  "
        f"NDCG@12={result.ndcg_at_12_retrieval:.4f}"
    )
    print(
        f"  +Rerank:     MRR@5={result.mrr_at_5_rerank:.4f}  "
        f"Recall@12={result.recall_at_12_rerank:.4f}  "
        f"NDCG@12={result.ndcg_at_12_rerank:.4f}"
    )

    if result.eval_query_count > 0:
        delta_mrr = result.mrr_at_5_rerank - result.mrr_at_5_retrieval
        delta_recall = result.recall_at_12_rerank - result.recall_at_12_retrieval
        delta_ndcg = result.ndcg_at_12_rerank - result.ndcg_at_12_retrieval
        print(
            f"  Delta:       MRR@5={delta_mrr:+.4f}  "
            f"Recall@12={delta_recall:+.4f}  "
            f"NDCG@12={delta_ndcg:+.4f}"
        )
    else:
        print("  Note:        Tidak ada eval query yang match seeded qrels.")

    if result.active_mem_mb is not None:
        print(f"MLX active:    {result.active_mem_mb:.1f} MB")
    if result.peak_mem_mb is not None:
        print(f"MLX peak:      {result.peak_mem_mb:.1f} MB")


def cmd_profile_benchmark(args: argparse.Namespace) -> None:
    """Benchmark embedding-only, reranker-only, or collaborative search pipeline."""
    workspace, _workspace_source = resolve_workspace_root(args.workspace)
    mode = args.mode

    # Backward compatibility for old flag.
    if getattr(args, "reranker", False):
        mode = "reranker"

    cfg = load_config()

    if not args.target:
        if mode == "embedding":
            default_target = cfg.get("profile", DEFAULT_PROFILE_KEY)
        elif mode == "reranker":
            default_target = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)
        else:  # collab
            default_target = cfg.get("profile", DEFAULT_PROFILE_KEY)
        targets_raw = [default_target]
    else:
        targets_raw = args.target

    repeats = max(1, args.repeats)
    samples = max(1, args.samples)
    max_chars = max(128, args.max_chars)
    requested_query_count = max(1, args.queries)

    if mode == "embedding":
        embedding_targets: list[BenchmarkTarget] = []
        for raw in targets_raw:
            try:
                t = _resolve_embedding_benchmark_target(raw)
            except ValueError as e:
                print(f"Target '{raw}' invalid: {e}")
                sys.exit(1)

            if args.skip_uncached and not is_cached(t.model_id):
                print(f"Skip '{raw}' (not cached): {t.model_id}")
                continue

            embedding_targets.append(t)

        if not embedding_targets:
            print("No embedding benchmark target to run.")
            return

        texts = collect_benchmark_texts(workspace=workspace, target_samples=samples, max_chars=max_chars)

        print("Embedding benchmark configuration:")
        print(f"  workspace:   {workspace}")
        print(f"  snippets:    {len(texts)}")
        print(f"  max_chars:   {max_chars}")
        print(f"  repeats:     {repeats}")
        if args.batch_size is not None:
            print(f"  batch_size:  {args.batch_size} (forced)")
        else:
            print("  batch_size:  auto (target/profile default)")

        results = []
        for t in embedding_targets:
            print(f"\nRunning embedding benchmark for {t.label} ...")
            res = run_embedding_benchmark(
                target=t,
                texts=texts,
                repeats=repeats,
                batch_size_override=args.batch_size,
            )
            _print_benchmark_result(res)
            results.append(res)

        ok_results = [r for r in results if r.success]
        if len(ok_results) >= 2:
            ranked = sorted(ok_results, key=lambda r: r.tokens_per_sec, reverse=True)
            print("\n== Embedding Ranking by tokens/s ==")
            for i, r in enumerate(ranked, 1):
                print(
                    f"{i}. {r.target.label:18s}  "
                    f"{r.tokens_per_sec:10.0f} tok/s  "
                    f"avg={r.avg_run_sec:.2f}s"
                )
        return

    if mode == "reranker":
        reranker_targets: list[BenchmarkTarget] = []
        for raw in targets_raw:
            try:
                t = _resolve_reranker_benchmark_target(raw)
            except ValueError as e:
                print(f"Target '{raw}' invalid: {e}")
                sys.exit(1)

            if args.skip_uncached and not is_cached(t.model_id):
                print(f"Skip '{raw}' (not cached): {t.model_id}")
                continue

            reranker_targets.append(t)

        if not reranker_targets:
            print("No reranker benchmark target to run.")
            return

        doc_count = max(1, args.reranker_docs)
        documents = collect_benchmark_texts(
            workspace=workspace,
            target_samples=doc_count,
            max_chars=max_chars,
        )
        query_count = requested_query_count
        queries = collect_benchmark_queries(query_count)

        print("Reranker benchmark configuration:")
        print(f"  workspace:   {workspace}")
        print(f"  queries:     {len(queries)}")
        print(f"  docs:        {len(documents)}")
        print(f"  max_chars:   {max_chars}")
        print(f"  repeats:     {repeats}")
        if args.rerank_batch_size is not None:
            print(f"  batch_size:  {args.rerank_batch_size} (forced)")
        else:
            print("  batch_size:  auto (target/profile default)")

        results = []
        for t in reranker_targets:
            print(f"\nRunning reranker benchmark for {t.label} ...")
            res = run_reranker_benchmark(
                target=t,
                repeats=repeats,
                batch_size_override=args.rerank_batch_size,
                queries=queries,
                documents=documents,
            )
            _print_reranker_benchmark_result(res)
            results.append(res)

        ok = [r for r in results if r.success]
        if len(ok) >= 2:
            ranked = sorted(ok, key=lambda r: r.docs_per_sec, reverse=True)
            print("\n== Reranker Ranking by docs/s ==")
            for i, r in enumerate(ranked, 1):
                print(
                    f"{i}. {r.target.label:22s}  "
                    f"{r.docs_per_sec:10.1f} docs/s  "
                    f"avg={r.avg_run_sec:.2f}s"
                )
        return

    # mode == collab
    collab_targets: list[CollabBenchmarkTarget] = []
    for raw in targets_raw:
        try:
            t = _resolve_collab_benchmark_target(raw)
        except ValueError as e:
            print(f"Target '{raw}' invalid: {e}")
            sys.exit(1)

        if args.skip_uncached:
            if not is_cached(t.embedding_model_id) or not is_cached(t.reranker_model_id):
                print(
                    f"Skip '{raw}' (not cached): "
                    f"embed={t.embedding_model_id}  reranker={t.reranker_model_id}"
                )
                continue

        collab_targets.append(t)

    if not collab_targets:
        print("No collaboration benchmark target to run.")
        return

    texts = collect_benchmark_texts(workspace=workspace, target_samples=samples, max_chars=max_chars)

    min_eval_queries = max(1, env_int("CODE_BENCHMARK_MIN_EVAL_QUERIES", 100))
    query_count = max(requested_query_count, min_eval_queries)
    if query_count != requested_query_count:
        print(
            f"  note: force queries -> {query_count} "
            f"(minimum eval stability requirement)"
        )

    queries = collect_benchmark_queries(query_count)
    retrieve_k = max(1, min(args.retrieve_k, 50))
    top_n = max(1, min(args.top_n, retrieve_k))

    print("Collaboration benchmark configuration:")
    print(f"  workspace:   {workspace}")
    print(f"  corpus:      {len(texts)}")
    print(f"  queries:     {len(queries)}")
    print(f"  retrieve_k:  {retrieve_k}")
    print(f"  top_n:       {top_n}")
    print(f"  max_chars:   {max_chars}")
    print(f"  repeats:     {repeats}")
    if args.batch_size is not None:
        print(f"  embed_batch: {args.batch_size} (forced)")
    else:
        print("  embed_batch: auto (target/profile default)")
    if args.rerank_batch_size is not None:
        print(f"  rerank_batch:{args.rerank_batch_size} (forced)")
    else:
        print("  rerank_batch:auto (target/profile/default)")

    results = []
    for t in collab_targets:
        print(f"\nRunning collaboration benchmark for {t.label} ...")
        res = run_collab_benchmark(
            target=t,
            texts=texts,
            queries=queries,
            repeats=repeats,
            retrieve_k=retrieve_k,
            top_n=top_n,
            embedding_batch_size_override=args.batch_size,
            reranker_batch_size_override=args.rerank_batch_size,
        )
        _print_collab_benchmark_result(res)
        results.append(res)

    ok_results = [r for r in results if r.success]
    if len(ok_results) >= 2:
        ranked = sorted(ok_results, key=lambda r: r.searches_per_sec, reverse=True)
        print("\n== Collaboration Ranking by searches/s ==")
        for i, r in enumerate(ranked, 1):
            print(
                f"{i}. {r.target.label:22s}  "
                f"{r.searches_per_sec:8.2f} searches/s  "
                f"avg={r.avg_run_sec:.2f}s"
            )


def cmd_profile_set(args: argparse.Namespace) -> None:
    """Set active profile and align active model/reranker to profile recommendation."""
    key = args.name.strip().lower()

    try:
        profile = get_profile_config(key)
    except ValueError as e:
        print(str(e))
        sys.exit(1)

    cfg = load_config()
    old_profile = cfg.get("profile", DEFAULT_PROFILE_KEY)
    old_model = cfg.get("model_key", DEFAULT_MODEL_KEY)
    old_rr = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)

    cfg["profile"] = profile.key
    cfg["model_key"] = profile.model_key
    cfg.pop("custom_model_id", None)
    if profile.reranker_model_key:
        cfg["reranker_model_key"] = profile.reranker_model_key
        cfg["reranker_enabled"] = True
    save_config(cfg)

    print(f"\u2713 Active profile set to: {profile.key} ({profile.name})")
    print(f"  Embed model: {profile.model_key}")
    if profile.reranker_model_key:
        print(f"  Reranker:    {profile.reranker_model_key}")

    if old_model != profile.model_key:
        print(f"  Embed updated: {old_model} -> {profile.model_key}")
    if profile.reranker_model_key and old_rr != profile.reranker_model_key:
        print(f"  Reranker updated: {old_rr} -> {profile.reranker_model_key}")
    if old_profile != profile.key:
        print(f"  Profile updated: {old_profile} -> {profile.key}")

    model_id = MODEL_REGISTRY[profile.model_key].id
    print(f"  Embed cache: {'\u2713' if is_cached(model_id) else '\u2717'}")
    if profile.reranker_model_key:
        rr_conf = get_reranker_model_config(profile.reranker_model_key)
        rr_cached = is_cached(rr_conf.id)
        print(f"  Reranker cache: {'\u2713' if rr_cached else '\u2717'}")

    print("\nNote: profile tuning is auto-applied when running 'code-radar serve'.")
    print("      Existing environment variables still take precedence.")
    print("\nImportant: after changing model/profile, clear + re-sync DB for best search quality:")
    print("  uv run code-radar db clear --force")
    print("  # then run sync_workspace from MCP client")
