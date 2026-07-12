"""System CLI commands: info, version, serve."""

import argparse
import sys
from pathlib import Path

from code_radar.cli.helpers import (
    print_size,
    dir_size,
    hf_cache_name,
    is_cached,
    HF_HUB_CACHE,
)
from code_radar.config import (
    CONFIG_FILE,
    load_config,
    resolve_db_path,
    resolve_model_id,
    resolve_reranker_model_id,
)
from code_radar.envvars import get_env
from code_radar.logging import get_logger, set_level
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
    apply_profile_env_defaults,
    get_profile_config,
)
from code_radar.workspace import ensure_safe_workspace_root, resolve_workspace_root

log = get_logger("cli")


def cmd_info(_args: argparse.Namespace) -> None:
    """Show system information."""
    from code_radar import __version__

    cfg = load_config()
    model_key = cfg.get("model_key", DEFAULT_MODEL_KEY)
    profile_key = cfg.get("profile", DEFAULT_PROFILE_KEY)

    print("  == Atom Embedding Engine ==")
    print(f"  Version:  {__version__}")
    print(f"  Python:   {sys.version.split()[0]}")
    print(f"  Platform: {sys.platform}\n")

    # Model
    if model_key in MODEL_REGISTRY:
        conf = get_model_config(model_key)
        hf_id = conf.id
        print("  == Model ==")
        print(f"     Key:      {model_key}")
        print(f"     Name:     {conf.name}")
        print(f"     HF ID:    {hf_id}")
        print(f"     RAM:      {conf.ram_gb:.1f} GB  |  "
              f"speed: {conf.speed_tier}  |  "
              f"accuracy: {conf.accuracy_tier}")
    else:
        hf_id = resolve_model_id(model_key)
        print("  == Model ==")
        print(f"     Key:      {model_key}")
        print(f"     HF ID:    {hf_id}")

    cache_dir = HF_HUB_CACHE / hf_cache_name(hf_id)
    if cache_dir.exists():
        print_size("     Cached", dir_size(cache_dir))
    else:
        print("     Cached:  ✗ (will download on load)")
    print()

    print("  == Profile ==")
    try:
        prof = get_profile_config(profile_key)
        print(f"     Key:      {prof.key}")
        print(f"     Name:     {prof.name}")
        print(f"     Model:    {prof.model_key}")
    except ValueError:
        print(f"     Key:      {profile_key} (unknown)")
    print()

    # Reranker
    rr_key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)
    print("  == Reranker ==")
    if rr_key in RERANKER_REGISTRY:
        rr_conf = get_reranker_model_config(rr_key)
        rr_hf_id = rr_conf.id
        rr_cached = is_cached(rr_hf_id)
        print(f"     Key:      {rr_key}")
        print(f"     Name:     {rr_conf.name}")
        print(f"     HF ID:    {rr_hf_id}")
        print(f"     RAM:      {rr_conf.ram_gb:.1f} GB")
        print(f"     Speed:    {rr_conf.speed_tier}")
        print(f"     Cached:   {'\u2713' if rr_cached else '\u2717'}")
    else:
        rr_hf_id = resolve_reranker_model_id(rr_key)
        print(f"     Key:      {rr_key}")
        print(f"     HF ID:    {rr_hf_id}")
    print(f"     Enabled:  {cfg.get('reranker_enabled', True)}")
    print()

    # Database
    from chromadb import PersistentClient
    from chromadb.config import Settings

    path = resolve_db_path(cfg)
    chroma_path = Path(path)
    print("  == Database ==")
    print(f"     Path:   {path}")
    if chroma_path.exists():
        print_size("     Size", dir_size(chroma_path))
        try:
            client = PersistentClient(
                path=path,
                settings=Settings(anonymized_telemetry=False),
            )
            for col in client.list_collections():
                print(f"     Collection '{col.name}':  {col.count():,} chunks")
        except Exception as e:
            print(f"     Error:  {e}")
    else:
        print("     Status: empty (not yet created)")
    print()

    print(f"  Config:   {CONFIG_FILE}")
    print(f"  Log level: {get_env('CODE_LOG_LEVEL', 'INFO')}")
    print(f"  Profile:   {profile_key}")


def cmd_version(_args: argparse.Namespace) -> None:
    """Show version."""
    from code_radar import __version__
    print(f"code-radar v{__version__}")


def cmd_serve(args: argparse.Namespace) -> None:
    """Start the MCP server."""
    from code_radar.chroma import ChromaStore
    from code_radar.engine import EmbeddingEngine
    from code_radar.server import (
        configure,
        mcp,
        register_shutdown_hooks,
        shutdown_runtime_sync,
    )

    if args.debug:
        set_level("DEBUG")

    cfg = load_config()
    db_path = resolve_db_path(cfg)
    profile_key = cfg.get("profile", DEFAULT_PROFILE_KEY)

    if isinstance(profile_key, str):
        apply_profile_env_defaults(profile_key)

    try:
        workspace_root, workspace_source = resolve_workspace_root(args.directory)
        ensure_safe_workspace_root(workspace_root, source=workspace_source)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(2) from exc

    engine = EmbeddingEngine()  # reads model_key from config internally
    log.info(
        "Starting Atom Embedding Engine  |  workspace=%s  |  source=%s  |  profile=%s  |  model=%s",
        workspace_root,
        workspace_source,
        profile_key,
        engine.model_name,
    )
    log.info("DB path:     %s", db_path)
    if args.workspace_id:
        log.info("Workspace ID override (CLI): %s", args.workspace_id)
    log.info("Embedding engine will be lazy-loaded on first tool call")

    store = ChromaStore(path=db_path, lazy_connect=True)
    configure(
        engine,
        store,
        root=workspace_root,
        workspace_id=args.workspace_id,
        workspace_source=workspace_source,
    )
    register_shutdown_hooks()

    try:
        mcp.run()
    finally:
        shutdown_runtime_sync()
