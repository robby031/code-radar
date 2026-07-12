"""
Entry point for ``python -m code_radar``.

Without arguments, starts the MCP server (backward-compatible with the
original ``main.py`` behaviour).

With arguments, delegates to the CLI::

    python -m code_radar model ls
    python -m code_radar db status
    ...
"""

import sys

from code_radar.envvars import get_env
from code_radar.logging import get_logger, set_level
from code_radar.workspace import ensure_safe_workspace_root, resolve_workspace_root

log = get_logger("main")


def _run_server() -> None:
    """Start the MCP server (original behaviour)."""
    from code_radar.chroma import ChromaStore
    from code_radar.config import load_config, resolve_db_path
    from code_radar.engine import EmbeddingEngine
    from code_radar.profiles import apply_profile_env_defaults
    from code_radar.server import (
        configure,
        mcp,
        register_shutdown_hooks,
        shutdown_runtime_sync,
    )

    if (get_env("CODE_LOG_LEVEL", "") or "").upper() == "DEBUG":
        set_level("DEBUG")

    cfg = load_config()
    db_path = resolve_db_path(cfg)
    profile_key = cfg.get("profile")
    if isinstance(profile_key, str):
        apply_profile_env_defaults(profile_key)

    try:
        workspace, workspace_source = resolve_workspace_root()
        ensure_safe_workspace_root(workspace, source=workspace_source)
    except ValueError as exc:
        log.error("%s", exc)
        raise SystemExit(2) from exc

    engine = EmbeddingEngine()  # reads active model from config
    log.info(
        "Starting Atom Embedding Engine  |  workspace=%s  |  source=%s  |  model=%s",
        workspace,
        workspace_source,
        engine.model_name,
    )
    log.info("Embedding engine will be lazy-loaded on first tool call")
    store = ChromaStore(path=db_path, lazy_connect=True)
    configure(
        engine,
        store,
        root=workspace,
        workspace_id=None,
        workspace_source=workspace_source,
    )
    register_shutdown_hooks()

    try:
        mcp.run()
    finally:
        shutdown_runtime_sync()


def main() -> None:
    # If arguments are passed, delegate to CLI.
    # Otherwise, start the server for backward compatibility.
    if len(sys.argv) > 1:
        from code_radar.cli import main as cli_main
        cli_main()
    else:
        _run_server()


if __name__ == "__main__":
    main()
