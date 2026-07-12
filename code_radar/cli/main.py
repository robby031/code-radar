"""Parser builder, entry point, and subparser registration."""

import argparse

from code_radar.logging import set_level
from code_radar.cli.db import (
    cmd_db_status,
    cmd_db_clear,
    cmd_db_reset,
)
from code_radar.cli.model import (
    cmd_model_ls,
    cmd_model_set,
    cmd_model_pull,
    cmd_model_rm,
)
from code_radar.cli.reranker import (
    cmd_reranker_ls,
    cmd_reranker_current,
    cmd_reranker_set,
    cmd_reranker_pull,
    cmd_reranker_rm,
)
from code_radar.cli.profile import (
    cmd_profile_ls,
    cmd_profile_current,
    cmd_profile_set,
    cmd_profile_benchmark,
)
from code_radar.cli.system import (
    cmd_info,
    cmd_version,
    cmd_serve,
)


def _top(name, sub, help_text, func):
    p = sub.add_parser(name, help=help_text)
    p.set_defaults(func=func)
    return p


# ---------------------------------------------------------------------------
# db subcommand builders
# ---------------------------------------------------------------------------

def _db_status(sub):
    p = sub.add_parser("status", help="Show database statistics")
    p.set_defaults(func=cmd_db_status)


def _db_clear(sub):
    p = sub.add_parser("clear", help="Delete all embeddings from the collection")
    p.add_argument("-f", "--force", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_db_clear)


def _db_reset(sub):
    p = sub.add_parser("reset", help="Delete and recreate the entire database directory")
    p.add_argument("-f", "--force", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_db_reset)


# ---------------------------------------------------------------------------
# model subcommand builders
# ---------------------------------------------------------------------------

def _model_ls(sub):
    p = sub.add_parser("ls", help="List all registered models (embedding + reranker)")
    p.set_defaults(func=cmd_model_ls)


def _model_set(sub):
    p = sub.add_parser("set", help="Set the active embedding model by registry key")
    p.add_argument("name", type=str, help="Registry key (e.g. qwen3-4b-4bit) or full HF ID")
    p.set_defaults(func=cmd_model_set)


def _model_pull(sub):
    p = sub.add_parser("pull", help="Download a model by registry key or full HF ID")
    p.add_argument("name", type=str, help="Registry key (e.g. qwen3-4b-4bit) or full HF model ID")
    p.set_defaults(func=cmd_model_pull)


def _model_rm(sub):
    p = sub.add_parser("rm", help="Remove a cached model")
    p.add_argument("name", type=str, help="Registry key or full HF model ID")
    p.add_argument("-f", "--force", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_model_rm)


# ---------------------------------------------------------------------------
# reranker subcommand builders
# ---------------------------------------------------------------------------

def _reranker_ls(sub):
    p = sub.add_parser("ls", help="List all available reranker models from the registry")
    p.set_defaults(func=cmd_reranker_ls)


def _reranker_current(sub):
    p = sub.add_parser("current", help="Show the active reranker model")
    p.set_defaults(func=cmd_reranker_current)


def _reranker_set(sub):
    p = sub.add_parser("set", help="Set the active reranker model by registry key")
    p.add_argument("name", type=str, help="Registry key (e.g. reranker-0.6b-4bit) or full HF ID")
    p.set_defaults(func=cmd_reranker_set)


def _reranker_pull(sub):
    p = sub.add_parser("pull", help="Download a reranker model by registry key or full HF ID")
    p.add_argument("name", type=str, help="Registry key (e.g. reranker-0.6b-4bit) or full HF model ID")
    p.set_defaults(func=cmd_reranker_pull)


def _reranker_rm(sub):
    p = sub.add_parser("rm", help="Remove a cached reranker model")
    p.add_argument("name", type=str, help="Registry key or full HF model ID")
    p.add_argument("-f", "--force", action="store_true", help="Skip confirmation")
    p.set_defaults(func=cmd_reranker_rm)


# ---------------------------------------------------------------------------
# profile subcommand builders
# ---------------------------------------------------------------------------

def _profile_ls(sub):
    p = sub.add_parser("ls", help="List available performance profiles")
    p.set_defaults(func=cmd_profile_ls)


def _profile_current(sub):
    p = sub.add_parser("current", help="Show active profile + runtime detail (embedding/reranker/tuning)")
    p.set_defaults(func=cmd_profile_current)


def _profile_set(sub):
    p = sub.add_parser("set", help="Set active profile (also sets recommended model)")
    p.add_argument("name", type=str, help="Profile key (e.g. fast, accurate)")
    p.set_defaults(func=cmd_profile_set)


def _profile_benchmark(sub):
    p = sub.add_parser(
        "benchmark",
        help="Run local benchmark: embedding-only, reranker-only, or collab pipeline",
    )
    p.add_argument(
        "--mode",
        choices=["embedding", "reranker", "collab"],
        default="embedding",
        help=(
            "Benchmark mode: embedding (default), reranker, or collab "
            "(query embedding + retrieval + rerank)."
        ),
    )
    p.add_argument(
        "--target",
        action="append",
        default=[],
        help=(
            "Benchmark target (repeatable). "
            "embedding mode: profile key/model key/full HF ID; "
            "reranker mode: profile key/reranker key/full HF ID; "
            "collab mode: profile key atau pair <embedding>+<reranker>."
        ),
    )
    p.add_argument(
        "--workspace",
        type=str,
        default=None,
        help="Workspace path used to sample code snippets (default: WORKSPACE_PATH or current dir)",
    )
    p.add_argument(
        "--samples",
        type=int,
        default=128,
        help="Number of sampled snippets for benchmark corpus (default: 128)",
    )
    p.add_argument(
        "--max-chars",
        type=int,
        default=1800,
        help="Max chars per sampled snippet/document (default: 1800)",
    )
    p.add_argument(
        "--queries",
        type=int,
        default=8,
        help=(
            "Number of benchmark queries (reranker/collab mode, default: 8). "
            "In collab mode, enforced minimum is controlled by CODE_BENCHMARK_MIN_EVAL_QUERIES (default 100)."
        ),
    )
    p.add_argument(
        "--reranker-docs",
        type=int,
        default=32,
        help="Number of benchmark documents for reranker-only mode (default: 32)",
    )
    p.add_argument(
        "--retrieve-k",
        type=int,
        default=12,
        help="Candidate count before rerank in collab mode (default: 12, hard cap 50)",
    )
    p.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Final top-N after reranking in collab mode (default: 5)",
    )
    p.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of measured runs after warmup (default: 3)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Force embedding batch size (embedding/collab mode)",
    )
    p.add_argument(
        "--rerank-batch-size",
        type=int,
        default=None,
        help="Force reranker batch size (reranker/collab mode)",
    )
    p.add_argument(
        "--skip-uncached",
        action="store_true",
        help="Skip targets that are not cached locally (avoid downloads)",
    )
    p.add_argument(
        "--reranker",
        action="store_true",
        help="[deprecated] same as --mode reranker",
    )
    p.set_defaults(func=cmd_profile_benchmark)


# ---------------------------------------------------------------------------
# parser builder & entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="code-radar",
        description="Atom Embedding Engine - manage embeddings, models, and database.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable DEBUG-level logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # db
    db = sub.add_parser("db", help="Database operations")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    _db_status(db_sub)
    _db_clear(db_sub)
    _db_reset(db_sub)

    # model
    mdl = sub.add_parser("model", help="Model operations")
    mdl_sub = mdl.add_subparsers(dest="model_command", required=True)
    _model_ls(mdl_sub)
    _model_set(mdl_sub)
    _model_pull(mdl_sub)
    _model_rm(mdl_sub)

    # reranker
    rrk = sub.add_parser("reranker", help="Reranker (cross-encoder) model operations")
    rrk_sub = rrk.add_subparsers(dest="reranker_command", required=True)
    _reranker_ls(rrk_sub)
    _reranker_current(rrk_sub)
    _reranker_set(rrk_sub)
    _reranker_pull(rrk_sub)
    _reranker_rm(rrk_sub)

    # profile
    prof = sub.add_parser("profile", help="Performance profile operations")
    prof_sub = prof.add_subparsers(dest="profile_command", required=True)
    _profile_ls(prof_sub)
    _profile_current(prof_sub)
    _profile_set(prof_sub)
    _profile_benchmark(prof_sub)

    # top-level
    _top("info", sub, "Show system information", cmd_info)
    _top("version", sub, "Show version", cmd_version)
    serve = _top("serve", sub, "Start the MCP server", cmd_serve)
    serve.add_argument(
        "directory", type=str, nargs="?",
        help="Workspace directory (default: current dir)",
    )
    serve.add_argument(
        "--workspace-id",
        type=str,
        default=None,
        help=(
            "Explicit workspace namespace for multi-project isolation. "
            "Overrides CODE_WORKSPACE_ID when provided."
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.debug:
        set_level("DEBUG")

    args.func(args)
