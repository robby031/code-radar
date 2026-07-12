"""
Atom Embedding CLI - manage database, models, and server from the terminal.

Usage::

    code-radar [--debug] <command> [<args>...]

Commands
--------

**Database (db)**

    db status    Show database statistics (chunks, disk usage, collections)
    db clear     Delete all embeddings from the collection
    db reset     Delete and recreate the entire database directory

**Models (model)**

    model ls        List all registered models (embedding + reranker)
    model set       Set the active embedding model by registry key
    model pull      Download a model by registry key (or full HF ID)
    model rm        Remove a cached model

**Reranker (reranker)** — cross-encoder untuk re-ranking hasil search:

    reranker ls        List available reranker models from registry
    reranker current   Show the active reranker model
    reranker set       Set the active reranker model
    reranker pull      Download a reranker model by key (or full HF ID)
    reranker rm        Remove a cached reranker model

**Profiles (profile)**:

    profile ls        List available runtime profiles
    profile current   Show active profile + runtime detail (embedding + reranker + tuning)
    profile set       Set active profile (also sets recommended model)
    profile benchmark Run local benchmark (embedding-only, reranker-only, atau collab)

**System**

    info        Show system information (version, model, DB, platform)
    version     Show version number
    serve       Start the MCP server
"""

from code_radar.cli.main import main, build_parser

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
    cmd_profile_benchmark,
    cmd_profile_set,
)
from code_radar.cli.system import (
    cmd_info,
    cmd_version,
    cmd_serve,
)

__all__ = [
    "main",
    "build_parser",
    "cmd_db_status",
    "cmd_db_clear",
    "cmd_db_reset",
    "cmd_model_ls",
    "cmd_model_set",
    "cmd_model_pull",
    "cmd_model_rm",
    "cmd_reranker_ls",
    "cmd_reranker_current",
    "cmd_reranker_set",
    "cmd_reranker_pull",
    "cmd_reranker_rm",
    "cmd_profile_ls",
    "cmd_profile_current",
    "cmd_profile_benchmark",
    "cmd_profile_set",
    "cmd_info",
    "cmd_version",
    "cmd_serve",
]
