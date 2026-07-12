"""Atom Embedding MCP server — split into sub-modules.

Public API (re-exported for backward compatibility):
    - ``mcp``: the FastMCP application instance
    - ``configure()``: initialise engine, store, reranker, hash cache
    - ``semantic_search()``: semantic search tool function
    - ``smart_search()``: hybrid search tool function
    - ``read_full_file()``: file-read tool function
"""

from .app import mcp
from .state import configure, register_shutdown_hooks, shutdown_runtime_sync
from .tools import read_full_file, semantic_search, smart_search

__all__ = [
    "configure",
    "mcp",
    "register_shutdown_hooks",
    "shutdown_runtime_sync",
    "read_full_file",
    "semantic_search",
    "smart_search",
]
