from __future__ import annotations

import hashlib
import os
import re
from collections.abc import Mapping
from pathlib import Path
from code_radar.constants import UNSAFE_WORKSPACE_PATHS

def sanitize_workspace_id(value: str, default: str = "default") -> str:
    raw = (value or "").strip().lower()
    cleaned = re.sub(r"[^a-z0-9]+", "_", raw).strip("_")
    return cleaned or default


def derive_workspace_id(root_path: str, explicit_id: str | None = None) -> str:
    """Derive a stable workspace id from root path or explicit override.

    Format: ``<slug>_<10-hex-hash>``.
    """
    if explicit_id and explicit_id.strip():
        return sanitize_workspace_id(explicit_id)

    resolved = Path(root_path).expanduser().resolve()
    slug = sanitize_workspace_id(resolved.name, default="workspace")
    digest = hashlib.sha1(str(resolved).encode("utf-8")).hexdigest()[:10]
    return f"{slug}_{digest}"


def build_collection_name(base: str, workspace_id: str) -> str:
    """Build bounded collection name for ChromaDB."""
    base_clean = sanitize_workspace_id(base, default="code_radars")
    ws_clean = sanitize_workspace_id(workspace_id)
    if len(ws_clean) > 40:
        ws_hash = hashlib.sha1(ws_clean.encode("utf-8")).hexdigest()[:10]
        ws_clean = f"{ws_clean[:29]}_{ws_hash}"
    return f"{base_clean}__{ws_clean}"


def build_scoped_db_filename(prefix: str, workspace_id: str, ext: str = ".db") -> str:
    """Build workspace-scoped SQLite filename."""
    return f"{prefix}__{sanitize_workspace_id(workspace_id)}{ext}"


def _contains_unresolved_placeholder(value: str) -> bool:
    """Detect placeholder token that was not expanded by MCP client.

    Examples:
    - ${workspaceFolder}
    - ${workspaceRoot}
    - ${SOME_VAR}
    - %SOME_VAR%
    """
    if re.search(r"\$\{[^}]+\}", value):
        return True
    if re.search(r"%[^%]+%", value):
        return True
    return False


def _expand_workspace_tokens(value: str) -> str:
    """Expand standard shell-like env/home tokens only.

    Note:
    - We intentionally do NOT expand editor-specific placeholders such as
      ``${workspaceFolder}`` to avoid ambiguous fallback behaviour.
    """
    replaced = os.path.expanduser(os.path.expandvars(value))

    # Best-effort %VAR% expansion (Windows-style env placeholders).
    replaced = re.sub(
        r"%([^%]+)%",
        lambda m: os.environ.get(m.group(1), m.group(0)),
        replaced,
    )
    return replaced


def _resolve_explicit_workspace(raw: str, source: str, cwd: Path) -> tuple[str, str]:
    expanded = _expand_workspace_tokens(raw)

    if _contains_unresolved_placeholder(expanded):
        raise ValueError(
            f"Invalid workspace path from {source}: unresolved placeholder in {raw!r}"
        )

    candidate = Path(expanded)
    if not candidate.is_absolute():
        candidate = (cwd / candidate).resolve()
    else:
        candidate = candidate.resolve()

    if candidate.is_dir():
        return str(candidate), source

    if candidate.is_file():
        return str(candidate.parent), f"{source} (parent of file path)"

    raise ValueError(f"Workspace path not found from {source}: {candidate}")


def resolve_workspace_root(
    directory_arg: str | None = None,
    env: Mapping[str, str] | None = None,
) -> tuple[str, str]:
    """Resolve workspace root deterministically.

    Priority:
    1. explicit CLI arg (serve [directory])
    2. env: WORKSPACE_PATH / MCP_WORKSPACE_ROOT / PROJECT_ROOT
    3. current working directory (single source of truth for editor-integrated runs)

    For explicit CLI/env values, invalid paths raise ``ValueError``.
    """
    env_map = os.environ if env is None else env
    cwd = Path.cwd().resolve()

    if directory_arg and directory_arg.strip():
        return _resolve_explicit_workspace(directory_arg.strip(), "cli", cwd)

    for key in ("WORKSPACE_PATH", "MCP_WORKSPACE_ROOT", "PROJECT_ROOT"):
        val = env_map.get(key)
        if val and val.strip():
            return _resolve_explicit_workspace(val.strip(), f"env:{key}", cwd)

    return str(cwd), "cwd"


def resolve_sync_directory(configured_root: Path, directory: str | None) -> Path:
    if directory is None or not directory.strip():
        return configured_root

    requested = Path(directory.strip())
    if requested.is_absolute():
        return requested.resolve()

    if requested == Path(configured_root.name):
        return configured_root

    return (configured_root / requested).resolve()


def is_unsafe_workspace_root(path: str) -> bool:
    """Return True for dangerous top-level paths that should not be indexed."""
    resolved = Path(path).expanduser().resolve()

    # Reject filesystem root on all platforms.
    if resolved == Path(resolved.anchor):
        return True

    return str(resolved) in UNSAFE_WORKSPACE_PATHS


def ensure_safe_workspace_root(
    path: str,
    *,
    source: str,
    env: Mapping[str, str] | None = None,
) -> None:
    """Fail-fast guard against indexing entire/system filesystem trees.

    Set CODE_ALLOW_UNSAFE_WORKSPACE=1 to bypass this guard intentionally.
    """
    env_map = os.environ if env is None else env
    allow_unsafe = (env_map.get("CODE_ALLOW_UNSAFE_WORKSPACE", "") or "").strip().lower()
    if allow_unsafe in {"1", "true", "yes", "y", "on"}:
        return

    if not is_unsafe_workspace_root(path):
        return

    raise ValueError(
        "Unsafe workspace root detected: "
        f"{Path(path).expanduser().resolve()} (source={source}). "
        "Refusing to index system-wide directories. "
        "Provide an explicit project path (e.g. code-radar serve /path/to/project). "
        "If you really want this, set CODE_ALLOW_UNSAFE_WORKSPACE=1."
    )
