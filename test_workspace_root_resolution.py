from __future__ import annotations

import os
import tempfile
from pathlib import Path

from code_radar.workspace import resolve_sync_directory, resolve_workspace_root


def test_resolve_workspace_root_uses_cwd_when_no_cli_or_env() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        cwd = Path(tmp).resolve()
        prev = Path.cwd()
        os.chdir(cwd)
        try:
            resolved, source = resolve_workspace_root(env={})
        finally:
            os.chdir(prev)

    assert resolved == str(cwd)
    assert source == "cwd"


def test_resolve_workspace_root_env_absolute_path() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        workspace = Path(tmp).resolve()
        resolved, source = resolve_workspace_root(env={"WORKSPACE_PATH": str(workspace)})

    assert resolved == str(workspace)
    assert source == "env:WORKSPACE_PATH"


def test_resolve_workspace_root_unresolved_placeholder_raises() -> None:
    try:
        resolve_workspace_root(env={"MCP_WORKSPACE_ROOT": "${workspaceFolder}"})
    except ValueError as exc:
        assert "unresolved placeholder" in str(exc)
    else:
        raise AssertionError("Expected ValueError for unresolved placeholder")


def test_resolve_workspace_root_missing_path_raises() -> None:
    try:
        resolve_workspace_root(directory_arg="/tmp/__code_radar_missing_workspace__")
    except ValueError as exc:
        assert "Workspace path not found" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing workspace path")


def test_resolve_sync_directory_deduplicates_workspace_name() -> None:
    configured_root = Path("/tmp/projects/gateway").resolve()

    resolved = resolve_sync_directory(configured_root, "gateway")

    assert resolved == configured_root


def test_resolve_sync_directory_keeps_nested_relative_paths() -> None:
    configured_root = Path("/tmp/projects/gateway").resolve()

    resolved = resolve_sync_directory(configured_root, "src")

    assert resolved == (configured_root / "src").resolve()


if __name__ == "__main__":
    test_resolve_workspace_root_uses_cwd_when_no_cli_or_env()
    test_resolve_workspace_root_env_absolute_path()
    test_resolve_workspace_root_unresolved_placeholder_raises()
    test_resolve_workspace_root_missing_path_raises()
    test_resolve_sync_directory_deduplicates_workspace_name()
    test_resolve_sync_directory_keeps_nested_relative_paths()
    print("\nALL PASS.")
