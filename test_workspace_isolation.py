"""Tests for workspace-level data isolation."""

from __future__ import annotations

import tempfile
from pathlib import Path

from code_radar.chroma import ChromaStore
from code_radar.hash_kv import HashKVStore
from code_radar.sparse_index import SparseChunkIndex


def test_hash_kv_workspace_isolation() -> None:
    root = tempfile.mkdtemp()

    store_a = HashKVStore(root, workspace_id="project-a")
    store_b = HashKVStore(root, workspace_id="project-b")

    store_a.put("main.py", "hash_a")
    store_b.put("main.py", "hash_b")

    assert store_a.get_all() == {"main.py": "hash_a"}
    assert store_b.get_all() == {"main.py": "hash_b"}
    assert store_a.db_path != store_b.db_path

    store_a.close()
    store_b.close()


def test_sparse_index_workspace_isolation() -> None:
    root = str(Path(tempfile.mkdtemp()) / "chroma")

    idx_a = SparseChunkIndex(root, workspace_id="alpha")
    idx_b = SparseChunkIndex(root, workspace_id="beta")

    idx_a.upsert_chunks(
        [
            {
                "id": "chunk-a",
                "filepath": "svc/a.py",
                "content": "def alpha_workspace_9812():\n    return 'alpha'\n",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
                "metadata": {"language": "py"},
            }
        ]
    )
    idx_b.upsert_chunks(
        [
            {
                "id": "chunk-b",
                "filepath": "svc/b.py",
                "content": "def beta_workspace_7741():\n    return 'beta'\n",
                "chunk_type": "function",
                "start_line": 1,
                "end_line": 2,
                "metadata": {"language": "py"},
            }
        ]
    )

    res_a = idx_a.search("alpha_workspace_9812", n=5)
    res_b = idx_b.search("beta_workspace_7741", n=5)

    assert res_a and res_a[0]["id"] == "chunk-a"
    assert res_b and res_b[0]["id"] == "chunk-b"

    # Cross-workspace leakage must not happen.
    leak_a = idx_a.search("beta_workspace_7741", n=5)
    leak_b = idx_b.search("alpha_workspace_9812", n=5)
    assert all(item["id"] != "chunk-b" for item in leak_a)
    assert all(item["id"] != "chunk-a" for item in leak_b)

    idx_a.close()
    idx_b.close()


def test_chroma_collection_name_scoped_by_workspace() -> None:
    root = tempfile.mkdtemp()

    store_a = ChromaStore(path=root, lazy_connect=True, workspace_id="project-a")
    store_b = ChromaStore(path=root, lazy_connect=True, workspace_id="project-b")

    assert store_a.collection_name != store_b.collection_name
    assert store_a.workspace_id == "project_a"
    assert store_b.workspace_id == "project_b"

    store_a.set_workspace("project-c")
    assert store_a.workspace_id == "project_c"
    assert store_a.collection_name.endswith("project_c")


if __name__ == "__main__":
    test_hash_kv_workspace_isolation()
    test_sparse_index_workspace_isolation()
    test_chroma_collection_name_scoped_by_workspace()
    print("\nALL PASS.")
