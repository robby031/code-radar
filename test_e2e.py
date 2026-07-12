import os
import sys
import time
from pathlib import Path

import mlx.core as mx

from code_radar.chroma import ChromaStore
from code_radar.codebase import sync_workspace
from code_radar.engine import EmbeddingEngine
from code_radar.server import configure, read_full_file, semantic_search, smart_search


def _ensure_hit(result: str, label: str) -> None:
    if not result or "No results found" in result:
        raise AssertionError(f"{label}: no results found\n{result}")


def _print_block(title: str) -> None:
    print(f"\n=== {title} ===")


def _resolve_engine_file() -> str:
    """Resolve engine module path across legacy/new layout."""
    candidates = [
        "code_radar/engine.py",  # legacy single-file layout
        "code_radar/engine/_engine.py",  # current package layout
    ]
    for rel in candidates:
        if Path(rel).is_file():
            return rel
    raise FileNotFoundError(f"None of engine file candidates exist: {candidates}")


ENGINE_FILEPATH = _resolve_engine_file()


# Fresh DB
if os.path.exists("./chroma_data"):
    import shutil

    shutil.rmtree("./chroma_data")

engine = EmbeddingEngine()
engine.load()

mem_before = mx.get_active_memory() / 1e6
print(f"MEMORY after model load: {mem_before:.0f}MB")

store = ChromaStore(path="./chroma_data")
configure(engine, store, root=".")

# SYNC
_print_block("SYNC WORKSPACE")
t0 = time.time()
added, unchanged, deleted = sync_workspace(".", engine, store)
elapsed = time.time() - t0

mem_sync = mx.get_active_memory() / 1e6
mem_peak = mx.get_peak_memory() / 1e6
print(f"added={added} unchanged={unchanged} deleted={deleted}")
print(f"elapsed={elapsed:.1f}s")
print(f"MEMORY active={mem_sync:.0f}MB peak={mem_peak:.0f}MB")

if added == 0:
    print("FAIL: no chunks added")
    sys.exit(1)


# SEMANTIC SEARCH
_print_block("SEMANTIC SEARCH")
semantic_1 = semantic_search("embedding function", n=5, filepath=ENGINE_FILEPATH)
print(semantic_1[:1000])
_ensure_hit(semantic_1, "semantic_1")
assert ENGINE_FILEPATH in semantic_1, (
    f"{ENGINE_FILEPATH} not found in semantic_1:\n{semantic_1}"
)
assert "score=" in semantic_1, f"semantic_1 missing score:\n{semantic_1}"

semantic_2 = semantic_search("upsert batch", n=5, filepath="code_radar/chroma.py")
print("semantic_2:", semantic_2[:500])
_ensure_hit(semantic_2, "semantic_2")
assert "code_radar/chroma.py" in semantic_2, (
    f"chroma.py not found in semantic_2:\n{semantic_2}"
)


# SMART SEARCH (HYBRID)
_print_block("SMART SEARCH (HYBRID)")
smart = smart_search(
    query="refactor upsert logic regex: `def\\s+upsert\\s*\\(`",
    n=5,
    filepath="code_radar/chroma.py",
)
print(smart[:1200])
_ensure_hit(smart, "smart_search")
assert "phase1:" in smart and "phase2:" in smart and "phase3:" in smart, (
    f"smart_search phase summary missing:\n{smart}"
)
assert "code_radar/chroma.py" in smart, f"smart_search missing chroma.py:\n{smart}"


# READ FILE + SAFETY
_print_block("READ FULL FILE")
content = read_full_file(ENGINE_FILEPATH)
assert "embed_text" in content, f"{ENGINE_FILEPATH} content missing embed_text:\n{content[:500]}"
assert "def load" in content, f"{ENGINE_FILEPATH} content missing def load:\n{content[:500]}"
print(f"{ENGINE_FILEPATH}: {len(content.splitlines())} lines OK")

blocked = read_full_file("../etc/passwd")
assert "denied" in blocked.lower(), f"path traversal should be denied:\n{blocked}"


# VERIFY CHUNKS
_print_block("VERIFY CHUNKS")
total_chunks = store.count()
print(f"Total chunks in DB: {total_chunks}")
assert total_chunks > 0, "DB should contain chunks after sync"

all_data = store.get_all_chunks()
ids = all_data.get("ids", [])
assert ids, "Chunk IDs should not be empty"
assert len(ids) == len(set(ids)), f"Duplicate IDs found! {len(ids)} != {len(set(ids))}"
print("PASS: No duplicate IDs")


# MEMORY (optional hard limit)
mem_final = mx.get_active_memory() / 1e6
print(f"\nMEMORY final: {mem_final:.0f}MB")

max_mem_mb_raw = os.environ.get("CODE_E2E_MAX_MEM_MB", "0").strip()
try:
    max_mem_mb = int(max_mem_mb_raw)
except ValueError:
    max_mem_mb = 0

if max_mem_mb > 0:
    assert mem_final < max_mem_mb, (
        f"Memory too high: {mem_final:.0f}MB >= CODE_E2E_MAX_MEM_MB={max_mem_mb}MB"
    )
    print(f"PASS: Memory under {max_mem_mb}MB")
else:
    print("INFO: memory hard-limit skipped (set CODE_E2E_MAX_MEM_MB to enforce)")

    print("\nALL PASS.")
