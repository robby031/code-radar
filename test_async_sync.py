import asyncio
import tempfile
from pathlib import Path

from code_radar.chroma import ChromaStore
from code_radar.codebase import async_sync_workspace
from code_radar.engine import EmbeddingEngine
from code_radar.state import HashCache, SyncProgress


def _make_workspace(tmp: Path) -> None:
    (tmp / "a.py").write_text("x = 1\n")
    (tmp / "b.py").write_text("def f():\n    return 2\n")


def test_syncprogress():
    p = SyncProgress()
    assert p.status == "idle"
    assert not p.is_running()
    assert not p.is_done()

    p.status = "scanning"
    assert p.is_running()

    p.status = "done"
    assert p.is_done()

    d = p.to_dict()
    assert d["status"] == "done"
    assert "elapsed_seconds" in d
    print("PASS: SyncProgress")


def test_hashcache():
    tmp = tempfile.mkdtemp()
    cache = HashCache(tmp)

    # Empty
    assert cache.get_all() == {}

    # Put + get
    cache.put_batch([("a.py", "hash1"), ("b.py", "hash2")])
    assert cache.get_all() == {"a.py": "hash1", "b.py": "hash2"}

    # In-memory cache hit
    assert cache.get_all() == {"a.py": "hash1", "b.py": "hash2"}

    # Delete
    cache.delete_batch(["a.py"])
    assert cache.get_all() == {"b.py": "hash2"}

    # Re-open: persisted to SQLite
    cache2 = HashCache(tmp)
    assert cache2.get_all() == {"b.py": "hash2"}
    cache2.close()

    cache.close()
    print("PASS: HashCache")


def test_async_sync_throttled():
    """Verify async_sync_workspace with throttling."""
    engine = EmbeddingEngine()
    engine.load()

    work = Path(tempfile.mkdtemp()).resolve()
    db_dir = Path(tempfile.mkdtemp()).resolve()
    _make_workspace(work)

    store = ChromaStore(path=str(db_dir / "ch"))
    cache = HashCache(store.path)
    progress = SyncProgress()

    async def run():
        return await async_sync_workspace(
            directory=str(work),
            engine=engine,
            store=store,
            hash_cache=cache,
            progress=progress,
            batch_size=4,
            throttle_sec=0.01,
        )

    added, unchanged, deleted = asyncio.run(run())
    assert added == 2, f"added={added}"
    assert store.count() == 2
    print(f"PASS: async_sync (initial)  |  added={added}")

    # Verify progress tracking
    assert progress.status == "done"
    assert progress.total_files == 2
    assert progress.added_chunks == 2
    print(f"PASS: async_sync progress  |  status={progress.status}")

    # Second sync — no changes
    async def run2():
        return await async_sync_workspace(
            directory=str(work),
            engine=engine,
            store=store,
            hash_cache=cache,
            progress=SyncProgress(),
            batch_size=4,
            throttle_sec=0.01,
        )

    added2, unchanged2, deleted2 = asyncio.run(run2())
    assert added2 == 0, f"added2={added2}"
    assert unchanged2 == 2
    print(f"PASS: async_sync (incremental, no changes)  |  added={added2} unchanged={unchanged2}")

    # Change a file
    (work / "a.py").write_text("x = 999")
    added3, unchanged3, deleted3 = asyncio.run(run2())
    assert added3 > 0, f"added3={added3}"
    assert store.count() == 2
    print(f"PASS: async_sync (file changed)  |  added={added3}")

    # Delete a file
    (work / "b.py").unlink()
    added4, unchanged4, deleted4 = asyncio.run(run2())
    assert added4 == 0, f"added4={added4}"
    assert store.count() == 1
    print(f"PASS: async_sync (file deleted)  |  count={store.count()}")

    cache.close()


if __name__ == "__main__":
    test_syncprogress()
    test_hashcache()
    test_async_sync_throttled()
    print("\nALL PASS.")
