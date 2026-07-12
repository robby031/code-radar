import asyncio
import tempfile
from unittest.mock import patch

from code_radar.state import HashCache
from code_radar.watcher import FileWatcher


class _DummyEngine:
    tokenizer = object()
    default_batch_size = 8


class _DummyStore:
    pass


def _event(event_type: str, path: str) -> dict[str, str | float]:
    return {
        "type": event_type,
        "path": path,
        "timestamp": 0.0,
    }


def test_watcher_coalesce_and_micro_batch():
    calls: list[tuple[tuple[str, ...], tuple[str, ...]]] = []

    async def fake_sync_files(*, filepaths, deleted_paths, **kwargs):
        calls.append((tuple(filepaths), tuple(deleted_paths)))
        await asyncio.sleep(0)
        return (0, 0, 0)

    async def run():
        cache = HashCache(tempfile.mkdtemp())
        watcher = FileWatcher(
            root_path=".",
            engine=_DummyEngine(),
            store=_DummyStore(),
            hash_cache=cache,
            debounce_seconds=0.03,
            micro_batch_size=2,
            yield_seconds=0.0,
            throttle_seconds=0.0,
        )

        watcher.is_running = True
        watcher.event_task = asyncio.create_task(watcher._event_processor())

        # Duplicate events for same file should coalesce into latest state.
        await watcher.event_queue.put(_event("modified", "a.py"))
        await watcher.event_queue.put(_event("modified", "a.py"))
        await watcher.event_queue.put(_event("created", "a.py"))
        await watcher.event_queue.put(_event("deleted", "b.py"))
        await watcher.event_queue.put(_event("created", "c.py"))

        await asyncio.sleep(0.2)
        await watcher.stop()
        cache.close()

    with patch("code_radar.watcher.async_sync_files", new=fake_sync_files):
        asyncio.run(run())

    # delete batch first, then touched batch (a.py + c.py)
    assert len(calls) == 2
    assert calls[0] == ((), ("b.py",))
    assert calls[1] == (("a.py", "c.py"), ())


def test_watcher_cancels_running_sync_on_new_changes():
    started: list[tuple[str, ...]] = []
    cancelled: list[tuple[str, ...]] = []

    async def fake_sync_files(*, filepaths, deleted_paths, **kwargs):
        files = tuple(filepaths)
        started.append(files)
        try:
            await asyncio.sleep(0.25)
        except asyncio.CancelledError:
            cancelled.append(files)
            raise
        return (0, 0, 0)

    async def run():
        cache = HashCache(tempfile.mkdtemp())
        watcher = FileWatcher(
            root_path=".",
            engine=_DummyEngine(),
            store=_DummyStore(),
            hash_cache=cache,
            debounce_seconds=0.03,
            micro_batch_size=10,
            yield_seconds=0.0,
            throttle_seconds=0.0,
        )

        watcher.is_running = True
        watcher.event_task = asyncio.create_task(watcher._event_processor())

        await watcher.event_queue.put(_event("modified", "a.py"))

        # Wait until first sync actually starts to make cancellation deterministic.
        deadline = asyncio.get_running_loop().time() + 0.4
        while not started and asyncio.get_running_loop().time() < deadline:
            await asyncio.sleep(0.01)

        await watcher.event_queue.put(_event("modified", "b.py"))
        await asyncio.sleep(0.35)

        await watcher.stop()
        cache.close()

    with patch("code_radar.watcher.async_sync_files", new=fake_sync_files):
        asyncio.run(run())

    assert ("a.py",) in started
    assert ("a.py",) in cancelled
    assert ("b.py",) in started
