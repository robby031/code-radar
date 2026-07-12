from __future__ import annotations

import asyncio
import tempfile
import time
import unittest
from typing import Any
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
        "timestamp": time.time(),
    }


async def _wait_until_idle(watcher: FileWatcher, timeout: float = 30.0) -> None:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        sync_running = watcher.sync_task is not None and not watcher.sync_task.done()
        debounce_running = watcher.debounce_task is not None and not watcher.debounce_task.done()
        if (
            watcher.event_queue.empty()
            and not watcher.pending_events
            and not sync_running
            and not debounce_running
        ):
            return
        await asyncio.sleep(0.01)
    raise TimeoutError("watcher did not reach idle state before timeout")


class AutoSyncScenarioTests(unittest.IsolatedAsyncioTestCase):
    async def _make_watcher(
        self,
        debounce_seconds: float = 2.5,
        micro_batch_size: int = 10,
        yield_seconds: float = 0.05,
    ) -> tuple[FileWatcher, HashCache]:
        cache = HashCache(tempfile.mkdtemp())
        watcher = FileWatcher(
            root_path=".",
            engine=_DummyEngine(),
            store=_DummyStore(),
            hash_cache=cache,
            debounce_seconds=debounce_seconds,
            micro_batch_size=micro_batch_size,
            yield_seconds=yield_seconds,
            throttle_seconds=0.0,
        )
        watcher.is_running = True
        watcher.event_task = asyncio.create_task(watcher._event_processor())
        return watcher, cache

    async def test_scenario_1_normal_development(self) -> None:
        """User edit 1 file -> debounce -> sync 1 file -> done."""
        calls: list[dict[str, Any]] = []

        async def fake_sync_files(*, filepaths, deleted_paths, **kwargs):
            calls.append({"filepaths": tuple(filepaths), "deleted_paths": tuple(deleted_paths)})
            await asyncio.sleep(0.02)
            return (0, 0, 0)

        watcher, cache = await self._make_watcher(
            debounce_seconds=2.5,
            micro_batch_size=10,
            yield_seconds=0.05,
        )

        try:
            with patch("code_radar.watcher.async_sync_files", new=fake_sync_files):
                t0 = time.perf_counter()
                await watcher.event_queue.put(_event("modified", "src/app.py"))
                await _wait_until_idle(watcher, timeout=10.0)
                elapsed = time.perf_counter() - t0

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["filepaths"], ("src/app.py",))
            self.assertEqual(calls[0]["deleted_paths"], ())

            # Target around ~2.6s (2.5 debounce + processing/yield overhead)
            self.assertGreaterEqual(elapsed, 2.45)
            self.assertLess(elapsed, 4.0)
        finally:
            await watcher.stop()
            cache.close()

    async def test_scenario_2_git_checkout_event_storm(self) -> None:
        """1000 changes -> coalesce -> 100 micro-batches of 10 + yield between batches."""
        calls: list[dict[str, Any]] = []

        async def fake_sync_files(*, filepaths, deleted_paths, **kwargs):
            calls.append({"filepaths": tuple(filepaths), "deleted_paths": tuple(deleted_paths)})
            # Simulate lightweight work per micro-batch so runtime stays realistic.
            await asyncio.sleep(0.025)
            return (0, 0, 0)

        watcher, cache = await self._make_watcher(
            debounce_seconds=2.5,
            micro_batch_size=10,
            yield_seconds=0.05,
        )

        try:
            with patch("code_radar.watcher.async_sync_files", new=fake_sync_files):
                t0 = time.perf_counter()

                for i in range(1000):
                    await watcher.event_queue.put(_event("modified", f"src/file_{i}.py"))

                await _wait_until_idle(watcher, timeout=35.0)
                elapsed = time.perf_counter() - t0

            # 1000 unique touched files, batch size 10 -> 100 sync calls
            self.assertEqual(len(calls), 100)
            self.assertTrue(all(len(c["filepaths"]) <= 10 for c in calls))
            self.assertTrue(all(len(c["deleted_paths"]) == 0 for c in calls))

            processed = {fp for call in calls for fp in call["filepaths"]}
            self.assertEqual(len(processed), 1000)
            self.assertEqual(watcher.sync_runs, 1)

            # Approx target with mocked lightweight processing:
            # debounce (2.5) + yields (100 * 0.05 = 5.0) + processing (100 * 0.025 = 2.5)
            # ~= 10s (still keeps editor responsive because cooperative yields are used)
            self.assertGreaterEqual(elapsed, 9.0)
            self.assertLess(elapsed, 16.5)
        finally:
            await watcher.stop()
            cache.close()

    async def test_scenario_3_rapid_saves_single_sync(self) -> None:
        """5 save events within ~1s should collapse to a single sync."""
        calls: list[dict[str, Any]] = []

        async def fake_sync_files(*, filepaths, deleted_paths, **kwargs):
            calls.append({"filepaths": tuple(filepaths), "deleted_paths": tuple(deleted_paths)})
            await asyncio.sleep(0.02)
            return (0, 0, 0)

        watcher, cache = await self._make_watcher(
            debounce_seconds=2.5,
            micro_batch_size=10,
            yield_seconds=0.05,
        )

        try:
            with patch("code_radar.watcher.async_sync_files", new=fake_sync_files):
                for i in range(5):
                    await watcher.event_queue.put(_event("modified", "src/rapid.py"))
                    if i < 4:
                        await asyncio.sleep(0.2)  # total burst ~0.8s

                t_last = time.perf_counter()
                await _wait_until_idle(watcher, timeout=10.0)
                elapsed_from_last = time.perf_counter() - t_last

            self.assertEqual(watcher.events_received, 5)
            self.assertEqual(watcher.sync_runs, 1)
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0]["filepaths"], ("src/rapid.py",))
            self.assertEqual(calls[0]["deleted_paths"], ())

            # Total from last save should be around debounce window + small overhead.
            self.assertGreaterEqual(elapsed_from_last, 2.45)
            self.assertLess(elapsed_from_last, 4.0)
        finally:
            await watcher.stop()
            cache.close()


if __name__ == "__main__":
    unittest.main()
