import time

from code_radar.state import SyncProgress


def test_syncprogress_live_elapsed() -> None:
    progress = SyncProgress(status="processing", start_time=time.perf_counter() - 1.0)

    payload = progress.to_dict()

    assert payload["elapsed_seconds"] > 0


if __name__ == "__main__":
    test_syncprogress_live_elapsed()
    print("\nALL PASS.")
