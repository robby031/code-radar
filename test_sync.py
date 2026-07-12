from pathlib import Path
import tempfile

from code_radar.chroma import ChromaStore
from code_radar.codebase import sync_workspace
from code_radar.engine import EmbeddingEngine


def test_batched_embedding():
    engine = EmbeddingEngine()
    engine.load()

    texts = ["hello world", "goodbye world", "foo bar baz"]
    single = [engine.embed_text(t) for t in texts]
    batch = engine.embed_texts(texts, batch_size=2)

    assert len(batch) == len(texts)
    for s, b in zip(single, batch):
        sim = engine.cosine_similarity(s, b)
        assert sim > 0.999, f"single vs batch sim={sim}"
    print("PASS: batch = single embedding")


def test_sync_incremental():
    engine = EmbeddingEngine()
    engine.load()

    # workspace and DB di direktori terpisah
    work = Path(tempfile.mkdtemp()).resolve()
    db_dir = Path(tempfile.mkdtemp()).resolve()

    (work / "a.py").write_text("x=1")
    (work / "b.py").write_text("def f(): return 2")
    store = ChromaStore(path=str(db_dir / "ch"))

    # Sync 1 full
    added, unchanged, deleted = sync_workspace(str(work), engine, store, batch_size=4)
    assert added == 2, f"sync1 added={added}"
    assert store.count() == 2
    first_count = store.count()
    print(f"PASS: sync1 -> {added} chunks added, {unchanged} unchanged")

    # Sync 2 no changes (incremental, skip all)
    added, unchanged, deleted = sync_workspace(str(work), engine, store, batch_size=4)
    assert added == 0, f"sync2 added={added}"
    assert unchanged == 2
    assert store.count() == first_count
    print(f"PASS: sync2 -> {added} chunks added, {unchanged} unchanged (no changes)")

    # Sync 3 file changed
    (work / "a.py").write_text("x = 999")
    added, unchanged, deleted = sync_workspace(str(work), engine, store, batch_size=4)
    assert added > 0, f"sync3 added={added}"
    assert store.count() == first_count
    print(f"PASS: sync3 -> {added} chunks added after a.py changed")

    # Sync 4 file deleted
    (work / "b.py").unlink()
    added, unchanged, deleted = sync_workspace(str(work), engine, store, batch_size=4)
    assert added == 0
    assert store.count() == 1
    print(f"PASS: sync4 -> b.py removed, DB now {store.count()} chunks")


if __name__ == "__main__":
    test_batched_embedding()
    test_sync_incremental()
    print("\nALL PASS.")
