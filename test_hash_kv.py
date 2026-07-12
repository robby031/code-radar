"""Test HashKVStore SQLite-backed KV store."""
import os
import tempfile

from code_radar.hash_kv import HashKVStore


def test_basic_crud():
    tmp = tempfile.mkdtemp()
    store = HashKVStore(tmp)

    # Empty store
    assert store.get_all() == {}

    # Put and get
    store.put("a.py", "hash1")
    store.put("b.py", "hash2")
    assert store.get_all() == {"a.py": "hash1", "b.py": "hash2"}

    # Update
    store.put("a.py", "hash1_updated")
    assert store.get_all() == {"a.py": "hash1_updated", "b.py": "hash2"}

    # Delete
    store.delete("b.py")
    assert store.get_all() == {"a.py": "hash1_updated"}

    # Batch put
    store.put_batch([("c.py", "hash3"), ("d.py", "hash4")])
    assert store.get_all() == {"a.py": "hash1_updated", "c.py": "hash3", "d.py": "hash4"}

    # Batch delete
    store.delete_batch(["c.py", "d.py"])
    assert store.get_all() == {"a.py": "hash1_updated"}

    # DB file exists in chroma_path dir (workspace-scoped)
    assert os.path.exists(os.path.join(tmp, "code_file_hashes__default.db"))

    store.close()
    print("PASS: basic CRUD")


def test_empty_ops():
    tmp = tempfile.mkdtemp()
    store = HashKVStore(tmp)

    # Empty batch ops shouldn't crash
    store.put_batch([])
    store.delete_batch([])
    assert store.get_all() == {}

    store.close()
    print("PASS: empty batch ops")


def test_reopen_persistence():
    tmp = tempfile.mkdtemp()
    store = HashKVStore(tmp)
    store.put("a.py", "hash1")
    store.put("b.py", "hash2")
    store.close()

    # Re-open and verify data persists
    store2 = HashKVStore(tmp)
    assert store2.get_all() == {"a.py": "hash1", "b.py": "hash2"}
    store2.close()
    print("PASS: persistence across reopen")


if __name__ == "__main__":
    test_basic_crud()
    test_empty_ops()
    test_reopen_persistence()
    print("\nALL PASS.")
