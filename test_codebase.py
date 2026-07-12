from pathlib import Path
import tempfile

from code_radar.chunker import chunk_file, count_tokens
from code_radar.codebase import sync_workspace
from code_radar.engine import EmbeddingEngine
from code_radar.chroma import ChromaStore
from code_radar.reader import CodeReader


def gen_temp_project():
    tmp = Path(tempfile.mkdtemp()).resolve()

    # file < 1000 token - harus jadi 1 chunk file-level
    (tmp / "small.py").write_text("x = 1\n")

    # file > 1000 token - harus pecah jadi function-level
    funcs = "\n\n".join(
        f"def func_{i}():\n    return {i}\n" for i in range(200)
    )
    (tmp / "large.py").write_text(
        "import sys\n\n" + funcs + "\n\nTOP_LEVEL = 42\n"
    )

    # file .gitignore
    (tmp / ".gitignore").write_text("ignored_dir/\n")

    # ignored dir/file
    (tmp / "ignored_dir").mkdir()
    (tmp / "ignored_dir" / "secret.py").write_text("SECRET = 1")

    return tmp


def test_reader_gitignore():
    tmp = gen_temp_project()
    reader = CodeReader(str(tmp))
    paths = [str(p.relative_to(tmp)) for p in reader.files()]
    assert "small.py" in paths, f"small.py tidak ada di {paths}"
    assert "large.py" in paths, f"large.py tidak ada di {paths}"
    assert "ignored_dir/secret.py" not in paths, ".gitignore diabaikan"
    print("PASS: CodeReader menerapkan .gitignore")


def test_chunk_file_level():
    tmp = gen_temp_project()
    engine = EmbeddingEngine()
    engine.load()
    small = tmp / "small.py"
    chunks = chunk_file(small, tmp, engine.tokenizer)
    assert len(chunks) == 1, f"small.py -> {len(chunks)} chunk (harus 1)"
    assert chunks[0]["chunk_type"] == "file"
    tok = count_tokens(engine.tokenizer, chunks[0]["content"])
    print(f"PASS: small.py -> 1 file-level chunk ({tok} token)")


def test_chunk_function_level():
    tmp = gen_temp_project()
    engine = EmbeddingEngine()
    engine.load()
    large = tmp / "large.py"
    chunks = chunk_file(large, tmp, engine.tokenizer)

    assert len(chunks) > 1, f"large.py cuma {len(chunks)} chunk"
    types = {c["chunk_type"] for c in chunks}
    assert "function" in types, f"tidak ada chunk function: {types}"

    func_count = sum(1 for c in chunks if c["chunk_type"] == "function")
    print(f"PASS: large.py -> {len(chunks)} chunks ({func_count} function)")


def test_initialize_codebase():
    work = gen_temp_project()
    db_dir = Path(tempfile.mkdtemp()).resolve()
    engine = EmbeddingEngine()
    engine.load()
    store = ChromaStore(path=str(db_dir / "ch"))

    added, unchanged, deleted = sync_workspace(str(work), engine, store)
    assert added >= 2, f"cuma {added} chunks"
    assert store.count() == added

    # query pakai salah satu chunk
    results = store.query(engine.embed_text("def func_1"), n=3)
    docs = results.get("documents")
    assert docs is not None, "documents None"
    top_docs = docs[0]
    assert any("func_1" in d for d in top_docs), "tidak ditemukan func_1"
    print(f"PASS: sync_workspace -> {added} chunk tersimpan dan bisa di query")


if __name__ == "__main__":
    test_reader_gitignore()
    test_chunk_file_level()
    test_chunk_function_level()
    test_initialize_codebase()
    print("\nALL PASS.")
