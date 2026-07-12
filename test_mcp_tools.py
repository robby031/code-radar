from pathlib import Path
import tempfile

from code_radar.codebase import sync_workspace
from code_radar.engine import EmbeddingEngine
from code_radar.chroma import ChromaStore
from code_radar.server import configure, semantic_search, smart_search, read_full_file


def test_tools():
    tmp = Path(tempfile.mkdtemp()).resolve()

    (tmp / "source.py").write_text(
        "def greet(name):\n"
        "    return f'hello {name}'\n"
        "\n"
        "def add(a, b):\n"
        "    return a + b\n"
    )
    (tmp / "main.go").write_text(
        "package main\n"
        "\n"
        "func greet(name string) string {\n"
        "    return fmt.Sprintf(\"hello %s\", name)\n"
        "}\n"
        "\n"
        "func add(a, b int) int {\n"
        "    return a + b\n"
        "}\n"
    )
    (tmp / "notes.txt").write_text("some random notes\nsecond line\n")

    engine = EmbeddingEngine()
    engine.load()
    db_dir = Path(tempfile.mkdtemp()).resolve()
    store = ChromaStore(path=str(db_dir / "ch"))
    sync_workspace(str(tmp), engine, store)
    configure(engine, store, str(tmp))

    # semantic_search
    result = semantic_search("greeting function", n=2)
    assert "source.py" in result, f"source.py tidak muncul:\n{result}"
    assert "score=" in result
    print("PASS: semantic_search returns hits with score")

    result_all = semantic_search("code", n=10)
    assert result_all
    print("PASS: semantic_search returns all available results")

    # semantic_search with language filter
    result_py = semantic_search("greeting function", n=5, language="py")
    assert "source.py" in result_py, f"language=py harusnya return source.py:\n{result_py}"
    assert "score=" in result_py
    assert "main.go" not in result_py, f"language=py tidak boleh return main.go:\n{result_py}"
    print("PASS: semantic_search with language=py filters correctly")

    result_go = semantic_search("greeting function", n=5, language="go")
    assert "main.go" in result_go, f"language=go harusnya return main.go:\n{result_go}"
    assert "source.py" not in result_go, f"language=go tidak boleh return source.py:\n{result_go}"
    print("PASS: semantic_search with language=go filters correctly")

    result_no_match = semantic_search("greeting function", n=5, language="rb")
    assert "No results found" in result_no_match, (
        f"language=rb harusnya No results found:\n{result_no_match}"
    )
    print("PASS: semantic_search with non-existent language returns no results")

    # semantic_search with language + filepath combination
    result_combined = semantic_search("function", n=5, language="py", filepath="source.py")
    assert "source.py" in result_combined, (
        f"language=py + filepath=source.py harus return source.py:\n{result_combined}"
    )
    assert "score=" in result_combined
    print("PASS: semantic_search with language + filepath combination works")

    result_combined_no = semantic_search("function", n=5, language="go", filepath="source.py")
    assert "No results found" in result_combined_no, (
        f"language=go + filepath=source.py harusnya No results found:\n{result_combined_no}"
    )
    print("PASS: semantic_search with conflicting language + filepath returns no results")

    # smart_search (hybrid)
    smart = smart_search("greet function", n=2)
    assert "source.py" in smart, f"smart_search source.py tidak muncul:\n{smart}"
    assert "phase1:" in smart
    print("PASS: smart_search returns hybrid hits with phase summary")

    # read_full_file
    content = read_full_file("source.py")
    assert "greet" in content, f"source.py:\n{content}"
    assert "def add" in content
    print("PASS: read_full_file returns full source.py with line numbers")

    content = read_full_file("notes.txt")
    assert "random notes" in content
    print("PASS: read_full_file returns notes.txt")

    # path traversal safety
    blocked = read_full_file("../etc/passwd")
    assert "denied" in blocked
    print("PASS: read_full_file blocks path traversal")

    missing = read_full_file("nope.py")
    assert "not found" in missing
    print("PASS: read_full_file handles missing file")


if __name__ == "__main__":
    test_tools()
    print("\nALL PASS.")
