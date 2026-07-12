from pathlib import Path
import tempfile

from code_radar.sparse_index import SparseChunkIndex


def test_sparse_index_basic() -> None:
    db_root = Path(tempfile.mkdtemp()).resolve()
    idx = SparseChunkIndex(str(db_root / "chroma"))

    chunks = [
        {
            "id": "a1",
            "filepath": "service/pricing.go",
            "content": "func CalculateProductPrice(qty int, base float64) float64 { return base * float64(qty) }",
            "chunk_type": "function",
            "start_line": 10,
            "end_line": 12,
            "metadata": {"language": "go"},
        },
        {
            "id": "a2",
            "filepath": "service/pricing.go",
            "content": "var oldProductPrice = 0",
            "chunk_type": "code_block",
            "start_line": 20,
            "end_line": 20,
            "metadata": {"language": "go"},
        },
        {
            "id": "b1",
            "filepath": "service/cart.go",
            "content": "func ApplyDiscount(amount float64) float64 { return amount * 0.9 }",
            "chunk_type": "function",
            "start_line": 5,
            "end_line": 7,
            "metadata": {"language": "go"},
        },
    ]

    idx.upsert_chunks(chunks)

    res = idx.search("CalculateProductPrice", n=5)
    assert res, "expected search results"
    assert res[0]["id"] == "a1", f"unexpected top hit: {res[0]}"

    scoped = idx.search("ApplyDiscount", n=5, filepath="service/cart.go")
    assert scoped and scoped[0]["id"] == "b1"

    basename_scoped = idx.search("CalculateProductPrice", n=5, filepath="pricing.go")
    assert basename_scoped and basename_scoped[0]["id"] == "a1"

    idx.delete_by_files(["service/pricing.go"])
    left = idx.search("CalculateProductPrice", n=5)
    assert not left, f"expected deleted chunks to be gone, got: {left}"

    idx.close()


def test_sparse_index_path_recall_for_plural_filename() -> None:
    db_root = Path(tempfile.mkdtemp()).resolve()
    idx = SparseChunkIndex(str(db_root / "chroma"))

    idx.upsert_chunks(
        [
            {
                "id": "daemon-file",
                "filepath": "internal/worker/daemons.go",
                "content": "func StartWorkers(ctx context.Context) error { return nil }",
                "chunk_type": "function_declaration",
                "start_line": 1,
                "end_line": 3,
                "metadata": {"language": "go"},
            },
            {
                "id": "banner-file",
                "filepath": "configs/banners.json",
                "content": '{"function": "banner configuration"}',
                "chunk_type": "file",
                "start_line": 1,
                "end_line": 1,
                "metadata": {"language": "json"},
            },
        ]
    )

    tokens = SparseChunkIndex._tokenize_query("daemon function")
    assert "daemon" in tokens
    assert "daemons" in tokens
    assert "func" in tokens

    res = idx.search("daemon function", n=5)
    assert res, "expected path recall results"
    assert res[0]["id"] == "daemon-file", f"unexpected top hit: {res[0]}"

    idx.close()


def test_sparse_index_full_filepath_does_not_leak_same_basename() -> None:
    db_root = Path(tempfile.mkdtemp()).resolve()
    idx = SparseChunkIndex(str(db_root / "chroma"))

    idx.upsert_chunks(
        [
            {
                "id": "swpulsa",
                "filepath": "src/core/integration/swpulsa/datasource_product.go",
                "content": "case 2: return calculateProductPrice(product.RumusProduk)",
                "chunk_type": "code_block",
                "start_line": 10,
                "end_line": 20,
                "metadata": {"language": "go"},
            },
            {
                "id": "otomax",
                "filepath": "src/core/integration/otomax/datasource_product.go",
                "content": "case 2: return calculateProductPrice(product.RumusProduk)",
                "chunk_type": "code_block",
                "start_line": 10,
                "end_line": 20,
                "metadata": {"language": "go"},
            },
            {
                "id": "host",
                "filepath": "src/core/integration/host/datasource_product.go",
                "content": "case 2: return calculateProductPrice(product.RumusProduk)",
                "chunk_type": "code_block",
                "start_line": 10,
                "end_line": 20,
                "metadata": {"language": "go"},
            },
        ]
    )

    full_path = idx.search(
        "RumusProduk case 2",
        n=10,
        filepath="src/core/integration/swpulsa/datasource_product.go",
    )
    assert [item["id"] for item in full_path] == ["swpulsa"]

    basename = idx.search(
        "RumusProduk case 2",
        n=10,
        filepath="datasource_product.go",
    )
    assert {item["id"] for item in basename} == {"swpulsa", "otomax", "host"}

    idx.close()


if __name__ == "__main__":
    test_sparse_index_basic()
    test_sparse_index_path_recall_for_plural_filename()
    test_sparse_index_full_filepath_does_not_leak_same_basename()
    print("\nALL PASS.")
