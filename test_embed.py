from code_radar.engine import EmbeddingEngine


def main():
    engine = EmbeddingEngine()
    engine.load()

    a = "Kucing itu berlari di taman yang indah"
    b = "Seekor kucing sedang berlari di taman indah"
    emb_a = engine.embed_text(a)
    emb_b = engine.embed_text(b)
    sim = engine.cosine_similarity(emb_a, emb_b)

    print(f"KALIMAT A: {a}")
    print(f"KALIMAT B: {b}")
    print(f"DIMENSI:   {len(emb_a)}")
    print(f"COSINE:    {sim:.4f}")

    assert len(emb_a) == 1024, f"dimensi {len(emb_a)} != 1024"
    assert len(emb_b) == 1024, f"dimensi {len(emb_b)} != 1024"
    assert sim > 0.5, f"similarity terlalu rendah: {sim}"
    assert isinstance(emb_a[0], float)
    assert isinstance(emb_b[0], float)

    print("PASS: embed_text() menghasilkan vektor 1024-d float.")
    print("PASS: cosine similarity dua kalimat mirip > 0.5.")


if __name__ == "__main__":
    main()
