"""Database CLI commands: status, clear, reset."""

import argparse
import shutil
from pathlib import Path

from code_radar.cli.helpers import (
    yes_no,
    print_size,
    dir_size,
)
from code_radar.config import (
    load_config,
    resolve_db_path,
    save_config,
)


# DB COMMANDS
def cmd_db_status(_args: argparse.Namespace) -> None:
    """Show ChromaDB statistics."""
    from chromadb import PersistentClient
    from chromadb.config import Settings

    cfg = load_config()
    path = resolve_db_path(cfg)
    chroma_path = Path(path)

    print(f"Database path:  {path}\n")

    if not chroma_path.exists():
        print("No ChromaDB data found (directory does not exist).")
        return

    print_size("On disk", dir_size(chroma_path))

    try:
        client = PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        collections = client.list_collections()
        print(f"\nCollections:  {len(collections)}")
        for col in collections:
            count = col.count()
            print(f"\n  == {col.name} ==")
            print(f"     Chunks:      {count:,}")
            meta = col.metadata or {}
            if meta:
                print(f"     HNSW space:  {meta.get('hnsw:space', '?')}")
            if count > 0:
                sample = col.get(limit=3, include=["metadatas"])
                metas = sample.get("metadatas", []) or []
                if metas and metas[0]:
                    sample_keys = list(metas[0].keys())[:6]
                    print(f"     Metadata:     {', '.join(sample_keys)}")
    except Exception as e:
        print(f"  Error reading database: {e}")


def cmd_db_clear(args: argparse.Namespace) -> None:
    """Delete all documents from every collection."""
    from chromadb import PersistentClient
    from chromadb.config import Settings

    cfg = load_config()
    path = resolve_db_path(cfg)
    print(f"Database path:  {path}")

    client = PersistentClient(
        path=path,
        settings=Settings(anonymized_telemetry=False),
    )
    collections = client.list_collections()
    if not collections:
        print("No collections to clear.")
        return

    total = sum(c.count() for c in collections)
    print(f"\nCollections: {len(collections)}")
    for c in collections:
        print(f"  • {c.name}: {c.count():,} chunks")
    print(f"\nTotal chunks to delete: {total:,}")

    if not args.force and not yes_no("Are you sure you want to clear ALL embeddings?"):
        print("Aborted.")
        return

    for col in collections:
        all_ids = col.get(limit=total, include=[])["ids"]
        for i in range(0, len(all_ids), 500):
            batch = all_ids[i : i + 500]
            col.delete(ids=batch)
        print(f"  ✓ {col.name}: deleted {len(all_ids):,} chunks")
    print("\nDone. All embeddings cleared.")


def cmd_db_reset(args: argparse.Namespace) -> None:
    """Delete the entire ChromaDB storage directory."""
    cfg = load_config()
    path = resolve_db_path(cfg)
    chroma_path = Path(path)

    if not chroma_path.exists():
        print("Database directory does not exist. Nothing to reset.")
        return

    print_size("Size on disk", dir_size(chroma_path))

    if not args.force and not yes_no("Delete the ENTIRE database directory?"):
        print("Aborted.")
        return

    shutil.rmtree(chroma_path)
    print("✓ Database directory deleted.")

    cfg.pop("db_path", None)
    save_config(cfg)
    print("✓ Database reset complete.")
