"""Model CLI commands: ls, set, pull, rm."""

import argparse
import shutil
import sys
import time
from pathlib import Path

from code_radar.cli.helpers import (
    yes_no,
    print_size,
    dir_size,
    hf_cache_name,
    is_cached,
    resolve_key,
    HF_HUB_CACHE,
)
from code_radar.config import (
    CONFIG_FILE,
    load_config,
    save_config,
)
from code_radar.models import (
    MODEL_REGISTRY,
    DEFAULT_MODEL_KEY,
    DEFAULT_RERANKER_MODEL_KEY,
    RERANKER_REGISTRY,
    get_model_config,
    list_models,
    list_reranker_models,
)

def cmd_model_ls(_args: argparse.Namespace) -> None:
    """List all registered models (embedding + reranker)."""
    cfg = load_config()
    active_embed_key = cfg.get("model_key", DEFAULT_MODEL_KEY)
    active_rr_key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)

    print("Registered models:\n")

    print("Embedding models:\n")
    for entry in list_models():
        key = str(entry["key"])
        cfg_model = MODEL_REGISTRY[key]
        hf_id = cfg_model.id
        cached = is_cached(hf_id)
        active = key == active_embed_key

        marker = "->" if active else " "

        print(f"  {marker} {key}")
        print(
            f"     {entry['name']:40s}  "
            f"{'[cached ✓]' if cached else '[not cached]'}"
            f"{'  [active embedding]' if active else ''}"
        )
        print(
            f"     RAM:  {entry['ram_gb']:.1f} GB  |  "
            f"speed: {entry['speed']:8s}  |  "
            f"accuracy: {entry['accuracy']}"
        )
        print(f"     {entry['description']}")
        if entry['multimodal']:
            print("     Supports images + text")
        print()

    print("Reranker models:\n")
    for entry in list_reranker_models():
        key = str(entry["key"])
        cfg_rr = RERANKER_REGISTRY[key]
        hf_id = cfg_rr.id
        cached = is_cached(hf_id)
        active = key == active_rr_key

        marker = "->" if active else " "

        print(f"  {marker} {key}")
        print(
            f"     {entry['name']:40s}  "
            f"{'[cached ✓]' if cached else '[not cached]'}"
            f"{'  [active reranker]' if active else ''}"
        )
        print(
            f"     RAM:  {entry['ram_gb']:.1f} GB  |  "
            f"speed: {entry['speed']:8s}  |  "
            f"accuracy: {entry['accuracy']}"
        )
        print(f"     {entry['description']}")
        print()

    print("===")
    print("Use  model set <embedding-key>      to activate embedding model")
    print("Use  reranker set <reranker-key>    to activate reranker model")
    print("Use  model pull / reranker pull     to download model weights")
    print()


def cmd_model_set(args: argparse.Namespace) -> None:
    """Set the active embedding model by registry key."""
    key = resolve_key(args.name)

    if key not in MODEL_REGISTRY:
        # Allow setting custom HF IDs directly
        if "/" in key:
            cfg = load_config()
            cfg["model_key"] = key
            cfg["custom_model_id"] = key
            save_config(cfg)
            print(f"✓ Active model set to custom ID: {key}")
            print(f"  Config saved to: {CONFIG_FILE}")
            return

        available = ", ".join(MODEL_REGISTRY.keys())
        print(f"Unknown model '{args.name}'.")
        print(f"Available keys: {available}")
        print("Or use a full HuggingFace ID (e.g. mlx-community/SomeModel).")
        sys.exit(1)

    cfg = load_config()
    cfg["model_key"] = key
    cfg.pop("custom_model_id", None)
    save_config(cfg)

    conf = get_model_config(key)
    print(f"✓ Active model set to: {key}")
    print(f"  Name:  {conf.name}")
    print(f"  HF ID: {conf.id}")
    print(f"\n  Next:  uv run code-radar model pull {key}   (if not cached)")


def cmd_model_pull(args: argparse.Namespace) -> None:
    """Download a model from HuggingFace by registry key or full HF ID."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        sys.exit(1)

    raw = args.name

    # Resolve key -> HF ID
    key = resolve_key(raw)
    if key in MODEL_REGISTRY:
        model_id = get_model_config(key).id
        model_label = f"{key} ({model_id})"
    else:
        model_id = key
        model_label = model_id

    print(f"Downloading: {model_label}")
    print(f"Cache:       {HF_HUB_CACHE}\n")

    t0 = time.perf_counter()
    try:
        path = snapshot_download(
            repo_id=model_id,
            cache_dir=str(HF_HUB_CACHE),
            ignore_patterns=["*.md", "*.txt"],
            token=None,
        )
    except Exception as e:
        print(f"✗ Download failed: {e}")
        sys.exit(1)

    elapsed = time.perf_counter() - t0
    dest = Path(path)
    print_size("Size", dir_size(dest))
    print(f"  Time:  {elapsed:.1f}s")
    print(f"\n✓ Downloaded to: {dest}")

    if key in MODEL_REGISTRY and yes_no("\nSet this model as active now?"):
        cmd_model_set(argparse.Namespace(name=key))


def cmd_model_rm(args: argparse.Namespace) -> None:
    """Remove a cached model."""
    raw = args.name

    # Resolve key -> HF ID
    key = resolve_key(raw)
    if key in MODEL_REGISTRY:
        model_id = get_model_config(key).id
    else:
        model_id = key

    cache_dir = HF_HUB_CACHE / hf_cache_name(model_id)
    if not cache_dir.exists():
        print(f"Model not found in cache: {model_id}")
        print(f"Looked in: {cache_dir}")
        sys.exit(1)

    print_size("Model size", dir_size(cache_dir))
    if not args.force and not yes_no(f"Remove '{key if key in MODEL_REGISTRY else model_id}'?"):
        print("Aborted.")
        return

    shutil.rmtree(cache_dir)
    label = key if key in MODEL_REGISTRY else model_id
    print(f"✓ Removed: {label}")

    # If it was the active model, reset to default
    cfg = load_config()
    if cfg.get("model_key") == key:
        cfg["model_key"] = DEFAULT_MODEL_KEY
        cfg.pop("custom_model_id", None)
        save_config(cfg)
        default_name = MODEL_REGISTRY[DEFAULT_MODEL_KEY].name
        print(f"  ↻ Active model reset to default: {DEFAULT_MODEL_KEY} ({default_name})")
