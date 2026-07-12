"""Reranker CLI commands: ls, current, set, pull, rm."""

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
    cached_size,
    resolve_key,
    HF_HUB_CACHE,
)
from code_radar.config import (
    CONFIG_FILE,
    load_config,
    resolve_reranker_model_id,
    save_config,
)
from code_radar.models import (
    RERANKER_REGISTRY,
    DEFAULT_RERANKER_MODEL_KEY,
    get_reranker_model_config,
    list_reranker_models,
)


def cmd_reranker_ls(_args: argparse.Namespace) -> None:
    """List all available reranker models from the registry."""
    cfg = load_config()
    active_key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)

    print("Available reranker models from registry:\n")

    for entry in list_reranker_models():
        key = str(entry["key"])
        cfg_rm = RERANKER_REGISTRY[key]
        hf_id = cfg_rm.id
        cached = is_cached(hf_id)
        active = key == active_key

        marker = "->" if active else " "

        print(f"  {marker} {key}")
        print(f"     {entry['name']:40s}  "
              f"{'[cached ✓]' if cached else '[not cached]'}"
              f"{'  [active]' if active else ''}")
        print(f"     RAM:  {entry['ram_gb']:.1f} GB  |  "
              f"speed: {entry['speed']:8s}  |  "
              f"accuracy: {entry['accuracy']}")
        print(f"     {entry['description']}")
        print()

    print("===")
    print("Use  reranker pull <key>  to download")
    print("Use  reranker set  <key>  to activate")
    print()


def cmd_reranker_current(_args: argparse.Namespace) -> None:
    """Show the currently configured reranker model with details."""
    cfg = load_config()
    key = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)

    if key in RERANKER_REGISTRY:
        conf = get_reranker_model_config(key)
        hf_id = conf.id
        cached = is_cached(hf_id)

        print(f"Active reranker:  {key}")
        print(f"  Name:        {conf.name}")
        print(f"  HF ID:       {hf_id}")
        print(f"  RAM:         {conf.ram_gb:.1f} GB")
        print(f"  Speed:       {conf.speed_tier}")
        print(f"  Accuracy:    {conf.accuracy_tier}")
        print(f"  Batch size:  {conf.default_batch_size}")
        print(f"  Cached:      {'✓' if cached else '✗ (auto-download on load)'}")
        if cached:
            print_size("  Cache size", cached_size(hf_id))
    else:
        # Custom / unknown model
        hf_id = resolve_reranker_model_id(key)
        cached = is_cached(hf_id)
        print(f"Active reranker:  {key}")
        print(f"  HF ID:       {hf_id}")
        print(f"  Cached:      {'✓' if cached else '✗ (auto-download on load)'}")

    print(f"  Enabled:     {cfg.get('reranker_enabled', True)}")
    print(f"\nConfig file:   {CONFIG_FILE}")


def cmd_reranker_set(args: argparse.Namespace) -> None:
    """Set the active reranker model by registry key."""
    key = resolve_key(args.name)

    if key not in RERANKER_REGISTRY:
        # Allow setting custom HF IDs directly
        if "/" in key:
            cfg = load_config()
            cfg["reranker_model_key"] = key
            save_config(cfg)
            print(f"✓ Active reranker set to custom ID: {key}")
            print(f"  Config saved to: {CONFIG_FILE}")
            return

        available = ", ".join(RERANKER_REGISTRY.keys())
        print(f"Unknown reranker model '{args.name}'.")
        print(f"Available keys: {available}")
        print("Or use a full HuggingFace ID (e.g. mlx-community/SomeModel).")
        sys.exit(1)

    cfg = load_config()
    cfg["reranker_model_key"] = key
    save_config(cfg)

    conf = get_reranker_model_config(key)
    print(f"✓ Active reranker set to: {key}")
    print(f"  Name:  {conf.name}")
    print(f"  HF ID: {conf.id}")
    print(f"\n  Next:  uv run code-radar reranker pull {key}   (if not cached)")


def cmd_reranker_pull(args: argparse.Namespace) -> None:
    """Download a reranker model from HuggingFace."""
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("Error: huggingface_hub not installed.")
        sys.exit(1)

    raw = args.name
    key = resolve_key(raw)

    if key in RERANKER_REGISTRY:
        model_id = get_reranker_model_config(key).id
        model_label = f"{key} ({model_id})"
    else:
        model_id = key
        model_label = model_id

    print(f"Downloading reranker: {model_label}")
    print(f"Cache:              {HF_HUB_CACHE}\n")

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

    if key in RERANKER_REGISTRY and yes_no("\nSet this reranker model as active now?"):
        cmd_reranker_set(argparse.Namespace(name=key))


def cmd_reranker_rm(args: argparse.Namespace) -> None:
    """Remove a cached reranker model."""
    raw = args.name
    key = resolve_key(raw)

    if key in RERANKER_REGISTRY:
        model_id = get_reranker_model_config(key).id
    else:
        model_id = key

    cache_dir = HF_HUB_CACHE / hf_cache_name(model_id)
    if not cache_dir.exists():
        print(f"Reranker model not found in cache: {model_id}")
        print(f"Looked in: {cache_dir}")
        sys.exit(1)

    print_size("Reranker size", dir_size(cache_dir))
    if not args.force and not yes_no(f"Remove reranker '{key if key in RERANKER_REGISTRY else model_id}'?"):
        print("Aborted.")
        return

    shutil.rmtree(cache_dir)
    label = key if key in RERANKER_REGISTRY else model_id
    print(f"✓ Removed reranker: {label}")

    # If it was the active reranker, reset to default
    cfg = load_config()
    if cfg.get("reranker_model_key") == key:
        cfg["reranker_model_key"] = DEFAULT_RERANKER_MODEL_KEY
        save_config(cfg)
        default_name = RERANKER_REGISTRY[DEFAULT_RERANKER_MODEL_KEY].name
        print(f"  ↻ Active reranker reset to default: {DEFAULT_RERANKER_MODEL_KEY} ({default_name})")
