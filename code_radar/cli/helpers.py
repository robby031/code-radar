"""Shared helper functions and constants for CLI commands."""

import os
from pathlib import Path

from code_radar.envvars import get_env
from code_radar.logging import get_logger
from code_radar.models import MODEL_REGISTRY

log = get_logger("cli")

# HF cache directory used by mlx-lm
HF_HUB_CACHE = Path(os.environ.get(
    "HF_HUB_CACHE",
    Path.home() / ".cache" / "huggingface" / "hub",
))


def yes_no(prompt: str) -> bool:
    while True:
        ans = input(f"{prompt} [y/N] ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("", "n", "no"):
            return False


def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = get_env(name)
    if raw is None:
        return max(minimum, default)
    try:
        value = int(raw)
    except ValueError:
        return max(minimum, default)
    return max(minimum, value)


def _format_size(n: int) -> str:
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


def print_size(label: str, n: int) -> None:
    print(f"  {label}:  {_format_size(n):>10}")


def dir_size(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def hf_cache_name(model_id: str) -> str:
    """Convert ``mlx-community/SomeModel`` to cache dir name."""
    return "models--" + model_id.replace("/", "--")


def is_cached(model_id: str) -> bool:
    return (HF_HUB_CACHE / hf_cache_name(model_id)).exists()


def cached_size(model_id: str) -> int:
    d = HF_HUB_CACHE / hf_cache_name(model_id)
    return dir_size(d) if d.exists() else 0


def resolve_key(key_or_id: str) -> str:
    """Return the registry key for a given key or full HF ID."""
    if key_or_id in MODEL_REGISTRY:
        return key_or_id
    # Maybe it's a full HF ID find matching key
    for k, cfg in MODEL_REGISTRY.items():
        if cfg.id == key_or_id:
            return k
    return key_or_id  # unknown, use as-is
