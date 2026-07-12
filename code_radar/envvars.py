from __future__ import annotations

import os


def get_env(name: str, default: str | None = None) -> str | None:
    """Read environment variable as-is (strict, no legacy fallback)."""
    return os.environ.get(name, default)


def has_env(name: str) -> bool:
    """Check whether environment variable is defined."""
    return name in os.environ
