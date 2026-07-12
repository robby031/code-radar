"""
Shared configuration management for Atom Embedding Engine.

Stores user settings in ``~/.config/code-radar/config.json``.

Schema
------
.. code-block:: json

    {
        "model_key": "qwen3-0.6b-4bit",
        "db_path": "./chroma_data",
        "profile": "fast"
    }
"""

import json
from pathlib import Path
from typing import Any

from code_radar.envvars import get_env
from code_radar.models import (
    DEFAULT_MODEL_KEY,
    DEFAULT_RERANKER_MODEL_KEY,
    MODEL_REGISTRY,
    RERANKER_REGISTRY,
    get_model_config,
    get_reranker_model_config,
)
from code_radar.profiles import DEFAULT_PROFILE_KEY, PROFILE_REGISTRY

CONFIG_DIR = Path.home() / ".config" / "code-radar"
CONFIG_FILE = CONFIG_DIR / "config.json"

APP_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = "./chroma_data"


def load_config() -> dict[str, Any]:
    """Load config, auto-migrating old ``model`` field to ``model_key``."""
    raw: dict[str, Any] = {}

    if CONFIG_FILE.exists():
        try:
            raw = json.loads(CONFIG_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    model_key = raw.get("model_key")
    if model_key is None and "model" in raw:
        legacy_id: str = raw["model"]
        for key, cfg in MODEL_REGISTRY.items():
            if cfg.id == legacy_id:
                model_key = key
                break
        if model_key is None:
            model_key = DEFAULT_MODEL_KEY
            raw["custom_model_id"] = legacy_id

    profile = raw.get("profile", DEFAULT_PROFILE_KEY)
    if profile not in PROFILE_REGISTRY:
        profile = DEFAULT_PROFILE_KEY

    # Reranker config
    reranker_model_key = raw.get("reranker_model_key")
    reranker_enabled = raw.get("reranker_enabled", True)
    if isinstance(reranker_enabled, str):
        reranker_enabled = reranker_enabled.lower() in ("true", "1", "yes")

    cfg = {
        "model_key": model_key or DEFAULT_MODEL_KEY,
        "db_path": raw.get("db_path", DEFAULT_DB_PATH),
        "custom_model_id": raw.get("custom_model_id"),
        "profile": profile,
        "reranker_enabled": reranker_enabled,
        "reranker_model_key": reranker_model_key or DEFAULT_RERANKER_MODEL_KEY,
    }
    return cfg


def _resolve_db_path_value(raw: str) -> str:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return str(path)

    # Keep DB location stable across editor workspaces:
    # resolve relative paths against code-radar app root, not process cwd.
    return str((APP_ROOT / path).resolve())


def resolve_db_path(cfg: dict[str, Any] | None = None) -> str:
    """Resolve DB path with environment override.

    Priority:
      1. CODE_DB_PATH env var
      2. Config db_path
      3. DEFAULT_DB_PATH

    Relative paths are resolved against ``APP_ROOT`` so workspace cwd changes
    (e.g. per-project MCP launches) do not move DB storage unexpectedly.
    """
    env_path = get_env("CODE_DB_PATH")
    if env_path:
        return _resolve_db_path_value(env_path)

    if cfg is None:
        cfg = load_config()

    raw = str(cfg.get("db_path", DEFAULT_DB_PATH))
    return _resolve_db_path_value(raw)


def save_config(cfg: dict[str, Any]) -> None:
    """Persist configuration to disk."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    # Only store what we need
    payload: dict[str, Any] = {
        "model_key": cfg.get("model_key", DEFAULT_MODEL_KEY),
        "db_path": cfg.get("db_path", DEFAULT_DB_PATH),
    }
    if cfg.get("custom_model_id"):
        payload["custom_model_id"] = cfg["custom_model_id"]
    if cfg.get("profile") in PROFILE_REGISTRY:
        payload["profile"] = cfg["profile"]
    # Reranker settings
    payload["reranker_enabled"] = cfg.get("reranker_enabled", True)
    payload["reranker_model_key"] = cfg.get("reranker_model_key", DEFAULT_RERANKER_MODEL_KEY)
    CONFIG_FILE.write_text(json.dumps(payload, indent=2) + "\n")


def resolve_model_id(model_key: str | None = None) -> str:
    """Resolve a model key to a full HuggingFace model ID.

    Falls back to the configured model if *model_key* is ``None``.
    Supports full HF IDs (containing ``/``) as passthrough.
    """
    if model_key is None:
        mk = load_config().get("model_key") or DEFAULT_MODEL_KEY
    else:
        mk = model_key

    # Full HF ID (contains "/") -> passthrough
    if "/" in mk:
        return mk

    # Registry key -> look up
    try:
        return get_model_config(mk).id
    except ValueError:
        # Custom ID stored in config?
        cfg = load_config()
        custom = cfg.get("custom_model_id")
        if custom:
            return custom
        return MODEL_REGISTRY[DEFAULT_MODEL_KEY].id


def resolve_reranker_model_id(model_key: str | None = None) -> str:
    """Resolve a reranker model key to a full HuggingFace model ID.

    Falls back to the configured reranker model if *model_key* is ``None``.
    Supports full HF IDs (containing ``/``) as passthrough.
    """
    if model_key is None:
        mk = load_config().get("reranker_model_key") or DEFAULT_RERANKER_MODEL_KEY
    else:
        mk = model_key

    # Full HF ID (contains "/") -> passthrough
    if "/" in mk:
        return mk

    # Registry key -> look up
    try:
        return get_reranker_model_config(mk).id
    except ValueError:
        return RERANKER_REGISTRY[DEFAULT_RERANKER_MODEL_KEY].id


def get_reranker_enabled() -> bool:
    """Check whether reranking is enabled.

    Priority:
      1. CODE_RERANKER_ENABLED env var
      2. Config file
      3. Default: True
    """
    env = get_env("CODE_RERANKER_ENABLED")
    if env is not None:
        return env.lower() in ("true", "1", "yes")
    return load_config().get("reranker_enabled", True)
