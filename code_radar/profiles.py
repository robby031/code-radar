from dataclasses import dataclass, field
from typing import Any
import os

from code_radar.envvars import has_env
from code_radar.logging import get_logger
from code_radar.models import MODEL_REGISTRY, RERANKER_REGISTRY

log = get_logger(__name__)


@dataclass(frozen=True)
class ProfileConfig:
    key: str
    name: str
    description: str
    model_key: str
    reranker_model_key: str | None = None
    tuning: dict[str, int | float] = field(default_factory=dict)


PROFILE_REGISTRY: dict[str, ProfileConfig] = {
    "fast": ProfileConfig(
        key="fast",
        name="Fast (0.6B)",
        description="Maximum throughput with minimal RAM/storage pressure. "
                    "Pair: Qwen3-Embedding-0.6B + Qwen3-Reranker-0.6B.",
        model_key="qwen3-0.6b-4bit",
        reranker_model_key="reranker-0.6b-4bit",
        tuning={
            "CODE_EMBED_BATCH_SIZE": 32,
            "CODE_EMBED_MAX_BATCH_TOKENS": 65536,
            "CODE_EMBED_MAX_TOKENS_PER_TEXT": 2048,
            "CODE_EMBED_CLEANUP_INTERVAL": 16,
            "CODE_RERANKER_BATCH_SIZE": 4,
            "CODE_RERANKER_MAX_BATCH_SIZE": 4,
            "CODE_RERANKER_DEFAULT_CANDIDATES": 12,
            "CODE_RERANKER_MAX_CANDIDATES": 30,
            "CODE_SEARCH_NO_RERANK_MAX_CANDIDATES": 20,
            "CODE_RERANKER_DOC_MAX_TOKENS": 256,
            "CODE_RERANKER_DOC_MAX_CHARS": 1000,
            "CODE_RERANKER_QUERY_MAX_TOKENS": 128,
            "CODE_RERANKER_ADAPTIVE_ENABLED": 1,
            "CODE_RERANKER_FORCE_COMPLEX_QUERY": 1,
            "CODE_RERANKER_COMPLEX_QUERY_MIN_WORDS": 8,
            "CODE_RERANKER_CONFIDENCE_TOP1_MIN": 0.80,
            "CODE_RERANKER_TOP1_TOP2_GAP_MIN": 0.04,
            "CODE_RERANKER_TOP5_SPREAD_MIN": 0.06,
            "CODE_RERANKER_SKIP_SIMPLE_QUERY": 1,
            "CODE_RERANKER_SIMPLE_QUERY_MAX_WORDS": 5,
            "CODE_RERANKER_TIMEOUT_MS": 180,
            "CODE_RERANKER_WORKERS": 1,
            "CODE_SYNC_CHUNK_BUFFER_SIZE": 768,
            "CODE_SYNC_DELETE_BATCH_SIZE": 1000,
            "CODE_SYNC_UPSERT_MAX_RETRIES": 3,
            "CODE_SYNC_DELETE_MAX_RETRIES": 3,
            "CODE_SYNC_UPSERT_BACKOFF_SEC": 0.25,
            "CODE_SYNC_DELETE_BACKOFF_SEC": 0.25,
            "CODE_CHROMA_UPSERT_BATCH_SIZE": 500,
        },
    ),
    "accurate": ProfileConfig(
        key="accurate",
        name="Accurate (4B)",
        description="Higher semantic quality with balanced speed. "
                    "Pair: Qwen3-Embedding-4B + Qwen3-Reranker-2B-4bit.",
        model_key="qwen3-4b-4bit",
        reranker_model_key="reranker-2b-4bit",
        tuning={
            "CODE_EMBED_BATCH_SIZE": 4,
            "CODE_EMBED_MAX_BATCH_TOKENS": 8192,
            "CODE_EMBED_MAX_TOKENS_PER_TEXT": 2048,
            "CODE_EMBED_CLEANUP_INTERVAL": 4,
            "CODE_RERANKER_BATCH_SIZE": 2,
            "CODE_RERANKER_MAX_BATCH_SIZE": 4,
            "CODE_RERANKER_DEFAULT_CANDIDATES": 12,
            "CODE_RERANKER_MAX_CANDIDATES": 30,
            "CODE_SEARCH_NO_RERANK_MAX_CANDIDATES": 20,
            "CODE_RERANKER_DOC_MAX_TOKENS": 256,
            "CODE_RERANKER_DOC_MAX_CHARS": 1000,
            "CODE_RERANKER_QUERY_MAX_TOKENS": 128,
            "CODE_RERANKER_ADAPTIVE_ENABLED": 1,
            "CODE_RERANKER_FORCE_COMPLEX_QUERY": 1,
            "CODE_RERANKER_COMPLEX_QUERY_MIN_WORDS": 8,
            "CODE_RERANKER_CONFIDENCE_TOP1_MIN": 0.80,
            "CODE_RERANKER_TOP1_TOP2_GAP_MIN": 0.04,
            "CODE_RERANKER_TOP5_SPREAD_MIN": 0.06,
            "CODE_RERANKER_SKIP_SIMPLE_QUERY": 1,
            "CODE_RERANKER_SIMPLE_QUERY_MAX_WORDS": 5,
            "CODE_RERANKER_TIMEOUT_MS": 220,
            "CODE_RERANKER_WORKERS": 1,
            "CODE_SYNC_CHUNK_BUFFER_SIZE": 256,
            "CODE_SYNC_DELETE_BATCH_SIZE": 1000,
            "CODE_SYNC_UPSERT_MAX_RETRIES": 3,
            "CODE_SYNC_DELETE_MAX_RETRIES": 3,
            "CODE_SYNC_UPSERT_BACKOFF_SEC": 0.25,
            "CODE_SYNC_DELETE_BACKOFF_SEC": 0.25,
            "CODE_CHROMA_UPSERT_BATCH_SIZE": 400,
        },
    ),
    "precise": ProfileConfig(
        key="precise",
        name="Precise (8B+2B-8bit)",
        description="Maximum search quality. "
                    "Pair: Qwen3-Embedding-8B + Qwen3-Reranker-2B-8bit. Requires 8GB+ RAM.",
        model_key="qwen3-8b-4bit",
        reranker_model_key="reranker-2b-8bit",
        tuning={
            "CODE_EMBED_BATCH_SIZE": 2,
            "CODE_EMBED_MAX_BATCH_TOKENS": 8192,
            "CODE_EMBED_MAX_TOKENS_PER_TEXT": 2048,
            "CODE_EMBED_CLEANUP_INTERVAL": 2,
            "CODE_RERANKER_BATCH_SIZE": 2,
            "CODE_RERANKER_MAX_BATCH_SIZE": 2,
            "CODE_RERANKER_DEFAULT_CANDIDATES": 12,
            "CODE_RERANKER_MAX_CANDIDATES": 30,
            "CODE_SEARCH_NO_RERANK_MAX_CANDIDATES": 20,
            "CODE_RERANKER_DOC_MAX_TOKENS": 256,
            "CODE_RERANKER_DOC_MAX_CHARS": 1000,
            "CODE_RERANKER_QUERY_MAX_TOKENS": 128,
            "CODE_RERANKER_ADAPTIVE_ENABLED": 1,
            "CODE_RERANKER_FORCE_COMPLEX_QUERY": 1,
            "CODE_RERANKER_COMPLEX_QUERY_MIN_WORDS": 8,
            "CODE_RERANKER_CONFIDENCE_TOP1_MIN": 0.80,
            "CODE_RERANKER_TOP1_TOP2_GAP_MIN": 0.04,
            "CODE_RERANKER_TOP5_SPREAD_MIN": 0.06,
            "CODE_RERANKER_SKIP_SIMPLE_QUERY": 1,
            "CODE_RERANKER_SIMPLE_QUERY_MAX_WORDS": 5,
            "CODE_RERANKER_TIMEOUT_MS": 280,
            "CODE_RERANKER_WORKERS": 1,
            "CODE_SYNC_CHUNK_BUFFER_SIZE": 128,
            "CODE_SYNC_DELETE_BATCH_SIZE": 500,
            "CODE_SYNC_UPSERT_MAX_RETRIES": 3,
            "CODE_SYNC_DELETE_MAX_RETRIES": 3,
            "CODE_SYNC_UPSERT_BACKOFF_SEC": 0.5,
            "CODE_SYNC_DELETE_BACKOFF_SEC": 0.5,
            "CODE_CHROMA_UPSERT_BATCH_SIZE": 200,
        },
    ),
}

DEFAULT_PROFILE_KEY = "fast"


def get_profile_config(key: str) -> ProfileConfig:
    if key not in PROFILE_REGISTRY:
        available = ", ".join(PROFILE_REGISTRY.keys())
        raise ValueError(f"Unknown profile '{key}'. Available: {available}")
    profile = PROFILE_REGISTRY[key]
    if profile.model_key not in MODEL_REGISTRY:
        raise ValueError(
            f"Profile '{key}' references unknown model_key '{profile.model_key}'"
        )
    if profile.reranker_model_key is not None and profile.reranker_model_key not in RERANKER_REGISTRY:
        raise ValueError(
            f"Profile '{key}' references unknown reranker_model_key "
            f"'{profile.reranker_model_key}'"
        )
    return profile


def list_profiles() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key, p in PROFILE_REGISTRY.items():
        out.append(
            {
                "key": key,
                "name": p.name,
                "description": p.description,
                "model_key": p.model_key,
                "reranker_model_key": p.reranker_model_key,
                "tuning": p.tuning,
            }
        )
    return out


def apply_profile_env_defaults(profile_key: str) -> ProfileConfig:
    profile = get_profile_config(profile_key)

    for k, v in profile.tuning.items():
        if has_env(k):
            continue
        os.environ[k] = str(v)

    log.info(
        "Applied profile defaults  |  profile=%s  |  model=%s  |  reranker=%s",
        profile.key,
        profile.model_key,
        profile.reranker_model_key,
    )
    return profile
