from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ModelConfig:
    """Konfigurasi model embedding yang tersedia."""
    id: str
    name: str
    description: str
    ram_gb: float
    speed_tier: Literal["fast", "medium", "slow"]
    accuracy_tier: Literal["good", "very_good", "excellent"]
    is_multimodal: bool = False


@dataclass(frozen=True)
class RerankerModelConfig:
    """Konfigurasi model reranker (cross-encoder) yang tersedia."""
    id: str
    name: str
    description: str
    ram_gb: float
    speed_tier: Literal["fast", "medium", "slow"]
    accuracy_tier: Literal["good", "very_good", "excellent"]
    default_batch_size: int = 4


# Registry semua model embedding yang didukung
MODEL_REGISTRY: dict[str, ModelConfig] = {
    "qwen3-0.6b-4bit": ModelConfig(
        id="mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
        name="Qwen3 0.6B (4-bit)",
        description="Ultra-fast, minimal RAM. Best for low-spec machines or large codebases.",
        ram_gb=0.5,
        speed_tier="fast",
        accuracy_tier="good",
    ),
    "qwen3-4b-4bit": ModelConfig(
        id="mlx-community/Qwen3-Embedding-4B-4bit-DWQ",
        name="Qwen3 4B (4-bit)",
        description="Balanced speed and accuracy. Recommended for most users.",
        ram_gb=2.5,
        speed_tier="medium",
        accuracy_tier="very_good",
    ),
    "qwen3-8b-4bit": ModelConfig(
        id="mlx-community/Qwen3-Embedding-8B-4bit-DWQ",
        name="Qwen3 8B (4-bit)",
        description="Maximum accuracy for semantic search. Requires 8GB+ RAM.",
        ram_gb=5.0,
        speed_tier="slow",
        accuracy_tier="excellent",
    ),
}

# Default model yang direkomendasikan
DEFAULT_MODEL_KEY = "qwen3-0.6b-4bit"


RERANKER_REGISTRY: dict[str, RerankerModelConfig] = {
    "reranker-0.6b-4bit": RerankerModelConfig(
        id="mlx-community/Qwen3-Reranker-0.6B-4bit",
        name="Qwen3 Reranker 0.6B (4-bit)",
        description="Ultra-fast cross-encoder. Minimal RAM impact (~0.6 GB). Ideal untuk reranking cepat pada low-spec machines.",
        ram_gb=0.6,
        speed_tier="fast",
        accuracy_tier="good",
        default_batch_size=8,
    ),
    "reranker-2b-4bit": RerankerModelConfig(
        id="mlx-community/Qwen3-VL-Reranker-2B-4bit",
        name="Qwen3 VL Reranker 2B (4-bit)",
        description="Sweet spot untuk reranking codebase. Multimodal-ready, balance speed & accuracy.",
        ram_gb=1.5,
        speed_tier="medium",
        accuracy_tier="very_good",
        default_batch_size=4,
    ),
    "reranker-2b-8bit": RerankerModelConfig(
        id="mlx-community/Qwen3-VL-Reranker-2B-8bit",
        name="Qwen3 VL Reranker 2B (8-bit)",
        description="High-precision cross-encoder. Akurasi maksimal untuk hasil search paling relevan.",
        ram_gb=2.8,
        speed_tier="slow",
        accuracy_tier="excellent",
        default_batch_size=2,
    ),
}

DEFAULT_RERANKER_MODEL_KEY = "reranker-0.6b-4bit"


# Resolver Functions
def get_model_config(key: str) -> ModelConfig:
    if key not in MODEL_REGISTRY:
        available = ", ".join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model '{key}'. Available: {available}")
    return MODEL_REGISTRY[key]


def get_reranker_model_config(key: str) -> RerankerModelConfig:
    if key not in RERANKER_REGISTRY:
        available = ", ".join(RERANKER_REGISTRY.keys())
        raise ValueError(f"Unknown reranker model '{key}'. Available: {available}")
    return RERANKER_REGISTRY[key]


def list_models() -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "name": cfg.name,
            "description": cfg.description,
            "ram_gb": cfg.ram_gb,
            "speed": cfg.speed_tier,
            "accuracy": cfg.accuracy_tier,
            "multimodal": cfg.is_multimodal,
        }
        for key, cfg in MODEL_REGISTRY.items()
    ]


def list_reranker_models() -> list[dict[str, object]]:
    return [
        {
            "key": key,
            "name": cfg.name,
            "description": cfg.description,
            "ram_gb": cfg.ram_gb,
            "speed": cfg.speed_tier,
            "accuracy": cfg.accuracy_tier,
        }
        for key, cfg in RERANKER_REGISTRY.items()
    ]
