from collections import deque

import mlx.core as mx

from code_radar.envvars import get_env
from code_radar.logging import get_logger

log = get_logger(__name__)


# Environment helpers
def env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = get_env(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        log.warning("Invalid int for %s=%r, using default=%d", name, raw, default)
        return default
    return max(minimum, value)


def env_flag(name: str) -> bool:
    raw = (get_env(name, "") or "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def is_oom_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "oom" in msg or "memory" in msg


# Batch planning
def plan_batches(
    items: list[tuple[int, list[int]]],
    max_items: int,
    max_batch_tokens: int,
) -> deque[list[tuple[int, list[int]]]]:
    """Sort items by length and pack them into batches respecting both limits."""
    items.sort(key=lambda x: len(x[1]))

    batches: deque[list[tuple[int, list[int]]]] = deque()
    cur: list[tuple[int, list[int]]] = []
    cur_max_len = 0

    for item in items:
        _, ids = item
        seq_len = len(ids)
        next_max_len = max(cur_max_len, seq_len)
        next_count = len(cur) + 1
        next_tokens = next_count * next_max_len

        if cur and (next_count > max_items or next_tokens > max_batch_tokens):
            batches.append(cur)
            cur = [item]
            cur_max_len = seq_len
        else:
            cur.append(item)
            cur_max_len = next_max_len

    if cur:
        batches.append(cur)

    return batches


# Similarity
def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        log.warning(
            "cosine_similarity: dimension mismatch or empty vectors  |  len(a)=%d  len(b)=%d",
            len(a) if a else 0,
            len(b) if b else 0,
        )
        return 0.0

    va = mx.array(a)
    vb = mx.array(b)
    dot = mx.sum(va * vb)
    na = mx.sqrt(mx.sum(va**2))
    nb = mx.sqrt(mx.sum(vb**2))

    denom = na * nb
    if float(denom) == 0.0:
        return 0.0

    return float(dot / denom)
