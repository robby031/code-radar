from code_radar.envvars import get_env
from code_radar.logging import get_logger

log = get_logger(__name__)


class RerankTimeoutError(TimeoutError):
    """Raised when reranking exceeds configured latency budget."""


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
