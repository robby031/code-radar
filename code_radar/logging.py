import logging
import sys
from typing import Literal

_LOG = logging.getLogger("code_radar")
_LOG_LEVEL: int = logging.INFO

_FORMAT = "%(asctime)s  %(levelname)-8s%(name)-20s  %(message)s"
_DATE_FMT = "%H:%M:%S"


def _setup() -> None:
    """Initialise the ``code_radar`` logger once."""
    if _LOG.handlers:
        return

    handler = logging.StreamHandler(stream=sys.stderr)
    handler.setFormatter(logging.Formatter(_FORMAT, datefmt=_DATE_FMT))
    _LOG.addHandler(handler)
    _LOG.setLevel(_LOG_LEVEL)

    # Do NOT propagate to root - otherwise libraries such as uvicorn /
    # rich / FastMCP that attach their own root handler cause double
    # logging (the ugly duplication the user saw).
    _LOG.propagate = False


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the ``code_radar`` hierarchy.

    ``name`` can be a dotted path (e.g. ``"code_radar.server.tools"``);
    only the last component is kept for brevity.
    """
    _setup()
    return _LOG.getChild(name.split(".")[-1])


LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def set_level(level: LogLevel | int) -> None:
    """Change the root log level at runtime.

    Accepts a string (``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``)
    or a numeric ``logging.*`` constant.
    """
    if isinstance(level, str):
        level = getattr(logging, level.upper(), logging.INFO)
    _LOG.setLevel(level)


# Bootstrap on import
_setup()
