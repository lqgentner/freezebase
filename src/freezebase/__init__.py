"""Core geospatial and I/O helpers."""

import functools
from importlib.metadata import version as _version
import logging
from typing import Literal

__version__ = _version("freezebase")

logger = logging.getLogger(__name__)


@functools.cache
def _ensure_handler() -> logging.Handler:
    """Return the package's cached stream handler."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return handler


def set_loglevel(
    level: Literal["notset", "debug", "info", "warning", "error", "critical"],
) -> None:
    """Set the package log level.

    Parameters
    ----------
    level : {"notset", "debug", "info", "warning", "error", "critical"}
        Log level.
    """
    logger.setLevel(level.upper())
    _ensure_handler().setLevel(level.upper())
