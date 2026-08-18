"""Initialize the freezebase module."""

import functools
from importlib.metadata import version as _version
import logging
from typing import Literal

__version__ = _version("freezebase")

logger = logging.getLogger(__name__)


@functools.cache
def _ensure_handler() -> logging.Handler:
    """
    Attach a `StreamHandler` to the root logger.

    Return this handler every time this function is called.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)
    return handler


def set_loglevel(
    level: Literal["notset", "debug", "info", "warning", "error", "critical"],
) -> None:
    """
    Configure freezebase's logging levels.

    Call `set_loglevel("info")` or `set_loglevel("debug")` to get additional debugging information.

    Parameters
    ----------
    level : {"notset", "debug", "info", "warning", "error", "critical"}
        The log level of the handler.

    Notes
    -----
    Copy of `matplotlib.set_loglevel`.

    """
    logger.setLevel(level.upper())
    _ensure_handler().setLevel(level.upper())
