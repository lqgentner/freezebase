"""Helpers for deprecating public names."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable


def deprecated_alias[**P, R](func: Callable[P, R], old: str, new: str) -> Callable[P, R]:
    """Return ``func`` under the deprecated name ``old``, warning on every call."""

    @functools.wraps(func)
    def alias(*args: P.args, **kwargs: P.kwargs) -> R:
        msg = f"{old} is deprecated and will be removed in a future release; use {new}."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old.rpartition(".")[2]
    alias.__doc__ = f"Deprecated alias of ``{new}``."
    return alias
