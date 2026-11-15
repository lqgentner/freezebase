"""Helpers for deprecating public names."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable


def deprecated_alias[F: Callable[..., Any]](func: F, old: str, new: str) -> F:
    """Return ``func`` under the deprecated name ``old``, warning on every call."""

    @functools.wraps(func)
    def alias(*args: Any, **kwargs: Any) -> Any:
        msg = f"{old} is deprecated and will be removed in a future release; use {new}."
        warnings.warn(msg, DeprecationWarning, stacklevel=2)
        return func(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old.rpartition(".")[2]
    alias.__doc__ = f"Deprecated alias of ``{new}``."
    # Typed as func itself, so a type checker binds an alias defined in a class body.
    return cast("F", alias)
