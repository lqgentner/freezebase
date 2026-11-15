"""Helpers for deprecating public names."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, cast
import warnings

if TYPE_CHECKING:
    from collections.abc import Callable


def warn_deprecated(old: str, new: str) -> None:
    """Warn that ``old`` is deprecated in favor of ``new``, from a wrapper of ``old``."""
    msg = f"{old} is deprecated and will be removed in a future release; use {new}."
    # Past this helper and the wrapper, so the warning points at the caller.
    warnings.warn(msg, DeprecationWarning, stacklevel=3)


def deprecated[**P, T](old: str, new: str) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Make the function public as ``old`` warn on every call that ``new`` replaces it."""

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            warn_deprecated(old, new)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecated_alias[F: Callable[..., Any]](func: F, old: str, new: str) -> F:
    """Return ``func`` under the deprecated name ``old``, warning on every call."""

    @functools.wraps(func)
    def alias(*args: Any, **kwargs: Any) -> Any:
        warn_deprecated(old, new)
        return func(*args, **kwargs)

    alias.__name__ = alias.__qualname__ = old.rpartition(".")[2]
    alias.__doc__ = f"Deprecated alias of ``{new}``."
    # Typed as func itself, so a type checker binds an alias defined in a class body.
    return cast("F", alias)
