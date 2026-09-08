"""General package helpers."""

from __future__ import annotations

from functools import wraps
import hashlib
from importlib.util import find_spec
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from platformdirs import user_cache_dir
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Callable

EPSG_WGS84 = 4326

logger_ = logging.getLogger(__name__)


def depends_on_optional[**P, T](
    module_name: str,
    install_hint: str | None = None,
) -> Callable[[Callable[P, T]], Callable[P, T]]:
    """Require an optional dependency when the decorated function is called.

    Parameters
    ----------
    module_name : str
        Module to require.
    install_hint : str or None, optional
        Installation instruction for the error message.

    Returns
    -------
    Callable[[Callable[P, T]], Callable[P, T]]
        Dependency-checking decorator.
    """

    def decorator(func: Callable[P, T]) -> Callable[P, T]:
        @wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> T:
            try:
                spec = find_spec(module_name)
            except (ModuleNotFoundError, ValueError):
                spec = None
            if spec is None:
                hint = f" Install with: {install_hint}" if install_hint else ""
                msg = f"The package '{module_name}' is required for '{func.__name__}()'.{hint}"
                raise ImportError(msg)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def get_cache_dir(cache_dir: str | Path | None = None) -> Path:
    r"""
    Get cache directory with fallback to system cache.

    Determines the cache directory to use based on the following priority:
    1. Provided `cache_dir` parameter
    2. 'FREEZEBASE_CACHE' environment variable
    3. System cache directory (via platformdirs)

    Parameters
    ----------
    cache_dir : str, Path, or None, default: None
        Explicit cache directory.

    Returns
    -------
    Path
        Selected cache directory.

    Notes
    -----
    The system cache location varies by platform:

    - Linux: ~/.cache/freezebase
    - macOS: ~/Library/Caches/freezebase
    - Windows: %LOCALAPPDATA%\\freezebase\\Cache
    """
    cache_dir = cache_dir or os.getenv("FREEZEBASE_CACHE")
    if cache_dir:
        return Path(cache_dir).expanduser()

    system_cache = user_cache_dir("freezebase")
    logger_.warning(
        "Using system cache directory: '%s' "
        "Set 'FREEZEBASE_CACHE' environment variable or pass `cache_dir` parameter "
        "to customize location.",
        system_cache,
    )
    return Path(system_cache)


def get_data_dir(data_dir: str | Path | None = None) -> Path:
    """Return the explicit or ``FREEZEBASE_DATA`` directory.

    Parameters
    ----------
    data_dir : str, Path, or None, default: None
        Explicit data directory.

    Returns
    -------
    Path
        Selected data directory.

    Raises
    ------
    ValueError
        If neither source provides a directory.
    """
    data_dir = data_dir or os.getenv("FREEZEBASE_DATA")
    if data_dir:
        return Path(data_dir).expanduser()
    msg = (
        "Data directory must be provided via 'data_dir' parameter "
        "or 'FREEZEBASE_DATA' environment variable"
    )
    raise ValueError(msg)


def shorten_string(string: str, n: int) -> str:
    """Shorten a string to ``n`` characters with a middle ellipsis.

    Parameters
    ----------
    string : str
        Input string.
    n : int
        Maximum length; negative values are treated as zero.

    Returns
    -------
    str
        Shortened string, hard-truncated when ``n < 3``.
    """
    n = max(n, 0)
    if len(string) <= n:
        return string
    ellipsis_ = "..."
    if n <= len(ellipsis_):
        return string[:n]
    budget = n - len(ellipsis_)
    n_1 = budget // 2 + budget % 2
    n_2 = budget // 2
    return string[:n_1] + ellipsis_ + string[len(string) - n_2 :]


def get_credentials_from_env(username_key: str, password_key: str) -> tuple[str, str]:
    """Read a non-empty username and password from the environment.

    Parameters
    ----------
    username_key : str
        Username variable.
    password_key : str
        Password variable.

    Returns
    -------
    tuple[str, str]
        Username and password.

    Raises
    ------
    KeyError
        If either variable is missing or empty.
    """
    username = os.getenv(username_key)
    password = os.getenv(password_key)

    if not username or not password:
        missing_keys = [
            key for key, value in ((username_key, username), (password_key, password)) if not value
        ]
        msg = f"Environment variables not set or empty: {', '.join(missing_keys)}"
        raise KeyError(msg)

    return username, password


_HASH_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def file_sha256(path: str | Path | UPath, *, chunk_size: int = _HASH_CHUNK_SIZE) -> str:
    """Return a file's SHA-256 hex digest, read in chunks.

    Parameters
    ----------
    path : str, Path, or UPath
        File to hash. A remote path is streamed, never held in memory whole.
    chunk_size : int, optional
        Bytes read per iteration.

    Returns
    -------
    str
        The lowercase hexadecimal SHA-256 digest.
    """
    digest = hashlib.sha256()
    with UPath(path).open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def format_valid_options(d: dict[Any, Any]) -> str:
    r"""Format dictionary items as a bullet list.

    Parameters
    ----------
    d : dict
        Items to format.

    Returns
    -------
    str
        Formatted list.
    """
    max_key_length = max(len(str(k)) for k in d)
    items = [f"- '{k}':{' ' * (max_key_length - len(str(k)) + 1)}{v}" for k, v in d.items()]
    return "\n".join(items)
