"""HTTP downloads with retries and progress reporting."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from functools import partial
import logging
from pathlib import Path
import re
from secrets import token_hex
from typing import TYPE_CHECKING, Any, Literal, Self, TypeVar, cast, overload
from urllib.parse import unquote, urljoin, urlparse

import requests
from rich.progress import (
    BarColumn,
    DownloadColumn,
    ProgressColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TransferSpeedColumn,
)
import tenacity

from freezebase.progress import create_progress
from freezebase.utils import shorten_string

if TYPE_CHECKING:
    from collections.abc import Iterable

    from requests.auth import AuthBase
    from upath import UPath

DEFAULT_TIMEOUT = 30
CHUNK_SIZE = 1024 * 1024  # 1 MiB
REMOTE_BLOCK_SIZE = 64 * 1024 * 1024  # 64 MiB

MAX_REDIRECTS = 30

logger = logging.getLogger(__name__)

TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 502, 503, 504})

# Redirects that may change POST to GET.
_HTTP_MOVED_PERMANENTLY = 301
_HTTP_FOUND = 302
_HTTP_SEE_OTHER = 303

_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)},
)

WrappedFn = TypeVar("WrappedFn", bound=Callable[..., Any])

_DOWNLOAD_COLUMNS: list[str | ProgressColumn] = [
    TextColumn("{task.fields[filename]}"),
    BarColumn(),
    TaskProgressColumn(),
    "•",
    DownloadColumn(),
    "•",
    TransferSpeedColumn(),
    "•",
    TimeElapsedColumn(),
    "•",
    TimeRemainingColumn(),
]


def _is_transient_request_error(
    exception: BaseException,
    *,
    extra_status_codes: frozenset[int] = frozenset(),
) -> bool:
    """Return whether a request error is transient."""
    if isinstance(exception, (requests.exceptions.ConnectionError, requests.exceptions.Timeout)):
        return True
    if isinstance(exception, requests.exceptions.HTTPError):
        response = exception.response
        return response is not None and (
            response.status_code in TRANSIENT_HTTP_STATUS_CODES
            or response.status_code in extra_status_codes
        )
    return False


def retry_request(
    logger: logging.Logger,
    *,
    extra_status_codes: frozenset[int] = frozenset(),
) -> Callable[[WrappedFn], WrappedFn]:
    """
    Create a `tenacity` retry decorator for HTTP requests using the `requests` library.

    This decorator will automatically retry functions that make HTTP requests
    when they encounter transient network or server-side errors. It is a
    pre-configured `tenacity.retry` function with the following behaviors:

    - ``reraise=True`` to see encountered exception at the end of the stack trace
    - Retries ``ConnectionError`` and ``Timeout`` errors of the ``requests``
      library, as well as HTTP Errors 408, 429, 502, 503, 504 (plus any status
      codes passed via ``extra_status_codes``).
    - Maximum attempts: 3
    - Wait strategy: Exponential backoff with jitter (the latter for resolving
      contention between multiple processes)

    Parameters
    ----------
    logger : logging.Logger
        Retry logger.
    extra_status_codes : frozenset of int, optional
        Additional transient status codes.

    Returns
    -------
    Callable[[WrappedFn], WrappedFn]
        Configured retry decorator.
    """
    return tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10) + tenacity.wait_random(0, 2),
        retry=tenacity.retry_if_exception(
            partial(_is_transient_request_error, extra_status_codes=extra_status_codes),
        ),
        # tenacity's LoggerProtocol demands a `log()` taking arbitrary keywords,
        # which `logging.Logger.log` does not; it only ever gets called positionally.
        # pyrefly: ignore [bad-argument-type]
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
    )


class HTTPDownloader:
    """Stream HTTP downloads through a reusable session.

    Built upon `requests`. Inspired by `pooch.HTTPDownloader`.
    Supports downloading with GET and POST requests. Manual redirects
    prevent credentials from reaching untrusted hosts. Use as
    a context manager or call :meth:`close` when finished.
    """

    def __init__(
        self,
        *,
        method: Literal["GET", "POST"] = "GET",
        auth: tuple[str, str] | AuthBase | None = None,
        trusted_hosts: str | Iterable[str] | None = None,
        timeout: float | tuple[float, float] = DEFAULT_TIMEOUT,
        progress: bool = True,
    ) -> None:
        """Initialize the downloader.

        Parameters
        ----------
        method : {"GET", "POST"}
            HTTP method used to request the file.
        auth : tuple[str, str] or instance of AuthBase subclass, optional
            Authentication object or ``(user, password)`` pair.
        trusted_hosts : str or iterable of str, optional
            Redirect hosts allowed to receive authentication.
        timeout : float or tuple[float, float]
            Request timeout or ``(connect, read)`` timeouts.
        progress : bool
            Show download progress.
        """
        self.session = requests.Session()
        self.method = method
        # Per-request auth can be dropped for an unsafe redirect.
        self._auth = auth
        if trusted_hosts is None:
            trusted_hosts = []
        elif isinstance(trusted_hosts, str):
            trusted_hosts = [trusted_hosts]
        self.trusted_hosts = trusted_hosts
        self.timeout = timeout
        self.show_progress = progress

    @overload
    def __call__(
        self,
        url: str,
        save_dir: str | Path,
        filename: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Path: ...
    @overload
    def __call__(
        self,
        url: str,
        save_dir: UPath,
        filename: str | None = None,
        *,
        overwrite: bool = False,
    ) -> UPath: ...
    @retry_request(logger=logger)
    def __call__(
        self,
        url: str,
        save_dir: str | Path | UPath,
        filename: str | None = None,
        *,
        overwrite: bool = False,
    ) -> Path | UPath:
        """Download a URL to a local or remote directory.

        Parameters
        ----------
        url : str
            Source URL.
        save_dir : str, Path, or UPath
            Destination directory.
        filename : str, optional
            Output name, inferred from the response or URL when omitted.
        overwrite : bool, default False
            Replace an existing target.

        Returns
        -------
        Path or UPath
            Downloaded path.

        Raises
        ------
        ValueError
            If the filename is unsafe.
        FileExistsError
            If the target exists unless ``overwrite`` is true.
        """
        # Closing the session alone does not close this streamed response.
        with contextlib.closing(self._follow_redirects(url)) as response:
            response.raise_for_status()

            if not _is_downloadable_content(response):
                msg = (
                    f"No downloadable file found for URL: '{url}'. "
                    "Make sure the authentication is correct."
                )
                raise RuntimeError(msg)

            explicit_filename = filename is not None
            if not filename:
                cd = response.headers.get("Content-Disposition")
                filename = _extract_filename_from_cd(cd) or _extract_filename_from_url(url)
                if not filename:
                    msg = "Could not infer filename. Please specify with `filename=` argument."
                    raise RuntimeError(msg)

            safe_name = _sanitize_filename(filename, explicit=explicit_filename)

            if isinstance(save_dir, str):
                save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            filepath = _resolve_within(save_dir, safe_name)
            logger.info("Downloading '%s' from '%s' to '%s'.", safe_name, url, str(save_dir))
            return _write_file(
                response,
                filepath,
                show_progress=self.show_progress,
                overwrite=overwrite,
            )

    def _follow_redirects(self, url: str) -> requests.Response:
        """Follow bounded redirects without leaking credentials."""
        auth = self._auth
        method: str = self.method

        for _ in range(MAX_REDIRECTS + 1):
            response = self.session.request(
                method,
                url=url,
                auth=auth,
                timeout=self.timeout,
                stream=True,
                allow_redirects=False,
            )
            if not response.is_redirect:
                return response

            location = response.headers["Location"]
            prev_parsed = urlparse(response.url)
            response.close()

            new_url = urljoin(response.url, location)
            new_parsed = urlparse(new_url)
            if new_parsed.hostname is None:
                msg = "Hostname not found in redirect Location header."
                raise RuntimeError(msg)

            is_trusted = (
                new_parsed.hostname == prev_parsed.hostname
                or new_parsed.hostname in self.trusted_hosts
            )
            is_downgrade = prev_parsed.scheme == "https" and new_parsed.scheme != "https"
            if not is_trusted or is_downgrade:
                auth = None

            method = _rewrite_redirect_method(response.status_code, method)
            url = new_url

        msg = f"Exceeded maximum of {MAX_REDIRECTS} redirects for URL."
        raise RuntimeError(msg)

    def close(self) -> None:
        """Close the session."""
        self.session.close()

    def __enter__(self) -> Self:
        """Return this downloader."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session."""
        self.close()

    def __del__(self) -> None:
        """Close the session, suppressing finalization errors."""
        with contextlib.suppress(Exception):
            self.close()


def _is_downloadable_content(response: requests.Response) -> bool:
    """Return whether a response contains downloadable content."""
    cd = response.headers.get("Content-Disposition")
    if cd:
        return True
    content_type = response.headers.get("Content-Type", "").lower()
    downloadable_types = [
        "application/",
        "audio/",
        "video/",
        "image/",
        "text/csv",
        "text/plain",
    ]
    return any(content_type.startswith(mime) for mime in downloadable_types)


def _get_filesize(response: requests.Response) -> float | None:
    """Return the response size in bytes, if known."""
    total = response.headers.get("content-length")
    if total is None:
        return total
    return float(total)


def _write_file[T: Path | UPath](
    response: requests.Response,
    filepath: T,
    *,
    show_progress: bool = True,
    overwrite: bool = False,
) -> T:
    """Stream a response to a local or remote file.

    Raises
    ------
    FileExistsError
        If ``filepath`` already exists and ``overwrite`` is ``False``.
    """
    if not overwrite and filepath.exists():
        msg = f"Target already exists: '{filepath}'. Pass `overwrite=True` to replace it."
        raise FileExistsError(msg)

    filename = shorten_string(filepath.name, 30)
    progress = create_progress(show_progress=show_progress, columns=_DOWNLOAD_COLUMNS)
    total = _get_filesize(response)

    # Only local UPaths inherit Path.
    if isinstance(filepath, Path):
        partial_filepath = filepath.with_name(f"{filepath.name}.{token_hex(8)}.partial")
        try:
            with progress:
                task_id = progress.add_task("Download", filename=filename, total=total)
                with partial_filepath.open("wb") as f:
                    for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        progress.update(task_id, advance=len(chunk))
                progress.update(task_id, refresh=True)
            partial_filepath.replace(filepath)
        except BaseException:
            partial_filepath.unlink(missing_ok=True)
            raise
        return filepath

    with progress:
        task_id = progress.add_task("Download", filename=filename, total=total)
        with filepath.open("wb", block_size=REMOTE_BLOCK_SIZE) as f:
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                progress.update(task_id, advance=len(chunk))
        progress.update(task_id, refresh=True)
    return filepath


def _rewrite_redirect_method(status_code: int, method: str) -> str:
    """Apply browser-compatible redirect method rewriting."""
    becomes_get = status_code == _HTTP_SEE_OTHER or (
        status_code in (_HTTP_MOVED_PERMANENTLY, _HTTP_FOUND) and method == "POST"
    )
    return "GET" if becomes_get else method


def _sanitize_filename(filename: str, *, explicit: bool) -> str:
    """Validate a filename as a safe path component.

    Raises
    ------
    ValueError
        If the name is unsafe.
    """
    source = "provided" if explicit else "inferred"
    if not filename or filename in (".", ".."):
        msg = f"Refusing {source} filename {filename!r}: not a valid file name."
        raise ValueError(msg)
    if _CONTROL_CHARS_RE.search(filename):
        msg = f"Refusing {source} filename {filename!r}: contains control characters."
        raise ValueError(msg)
    if "/" in filename or "\\" in filename:
        msg = f"Refusing {source} filename {filename!r}: contains a directory separator."
        raise ValueError(msg)
    if ":" in filename:
        # Reject drive letters and NTFS alternate streams.
        msg = f"Refusing {source} filename {filename!r}: contains ':'."
        raise ValueError(msg)
    stem = filename.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        msg = f"Refusing {source} filename {filename!r}: reserved device name."
        raise ValueError(msg)
    return filename


def _resolve_within[T: Path | UPath](save_dir: T, filename: str) -> T:
    """Join a filename beneath a directory, rejecting local symlink escapes."""
    target = save_dir / filename
    if isinstance(save_dir, Path):
        resolved_dir = save_dir.resolve()
        resolved_target = target.resolve()
        if resolved_target.parent != resolved_dir:
            msg = f"Filename {filename!r} escapes destination directory '{save_dir}'."
            raise ValueError(msg)
    return cast("T", target)


def _extract_filename_from_cd(cd: str | None) -> str | None:
    """Extract a filename from Content-Disposition, preferring ``filename*``."""
    if not cd:
        return None

    # RFC 5987: filename*=charset'language'encoded-value.
    rfc5987_match = re.search(r"filename\*=([^']*)'[^']*'([^;\s]+)", cd, re.IGNORECASE)
    if rfc5987_match:
        charset = rfc5987_match.group(1) or "utf-8"
        return unquote(rfc5987_match.group(2), encoding=charset, errors="replace")

    plain_match = re.search(r'filename="([^"]+)"|filename=([^;\s]+)', cd, re.IGNORECASE)
    if plain_match:
        return plain_match.group(1) or plain_match.group(2)

    return None


def _extract_filename_from_url(
    url: str | None,
) -> str | None:
    """Extract a filename with an extension from a URL."""
    if not url:
        return None
    path = unquote(urlparse(url).path)
    filename = path.split("/")[-1]
    if "." not in filename:
        return None
    return filename
