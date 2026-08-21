"""HTTP download tooling: retry helpers and the HTTPDownloader."""

from __future__ import annotations

from collections.abc import Callable
import contextlib
from functools import partial
import logging
import ntpath
from pathlib import Path
import posixpath
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

# Bound manual redirect following (mirrors the `requests` default).
MAX_REDIRECTS = 30

logger = logging.getLogger(__name__)

# HTTP status codes considered transient and worth retrying.
TRANSIENT_HTTP_STATUS_CODES = frozenset({408, 429, 502, 503, 504})

# Redirect status codes that rewrite the request method (per RFC 7231 / browsers).
_HTTP_MOVED_PERMANENTLY = 301
_HTTP_FOUND = 302
_HTTP_SEE_OTHER = 303

# Control characters (C0 range plus DEL) are never valid in a saved filename.
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x1f\x7f]")

# Case-insensitive Windows reserved device names (a leading stem match is enough).
_WINDOWS_RESERVED_NAMES = frozenset(
    {"CON", "PRN", "AUX", "NUL"}
    | {f"COM{i}" for i in range(1, 10)}
    | {f"LPT{i}" for i in range(1, 10)},
)

# Return type for retry_request
WrappedFn = TypeVar("WrappedFn", bound=Callable[..., Any])

# Byte-oriented progress bar layout for HTTP downloads.
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
    """Check whether an exception raised by `requests` is worth retrying."""
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
        Logger instance used to log retry attempts before each sleep.
    extra_status_codes : frozenset of int, optional
        Additional HTTP status codes to treat as transient for this call site,
        on top of the codes in ``TRANSIENT_HTTP_STATUS_CODES``. Use this rather
        than widening ``TRANSIENT_HTTP_STATUS_CODES`` itself when a status code
        (e.g. 500) is only known to be transient for one particular endpoint.

    Returns
    -------
    Callable[[WrappedFn], WrappedFn]
        A tenacity retry decorator configured for HTTP request retries.
    """
    return tenacity.retry(
        reraise=True,
        stop=tenacity.stop_after_attempt(3),
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10) + tenacity.wait_random(0, 2),
        retry=tenacity.retry_if_exception(
            partial(_is_transient_request_error, extra_status_codes=extra_status_codes),
        ),
        before_sleep=tenacity.before_sleep_log(logger, logging.INFO),
    )


class HTTPDownloader:
    """
    Download manager for fetching files over HTTP/HTTPS.

    Built upon `requests`. Inspired by `pooch.HTTPDownloader`.
    Supports downloading with GET and POST requests.

    Response bodies are always streamed to disk in chunks, so memory use stays
    flat no matter how large the file is, and redirects are always followed by
    the downloader itself, so `trusted_hosts` governs where credentials go.

    Holds an open `requests.Session`, which keeps a connection to the server
    alive between downloads. Call `close()` when done, or use the downloader as
    a context manager. If neither happens, the session is closed once Python
    discards the downloader.
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
        """
        Initialize an HTTPDownloader instance.

        Parameters
        ----------
        method : {"GET", "POST"}
            HTTP method used to request the file.
        auth : tuple[str, str] or instance of AuthBase subclass, optional
            HTTP authentication object (default is None). For HTTP Basic Authentication,
            provide `(user, pass)` tuple
        trusted_hosts : str or iterable of str, optional
            Hostnames for which auth should be preserved during redirects. If not specified,
            auth is only sent to the original host.
        timeout : float or tuple[float, float]
            Seconds to wait for the server, as a single value or a
            `(connect, read)` pair.
        progress : bool
            If True, show a progress bar during download.

        """
        self.session = requests.Session()
        self.method = method
        # Kept per-instance and passed per-request, never stored on the session, so
        # dropping credentials for one hop does not affect later calls.
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
        """
        Download a file from a URL to a directory with optional progress bar.

        Supports both local paths and remote paths (e.g. UPath for S3).

        Parameters
        ----------
        url : str
            The URL of the file to download.
        save_dir : str, Path, or UPath
            The directory to save the downloaded file. Can be a local path
            or a remote UPath (e.g. S3).
        filename : str, optional
            The filename of the downloaded file.
            If not provided, the filename will be inferred from the HTML header or URL.
            Whether provided or inferred, the name is validated to be a single
            path component confined to ``save_dir``; absolute paths, directory
            separators, ``..``, control characters, and reserved device names
            are rejected.
        overwrite : bool, default False
            If ``False`` (the default), raise ``FileExistsError`` when the
            target already exists. If ``True``, replace it.

        Returns
        -------
        Path or UPath
            The path of the downloaded file.

        Raises
        ------
        ValueError
            If the resolved filename is unsafe or escapes ``save_dir``.
        FileExistsError
            If the target exists and ``overwrite`` is ``False``.

        """
        # Ensures the streamed response is properly closed
        # (`close()` only closes the session)
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
        """Follow redirects manually, bounding depth and protecting credentials.

        Credentials are sent only while the hop stays on a trusted host and on
        an HTTPS connection; leaving a trusted host or downgrading to plaintext
        drops them for the rest of the chain. Each intermediate streamed
        response is closed before the next request, and the redirect method is
        rewritten to match browser behavior (303, and 301/302 on POST, become
        GET; 307/308 keep the method).
        """
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
            # Release the streamed connection before following the redirect.
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
        """Close the session and its connections to the server."""
        self.session.close()

    def __enter__(self) -> Self:
        """Return self for use as a context manager."""
        return self

    def __exit__(self, *exc_info: object) -> None:
        """Close the session on context exit."""
        self.close()

    def __del__(self) -> None:
        """Close the session when Python discards the downloader.

        Errors are ignored: this can run while the interpreter is shutting
        down, or on a downloader whose `__init__` did not finish.
        """
        with contextlib.suppress(Exception):
            self.close()


def _is_downloadable_content(response: requests.Response) -> bool:
    """Check if a file is downloadable."""
    cd = response.headers.get("Content-Disposition")
    if cd:
        # File is always downloadable if Content-Disposition exists
        return True
    content_type = response.headers.get("Content-Type", "").lower()
    # Common downloadable MIME types
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
    """Get the filesize in bytes."""
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
    """
    Stream a response body to a file with an optional rich progress bar.

    Local destinations stream to a unique sibling ``.partial`` file and rename
    on success. Remote destinations (e.g. ``UPath`` on S3) write directly
    to the final path with an increased fsspec ``block_size``.

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

    # `UPath("/tmp/x")` is a Path subclass (PosixUPath), `UPath("s3://...")` is not,
    # so isinstance(_, Path) separates local from remote.
    if isinstance(filepath, Path):
        # A unique suffix keeps concurrent writers to the same target from
        # clobbering one another's partial file.
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
    """Return the method to use for a redirect, matching browser behavior.

    A 303, and a 301/302 on a POST, becomes a GET; 307/308 keep the method.
    """
    becomes_get = status_code == _HTTP_SEE_OTHER or (
        status_code in (_HTTP_MOVED_PERMANENTLY, _HTTP_FOUND) and method == "POST"
    )
    return "GET" if becomes_get else method


def _sanitize_filename(filename: str, *, explicit: bool) -> str:
    """Validate that ``filename`` is a safe, single path component.

    Rejects absolute paths, directory separators, ``..``, control characters,
    and Windows reserved device names, whether the name was supplied explicitly
    or inferred from a server header or URL.

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
        # Guards against Windows drive letters (``C:...``) and NTFS alternate
        # data streams (``name:stream``).
        msg = f"Refusing {source} filename {filename!r}: contains ':'."
        raise ValueError(msg)
    if posixpath.isabs(filename) or ntpath.isabs(filename):
        msg = f"Refusing {source} filename {filename!r}: is an absolute path."
        raise ValueError(msg)
    # ``basename`` on either platform must be a no-op for a bare file name.
    if posixpath.basename(filename) != filename or ntpath.basename(filename) != filename:
        msg = f"Refusing {source} filename {filename!r}: not a bare file name."
        raise ValueError(msg)
    stem = filename.split(".", 1)[0].upper()
    if stem in _WINDOWS_RESERVED_NAMES:
        msg = f"Refusing {source} filename {filename!r}: reserved device name."
        raise ValueError(msg)
    return filename


def _resolve_within[T: Path | UPath](save_dir: T, filename: str) -> T:
    """Join ``filename`` under ``save_dir`` and confirm it does not escape it.

    ``filename`` is expected to already be a sanitized bare name; this is a
    defense-in-depth check that also catches symlink-based escapes locally.
    """
    target = save_dir / filename
    if isinstance(save_dir, Path):
        resolved_dir = save_dir.resolve()
        resolved_target = target.resolve()
        if resolved_target.parent != resolved_dir:
            msg = f"Filename {filename!r} escapes destination directory '{save_dir}'."
            raise ValueError(msg)
    return cast("T", target)


def _extract_filename_from_cd(cd: str | None) -> str | None:
    """
    Extract the filename from the Content-Disposition HTML headers field.

    Handles both RFC 5987 encoded filename* and plain filename parameters,
    preferring filename* when both are present.
    """
    if not cd:
        return None

    # RFC 5987: filename*=charset'language'encoded-value (e.g. UTF-8''name.zip).
    # The value is percent-encoded, so decode it before returning (the caller
    # then validates it; percent-encoded traversal like %2e%2e%2f is caught there).
    rfc5987_match = re.search(r"filename\*=([^']*)'[^']*'([^;\s]+)", cd, re.IGNORECASE)
    if rfc5987_match:
        charset = rfc5987_match.group(1) or "utf-8"
        return unquote(rfc5987_match.group(2), encoding=charset, errors="replace")

    # Plain filename= with optional quotes
    plain_match = re.search(r'filename="([^"]+)"|filename=([^;\s]+)', cd, re.IGNORECASE)
    if plain_match:
        return plain_match.group(1) or plain_match.group(2)

    return None


def _extract_filename_from_url(
    url: str | None,
) -> str | None:
    """Extract the filename from an URL."""
    if not url:
        return None
    path = unquote(urlparse(url).path)
    filename = path.split("/")[-1]
    # Only return if it has extension
    if "." not in filename:
        return None
    return filename
