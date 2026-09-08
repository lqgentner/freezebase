"""S3 UPath construction, retries, and rasterio environments."""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePosixPath
from secrets import token_hex
from typing import TYPE_CHECKING, Any, cast

from botocore.configprovider import ConfiguredEndpointProvider
from botocore.exceptions import ClientError, ProfileNotFound
import botocore.session
import rasterio
from rasterio.session import AWSSession
from s3fs.core import set_custom_error_handler
from upath import UPath

from freezebase.download import TRANSIENT_HTTP_STATUS_CODES

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

# Transient on CDSE and Ceph gateways.
TRANSIENT_S3_ERROR_CODES = frozenset(
    {"SignatureDoesNotMatch", "RequestTimeTooSkewed", "RequestTimeout"},
)

# Retry budget for each botocore call.
S3_MAX_ATTEMPTS = 10
S3_RETRY_MODE = "adaptive"

# GDAL retries independently from botocore/s3fs.
GDAL_HTTP_RETRY_CODES = ",".join(str(code) for code in sorted({403, *TRANSIENT_HTTP_STATUS_CODES}))
GDAL_HTTP_MAX_RETRY = 5
GDAL_HTTP_RETRY_DELAY_S = 1

# GDAL configuration every /vsis3/ reader needs, whether it runs in this process
# (`s3_env`) or in a child (`subprocess_s3_env`). One table so the two cannot
# drift: a child that lacks GDAL_DISABLE_READDIR_ON_OPEN lists the directory of
# every object it opens, which is one extra request per source.
GDAL_S3_OPTIONS: Mapping[str, str] = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    # /vsis3/ cannot write randomly without a temporary file.
    "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": "YES",
    "GDAL_HTTP_MAX_RETRY": str(GDAL_HTTP_MAX_RETRY),
    "GDAL_HTTP_RETRY_DELAY": str(GDAL_HTTP_RETRY_DELAY_S),
    "GDAL_HTTP_RETRY_CODES": GDAL_HTTP_RETRY_CODES,
}

# These client kwargs bypass the credentials visible to s3_env.
CLIENT_CREDENTIAL_KEYS = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")

DEFAULT_CONTENT_TYPE = "application/octet-stream"

# Content type per suffix, as an object should be served to a browser or a
# client. A bucket serves what the upload declared, and `mimetypes` knows none
# of the cloud-native suffixes, so without this a COG downloads instead of
# opening by range request and a README saves instead of rendering.
CONTENT_TYPES: Mapping[str, str] = {
    ".json": "application/json",
    ".geojson": "application/geo+json",
    ".md": "text/markdown",
    ".parquet": "application/vnd.apache.parquet",
    ".tif": "image/tiff; application=geotiff; profile=cloud-optimized",
    ".tiff": "image/tiff; application=geotiff; profile=cloud-optimized",
    ".pmtiles": "application/vnd.pmtiles",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".svg": "image/svg+xml",
    ".yaml": "application/yaml",
    ".yml": "application/yaml",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".html": "text/html",
}


def content_type_for(path: str | Path | UPath) -> str:
    """Return the content type an object should be served as, by its suffix.

    Parameters
    ----------
    path : str, Path, or UPath
        Object name or path. Only the suffix is read, case-insensitively.

    Returns
    -------
    str
        A media type from :data:`CONTENT_TYPES`, or ``application/octet-stream``
        for an unlisted suffix rather than a guess.
    """
    return CONTENT_TYPES.get(PurePosixPath(str(path)).suffix.lower(), DEFAULT_CONTENT_TYPE)


def _retry_transient_s3_errors(exc: Exception) -> bool:
    """Return whether s3fs should retry a ClientError."""
    if isinstance(exc, ClientError):
        code = exc.response.get("Error", {}).get("Code", "")
        return code in TRANSIENT_S3_ERROR_CODES
    return False


set_custom_error_handler(_retry_transient_s3_errors)


@lru_cache(maxsize=32)
def _aws_session(  # noqa: PLR0917 -- all six values define the cache identity
    unsigned: bool,  # noqa: FBT001
    key: str | None,
    secret: str | None,
    token: str | None,
    region_name: str | None,
    profile: str | None,
    /,
) -> AWSSession:
    """Build an AWSSession cached by credential configuration."""
    return AWSSession(
        aws_unsigned=unsigned,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name=region_name,
        profile_name=profile,
    )


def aws_session(path: UPath) -> AWSSession:
    """Return a cached rasterio AWSSession for an S3 path.

    Refreshable credentials remain refreshable. Clear the cache after changing
    static credentials during a process.

    Parameters
    ----------
    path : UPath
        S3 path created by :func:`make_s3_upath`.

    Returns
    -------
    AWSSession
        Session using the path's authentication settings.
    """
    so = path.storage_options
    return _aws_session(
        # Match s3fs signing behavior.
        bool(so.get("anon", False)),
        so.get("key"),
        so.get("secret"),
        so.get("token"),
        (so.get("client_kwargs") or {}).get("region_name"),
        so.get("profile"),
    )


def clear_aws_session_cache() -> None:
    """Clear cached AWS sessions."""
    _aws_session.cache_clear()


def make_s3_upath(
    path: str,
    *,
    anon: bool = False,
    profile: str | None = None,
    key: str | None = None,
    secret: str | None = None,
    token: str | None = None,
    region: str | None = None,
    endpoint_url: str | None = None,
    request_checksum_calculation: str = "when_required",
    response_checksum_validation: str = "when_required",
    client_kwargs: dict[str, Any] | None = None,
) -> UPath:
    """Build an S3 UPath with shared authentication and retries.

    Parameters
    ----------
    path : str
        S3 URI or prefix. The ``s3`` protocol is forced.
    anon : bool, default False
        Use unsigned requests.
    profile : str, optional
        Named AWS profile. Mutually exclusive with anonymous or explicit keys.
    key : str, optional
        Access key ID.
    secret : str, optional
        Secret access key.
    token : str, optional
        Session token.
    region : str, optional
        AWS region.
    endpoint_url : str, optional
        S3-compatible endpoint.
    request_checksum_calculation, response_checksum_validation : str, default "when_required"
        boto3 checksum policies.
    client_kwargs : dict, optional
        Additional botocore client arguments. Credential entries cannot be
        combined with ``anon`` or ``profile``.

    Returns
    -------
    UPath
        Configured S3 path.

    Raises
    ------
    ValueError
        If authentication options conflict or ``profile`` is empty.
    """
    if profile is not None and not profile:
        msg = "`profile` must be a non-empty profile name, or None."
        raise ValueError(msg)
    explicit_credentials = (
        key is not None
        or secret is not None
        or token is not None
        or any(name in (client_kwargs or {}) for name in CLIENT_CREDENTIAL_KEYS)
    )
    if anon and (profile is not None or explicit_credentials):
        msg = (
            "anon=True cannot be combined with `profile`, `key`, `secret`, `token`, "
            "or credentials in `client_kwargs`."
        )
        raise ValueError(msg)
    if profile is not None and explicit_credentials:
        msg = (
            "`profile` cannot be combined with explicit `key`, `secret`, `token`, "
            "or credentials in `client_kwargs`."
        )
        raise ValueError(msg)
    resolved_client_kwargs = (
        client_kwargs if client_kwargs is not None else {"endpoint_url": endpoint_url}
    )
    if region is not None and "region_name" not in resolved_client_kwargs:
        resolved_client_kwargs = {**resolved_client_kwargs, "region_name": region}
    return UPath(
        path,
        protocol="s3",
        profile=profile,
        key=key,
        secret=secret,
        token=token,
        anon=anon,
        endpoint_url=endpoint_url,
        client_kwargs=resolved_client_kwargs,
        config_kwargs={
            "request_checksum_calculation": request_checksum_calculation,
            "response_checksum_validation": response_checksum_validation,
            "retries": {"max_attempts": S3_MAX_ATTEMPTS, "mode": S3_RETRY_MODE},
        },
    )


@contextmanager
def s3_env(path: UPath) -> Generator[None]:
    """Enter a rasterio environment configured from an S3 UPath.

    Parameters
    ----------
    path : UPath
        S3 path created by :func:`make_s3_upath`.
    """
    session = aws_session(path)

    options = {**GDAL_S3_OPTIONS, **_endpoint_options(resolve_endpoint_url(path))}

    with rasterio.Env(session=session, **options):
        yield


def configured_endpoint_url(profile: str | None = None) -> str | None:
    """Return the S3 endpoint the AWS configuration sets for a profile, if any.

    The same lookup botocore performs when a client is created without an
    explicit endpoint: ``AWS_ENDPOINT_URL_S3``, then ``AWS_ENDPOINT_URL``, then
    the ``services`` section and the ``endpoint_url`` key of the profile in the
    shared config file.

    Parameters
    ----------
    profile : str, optional
        Named AWS profile. ``None`` reads the default profile.

    Returns
    -------
    str or None
        The configured endpoint, or ``None`` when nothing sets one or the
        profile does not exist.
    """
    session = botocore.session.Session(profile=profile)
    try:
        scoped = session.get_scoped_config()
    except ProfileNotFound:
        return None
    provider = ConfiguredEndpointProvider(
        full_config=session.full_config,
        scoped_config=scoped,
        client_name="s3",
    )
    endpoint = provider.provide()
    return str(endpoint) if endpoint else None


def resolve_endpoint_url(path: UPath) -> str | None:
    """Return the endpoint an S3 path's requests go to.

    An ``endpoint_url`` in the path's storage options wins. Without one, the
    path's profile is looked up with :func:`configured_endpoint_url`, so a
    profile whose config section carries ``endpoint_url`` points GDAL at the
    same gateway s3fs already talks to.

    Parameters
    ----------
    path : UPath
        S3 path created by :func:`make_s3_upath`.

    Returns
    -------
    str or None
        The endpoint, or ``None`` for plain AWS.
    """
    so = path.storage_options
    if endpoint_url := so.get("endpoint_url"):
        return str(endpoint_url)
    profile = so.get("profile")
    if profile is None and so.get("key") is not None:
        # Explicit credentials: nothing in the shared config applies.
        return None
    return configured_endpoint_url(str(profile) if profile else None)


def _endpoint_options(endpoint_url: object) -> dict[str, str]:
    """GDAL options that point /vsis3/ at an S3-compatible endpoint, if any."""
    if not endpoint_url:
        return {}
    endpoint = str(endpoint_url)
    options = {
        # GDAL < 3.11 expects the endpoint without a scheme.
        "AWS_S3_ENDPOINT": endpoint.removeprefix("https://").removeprefix("http://"),
        "AWS_VIRTUAL_HOSTING": "FALSE",
    }
    if endpoint.startswith("http://"):
        options["AWS_HTTPS"] = "NO"
    return options


def subprocess_s3_env(path: UPath) -> dict[str, str]:
    """Build the S3 settings a child process needs, as environment variables.

    :func:`s3_env` only configures this process; GDAL in a child instead
    honours ``AWS_PROFILE`` against ``~/.aws/credentials``, so no access key
    needs to pass through the environment. The child also receives the same
    GDAL reader configuration :func:`s3_env` applies, so a subprocess opens a
    remote object with the same number of requests as the parent would.

    Parameters
    ----------
    path : UPath
        S3 path created by :func:`make_s3_upath`, whose storage options carry
        the profile and endpoint. A non-S3 path yields an empty mapping.

    Returns
    -------
    dict of str to str
        Variables to merge into the child's environment.
    """
    if path.protocol != "s3":
        return {}
    so = path.storage_options
    env: dict[str, str] = dict(GDAL_S3_OPTIONS)
    if profile := so.get("profile"):
        env["AWS_PROFILE"] = str(profile)
    env.update(_endpoint_options(resolve_endpoint_url(path)))
    return env


def list_object_sizes(directory: str | Path | UPath, *, recursive: bool = False) -> dict[str, int]:
    """List a prefix as ``{relative path: size}`` from one listing call.

    One listing replaces one remote open per object, which is what makes
    checking tens of thousands of objects affordable. Only the size is taken:
    an ETag is the MD5 for a single-part upload and something else for a
    multipart one, so keying on it would constrain how every writer uploads.

    Parameters
    ----------
    directory : str, Path, or UPath
        Prefix to list. Any fsspec-backed path works, including a local one.
    recursive : bool, default False
        Include everything below the prefix, keyed by its path relative to the
        prefix with ``/`` separators. The default lists the prefix's direct
        entries by name.

    Returns
    -------
    dict of str to int
        Relative path to size in bytes. Empty when the prefix does not exist.
    """
    root = UPath(directory)
    prefix = str(root.path).rstrip("/") + "/"
    if not root.fs.exists(str(root.path)):
        return {}
    if recursive:
        # `detail=True` makes fsspec return `{path: info}`; its signature
        # declares only the `detail=False` list, so the mapping is named here.
        found = cast(
            "dict[str, dict[str, Any]]",
            root.fs.find(str(root.path), detail=True),
        )
        infos = list(found.values())
    else:
        infos = cast("list[dict[str, Any]]", root.fs.ls(str(root.path), detail=True))
    entries: dict[str, int] = {}
    for info in infos:
        if info.get("type") == "directory":
            continue
        name = str(info["name"])
        if not name.startswith(prefix):
            continue
        entries[name[len(prefix) :]] = int(info.get("size") or info.get("Size") or 0)
    return entries


def atomic_write_text(path: str | Path | UPath, text: str) -> None:
    """Write a text file so that a reader never sees it half-written.

    On S3 a ``PutObject`` is atomic on its own. On a local filesystem the text
    goes to a sibling temporary file that is renamed over the target, and the
    temporary file is removed if the write fails.

    Parameters
    ----------
    path : str, Path, or UPath
        Destination.
    text : str
        Content to write.
    """
    target = UPath(path)
    if target.protocol == "s3":
        target.write_text(text)
        return
    partial = target.with_name(f"{target.name}.{token_hex(8)}.partial")
    try:
        partial.write_text(text)
        partial.replace(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
