"""Shared S3 UPath construction, retry configuration, and rasterio env.

Single source of truth for building S3-backed `UPath` instances, for which
S3/botocore errors are transient and how many times to retry them, and for the
GDAL/rasterio environment used to read and write rasters on S3-compatible
object storage. Build every S3 path via :func:`make_s3_upath` rather than
constructing a bare `UPath` directly, and wrap raster I/O in :func:`s3_env`.

Importing this module registers a `s3fs.core.set_custom_error_handler` callback,
which is what gates `ClientError` retries inside s3fs's own retry loop -- separate
from (and consulted after) botocore's `retries.max_attempts` config.
"""

from __future__ import annotations

from contextlib import contextmanager
from functools import lru_cache
from typing import TYPE_CHECKING, Any

from botocore.exceptions import ClientError
import rasterio
from rasterio.session import AWSSession
from s3fs.core import set_custom_error_handler
from upath import UPath

from freezebase.download import TRANSIENT_HTTP_STATUS_CODES

if TYPE_CHECKING:
    from collections.abc import Generator

# Observed as transient on both the CDSE eodata gateway and Ceph-based S3
# gateways under clock-skew, auth-refresh races, or load.
TRANSIENT_S3_ERROR_CODES = frozenset(
    {"SignatureDoesNotMatch", "RequestTimeTooSkewed", "RequestTimeout"},
)

# botocore retry budget for a single S3 API call (e.g. one HeadObject). "adaptive"
# mode adds client-side rate limiting on top of exponential backoff, which helps
# when a gateway is transiently overloaded rather than fully down.
S3_MAX_ATTEMPTS = 10
S3_RETRY_MODE = "adaptive"

# GDAL's own /vsis3 read/write path has a retry budget separate from botocore's
# (which only covers s3fs metadata calls). Shared transient codes, plus 403
# (S3 can return this transiently under auth-refresh races).
GDAL_HTTP_RETRY_CODES = ",".join(str(code) for code in sorted({403, *TRANSIENT_HTTP_STATUS_CODES}))
GDAL_HTTP_MAX_RETRY = 5
GDAL_HTTP_RETRY_DELAY_S = 1

# botocore client kwargs that carry credentials. s3fs forwards these straight to
# the client, where they win over everything else -- but `s3_env` never sees them,
# so they must not be mixed with a profile or a key pair.
CLIENT_CREDENTIAL_KEYS = ("aws_access_key_id", "aws_secret_access_key", "aws_session_token")


def _retry_transient_s3_errors(exc: Exception) -> bool:
    """s3fs custom error handler that retries transient `ClientError` responses.

    See the module docstring for why this -- not `retries.max_attempts` -- is the
    layer that actually gates whether s3fs retries a `ClientError` at all.
    """
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
    """Build an `AWSSession`, memoized on the options that define its identity.

    Positional-only so a caller can't split the cache by switching call form.
    Bounded so that per-request STS credentials evict rather than accumulate.
    """
    return AWSSession(
        aws_unsigned=unsigned,
        aws_access_key_id=key,
        aws_secret_access_key=secret,
        aws_session_token=token,
        region_name=region_name,
        profile_name=profile,
    )


def aws_session(path: UPath) -> AWSSession:
    """Return a cached rasterio `AWSSession` for an S3 `UPath`.

    Constructing an `AWSSession` runs boto's whole credential resolution chain
    eagerly, so paths sharing a configuration share one session: it is resolved
    once per process instead of once per :func:`s3_env` block, which
    :func:`freezebase.raster` enters on *every* raster operation.

    Caching is safe for rotating credentials: rasterio re-freezes the session's
    credentials on each `rasterio.Env` entry, so refreshable STS/SSO profiles
    still pick up new keys on their own. Static credentials read from a file are
    frozen for the process lifetime -- call :func:`clear_aws_session_cache` if
    such a file is rewritten underneath a long-running process.

    Parameters
    ----------
    path : UPath
        A UPath with ``protocol="s3"`` whose ``storage_options`` carry an
        optional named ``profile``, fsspec/s3fs credentials (``key``,
        ``secret``, ``token``), an optional ``anon`` flag, and an optional
        ``client_kwargs["region_name"]`` (as produced by :func:`make_s3_upath`).

    Returns
    -------
    AWSSession
        Unsigned if the path carries ``anon=True``; otherwise signed with the
        path's ``key``/``secret``, its named profile, or -- absent both -- boto's
        default credential resolver.
    """
    so = path.storage_options
    return _aws_session(
        # Read the same flag s3fs does, defaulting the same way it does, so a
        # path cannot be signed for one layer and unsigned for the other.
        bool(so.get("anon", False)),
        so.get("key"),
        so.get("secret"),
        so.get("token"),
        (so.get("client_kwargs") or {}).get("region_name"),
        so.get("profile"),
    )


def clear_aws_session_cache() -> None:
    """Discard every `AWSSession` memoized by :func:`aws_session`.

    The next :func:`aws_session` call per configuration re-resolves credentials
    from scratch. Needed only when static credentials change mid-process -- a
    rewritten shared credentials file or a swapped ``AWS_*`` environment.
    """
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
    """Build an S3 `UPath` with the shared retry policy.

    Parameters
    ----------
    path : str
        S3 URI or prefix, e.g. ``"s3://my-bucket/some/key.tif"``. The ``s3``
        protocol is forced, so a bare ``"my-bucket/key"`` also works.
    anon : bool, default False
        Whether to use an anonymous connection (public buckets only). If
        ``False``, uses ``profile``, the ``key``/``secret`` given, or boto's
        default credential resolver. Mirrors ``s3fs.S3FileSystem``'s ``anon``.
    profile : str, optional
        Named AWS configuration profile to use for this path. The profile name
        is stored on the path and used independently by both s3fs and Rasterio.
        May be combined with ``endpoint_url`` for S3-compatible stores, but not
        with ``anon=True`` or explicit ``key``, ``secret``, or ``token``.
        Setting a profile explicitly makes boto skip its environment-variable
        credential provider on both layers, so such a path can never be signed
        by ambient ``AWS_ACCESS_KEY_ID``/``AWS_SECRET_ACCESS_KEY`` values --
        which is what makes profiles the safer way to authenticate here.
    key : str, optional
        If not anonymous, use this access key ID, if specified. Mutually
        exclusive with ``aws_access_key_id`` in ``client_kwargs``.
    secret : str, optional
        If not anonymous, use this secret access key, if specified. Mutually
        exclusive with ``aws_secret_access_key`` in ``client_kwargs``.
    token : str, optional
        If not anonymous, use this security token, if specified, for
        temporary/STS credentials.
    region : str, optional
        AWS region name. Relevant for region-scoped AWS-native buckets.
    endpoint_url : str, optional
        S3-compatible endpoint URL. Omit for AWS-native S3.
    request_checksum_calculation, response_checksum_validation : str, default "when_required"
        boto3 >=1.36 checksum behavior. ``"when_required"`` avoids the
        ``x-amz-checksum-*`` trailers that some S3-compatible gateways (e.g.
        Ceph) reject; it's a safe default even for gateways not known to need it.
    client_kwargs : dict, optional
        Extra kwargs forwarded to the underlying botocore client. Defaults to
        ``{"endpoint_url": endpoint_url}``. ``region`` is injected here as
        ``region_name``. Credentials passed here (:data:`CLIENT_CREDENTIAL_KEYS`)
        reach s3fs only, never :func:`s3_env`, so they are rejected alongside
        ``anon=True`` and ``profile`` rather than silently signing the two
        layers differently.

    Returns
    -------
    UPath
        An S3 `UPath` configured with :data:`S3_MAX_ATTEMPTS` botocore retries in
        :data:`S3_RETRY_MODE` mode, on top of the s3fs-level retry handler
        registered by this module. Pair it with :func:`s3_env` for raster I/O.

    Raises
    ------
    ValueError
        If ``anon=True`` is combined with a profile or credentials, if a profile
        is combined with explicit ``key``, ``secret``, ``token``, or credentials
        in ``client_kwargs``, or if ``profile`` is an empty string.

    Examples
    --------
    Anonymous access to a public bucket::

        path = make_s3_upath(
            "s3://copernicus-dem-30m/x.tif", region="eu-central-1", anon=True
        )

    Signed access via boto's credential resolver, with no explicit key pair::

        path = make_s3_upath("s3://my-private-bucket/x.tif")

    Per-path profile selection, including for an S3-compatible endpoint::

        path = make_s3_upath(
            "s3://my-bucket/x.tif",
            profile="research",
            endpoint_url="https://objects.example.org",
        )
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
    """Enter a rasterio (bundled GDAL) environment for S3-compatible storage.

    Extracts credentials and endpoint configuration from the fsspec storage
    options attached to ``path`` and applies them to rasterio for the duration
    of the ``with`` block. Requests are sent unsigned only when the path carries
    ``anon=True``; otherwise GDAL signs them, using ``key``/``secret`` when
    present, selecting the path's named profile when configured, and falling
    back to boto's default credential resolver otherwise.

    The session itself comes from :func:`aws_session`, so entering this block
    repeatedly for the same configuration doesn't re-resolve credentials.

    Parameters
    ----------
    path : UPath
        A UPath with ``protocol="s3"`` whose ``storage_options`` carry
        an optional named ``profile``, fsspec/s3fs credentials (``key``,
        ``secret``, ``token``), an optional ``anon`` flag, an optional
        ``client_kwargs["region_name"]``, and an optional custom
        ``endpoint_url`` (as produced by :func:`make_s3_upath`).
    """
    so = path.storage_options
    session = aws_session(path)

    options: dict[str, str] = {
        # Don't probe for sidecar files on open.
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        # /vsis3/ has no native random-write support; needed for writes.
        "CPL_VSIL_USE_TEMP_FILE_FOR_RANDOM_WRITE": "YES",
        # Retry transient HTTP failures on the /vsis3 read/write path.
        "GDAL_HTTP_MAX_RETRY": str(GDAL_HTTP_MAX_RETRY),
        "GDAL_HTTP_RETRY_DELAY": str(GDAL_HTTP_RETRY_DELAY_S),
        "GDAL_HTTP_RETRY_CODES": GDAL_HTTP_RETRY_CODES,
    }
    if endpoint_url := so.get("endpoint_url"):
        # Pass the host without scheme so this works on GDAL < 3.11 too, and
        # flag plain-HTTP endpoints explicitly.
        options["AWS_S3_ENDPOINT"] = endpoint_url.removeprefix("https://").removeprefix("http://")
        options["AWS_VIRTUAL_HOSTING"] = "FALSE"
        if endpoint_url.startswith("http://"):
            options["AWS_HTTPS"] = "NO"

    with rasterio.Env(session=session, **options):
        yield
