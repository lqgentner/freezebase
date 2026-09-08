"""S3 UPath construction, retries, and rasterio environments."""

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
    so = path.storage_options
    session = aws_session(path)

    options = {**GDAL_S3_OPTIONS, **_endpoint_options(so.get("endpoint_url"))}

    with rasterio.Env(session=session, **options):
        yield


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
    env.update(_endpoint_options(so.get("endpoint_url")))
    return env
