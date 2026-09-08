"""S3 connection settings read from the environment, and the paths they open."""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Self

from pydantic import PrivateAttr
from pydantic_settings import BaseSettings, SettingsConfigDict
from upath import UPath

from freezebase.s3 import make_s3_upath

type AnyDir = Path | UPath

ChecksumPolicy = Literal["when_required", "when_supported"]


class S3Settings(BaseSettings):
    """S3 profile and botocore tuning from ``<PREFIX>ENDPOINT_URL`` and friends.

    Authentication is by named AWS profile only: ``<PREFIX>PROFILE`` selects a
    section of ``~/.aws/credentials`` (or of ``AWS_SHARED_CREDENTIALS_FILE``),
    so no access key ever passes through the environment. Selecting a profile
    explicitly also drops boto's environment-variable provider from the
    credential chain, so a stray ``AWS_ACCESS_KEY_ID`` in a shell cannot sign
    these requests.

    ``<PREFIX>ENDPOINT_URL`` is optional. Without it, the endpoint is whatever
    the profile's config section sets, which is where botocore, s3fs and
    :func:`~freezebase.s3.s3_env` all read it from.

    Build one with :meth:`from_env`, which fixes the prefix.
    """

    model_config = SettingsConfigDict(extra="ignore")

    endpoint_url: str | None = None
    profile: str | None = None

    # boto3 >= 1.36 defaults to "when_supported", which adds x-amz-checksum-*
    # trailers that some S3-compatible gateways reject. Typed as the two
    # values botocore accepts, so a typo in the environment is refused here
    # rather than by a signing failure half-way through a copy.
    request_checksum_calculation: ChecksumPolicy = "when_required"
    response_checksum_validation: ChecksumPolicy = "when_required"

    _env_prefix: str = PrivateAttr(default="")

    @classmethod
    def from_env(cls, prefix: str) -> Self:
        """Read the settings whose variable names start with ``prefix``.

        Parameters
        ----------
        prefix : str
            Environment variable prefix, such as ``"GLACE_S3_"``.

        Returns
        -------
        S3Settings
            The settings, remembering the prefix for its error messages.
        """
        settings = cls(_env_prefix=prefix)
        settings._env_prefix = prefix
        return settings

    def make_upath(self, uri: str) -> UPath:
        """Open an ``s3://`` URI with these settings.

        Parameters
        ----------
        uri : str
            An ``s3://`` URI.

        Returns
        -------
        UPath
            The path, carrying the profile, endpoint, checksum policies and the
            retry budget :func:`~freezebase.s3.make_s3_upath` applies.

        Raises
        ------
        ValueError
            If no profile is configured.
        """
        if not self.profile:
            msg = f"An s3:// path needs {self._env_prefix}PROFILE."
            raise ValueError(msg)
        return make_s3_upath(
            uri,
            profile=self.profile,
            endpoint_url=self.endpoint_url,
            request_checksum_calculation=self.request_checksum_calculation,
            response_checksum_validation=self.response_checksum_validation,
        )

    def resolve_path(self, raw: str) -> AnyDir:
        """Resolve a local path or an ``s3://`` URI.

        Parameters
        ----------
        raw : str
            A filesystem path, possibly ``~``-prefixed, or an ``s3://`` URI.

        Returns
        -------
        Path or UPath
            A :class:`~pathlib.Path` for a local value, else the URI opened with
            :meth:`make_upath`, so downstream code does not branch on the
            storage backend.

        Raises
        ------
        ValueError
            If ``raw`` is an ``s3://`` URI and no profile is configured.
        """
        if not raw.startswith("s3://"):
            return Path(raw).expanduser()
        return self.make_upath(raw)
