"""Tests for freezebase.s3 UPath construction, env injection, and retries."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import pytest

pytest.importorskip("s3fs")
pytest.importorskip("botocore")
pytest.importorskip("boto3")

from botocore.exceptions import ClientError
from rasterio.env import getenv
from upath import UPath

from freezebase.s3 import (
    CONTENT_TYPES,
    DEFAULT_CONTENT_TYPE,
    GDAL_HTTP_MAX_RETRY,
    GDAL_HTTP_RETRY_CODES,
    GDAL_S3_OPTIONS,
    TRANSIENT_S3_ERROR_CODES,
    _retry_transient_s3_errors,
    atomic_write_text,
    aws_session,
    clear_aws_session_cache,
    configured_endpoint_url,
    content_type_for,
    list_object_sizes,
    make_s3_upath,
    resolve_endpoint_url,
    s3_env,
    subprocess_s3_env,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path


@pytest.fixture(autouse=True)
def _isolated_aws_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the developer's own AWS configuration out of every test.

    `resolve_endpoint_url` consults the shared config file and the
    `AWS_ENDPOINT_URL*` variables for a profile-only path, so a real profile or
    a shell-wide endpoint would otherwise leak into the expectations here.
    """
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-config"))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-credentials"))
    for name in ("AWS_ENDPOINT_URL", "AWS_ENDPOINT_URL_S3", "AWS_PROFILE"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(autouse=True)
def _clear_session_cache() -> Generator[None]:
    """Keep `aws_session`'s memo from leaking between tests.

    Load-bearing, not hygiene: a warm cache bypasses the `captured_session` spy
    entirely, and would hand one test's credentials to the next.
    """
    clear_aws_session_cache()
    yield
    clear_aws_session_cache()


def _client_error(code: str) -> ClientError:
    return ClientError({"Error": {"Code": code, "Message": "x"}}, "HeadObject")


class TestTransientS3Errors:
    @pytest.mark.parametrize("code", sorted(TRANSIENT_S3_ERROR_CODES))
    def test_transient_codes_retry(self, code: str) -> None:
        assert _retry_transient_s3_errors(_client_error(code)) is True

    @pytest.mark.parametrize("code", ["AccessDenied", "403", "NoSuchKey", "InvalidAccessKeyId"])
    def test_permanent_codes_not_retried(self, code: str) -> None:
        # A permanent authorization failure must not be retried (previously a
        # bare "403" was treated as transient and retried up to ten times).
        assert _retry_transient_s3_errors(_client_error(code)) is False

    def test_non_client_error_not_retried(self) -> None:
        assert _retry_transient_s3_errors(ValueError("nope")) is False


class TestMakeS3Upath:
    def test_profile_forwarded_and_preserved_by_child_path(self) -> None:
        path = make_s3_upath("s3://b", profile="research")
        assert path.storage_options["profile"] == "research"
        assert (path / "child.tif").storage_options["profile"] == "research"

    def test_profile_can_be_combined_with_custom_endpoint(self) -> None:
        so = make_s3_upath(
            "s3://b/k",
            profile="ceph-research",
            endpoint_url="https://ceph.example.org",
        ).storage_options
        assert so["profile"] == "ceph-research"
        assert so["endpoint_url"] == "https://ceph.example.org"

    def test_region_goes_into_client_kwargs_not_top_level(self) -> None:
        # Regression: a top-level ``region`` kwarg reaches aiobotocore's session
        # and raises; it must live in client_kwargs as ``region_name``.
        so = make_s3_upath("s3://b/k", key="a", secret="b", region="eu-central-1").storage_options
        assert so["client_kwargs"]["region_name"] == "eu-central-1"
        assert "region" not in so

    def test_endpoint_url_forwarded(self) -> None:
        so = make_s3_upath(
            "s3://b/k",
            key="a",
            secret="b",
            endpoint_url="https://ceph.example.org",
        ).storage_options
        assert so["endpoint_url"] == "https://ceph.example.org"
        assert so["client_kwargs"]["endpoint_url"] == "https://ceph.example.org"

    def test_token_forwarded(self) -> None:
        so = make_s3_upath("s3://b/k", key="a", secret="b", token="TK").storage_options
        assert so["token"] == "TK"

    def test_signed_by_default_without_credentials(self) -> None:
        # Matches `s3fs.S3FileSystem(anon=False)`: absent credentials mean
        # "let boto's resolver find them", not "public bucket".
        so = make_s3_upath("s3://b/k").storage_options
        assert so["key"] is None
        assert so["secret"] is None
        assert so["anon"] is False

    def test_signed_by_default_with_credentials(self) -> None:
        so = make_s3_upath("s3://b/k", key="a", secret="b").storage_options
        assert so["anon"] is False

    def test_explicit_anon_true(self) -> None:
        # The flag is what makes a path anonymous, and it must be recorded so
        # both s3fs and `s3_env` read the same answer.
        so = make_s3_upath("s3://public/k", anon=True).storage_options
        assert so["anon"] is True

    @pytest.mark.parametrize(
        "creds",
        [{"key": "a"}, {"secret": "b"}, {"token": "TK"}, {"key": "a", "secret": "b"}],
    )
    def test_anon_true_with_credentials_rejected(self, creds: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match="anon=True cannot be combined"):
            make_s3_upath("s3://b/k", anon=True, **creds)

    def test_anon_true_with_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="anon=True cannot be combined"):
            make_s3_upath("s3://b/k", anon=True, profile="research")

    @pytest.mark.parametrize(
        "creds",
        [{"key": "a"}, {"secret": "b"}, {"token": "TK"}, {"key": "a", "secret": "b"}],
    )
    def test_profile_with_explicit_credentials_rejected(self, creds: dict[str, Any]) -> None:
        with pytest.raises(ValueError, match=r"profile.*explicit"):
            make_s3_upath("s3://b/k", profile="research", **creds)

    @pytest.mark.parametrize("auth", [{"profile": "research"}, {"anon": True}])
    def test_client_kwargs_credentials_rejected(self, auth: dict[str, Any]) -> None:
        # s3fs forwards these to the client, where they win; `s3_env` never sees
        # them. Rejecting them keeps both layers signing the same way.
        with pytest.raises(ValueError, match="client_kwargs"):
            make_s3_upath(
                "s3://b/k",
                client_kwargs={"aws_access_key_id": "AK", "aws_secret_access_key": "SK"},
                **auth,
            )

    def test_empty_profile_rejected(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            make_s3_upath("s3://b/k", profile="")

    def test_protocol_forced_for_bare_string(self) -> None:
        assert make_s3_upath("bucket/key.tif", key="a", secret="b").protocol == "s3"

    def test_caller_region_name_takes_precedence(self) -> None:
        so = make_s3_upath(
            "s3://b/k",
            key="a",
            secret="b",
            region="us-east-1",
            client_kwargs={"region_name": "eu-west-1"},
        ).storage_options
        assert so["client_kwargs"]["region_name"] == "eu-west-1"


@pytest.fixture
def captured_session(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    """Patch ``AWSSession`` in :mod:`freezebase.s3` to record its kwargs."""
    import inspect  # noqa: PLC0415

    from rasterio.session import AWSSession  # noqa: PLC0415

    captured: dict[str, object] = {}

    def spy(**kwargs: object) -> AWSSession:
        captured.update(kwargs)
        # The returned session is unsigned so the fake profile names used by
        # these unit tests are never resolved (the integration test covers real
        # resolution), but the kwargs must still name real AWSSession
        # parameters -- a renamed or misspelled one has to fail here.
        inspect.signature(AWSSession).bind(**kwargs)
        return AWSSession(aws_unsigned=True)

    monkeypatch.setattr("freezebase.s3.AWSSession", spy)
    return captured


class TestS3Env:
    def test_session_receives_profile(
        self,
        captured_session: dict[str, object],
    ) -> None:
        p = make_s3_upath(
            "s3://b/k",
            profile="research",
            endpoint_url="https://ceph.example.org",
        )
        with s3_env(p):
            pass
        assert captured_session["profile_name"] == "research"
        assert captured_session["aws_access_key_id"] is None
        assert captured_session["aws_secret_access_key"] is None
        assert captured_session["aws_unsigned"] is False

    def test_https_endpoint_options(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b", endpoint_url="https://ceph.example.org")
        with s3_env(p):
            env = getenv()
        assert env["AWS_S3_ENDPOINT"] == "ceph.example.org"  # scheme stripped
        assert env["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert env.get("AWS_HTTPS") != "NO"
        assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        assert "403" in env["GDAL_HTTP_RETRY_CODES"]

    def test_http_endpoint_flags_plaintext(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b", endpoint_url="http://minio:9000")
        with s3_env(p):
            env = getenv()
        assert env["AWS_S3_ENDPOINT"] == "minio:9000"
        assert env["AWS_HTTPS"] == "NO"

    def test_no_endpoint_omits_endpoint_options(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b")
        with s3_env(p):
            env = getenv()
        assert "AWS_S3_ENDPOINT" not in env
        assert "AWS_VIRTUAL_HOSTING" not in env

    def test_profile_endpoint_reaches_gdal(
        self,
        captured_session: dict[str, object],
        profile_with_endpoint: str,
    ) -> None:
        # Previously only an explicit endpoint_url reached GDAL, so a
        # profile-only path sent /vsis3/ to AWS while s3fs talked to the gateway.
        p = make_s3_upath("s3://b/k", profile=profile_with_endpoint)
        with s3_env(p):
            env = getenv()
        assert env["AWS_S3_ENDPOINT"] == "gateway.example.org"
        assert env["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert captured_session["profile_name"] == profile_with_endpoint

    def test_session_receives_region_and_credentials(
        self,
        captured_session: dict[str, object],
    ) -> None:
        p = make_s3_upath("s3://b/k", key="AK", secret="SK", token="TK", region="eu-central-1")
        with s3_env(p):
            pass
        assert captured_session["aws_access_key_id"] == "AK"
        assert captured_session["aws_secret_access_key"] == "SK"
        assert captured_session["aws_session_token"] == "TK"
        assert captured_session["region_name"] == "eu-central-1"
        assert captured_session["aws_unsigned"] is False

    def test_session_unsigned_when_anon(
        self,
        captured_session: dict[str, object],
    ) -> None:
        p = make_s3_upath("s3://public/k", region="eu-central-1", anon=True)
        with s3_env(p):
            pass
        assert captured_session["aws_unsigned"] is True
        # Region still applies for public, region-scoped buckets.
        assert captured_session["region_name"] == "eu-central-1"

    def test_session_signed_without_credentials_by_default(
        self,
        captured_session: dict[str, object],
    ) -> None:
        # Absent credentials no longer imply anonymous: GDAL signs and lets
        # boto's resolver supply the credentials, matching s3fs.
        p = make_s3_upath("s3://b/k")
        with s3_env(p):
            pass
        assert captured_session["aws_unsigned"] is False

    def test_bare_upath_without_anon_defaults_to_signed(
        self,
        captured_session: dict[str, object],
    ) -> None:
        # A UPath not built by `make_s3_upath` carries no `anon` key; fall back
        # to the same default s3fs uses rather than inferring from credentials.
        p = UPath("s3://public/k", protocol="s3")
        assert "anon" not in p.storage_options
        with s3_env(p):
            pass
        assert captured_session["aws_unsigned"] is False


class TestSubprocessS3Env:
    def test_local_path_yields_empty_mapping(self, tmp_path: Path) -> None:
        assert subprocess_s3_env(UPath(tmp_path / "f.tif")) == {}

    def test_profile_only(self) -> None:
        p = make_s3_upath("s3://b/k", profile="research")
        assert subprocess_s3_env(p) == {**GDAL_S3_OPTIONS, "AWS_PROFILE": "research"}

    def test_profile_endpoint_reaches_the_child(self, profile_with_endpoint: str) -> None:
        p = make_s3_upath("s3://b/k", profile=profile_with_endpoint)
        env = subprocess_s3_env(p)
        assert env["AWS_PROFILE"] == profile_with_endpoint
        assert env["AWS_S3_ENDPOINT"] == "gateway.example.org"
        assert env["AWS_VIRTUAL_HOSTING"] == "FALSE"

    def test_child_gets_the_reader_options_s3_env_applies(self) -> None:
        # Previously the child listed the directory of every object it opened:
        # GDAL_DISABLE_READDIR_ON_OPEN reached the parent's rasterio.Env only.
        p = make_s3_upath("s3://b/k", profile="research")
        env = subprocess_s3_env(p)
        assert env["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"
        assert env["GDAL_HTTP_MAX_RETRY"] == str(GDAL_HTTP_MAX_RETRY)
        assert env["GDAL_HTTP_RETRY_CODES"] == GDAL_HTTP_RETRY_CODES
        assert GDAL_S3_OPTIONS.items() <= env.items()

    def test_https_endpoint_options(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b", endpoint_url="https://ceph.example.org")
        env = subprocess_s3_env(p)
        assert env["AWS_S3_ENDPOINT"] == "ceph.example.org"  # scheme stripped
        assert env["AWS_VIRTUAL_HOSTING"] == "FALSE"
        assert "AWS_HTTPS" not in env
        assert "AWS_PROFILE" not in env

    def test_http_endpoint_flags_plaintext(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b", endpoint_url="http://minio:9000")
        env = subprocess_s3_env(p)
        assert env["AWS_S3_ENDPOINT"] == "minio:9000"
        assert env["AWS_HTTPS"] == "NO"

    def test_profile_and_endpoint_combine(self) -> None:
        p = make_s3_upath("s3://b/k", profile="research", endpoint_url="https://ceph.example.org")
        env = subprocess_s3_env(p)
        assert env["AWS_PROFILE"] == "research"
        assert env["AWS_S3_ENDPOINT"] == "ceph.example.org"

    def test_explicit_credentials_never_reach_the_child(self) -> None:
        p = make_s3_upath("s3://b/k", key="a", secret="b")
        env = subprocess_s3_env(p)
        assert env == dict(GDAL_S3_OPTIONS)
        assert not any(name.startswith("AWS_") for name in env)


@pytest.fixture
def aws_profiles(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, str]:
    """Point the AWS SDK at an isolated credentials file holding two profiles.

    These tests build *real* `AWSSession` objects rather than using the
    `captured_session` spy, so the profiles they name have to actually resolve.
    Isolating both AWS config paths keeps a developer's real credentials from
    satisfying -- or interfering with -- that resolution.

    Returns the two profile names, which carry different keys.
    """
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[research]\n"
        "aws_access_key_id = AKIARESEARCH\n"
        "aws_secret_access_key = s3cret\n"
        "\n"
        "[archive]\n"
        "aws_access_key_id = AKIAARCHIVE\n"
        "aws_secret_access_key = s3cret\n",
    )
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "config"))
    return "research", "archive"


class TestAwsSessionCache:
    def test_credentials_resolved_once_across_many_envs(
        self,
        aws_profiles: tuple[str, str],
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        # The regression this cache exists for: a named profile takes botocore's
        # env-var provider out of the chain, so every uncached session used to
        # re-read the shared credentials file -- once per raster operation.
        profile, _ = aws_profiles
        p = make_s3_upath("s3://b/k", profile=profile)
        with caplog.at_level(logging.INFO, logger="botocore.credentials"):
            for _ in range(5):
                with s3_env(p):
                    pass
        resolutions = [r for r in caplog.records if "Found credentials" in r.getMessage()]
        assert len(resolutions) == 1

    def test_equivalent_paths_share_a_session(self, aws_profiles: tuple[str, str]) -> None:
        profile, _ = aws_profiles
        first = make_s3_upath("s3://b/k", profile=profile)
        second = make_s3_upath("s3://other-bucket/other-key", profile=profile)
        assert aws_session(first) is aws_session(second)

    def test_child_path_shares_parent_session(self, aws_profiles: tuple[str, str]) -> None:
        # The case that actually recurs: `raster.py` enters `s3_env` for each
        # product path derived from one configured prefix.
        profile, _ = aws_profiles
        parent = make_s3_upath("s3://b/prefix", profile=profile)
        assert aws_session(parent / "scene.tif") is aws_session(parent)

    @pytest.mark.parametrize(
        "other",
        [
            {"profile": "archive"},
            {"key": "AK", "secret": "SK"},
            {"anon": True},
            {"profile": "research", "region": "eu-central-1"},
        ],
    )
    def test_differing_configurations_get_distinct_sessions(
        self,
        other: dict[str, Any],
        aws_profiles: tuple[str, str],
    ) -> None:
        profile, _ = aws_profiles
        base = make_s3_upath("s3://b/k", profile=profile)
        assert aws_session(base) is not aws_session(make_s3_upath("s3://b/k", **other))

    def test_clear_cache_forces_rebuild(self) -> None:
        p = make_s3_upath("s3://b/k", key="AK", secret="SK")
        before = aws_session(p)
        clear_aws_session_cache()
        assert aws_session(p) is not before


@pytest.fixture
def profile_with_endpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> str:
    """Write a shared config file whose one profile carries an ``endpoint_url``."""
    config = tmp_path / "config"
    config.write_text(
        "[profile gateway]\nregion = eu-central-1\nendpoint_url = https://gateway.example.org\n",
    )
    credentials = tmp_path / "credentials"
    credentials.write_text(
        "[gateway]\naws_access_key_id = AKIAGATEWAY\naws_secret_access_key = s3cret\n",
    )
    monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(credentials))
    return "gateway"


class TestConfiguredEndpointUrl:
    def test_profile_section_endpoint(self, profile_with_endpoint: str) -> None:
        assert configured_endpoint_url(profile_with_endpoint) == "https://gateway.example.org"

    def test_missing_profile_is_none(self) -> None:
        assert configured_endpoint_url("no-such-profile") is None

    def test_nothing_configured_is_none(self, profile_with_endpoint: str) -> None:
        del profile_with_endpoint
        assert configured_endpoint_url() is None

    def test_service_variable_wins(
        self,
        profile_with_endpoint: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://minio:9000")
        assert configured_endpoint_url(profile_with_endpoint) == "http://minio:9000"

    def test_ignore_variable_disables_the_lookup(
        self,
        profile_with_endpoint: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # What a client built by botocore does with the same setting.
        monkeypatch.setenv("AWS_IGNORE_CONFIGURED_ENDPOINT_URLS", "true")
        assert configured_endpoint_url(profile_with_endpoint) is None

    def test_profile_ignore_key_disables_the_lookup(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        config = tmp_path / "config"
        config.write_text(
            "[profile gateway]\n"
            "endpoint_url = https://gateway.example.org\n"
            "ignore_configured_endpoint_urls = true\n",
        )
        monkeypatch.setenv("AWS_CONFIG_FILE", str(config))
        assert configured_endpoint_url("gateway") is None

    def test_result_is_cached_until_cleared(
        self,
        profile_with_endpoint: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        assert configured_endpoint_url(profile_with_endpoint) == "https://gateway.example.org"
        monkeypatch.setenv("AWS_ENDPOINT_URL_S3", "http://minio:9000")
        assert configured_endpoint_url(profile_with_endpoint) == "https://gateway.example.org"
        clear_aws_session_cache()
        assert configured_endpoint_url(profile_with_endpoint) == "http://minio:9000"


class TestResolveEndpointUrl:
    def test_explicit_endpoint_wins(self, profile_with_endpoint: str) -> None:
        p = make_s3_upath(
            "s3://b/k",
            profile=profile_with_endpoint,
            endpoint_url="https://ceph.example.org",
        )
        assert resolve_endpoint_url(p) == "https://ceph.example.org"

    def test_profile_endpoint_is_the_fallback(self, profile_with_endpoint: str) -> None:
        p = make_s3_upath("s3://b/k", profile=profile_with_endpoint)
        assert resolve_endpoint_url(p) == "https://gateway.example.org"

    def test_explicit_credentials_still_honour_the_endpoint_variable(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # botocore applies AWS_ENDPOINT_URL to a client whatever its
        # credentials, and so does the client s3fs builds for this path.
        monkeypatch.setenv("AWS_ENDPOINT_URL", "http://minio:9000")
        p = make_s3_upath("s3://b/k", key="a", secret="b")
        assert resolve_endpoint_url(p) == "http://minio:9000"

    def test_plain_aws_is_none(self) -> None:
        assert resolve_endpoint_url(make_s3_upath("s3://b/k", profile="research")) is None


class TestListObjectSizes:
    @pytest.fixture
    def tree(self, tmp_path: Path) -> Path:
        (tmp_path / "tiles" / "2024").mkdir(parents=True)
        (tmp_path / "catalog.json").write_bytes(b"x" * 12)
        (tmp_path / "tiles" / "collection.json").write_bytes(b"x" * 7)
        (tmp_path / "tiles" / "2024" / "a.tif").write_bytes(b"x" * 100)
        return tmp_path

    def test_direct_entries_by_name(self, tree: Path) -> None:
        assert list_object_sizes(tree) == {"catalog.json": 12}

    def test_recursive_entries_by_relative_path(self, tree: Path) -> None:
        assert list_object_sizes(tree, recursive=True) == {
            "catalog.json": 12,
            "tiles/collection.json": 7,
            "tiles/2024/a.tif": 100,
        }

    def test_subtree_is_relative_to_itself(self, tree: Path) -> None:
        assert list_object_sizes(tree / "tiles", recursive=True) == {
            "collection.json": 7,
            "2024/a.tif": 100,
        }

    def test_missing_prefix_is_empty(self, tree: Path) -> None:
        assert list_object_sizes(tree / "nothing", recursive=True) == {}

    def test_memory_filesystem(self) -> None:
        root = UPath("memory://list-object-sizes")
        (root / "a").mkdir(parents=True, exist_ok=True)
        (root / "a" / "b.parquet").write_bytes(b"x" * 5)
        (root / "c.json").write_bytes(b"x" * 2)
        assert list_object_sizes(root) == {"c.json": 2}
        assert list_object_sizes(root, recursive=True) == {"a/b.parquet": 5, "c.json": 2}

    def test_folder_marker_keys_are_not_objects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # An object store lists a zero-byte "tiles/2024/" marker as a file.
        root = UPath("memory://folder-markers")
        entries: list[dict[str, object]] = [
            {"name": f"{root.path}/tiles/2024/", "size": 0, "type": "file"},
            {"name": f"{root.path}/tiles/a.tif", "size": 3, "type": "file"},
            {"name": f"{root.path}/tiles", "size": 0, "type": "directory"},
        ]

        def exists(*args: object, **kwargs: object) -> bool:
            del args, kwargs
            return True

        def ls(*args: object, **kwargs: object) -> list[dict[str, object]]:
            del args, kwargs
            return entries

        monkeypatch.setattr(type(root.fs), "exists", exists)
        monkeypatch.setattr(type(root.fs), "ls", ls)
        assert list_object_sizes(root) == {"tiles/a.tif": 3}


class TestAtomicWriteText:
    def test_writes_and_leaves_no_partial(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        atomic_write_text(target, "{}")
        assert target.read_text() == "{}"
        assert [p.name for p in tmp_path.iterdir()] == ["state.json"]

    def test_replaces_existing(self, tmp_path: Path) -> None:
        target = tmp_path / "state.json"
        target.write_text("old")
        atomic_write_text(target, "new")
        assert target.read_text() == "new"

    def test_failed_write_removes_the_partial(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        target = tmp_path / "state.json"

        def boom(self: UPath, text: str, **kwargs: object) -> int:
            del self, text, kwargs
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(type(UPath(target)), "write_text", boom)
        with pytest.raises(OSError, match="disk full"):
            atomic_write_text(target, "{}")
        assert list(tmp_path.iterdir()) == []

    def test_other_object_stores_write_directly(self) -> None:
        # Nothing but the local filesystem has a rename; a put is atomic.
        target = UPath("memory://atomic-write/state.json")
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(target, "{}")
        assert target.read_text() == "{}"
        assert [p.name for p in target.parent.iterdir()] == ["state.json"]

    def test_s3_is_one_put(self, monkeypatch: pytest.MonkeyPatch) -> None:
        written: dict[str, str] = {}
        p = make_s3_upath("s3://b/state.json", profile="research")

        def record(self: UPath, text: str, **kwargs: object) -> int:
            del kwargs
            written[str(self)] = text
            return len(text)

        monkeypatch.setattr(type(p), "write_text", record)
        atomic_write_text(p, "{}")
        assert written == {"s3://b/state.json": "{}"}


class TestContentTypeFor:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("tiles/2024/item.json", "application/json"),
            ("README.md", "text/markdown"),
            ("items.parquet", "application/vnd.apache.parquet"),
            ("coh12_vv.tif", "image/tiff; application=geotiff; profile=cloud-optimized"),
            ("COH12_VV.TIF", "image/tiff; application=geotiff; profile=cloud-optimized"),
            ("coh12_rgb.pmtiles", "application/vnd.pmtiles"),
            ("wordmark.svg", "image/svg+xml"),
            ("LICENSE", DEFAULT_CONTENT_TYPE),
            ("archive.zip", DEFAULT_CONTENT_TYPE),
        ],
    )
    def test_by_suffix(self, name: str, expected: str) -> None:
        assert content_type_for(name) == expected

    def test_accepts_paths(self, tmp_path: Path) -> None:
        assert content_type_for(tmp_path / "x.png") == "image/png"
        assert content_type_for(UPath("s3://b/x.parquet")) == CONTENT_TYPES[".parquet"]
