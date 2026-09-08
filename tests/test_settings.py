"""Tests for freezebase.settings."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pydantic_settings")
pytest.importorskip("s3fs")

from upath import UPath

from freezebase.settings import S3Settings


@pytest.fixture
def _env(monkeypatch: pytest.MonkeyPatch) -> None:
    """One prefixed set of variables, plus a stray value under another prefix."""
    monkeypatch.setenv("FB_TEST_S3_PROFILE", "research")
    monkeypatch.setenv("FB_TEST_S3_ENDPOINT_URL", "https://ceph.example.org")
    monkeypatch.setenv("FB_TEST_S3_REQUEST_CHECKSUM_CALCULATION", "when_supported")
    monkeypatch.setenv("OTHER_S3_PROFILE", "archive")


@pytest.mark.usefixtures("_env")
class TestFromEnv:
    def test_reads_the_prefixed_variables(self) -> None:
        settings = S3Settings.from_env("FB_TEST_S3_")
        assert settings.profile == "research"
        assert settings.endpoint_url == "https://ceph.example.org"
        assert settings.request_checksum_calculation == "when_supported"
        assert settings.response_checksum_validation == "when_required"

    def test_another_prefix_is_another_configuration(self) -> None:
        settings = S3Settings.from_env("OTHER_S3_")
        assert settings.profile == "archive"
        assert settings.endpoint_url is None

    def test_checksum_typo_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FB_TEST_S3_RESPONSE_CHECKSUM_VALIDATION", "always")
        with pytest.raises(ValueError, match="response_checksum_validation"):
            S3Settings.from_env("FB_TEST_S3_")


@pytest.mark.usefixtures("_env")
class TestResolvePath:
    def test_local_path_expands_home(self) -> None:
        resolved = S3Settings.from_env("FB_TEST_S3_").resolve_path("~/data")
        assert isinstance(resolved, Path)
        assert resolved == Path.home() / "data"

    def test_s3_uri_carries_the_settings(self) -> None:
        resolved = S3Settings.from_env("FB_TEST_S3_").resolve_path("s3://bucket/prefix")
        assert isinstance(resolved, UPath)
        assert resolved.protocol == "s3"
        so = resolved.storage_options
        assert so["profile"] == "research"
        assert so["endpoint_url"] == "https://ceph.example.org"
        assert so["config_kwargs"]["request_checksum_calculation"] == "when_supported"
        assert so["config_kwargs"]["retries"] == {"max_attempts": 10, "mode": "adaptive"}

    def test_endpoint_is_optional(self) -> None:
        # The profile's config section may carry the endpoint instead.
        resolved = S3Settings.from_env("OTHER_S3_").make_upath("s3://bucket/prefix")
        assert resolved.storage_options["profile"] == "archive"
        assert resolved.storage_options["endpoint_url"] is None

    def test_missing_profile_names_the_variable(self) -> None:
        settings = S3Settings.from_env("UNSET_S3_")
        with pytest.raises(ValueError, match="UNSET_S3_PROFILE"):
            settings.resolve_path("s3://bucket/prefix")
