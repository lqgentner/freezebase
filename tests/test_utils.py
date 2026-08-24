"""Tests for shared helpers in freezebase.utils and freezebase.download."""

from __future__ import annotations

import hashlib
import sys
import types
from typing import TYPE_CHECKING

import pytest
import requests

from freezebase.download import _is_transient_request_error
from freezebase.utils import (
    depends_on_optional,
    file_sha256,
    format_valid_options,
    get_cache_dir,
    get_credentials_from_env,
    get_data_dir,
    shorten_string,
)

if TYPE_CHECKING:
    from pathlib import Path


class TestTransientRequestError:
    def _http_error(self, status_code: int) -> requests.exceptions.HTTPError:
        response = requests.Response()
        response.status_code = status_code
        return requests.exceptions.HTTPError(response=response)

    @pytest.mark.parametrize("status_code", [408, 429, 502, 503, 504])
    def test_transient_status_codes(self, status_code: int) -> None:
        assert _is_transient_request_error(self._http_error(status_code))

    @pytest.mark.parametrize("status_code", [400, 401, 404, 500])
    def test_non_transient_status_codes(self, status_code: int) -> None:
        assert not _is_transient_request_error(self._http_error(status_code))

    def test_extra_status_codes(self) -> None:
        assert _is_transient_request_error(
            self._http_error(500),
            extra_status_codes=frozenset({500}),
        )

    def test_connection_and_timeout_errors(self) -> None:
        assert _is_transient_request_error(requests.exceptions.ConnectionError())
        assert _is_transient_request_error(requests.exceptions.Timeout())

    def test_unrelated_exception(self) -> None:
        assert not _is_transient_request_error(ValueError("nope"))


class TestShortenString:
    def test_short_string_unchanged(self) -> None:
        assert shorten_string("short.zip", 30) == "short.zip"

    def test_long_string_shortened_with_ellipsis(self) -> None:
        result = shorten_string("a" * 20 + "b" * 20, 21)

        assert len(result) == 21
        assert "..." in result

    @pytest.mark.parametrize("n", [-5, 0, 1, 2, 3, 4, 5, 8])
    def test_never_exceeds_limit(self, n: int) -> None:
        # Previously `shorten_string("abcdef", 3)` returned "...abcdef".
        result = shorten_string("abcdef", n)
        assert len(result) <= max(n, 0)

    def test_tiny_limit_hard_truncates(self) -> None:
        assert shorten_string("abcdef", 3) == "abc"


class TestGetCredentialsFromEnv:
    def test_returns_both(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FC_USER", "alice")
        monkeypatch.setenv("FC_PASS", "secret")
        assert get_credentials_from_env("FC_USER", "FC_PASS") == ("alice", "secret")

    def test_rejects_empty_string(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Previously an empty string was accepted as a valid credential.
        monkeypatch.setenv("FC_USER", "")
        monkeypatch.setenv("FC_PASS", "secret")
        with pytest.raises(KeyError, match="FC_USER"):
            get_credentials_from_env("FC_USER", "FC_PASS")

    def test_error_mentions_environment_not_dotenv(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.delenv("FC_USER", raising=False)
        monkeypatch.delenv("FC_PASS", raising=False)
        with pytest.raises(KeyError, match="Environment variables"):
            get_credentials_from_env("FC_USER", "FC_PASS")


class TestDependsOnOptional:
    def test_present_module_runs_the_function(self) -> None:
        @depends_on_optional("json")
        def needs_json() -> str:
            return "ran"

        assert needs_json() == "ran"

    def test_missing_module_raises_import_error(self) -> None:
        @depends_on_optional("definitely_not_a_real_module_xyz")
        def needs_missing() -> str:  # pragma: no cover -- must never run
            return "ran"

        # Must name both the module and the function to be actionable.
        with pytest.raises(ImportError, match=r"definitely_not_a_real_module_xyz.*needs_missing"):
            needs_missing()

    def test_install_hint_is_appended(self) -> None:
        @depends_on_optional("definitely_not_a_real_module_xyz", "pip install freezebase[s3]")
        def needs_missing() -> str:  # pragma: no cover -- must never run
            return "ran"

        with pytest.raises(ImportError, match=r"Install with: pip install freezebase\[s3\]"):
            needs_missing()

    def test_unspecced_module_treated_as_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # A module without __spec__ makes find_spec raise ValueError.
        module = types.ModuleType("freezebase_unspecced_probe")
        module.__spec__ = None
        monkeypatch.setitem(sys.modules, "freezebase_unspecced_probe", module)

        @depends_on_optional("freezebase_unspecced_probe")
        def needs_unspecced() -> str:  # pragma: no cover -- must never run
            return "ran"

        with pytest.raises(ImportError, match="freezebase_unspecced_probe"):
            needs_unspecced()

    def test_preserves_function_identity(self) -> None:
        @depends_on_optional("json")
        def documented() -> None:
            """Original docstring."""

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "Original docstring."


class TestGetCacheDir:
    def test_explicit_argument_wins_over_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("FREEZEBASE_CACHE", str(tmp_path / "from_env"))
        assert get_cache_dir(tmp_path / "explicit") == tmp_path / "explicit"

    def test_environment_variable_used_and_expanded(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setenv("FREEZEBASE_CACHE", "~/fbcache")
        result = get_cache_dir()
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_falls_back_to_system_cache_with_warning(
        self,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        monkeypatch.delenv("FREEZEBASE_CACHE", raising=False)

        with caplog.at_level("WARNING", logger="freezebase.utils"):
            result = get_cache_dir()

        assert result.name == "freezebase"
        # The warning is what tells users the cache landed somewhere implicit.
        assert "FREEZEBASE_CACHE" in caplog.text


class TestGetDataDir:
    def test_expands_user(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FREEZEBASE_DATA", "~/fcdata")
        result = get_data_dir()
        assert "~" not in str(result)
        assert result.is_absolute()

    def test_explicit_argument_wins_over_environment(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        monkeypatch.setenv("FREEZEBASE_DATA", str(tmp_path / "from_env"))
        assert get_data_dir(tmp_path / "explicit") == tmp_path / "explicit"

    def test_raises_when_neither_argument_nor_environment_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Unlike the cache dir, there is no fallback location.
        monkeypatch.delenv("FREEZEBASE_DATA", raising=False)
        with pytest.raises(ValueError, match="FREEZEBASE_DATA"):
            get_data_dir()


class TestFileSha256:
    def test_matches_hashlib(self, tmp_path: Path) -> None:
        data = b"freezebase" * 100_000  # exceeds one chunk
        path = tmp_path / "blob.bin"
        path.write_bytes(data)
        assert file_sha256(path) == hashlib.sha256(data).hexdigest()

    def test_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.bin"
        path.write_bytes(b"")
        assert file_sha256(path) == hashlib.sha256(b"").hexdigest()


def test_format_valid_options() -> None:
    formatted = format_valid_options({"IW": "Interferometric Wide", "EW": "Extra Wide"})

    assert formatted == "- 'IW': Interferometric Wide\n- 'EW': Extra Wide"
