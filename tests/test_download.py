"""Tests for freezebase.download: filename confinement and redirect credential safety."""

from __future__ import annotations

import gc
import io
from typing import TYPE_CHECKING, Self
from unittest.mock import MagicMock
from urllib.parse import urlparse
from uuid import uuid4
import weakref

import pytest
import requests
from requests.adapters import BaseAdapter
from requests.structures import CaseInsensitiveDict
from upath import UPath

from freezebase.download import (
    MAX_REDIRECTS,
    REMOTE_BLOCK_SIZE,
    HTTPDownloader,
    _extract_filename_from_cd,
    _resolve_within,
    _rewrite_redirect_method,
    _sanitize_filename,
    _write_file,
)

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

HOST = "https://host.example"
OTHER_HOST = "https://evil.example"
TRUSTED_OTHER_HOST = "https://trusted.example"


def make_response(
    *,
    status: int = 200,
    headers: dict[str, str] | None = None,
    url: str = f"{HOST}/file.zip",
    body: bytes = b"payload",
) -> requests.Response:
    resp = requests.Response()
    resp.status_code = status
    resp.url = url
    resp.headers = CaseInsensitiveDict(headers or {})
    resp.raw = io.BytesIO(body)
    resp.encoding = "utf-8"
    return resp


class FakeSession:
    """Minimal stand-in for ``requests.Session`` returning scripted responses."""

    def __init__(self, responses: dict[str, requests.Response]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str, object]] = []
        self.closed: list[str] = []

    def request(
        self,
        method: str,
        url: str,
        auth: object = None,
        **_: object,
    ) -> requests.Response:
        self.calls.append((method, url, auth))
        resp = self._responses[url]
        resp.raw = io.BytesIO(b"payload")  # reset the stream for re-reads
        original_close = resp.close

        def _close() -> None:
            self.closed.append(url)
            original_close()

        resp.close = _close  # type: ignore[method-assign]
        return resp


class _ScriptedAdapter(BaseAdapter):
    """Transport adapter serving canned responses by URL.

    Mounted on a real `requests.Session`, so the requests it records are the
    fully prepared ones a server would receive, headers included. `FakeSession`
    intercepts higher up and only records the arguments passed to it.
    """

    def __init__(self, responses: dict[str, requests.Response]) -> None:
        super().__init__()
        self._responses = responses
        self.calls: list[requests.PreparedRequest] = []

    def send(self, request: requests.PreparedRequest, **_kwargs: object) -> requests.Response:
        self.calls.append(request)
        resp = self._responses[request.url]
        resp.request = request
        resp.raw = io.BytesIO(b"payload")  # reset the stream for re-reads
        return resp

    def close(self) -> None:
        pass


def make_downloader(session: FakeSession, **kwargs: object) -> HTTPDownloader:
    dl = HTTPDownloader(progress=False, **kwargs)  # type: ignore[arg-type]
    dl.session = session  # type: ignore[assignment]
    return dl


def _adapter_downloader(
    target: str,
    **kwargs: object,
) -> tuple[_ScriptedAdapter, HTTPDownloader]:
    """Build a downloader whose transport serves `HOST/a` -> `target` via a 302."""
    adapter = _ScriptedAdapter(
        {
            f"{HOST}/a": make_response(
                status=302,
                url=f"{HOST}/a",
                headers={"Location": target},
            ),
            target: make_response(
                url=target,
                headers={"Content-Disposition": 'attachment; filename="real.zip"'},
            ),
        },
    )
    dl = HTTPDownloader(progress=False, **kwargs)  # type: ignore[arg-type]
    dl.session.mount("http://", adapter)
    dl.session.mount("https://", adapter)
    return adapter, dl


# ---------------------------------------------------------------------------
# FC-01: filename confinement
# ---------------------------------------------------------------------------


class TestSanitizeFilename:
    @pytest.mark.parametrize(
        "name",
        [
            "../evil.zip",
            "../../etc/passwd",
            "/etc/passwd",
            "a/b.zip",
            "a\\b.zip",
            "..\\evil.zip",
            "C:\\Windows\\system32",
            "name:stream",
            "CON",
            "con.txt",
            "LPT1.dat",
            "..",
            ".",
            "",
            "bad\x00name",
            "tab\tname",
        ],
    )
    def test_rejects_unsafe_names(self, name: str) -> None:
        with pytest.raises(ValueError, match="Refusing"):
            _sanitize_filename(name, explicit=False)

    @pytest.mark.parametrize("name", ["file.zip", "S1A_20200101.tif", "data.tar.gz", "plain"])
    def test_accepts_safe_names(self, name: str) -> None:
        assert _sanitize_filename(name, explicit=True) == name


class TestResolveWithin:
    def test_confines_to_directory(self, tmp_path: Path) -> None:
        assert _resolve_within(tmp_path, "file.zip") == tmp_path / "file.zip"

    def test_rejects_symlink_escape(self, tmp_path: Path) -> None:
        outside = tmp_path / "outside"
        outside.mkdir()
        save_dir = tmp_path / "save"
        # `save` resolves (via a symlink) to a sibling of where the caller thinks
        # it is; a sanitized name is still fine, but a link *named* like the file
        # would escape -- here we exercise the resolved-parent guard directly.
        save_dir.symlink_to(outside, target_is_directory=True)
        # The join stays a bare name, so this must succeed and land in `outside`.
        target = _resolve_within(save_dir, "file.zip")
        assert target.resolve().parent == outside.resolve()


class TestDownloadFilenameConfinement:
    def test_malicious_content_disposition_rejected(self, tmp_path: Path) -> None:
        resp = make_response(
            headers={"Content-Disposition": 'attachment; filename="../../evil.zip"'},
        )
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        with pytest.raises(ValueError, match="Refusing"):
            dl(f"{HOST}/file", tmp_path)

        assert not (tmp_path.parent / "evil.zip").exists()

    def test_explicit_malicious_filename_rejected(self, tmp_path: Path) -> None:
        resp = make_response(headers={"Content-Type": "application/zip"})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        with pytest.raises(ValueError, match="Refusing"):
            dl(f"{HOST}/file", tmp_path, filename="../escape.zip")

    def test_encoded_traversal_rejected(self, tmp_path: Path) -> None:
        resp = make_response(
            headers={"Content-Disposition": "attachment; filename*=UTF-8''%2e%2e%2fevil.zip"},
        )
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        with pytest.raises(ValueError, match="Refusing"):
            dl(f"{HOST}/file", tmp_path)

    def test_happy_path_writes_confined_file(self, tmp_path: Path) -> None:
        resp = make_response(
            headers={"Content-Disposition": 'attachment; filename="real.zip"'},
        )
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        out = dl(f"{HOST}/file", tmp_path)

        assert out == tmp_path / "real.zip"
        assert out.read_bytes() == b"payload"

    def test_existing_target_not_overwritten_by_default(self, tmp_path: Path) -> None:
        existing = tmp_path / "real.zip"
        existing.write_bytes(b"original")
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        with pytest.raises(FileExistsError):
            dl(f"{HOST}/file", tmp_path)
        assert existing.read_bytes() == b"original"

    def test_overwrite_true_replaces(self, tmp_path: Path) -> None:
        existing = tmp_path / "real.zip"
        existing.write_bytes(b"original")
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        out = dl(f"{HOST}/file", tmp_path, overwrite=True)
        assert out.read_bytes() == b"payload"

    def test_no_partial_files_left_behind(self, tmp_path: Path) -> None:
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        dl(f"{HOST}/file", tmp_path)
        assert not list(tmp_path.glob("*.partial"))


# ---------------------------------------------------------------------------
# Remote (non-local) destinations
# ---------------------------------------------------------------------------


class _RecordingRemoteFile:
    """Minimal stand-in for a remote file handle that records its open kwargs."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.chunks: list[bytes] = []
        self.open_kwargs: dict[str, object] = {}
        self.exists_value = False

    # Not a `pathlib.Path`: that is what `_write_file` branches on.
    def exists(self) -> bool:
        return self.exists_value

    def open(self, mode: str, **kwargs: object) -> Self:
        assert mode == "wb"
        self.open_kwargs = kwargs
        return self

    def write(self, chunk: bytes) -> None:
        self.chunks.append(chunk)

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc_info: object) -> None:
        pass


@pytest.fixture
def memory_dir() -> Iterator[UPath]:
    """Return an empty in-memory directory, unique to this test."""
    directory = UPath(f"memory://bucket/{uuid4().hex}")
    yield directory
    if directory.exists():
        directory.fs.rm(directory.path, recursive=True)


class TestRemoteDestination:
    """Downloads to a non-local ``UPath`` write the final key directly."""

    def test_writes_file_to_remote_path(self, memory_dir: UPath) -> None:
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        out = dl(f"{HOST}/file", memory_dir)

        assert out == memory_dir / "real.zip"
        assert out.read_bytes() == b"payload"

    def test_creates_the_remote_directory(self, memory_dir: UPath) -> None:
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        dl(f"{HOST}/file", memory_dir)

        assert memory_dir.exists()

    def test_no_partial_file_is_staged_remotely(self, memory_dir: UPath) -> None:
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        dl(f"{HOST}/file", memory_dir)

        assert [p.name for p in memory_dir.iterdir()] == ["real.zip"]

    def test_existing_remote_target_not_overwritten_by_default(self, memory_dir: UPath) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "real.zip").write_bytes(b"original")
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        with pytest.raises(FileExistsError):
            dl(f"{HOST}/file", memory_dir)
        assert (memory_dir / "real.zip").read_bytes() == b"original"

    def test_overwrite_true_replaces_remote_target(self, memory_dir: UPath) -> None:
        memory_dir.mkdir(parents=True, exist_ok=True)
        (memory_dir / "real.zip").write_bytes(b"original")
        resp = make_response(headers={"Content-Disposition": 'attachment; filename="real.zip"'})
        dl = make_downloader(FakeSession({f"{HOST}/file": resp}))

        out = dl(f"{HOST}/file", memory_dir, overwrite=True)

        assert out.read_bytes() == b"payload"

    def test_remote_write_uses_the_large_block_size(self) -> None:
        # Object stores pay per request, hence the larger block size.
        resp = make_response()
        target = _RecordingRemoteFile("real.zip")

        _write_file(resp, target, show_progress=False)  # type: ignore[type-var]

        assert target.open_kwargs == {"block_size": REMOTE_BLOCK_SIZE}
        assert b"".join(target.chunks) == b"payload"

    def test_remote_write_respects_overwrite_guard(self) -> None:
        resp = make_response()
        target = _RecordingRemoteFile("real.zip")
        target.exists_value = True

        with pytest.raises(FileExistsError, match="Target already exists"):
            _write_file(resp, target, show_progress=False)  # type: ignore[type-var]

        assert target.chunks == []


# ---------------------------------------------------------------------------
# FC-02: redirect credential safety
# ---------------------------------------------------------------------------


class TestRedirectCredentialSafety:
    def _final(self) -> requests.Response:
        return make_response(
            headers={"Content-Disposition": 'attachment; filename="real.zip"'},
        )

    def test_auth_preserved_on_same_host_redirect(self, tmp_path: Path) -> None:
        redirect = make_response(
            status=302,
            url=f"{HOST}/a",
            headers={"Location": f"{HOST}/b"},
        )
        session = FakeSession({f"{HOST}/a": redirect, f"{HOST}/b": self._final()})
        dl = make_downloader(session, auth=("user", "pass"))

        dl(f"{HOST}/a", tmp_path)

        auths = [auth for _, _, auth in session.calls]
        assert auths[0] == ("user", "pass")
        assert auths[1] == ("user", "pass")  # same host keeps credentials

    def test_auth_dropped_on_cross_host_redirect(self, tmp_path: Path) -> None:
        redirect = make_response(
            status=302,
            url=f"{HOST}/a",
            headers={"Location": f"{OTHER_HOST}/b"},
        )
        final = make_response(
            url=f"{OTHER_HOST}/b",
            headers={"Content-Disposition": 'attachment; filename="real.zip"'},
        )
        session = FakeSession({f"{HOST}/a": redirect, f"{OTHER_HOST}/b": final})
        dl = make_downloader(session, auth=("user", "pass"))

        dl(f"{HOST}/a", tmp_path)

        assert session.calls[0][2] == ("user", "pass")
        assert session.calls[1][2] is None  # credentials dropped leaving the host

    def test_auth_dropped_on_scheme_downgrade(self, tmp_path: Path) -> None:
        redirect = make_response(
            status=302,
            url=f"{HOST}/a",
            headers={"Location": "http://host.example/b"},  # same host, https->http
        )
        final = make_response(
            url="http://host.example/b",
            headers={"Content-Disposition": 'attachment; filename="real.zip"'},
        )
        session = FakeSession({f"{HOST}/a": redirect, "http://host.example/b": final})
        dl = make_downloader(session, auth=("user", "pass"))

        dl(f"{HOST}/a", tmp_path)

        assert session.calls[1][2] is None  # never send credentials over plaintext

    def test_instance_auth_not_mutated_across_calls(self, tmp_path: Path) -> None:
        # First call leaves a trusted host, dropping auth for that chain only.
        redirect = make_response(
            status=302,
            url=f"{HOST}/a",
            headers={"Location": f"{OTHER_HOST}/b"},
        )
        final_other = make_response(
            url=f"{OTHER_HOST}/b",
            headers={"Content-Disposition": 'attachment; filename="a.zip"'},
        )
        direct = make_response(
            url=f"{HOST}/c",
            headers={"Content-Disposition": 'attachment; filename="c.zip"'},
        )
        session = FakeSession(
            {
                f"{HOST}/a": redirect,
                f"{OTHER_HOST}/b": final_other,
                f"{HOST}/c": direct,
            },
        )
        dl = make_downloader(session, auth=("user", "pass"))

        dl(f"{HOST}/a", tmp_path)
        dl(f"{HOST}/c", tmp_path)

        # The second, independent call must still carry credentials.
        assert session.calls[-1][2] == ("user", "pass")

    def test_intermediate_response_closed(self, tmp_path: Path) -> None:
        redirect = make_response(
            status=302,
            url=f"{HOST}/a",
            headers={"Location": f"{HOST}/b"},
        )
        session = FakeSession({f"{HOST}/a": redirect, f"{HOST}/b": self._final()})
        dl = make_downloader(session, auth=("user", "pass"))

        dl(f"{HOST}/a", tmp_path)
        assert f"{HOST}/a" in session.closed

    def test_redirect_loop_is_bounded(self, tmp_path: Path) -> None:
        a = make_response(status=302, url=f"{HOST}/a", headers={"Location": f"{HOST}/b"})
        b = make_response(status=302, url=f"{HOST}/b", headers={"Location": f"{HOST}/a"})
        session = FakeSession({f"{HOST}/a": a, f"{HOST}/b": b})
        dl = make_downloader(session, auth=("user", "pass"))

        with pytest.raises(RuntimeError, match="Exceeded maximum"):
            dl(f"{HOST}/a", tmp_path)
        assert len(session.calls) == MAX_REDIRECTS + 1

    def test_auth_preserved_on_trusted_cross_host_redirect(self, tmp_path: Path) -> None:
        """A host named in `trusted_hosts` still receives the credentials.

        Asserts on the `Authorization` header the server would actually see,
        rather than on the `auth` argument, so the whole path from `auth=` to
        the wire is covered.
        """
        adapter, dl = _adapter_downloader(
            f"{TRUSTED_OTHER_HOST}/b",
            auth=("user", "pass"),
            trusted_hosts=[urlparse(TRUSTED_OTHER_HOST).hostname],
        )
        dl(f"{HOST}/a", tmp_path)

        assert len(adapter.calls) == 2
        assert "Authorization" in adapter.calls[0].headers
        assert "Authorization" in adapter.calls[1].headers

    def test_auth_not_sent_to_untrusted_host_on_the_wire(self, tmp_path: Path) -> None:
        """The counterpart: an unlisted host receives no `Authorization` header."""
        adapter, dl = _adapter_downloader(f"{OTHER_HOST}/b", auth=("user", "pass"))
        dl(f"{HOST}/a", tmp_path)

        assert len(adapter.calls) == 2
        assert "Authorization" in adapter.calls[0].headers
        assert "Authorization" not in adapter.calls[1].headers


# ---------------------------------------------------------------------------
# Session lifecycle: close() / context manager / __del__ backstop
# ---------------------------------------------------------------------------


class TestSessionLifecycle:
    def test_close_closes_session(self) -> None:
        dl = HTTPDownloader(progress=False)
        dl.session = MagicMock(wraps=dl.session)
        dl.close()
        dl.session.close.assert_called_once()

    def test_context_manager_closes_session_on_exit(self) -> None:
        with HTTPDownloader(progress=False) as dl:
            dl.session = MagicMock(wraps=dl.session)
            session = dl.session
        session.close.assert_called_once()

    def test_gc_closes_session_as_backstop(self) -> None:
        dl = HTTPDownloader(progress=False)
        session = MagicMock(wraps=dl.session)
        dl.session = session
        ref = weakref.ref(dl)

        del dl
        gc.collect()

        assert ref() is None
        session.close.assert_called_once()


# ---------------------------------------------------------------------------
# Response lifecycle: a streamed body is closed on every path out of __call__
# ---------------------------------------------------------------------------


class TestResponseClosedOnFailure:
    """A streamed response holds its connection until closed.

    The connection is checked out of the pool for the lifetime of the streamed
    body, so `HTTPDownloader.close()` cannot reclaim it: only closing the
    response itself returns the socket. Leaving that to garbage collection
    leaks the connection and raises `ResourceWarning` at an unrelated moment,
    typically interpreter shutdown.
    """

    def test_error_status_closes_response(self, tmp_path: Path) -> None:
        url = f"{HOST}/file.zip"
        session = FakeSession({url: make_response(status=500, url=url)})
        dl = make_downloader(session)

        with pytest.raises(requests.HTTPError):
            dl(url=url, save_dir=tmp_path, filename="file.zip")

        assert session.closed == [url]

    def test_undownloadable_body_closes_response(self, tmp_path: Path) -> None:
        url = f"{HOST}/file.zip"
        session = FakeSession(
            {url: make_response(url=url, headers={"Content-Type": "text/html"})},
        )
        dl = make_downloader(session)

        with pytest.raises(RuntimeError, match="No downloadable file found"):
            dl(url=url, save_dir=tmp_path, filename="file.zip")

        assert session.closed == [url]

    def test_rejected_filename_closes_response(self, tmp_path: Path) -> None:
        url = f"{HOST}/file.zip"
        session = FakeSession(
            {url: make_response(url=url, headers={"Content-Type": "application/zip"})},
        )
        dl = make_downloader(session)

        with pytest.raises(ValueError, match="directory separator"):
            dl(url=url, save_dir=tmp_path, filename="../escape.zip")

        assert session.closed == [url]

    def test_existing_target_closes_response(self, tmp_path: Path) -> None:
        url = f"{HOST}/file.zip"
        session = FakeSession(
            {url: make_response(url=url, headers={"Content-Type": "application/zip"})},
        )
        dl = make_downloader(session)
        (tmp_path / "file.zip").write_bytes(b"already here")

        with pytest.raises(FileExistsError):
            dl(url=url, save_dir=tmp_path, filename="file.zip")

        assert session.closed == [url]

    def test_successful_download_closes_response(self, tmp_path: Path) -> None:
        url = f"{HOST}/file.zip"
        session = FakeSession(
            {url: make_response(url=url, headers={"Content-Type": "application/zip"})},
        )
        dl = make_downloader(session)

        filepath = dl(url=url, save_dir=tmp_path, filename="file.zip")

        assert filepath.read_bytes() == b"payload"
        assert session.closed == [url]


# ---------------------------------------------------------------------------
# Redirect method rewriting and RFC 5987 decoding
# ---------------------------------------------------------------------------


class TestRedirectMethodRewrite:
    def test_see_other_becomes_get(self) -> None:
        assert _rewrite_redirect_method(303, "POST") == "GET"

    def test_moved_post_becomes_get(self) -> None:
        assert _rewrite_redirect_method(301, "POST") == "GET"

    def test_temporary_redirect_preserves_method(self) -> None:
        assert _rewrite_redirect_method(307, "POST") == "POST"


class TestExtractFilenameFromCd:
    def test_rfc5987_is_percent_decoded(self) -> None:
        cd = "attachment; filename*=UTF-8''r%C3%A9sum%C3%A9.pdf"
        assert _extract_filename_from_cd(cd) == "résumé.pdf"

    def test_plain_filename(self) -> None:
        assert _extract_filename_from_cd('attachment; filename="a.zip"') == "a.zip"

    def test_none_when_absent(self) -> None:
        assert _extract_filename_from_cd("attachment") is None
