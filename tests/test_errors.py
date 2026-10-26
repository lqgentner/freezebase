"""Tests for the exception hierarchy."""

from __future__ import annotations

from freezebase.errors import DownloadError, FreezebaseError


def test_download_error_keeps_its_builtin_base() -> None:
    assert issubclass(DownloadError, FreezebaseError)
    assert issubclass(DownloadError, RuntimeError)
