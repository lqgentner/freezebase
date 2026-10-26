"""Tests for the exception hierarchy."""

from __future__ import annotations

import pytest

from freezebase.errors import DownloadError, FreezebaseError, RasterWriteError


@pytest.mark.parametrize("error", [DownloadError, RasterWriteError])
def test_errors_keep_their_builtin_base(error: type[FreezebaseError]) -> None:
    assert issubclass(error, FreezebaseError)
    assert issubclass(error, RuntimeError)
