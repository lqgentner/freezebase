"""Tests for the shared fixtures' helpers."""

from __future__ import annotations

import pytest

from tests.conftest import skip_or_fail


def test_skip_or_fail_skips_locally(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without CI set, a missing resource skips."""
    monkeypatch.delenv("CI", raising=False)
    with pytest.raises(pytest.skip.Exception):
        skip_or_fail("grids missing")


def test_skip_or_fail_fails_under_ci(monkeypatch: pytest.MonkeyPatch) -> None:
    """With CI set, a missing resource fails."""
    monkeypatch.setenv("CI", "true")
    with pytest.raises(pytest.fail.Exception, match="grids missing"):
        skip_or_fail("grids missing")
