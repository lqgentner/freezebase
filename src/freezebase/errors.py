"""Exceptions raised by freezebase."""

from __future__ import annotations


class FreezebaseError(Exception):
    """Base class for freezebase-specific errors."""


class DownloadError(FreezebaseError, RuntimeError):
    """A download could not be completed."""
