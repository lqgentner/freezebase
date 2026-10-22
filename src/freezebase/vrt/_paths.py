"""How a VRT names its sources, on write and on read."""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

from upath import UPath

from freezebase.raster import _to_vsi_uri

if TYPE_CHECKING:
    from collections.abc import Mapping


def _same_store(path: UPath, other: UPath) -> bool:
    """Whether both paths are local, or both on the same S3 bucket."""
    if path.protocol != other.protocol:
        return False
    if path.protocol == "s3":
        return path.path.split("/", 1)[0] == other.path.split("/", 1)[0]
    return path.protocol == ""


def gdal_path(path: UPath, *, relative_to: UPath | None) -> tuple[str, bool]:
    """Return the text GDAL reads ``path`` by and whether it is relative to the VRT.

    Parameters
    ----------
    path : UPath
        The source file.
    relative_to : UPath | None
        The VRT file; ``None`` always gives an absolute path.

    Returns
    -------
    tuple[str, bool]
        The path text and ``True`` when it is relative to ``relative_to``.
    """
    if path.protocol == "":
        path = path.resolve()
    if relative_to is not None:
        if relative_to.protocol == "":
            relative_to = relative_to.resolve()
        if _same_store(path, relative_to):
            return posixpath.relpath(path.path, relative_to.parent.path), True
    return _to_vsi_uri(path), False


def inherited_options(
    base: UPath | None, storage_options: Mapping[str, Any] | None
) -> Mapping[str, Any]:
    """Return ``storage_options``, else those of an S3 ``base``, for the VRT's S3 sources."""
    if storage_options is not None:
        return storage_options
    if base is not None and base.protocol == "s3":
        return base.storage_options
    return {}


def parse_gdal_path(
    text: str,
    *,
    relative: bool,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> UPath:
    """Return the path GDAL reads ``text`` as.

    Parameters
    ----------
    text : str
        A source path as written in a VRT.
    relative : bool
        Whether ``text`` is relative to the VRT.
    base : UPath | None
        The VRT file; ``None`` when the VRT has no location.
    storage_options : Mapping[str, Any] | None
        Options for an absolute S3 ``text``; ``None`` inherits those of an S3 ``base``. A
        relative ``text`` always keeps the options of ``base``.

    Returns
    -------
    UPath
        The source path.

    Raises
    ------
    ValueError
        If ``text`` uses a GDAL virtual file system other than ``/vsis3/`` and ``/vsicurl/``,
        or is relative without a ``base``.
    """
    if text.startswith("/vsis3/"):
        return UPath(
            f"s3://{text.removeprefix('/vsis3/')}", **inherited_options(base, storage_options)
        )
    if text.startswith("/vsicurl/"):
        return UPath(text.removeprefix("/vsicurl/"))
    if text.startswith("/vsi"):
        msg = f"Unsupported GDAL path {text!r}"
        raise ValueError(msg)
    if not relative:
        return UPath(text)
    if base is None:
        msg = f"Source path {text!r} is relative to a VRT without a location"
        raise ValueError(msg)
    if base.protocol in {"http", "https"}:
        return UPath(urljoin(str(base), text))
    joined = posixpath.normpath(posixpath.join(base.parent.path, text))
    if base.protocol == "s3":
        return UPath(f"s3://{joined}", **base.storage_options)
    return base.with_segments(joined)
