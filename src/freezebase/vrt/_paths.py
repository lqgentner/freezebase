"""How a VRT names its sources on write."""

from __future__ import annotations

import posixpath
from typing import TYPE_CHECKING

from freezebase.raster import _to_vsi_uri

if TYPE_CHECKING:
    from upath import UPath


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
