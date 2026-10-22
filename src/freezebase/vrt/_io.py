"""Reading and writing a VRT on local disk or S3."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING, Any

from upath import UPath

from freezebase import raster
from freezebase.vrt._xml import from_xml

if TYPE_CHECKING:
    from collections.abc import Mapping

    from freezebase.raster import AnyPath
    from freezebase.vrt._model import VRT


def write(vrt: VRT, path: AnyPath, *, relative: bool = True) -> UPath:
    """Write ``vrt`` to ``path``; see ``VRT.write``."""
    dst = UPath(path)
    xml = vrt.to_xml(relative_to=dst if relative else None)
    if dst.protocol == "s3":
        raster._upload(xml.encode(), dst)  # noqa: SLF001  # patched via the module in tests
        return dst
    tmp = dst.with_name(f".{dst.name}.{secrets.token_hex(8)}.tmp")
    try:
        tmp.write_bytes(xml.encode())
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dst


def read(path: AnyPath, *, storage_options: Mapping[str, Any] | None = None) -> VRT:
    """Read a VRT file into the model.

    Relative source paths resolve against ``path``. ``/vsis3/`` sources become ``s3://``
    paths carrying ``storage_options``, or else the storage options of an ``s3://`` ``path``.

    Parameters
    ----------
    path : str | Path | UPath
        The VRT, local or ``s3://``.
    storage_options : Mapping[str, Any] | None, optional
        Options for every S3 source, and for reading an ``s3://`` ``path``; they override
        the storage options of ``path``.

    Returns
    -------
    VRT
        The parsed VRT.

    Raises
    ------
    ValueError
        If the VRT holds an element the model does not support, or a source path on a GDAL
        virtual file system other than ``/vsis3/`` and ``/vsicurl/``.
    """
    src = UPath(path)
    if src.protocol == "s3" and storage_options is not None:
        src = UPath(src, **storage_options)
    elif src.protocol == "":
        src = src.absolute()
    return from_xml(src.read_text(encoding="utf-8"), base=src, storage_options=storage_options)
