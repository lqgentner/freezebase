"""Writing a VRT to local disk or S3."""

from __future__ import annotations

import secrets
from typing import TYPE_CHECKING

from upath import UPath

from freezebase import raster

if TYPE_CHECKING:
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
        tmp.write_text(xml)
        tmp.replace(dst)
    finally:
        if tmp.exists():
            tmp.unlink()
    return dst
