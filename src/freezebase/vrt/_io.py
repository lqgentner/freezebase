"""Reading and writing a VRT on local disk or S3."""

from __future__ import annotations

from contextlib import contextmanager
import secrets
from typing import TYPE_CHECKING, Any

import rasterio
import rasterio.env
from upath import UPath

from freezebase import raster
from freezebase.vrt._xml import from_xml

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping
    from contextlib import AbstractContextManager

    from rasterio.io import DatasetReader

    from freezebase.raster import AnyPath
    from freezebase.vrt._model import VRT, SourceRef


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

    Relative source paths resolve against ``path`` and keep its storage options.
    ``/vsis3/`` sources become ``s3://`` paths carrying ``storage_options``, or else the
    storage options of an ``s3://`` ``path``.

    Parameters
    ----------
    path : str | Path | UPath
        The VRT, local or ``s3://``.
    storage_options : Mapping[str, Any] | None, optional
        Options for every S3 source. For an ``s3://`` ``path`` they replace its storage
        options, for reading the VRT and its relative sources.

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
        # Rebuilt rather than UPath(src, **options), which would merge the options.
        src = UPath(f"s3://{src.path}", **storage_options)
    elif src.protocol == "":
        src = src.absolute()
    return from_xml(src.read_text(encoding="utf-8"), base=src, storage_options=storage_options)


def sources(vrt: VRT) -> list[SourceRef]:
    """Return the direct sources of ``vrt``; see ``VRT.sources``."""
    found: list[SourceRef] = [source.path for band in vrt.bands for source in band.sources]
    if vrt.warp is not None:
        found.append(vrt.warp.source)
    seen: set[str | int] = set()
    unique: list[SourceRef] = []
    for ref in found:
        key = str(ref) if isinstance(ref, UPath) else id(ref)
        if key not in seen:
            seen.add(key)
            unique.append(ref)
    return unique


def source_files(
    vrt: VRT, *, follow: bool, storage_options: Mapping[str, Any] | None
) -> list[UPath]:
    """Return the leaf files under ``vrt``; see ``VRT.source_files``."""
    files: dict[str, UPath] = {}

    def visit(node: VRT) -> None:
        for ref in sources(node):
            if isinstance(ref, UPath):
                if follow and ref.suffix.lower() == ".vrt":
                    visit(read(ref, storage_options=storage_options))
                else:
                    files.setdefault(str(ref), ref)
            else:
                visit(ref)

    visit(vrt)
    return list(files.values())


@contextmanager
def _plain_env() -> Generator[None]:
    """Enter a plain rasterio environment."""
    with rasterio.env.Env():
        yield


def env(vrt: VRT) -> AbstractContextManager[None]:
    """Return the rasterio environment for ``vrt``; see ``VRT.env``."""
    files = source_files(vrt, follow=True, storage_options=None)
    by_protocol: dict[str, list[UPath]] = {}
    for path in files:
        by_protocol.setdefault(path.protocol, []).append(path)
    for protocol in by_protocol:
        if protocol not in {"", "s3", "http", "https"}:
            msg = f"Unsupported protocol '{protocol}'."
            raise ValueError(msg)
    s3 = by_protocol.get("s3", [])
    web = by_protocol.get("http", []) + by_protocol.get("https", [])
    if s3 and web:
        msg = f"Sources need different credentials: {s3[0]} and {web[0]}"
        raise ValueError(msg)
    for other in s3[1:]:
        if other.storage_options != s3[0].storage_options:
            msg = f"Sources need different credentials: {s3[0]} and {other}"
            raise ValueError(msg)
    if s3:
        # Lazy import: keeps botocore/s3fs (the `[s3]` extra) out of the base import.
        from freezebase.s3 import s3_env  # noqa: PLC0415

        return s3_env(s3[0])
    if web:
        return raster.http_env()
    return _plain_env()


@contextmanager
def open_vrt(vrt: VRT, **kwargs: Any) -> Generator[DatasetReader]:
    """Open ``vrt`` in memory inside its environment; see ``VRT.open``."""
    with env(vrt), rasterio.open(vrt.to_xml(), **kwargs) as dataset:
        yield dataset
