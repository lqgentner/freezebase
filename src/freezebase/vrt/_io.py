"""Reading and writing a VRT on local disk or S3."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.util import find_spec
import secrets
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

import rasterio
import rasterio.env
from upath import UPath

from freezebase import raster
from freezebase.vrt._paths import parse_gdal_path
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
    if dst.protocol not in {"", "s3"}:
        msg = f"VRTs are written local or s3://, not {dst.protocol}://: {dst}"
        raise ValueError(msg)
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
    ImportError
        If ``path`` is ``http(s)://`` and aiohttp is not installed.
    """
    src = _located(UPath(path), storage_options)
    return from_xml(_read_text(src), base=src, storage_options=storage_options)


def _located(path: UPath, storage_options: Mapping[str, Any] | None) -> UPath:
    """Return ``path`` absolute if local, or carrying ``storage_options`` if on S3."""
    if path.protocol == "s3" and storage_options is not None:
        # Rebuilt rather than UPath(src, **options), which would merge the options.
        return UPath(f"s3://{path.path}", **storage_options)
    if path.protocol == "":
        return path.absolute()
    return path


def _read_text(path: UPath) -> str:
    """Read a VRT file, raising clearly when HTTP(S) support is missing."""
    # fsspec reads HTTP(S) through aiohttp, which only the s3 extra installs.
    if path.protocol in {"http", "https"} and find_spec("aiohttp") is None:
        msg = f"Reading {path} needs aiohttp; install freezebase[s3] or aiohttp."
        raise ImportError(msg)
    return path.read_text(encoding="utf-8")


class _Unique:
    """Paths in first-seen order; the same path with other storage options is kept."""

    def __init__(self) -> None:
        self.paths: list[UPath] = []
        self._options: dict[str, list[Mapping[str, Any]]] = {}

    def add(self, path: UPath) -> bool:
        """Add ``path`` unless already present; return whether it was added."""
        seen = self._options.setdefault(str(path), [])
        if path.storage_options in seen:
            return False
        seen.append(path.storage_options)
        self.paths.append(path)
        return True


def sources(vrt: VRT) -> list[SourceRef]:
    """Return the direct sources of ``vrt``; see ``VRT.sources``."""
    found: list[SourceRef] = [source.path for band in vrt.bands for source in band.sources]
    if vrt.warp is not None:
        found.append(vrt.warp.source)
    paths = _Unique()
    nested: set[int] = set()
    unique: list[SourceRef] = []
    for ref in found:
        if isinstance(ref, UPath):
            if paths.add(ref):
                unique.append(ref)
        elif id(ref) not in nested:
            nested.add(id(ref))
            unique.append(ref)
    return unique


def source_files(
    vrt: VRT, *, follow: bool, storage_options: Mapping[str, Any] | None
) -> list[UPath]:
    """Return the leaf files under ``vrt``; see ``VRT.source_files``."""
    discovery = _Discovery(follow=follow, storage_options=storage_options)
    discovery.visit(vrt)
    return discovery.files.paths


class _Discovery:
    """Leaf files of a VRT, following VRT files without the strict parser."""

    def __init__(self, *, follow: bool, storage_options: Mapping[str, Any] | None) -> None:
        self.follow = follow
        self.storage_options = storage_options
        self.files = _Unique()
        self._followed: set[str] = set()

    def visit(self, node: VRT) -> None:
        """Add the sources of an in-memory VRT."""
        for ref in sources(node):
            if isinstance(ref, UPath):
                self.add(ref, self.storage_options)
            else:
                self.visit(ref)

    def add(self, path: UPath, inherited: Mapping[str, Any] | None) -> None:
        """Add a file, or follow a VRT file once."""
        if not (self.follow and path.suffix.lower() == ".vrt"):
            self.files.add(path)
            return
        # The options given win; then the file's own; then the referencing VRT's.
        own = path.storage_options if path.protocol == "s3" else None
        options = self.storage_options if self.storage_options is not None else own or inherited
        located = _located(path, options)
        if str(located) in self._followed:
            return
        self._followed.add(str(located))
        # Not read(): discovery must not refuse elements GDAL reads but the model can't hold.
        self.walk(ET.fromstring(_read_text(located)), located, options)  # noqa: S314  # see from_xml

    def walk(
        self, root: ET.Element, base: UPath | None, options: Mapping[str, Any] | None
    ) -> None:
        """Add every source named in a VRT document."""
        for element in root.iter():
            if element.tag not in {"SourceFilename", "SourceDataset"}:
                continue
            text = element.text or ""
            if text.lstrip().startswith("<VRTDataset"):
                # An in-memory VRT, named by its escaped XML.
                self.walk(ET.fromstring(text), None, options)  # noqa: S314  # see from_xml
                continue
            relative = element.get("relativeToVRT") == "1"
            path = parse_gdal_path(text, relative=relative, base=base, storage_options=options)
            self.add(path, options)


@contextmanager
def _plain_env() -> Generator[None]:
    """Enter a plain rasterio environment."""
    with rasterio.env.Env():
        yield


def env(
    vrt: VRT, *, storage_options: Mapping[str, Any] | None = None
) -> AbstractContextManager[None]:
    """Return the rasterio environment for ``vrt``; see ``VRT.env``."""
    files = source_files(vrt, follow=True, storage_options=storage_options)
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
def open_vrt(
    vrt: VRT, *, storage_options: Mapping[str, Any] | None = None, **kwargs: Any
) -> Generator[DatasetReader]:
    """Open ``vrt`` in memory inside its environment; see ``VRT.open``."""
    with (
        env(vrt, storage_options=storage_options),
        rasterio.open(vrt.to_xml(), **kwargs) as dataset,
    ):
        yield dataset
