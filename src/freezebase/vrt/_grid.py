"""Raster headers and grid arithmetic shared by the VRT builders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from rasterio.coords import BoundingBox
from upath import UPath

from freezebase.raster import rasterio_open
from freezebase.vrt._model import _GDAL_DTYPE_NAMES, VRT, _nodata_equal

if TYPE_CHECKING:
    from collections.abc import Sequence

    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import ColorInterp
    from rasterio.io import DatasetReader

    from freezebase.vrt._model import AnyPath, SourceRef

_PIXEL_SIZE_TOL = 1e-6
_ALIGNMENT_TOL = 1e-3
# Reprojected edges curve; four corners underestimate the envelope.
_DENSIFY_PTS = 51

__all__ = [
    "_ALIGNMENT_TOL",
    "_DENSIFY_PTS",
    "_PIXEL_SIZE_TOL",
    "RasterInfo",
    "_assert_extent_aligned",
    "_assert_grid_aligned",
    "_check_no_colour_table",
    "_check_north_up",
    "_has_colour_table",
    "_label",
    "_mosaic_grid",
    "_snap_out",
    "_validate_tiles",
    "raster_info",
]


class RasterInfo(NamedTuple):
    """Header of a raster file or VRT."""

    ref: SourceRef
    crs: CRS | None
    transform: Affine
    width: int
    height: int
    count: int
    dtypes: tuple[str, ...]
    nodatavals: tuple[float | None, ...]
    descriptions: tuple[str | None, ...]
    colorinterp: tuple[ColorInterp, ...]
    units: tuple[str | None, ...]
    band_tags: list[dict[str, str]]
    tags: dict[str, str]
    block_shapes: list[tuple[int, int]]
    scales: tuple[float, ...]
    offsets: tuple[float, ...]
    colour_tables: tuple[bool, ...]

    @property
    def px(self) -> float:
        """Pixel width, positive."""
        return abs(self.transform.a)

    @property
    def py(self) -> float:
        """Pixel height, positive."""
        return abs(self.transform.e)

    @property
    def bounds(self) -> BoundingBox:
        """Extent of an axis-aligned grid."""
        t = self.transform
        x1, y1 = t.c, t.f
        x2, y2 = t.c + t.a * self.width, t.f + t.e * self.height
        return BoundingBox(min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))


def raster_info(src: AnyPath | VRT) -> RasterInfo:
    """Read the header of a raster path, or of a ``VRT`` opened in memory."""
    if isinstance(src, VRT):
        ref: SourceRef = src
        opened = src.open()
    else:
        path = UPath(src)
        ref = path.absolute() if path.protocol == "" else path
        opened = rasterio_open(ref)
    with opened as ds:
        return RasterInfo(
            ref=ref,
            crs=ds.crs,
            transform=ds.transform,
            width=ds.width,
            height=ds.height,
            count=ds.count,
            dtypes=tuple(ds.dtypes),
            nodatavals=tuple(ds.nodatavals),
            descriptions=tuple(ds.descriptions),
            colorinterp=tuple(ds.colorinterp),
            units=tuple(ds.units),
            band_tags=[ds.tags(i) for i in ds.indexes],
            tags=ds.tags(),
            block_shapes=list(ds.block_shapes),
            scales=tuple(ds.scales),
            offsets=tuple(ds.offsets),
            colour_tables=tuple(_has_colour_table(ds, i) for i in ds.indexes),
        )


def _has_colour_table(ds: DatasetReader, band: int) -> bool:
    """Whether band ``band`` (1-based) of ``ds`` has a colour table."""
    try:
        ds.colormap(band)
    except ValueError:  # rasterio raises for a band without one
        return False
    return True


def _check_no_colour_table(colour_tables: Sequence[bool], name: str) -> None:
    """Raise for a raster with a colour table, which the model cannot hold."""
    for band, has_table in enumerate(colour_tables, start=1):
        if has_table:
            msg = (
                f"'{name}' band {band} has a colour table, which freezebase.vrt cannot keep; "
                "expand it to RGB(A) first, e.g. with gdal_translate -expand rgba."
            )
            raise ValueError(msg)


def _assert_extent_aligned(
    px: float,
    py: float,
    *,
    left: float,
    bottom: float,
    right: float,
    top: float,
) -> None:
    """Reject an extent whose width or height is not a whole number of pixels."""
    w, h = (right - left) / px, (top - bottom) / py
    if abs(w - round(w)) > _ALIGNMENT_TOL or abs(h - round(h)) > _ALIGNMENT_TOL:
        msg = (
            f"bounds ({left}, {bottom}, {right}, {top}) is not a whole number of {px}x{py} pixels."
        )
        raise ValueError(msg)


def _snap_out(
    bounds: tuple[float, float, float, float],
    res_x: float,
    res_y: float,
) -> tuple[float, float, float, float]:
    """Grow ``bounds`` outward onto the grid of ``res_x`` by ``res_y``."""
    left, bottom, right, top = bounds
    return (
        math.floor(left / res_x) * res_x,
        math.floor(bottom / res_y) * res_y,
        math.ceil(right / res_x) * res_x,
        math.ceil(top / res_y) * res_y,
    )


def _label(info: RasterInfo, index: int) -> str:
    """Name an input in error messages: the file name, or its position for a ``VRT``."""
    return f"VRT input #{index}" if isinstance(info.ref, VRT) else info.ref.name


def _check_north_up(info: RasterInfo, name: str) -> None:
    """Raise for a south-up raster, whose rows a north-up VRT grid would flip."""
    if info.transform.e > 0:
        msg = f"'{name}' is south-up (positive pixel height); warp it north-up first."
        raise ValueError(msg)


def _validate_tiles(infos: Sequence[RasterInfo], *, same_pixel_size: bool = True) -> None:
    """Raise unless the tiles can share one axis-aligned mosaic grid.

    Parameters
    ----------
    infos : Sequence[RasterInfo]
        Headers of the tiles; the first is the reference.
    same_pixel_size : bool, optional
        Require every tile to have the reference's pixel size.

    Raises
    ------
    ValueError
        If CRS, band count, dtypes, scales, offsets, NODATA or (with ``same_pixel_size``)
        pixel size differ, a tile is rotated, south-up or has a colour table, or a dtype
        has no GDAL name.
    """
    ref = infos[0]
    ref_name = _label(ref, 0)
    for dtype in ref.dtypes:
        if dtype not in _GDAL_DTYPE_NAMES:
            msg = f"Unsupported tile dtype '{dtype}' in '{ref_name}'."
            raise ValueError(msg)
    for i, t in enumerate(infos):
        _validate_tile(t, _label(t, i), ref, ref_name, same_pixel_size=same_pixel_size)


def _validate_tile(
    t: RasterInfo, name: str, ref: RasterInfo, ref_name: str, *, same_pixel_size: bool
) -> None:
    """Validate one tile against the mosaic reference."""
    if t.crs != ref.crs:
        msg = f"All tiles must share a CRS; '{name}' differs from '{ref_name}'."
        raise ValueError(msg)
    if t.count != ref.count:
        msg = (
            f"All tiles must share a band count; '{name}' has {t.count} bands, "
            f"'{ref_name}' has {ref.count}."
        )
        raise ValueError(msg)
    if t.dtypes != ref.dtypes:
        got, want = next((a, b) for a, b in zip(t.dtypes, ref.dtypes, strict=True) if a != b)
        msg = f"All tiles must share a dtype; '{name}' is '{got}', '{ref_name}' is '{want}'."
        raise ValueError(msg)
    if (t.scales, t.offsets) != (ref.scales, ref.offsets):
        msg = (
            f"All tiles must share a scale and offset; '{name}' has {t.scales} and "
            f"{t.offsets}, '{ref_name}' has {ref.scales} and {ref.offsets}."
        )
        raise ValueError(msg)
    for got_nd, want_nd in zip(t.nodatavals, ref.nodatavals, strict=True):
        if not _nodata_equal(got_nd, want_nd):
            msg = (
                f"All tiles must share a NODATA value; '{name}' is {got_nd}, "
                f"'{ref_name}' is {want_nd}."
            )
            raise ValueError(msg)
    if same_pixel_size and (
        abs(t.px - ref.px) > _PIXEL_SIZE_TOL or abs(t.py - ref.py) > _PIXEL_SIZE_TOL
    ):
        msg = "All tiles must share the same pixel size to be mosaicked into a VRT"
        raise ValueError(msg)
    if abs(t.transform.b) > _PIXEL_SIZE_TOL or abs(t.transform.d) > _PIXEL_SIZE_TOL:
        msg = f"Rotated/sheared tiles cannot be mosaicked into an axis-aligned VRT: '{name}'."
        raise ValueError(msg)
    _check_north_up(t, name)
    _check_no_colour_table(t.colour_tables, name)


def _assert_grid_aligned(infos: Sequence[RasterInfo], minx: float, maxy: float) -> None:
    """Reject tile origins outside the mosaic pixel grid."""
    px, py = infos[0].px, infos[0].py
    for i, t in enumerate(infos):
        bounds = t.bounds
        col = (bounds.left - minx) / px
        row = (maxy - bounds.top) / py
        if abs(col - round(col)) > _ALIGNMENT_TOL or abs(row - round(row)) > _ALIGNMENT_TOL:
            msg = (
                f"Tile '{_label(t, i)}' is not aligned to the common pixel grid; "
                "its origin does not fall on an integer pixel offset."
            )
            raise ValueError(msg)


def _mosaic_grid(
    infos: Sequence[RasterInfo],
    bounds: tuple[float, float, float, float] | None,
    rx: float,
    ry: float,
) -> tuple[float, float, int, int]:
    """Return the left, top, width and height of an ``rx`` by ``ry`` grid over the tiles."""
    if bounds is not None:
        left, bottom, right, top = bounds
        _assert_extent_aligned(rx, ry, left=left, bottom=bottom, right=right, top=top)
        return left, top, round((right - left) / rx), round((top - bottom) / ry)
    left = min(t.bounds.left for t in infos)
    bottom = min(t.bounds.bottom for t in infos)
    right = max(t.bounds.right for t in infos)
    top = max(t.bounds.top for t in infos)
    anchor = next(
        (
            t
            for t in infos
            if abs(t.px - rx) <= _PIXEL_SIZE_TOL and abs(t.py - ry) <= _PIXEL_SIZE_TOL
        ),
        None,
    )
    if anchor is None:
        width = math.ceil((right - left) / rx - _ALIGNMENT_TOL)
        height = math.ceil((top - bottom) / ry - _ALIGNMENT_TOL)
        return left, top, width, height
    # Grow the union outward onto the anchor tile's grid; the slack keeps exact edges.
    ax, ay = anchor.bounds.left, anchor.bounds.top
    col0 = math.floor((left - ax) / rx + _ALIGNMENT_TOL)
    col1 = math.ceil((right - ax) / rx - _ALIGNMENT_TOL)
    row0 = math.floor((ay - top) / ry + _ALIGNMENT_TOL)
    row1 = math.ceil((ay - bottom) / ry - _ALIGNMENT_TOL)
    return ax + col0 * rx, ay - row0 * ry, col1 - col0, row1 - row0
