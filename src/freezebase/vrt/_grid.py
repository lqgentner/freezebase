"""Raster headers and grid arithmetic shared by the VRT builders."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple

from rasterio.coords import BoundingBox
from upath import UPath

from freezebase.raster import rasterio_open
from freezebase.vrt._model import VRT, _nodata_equal

if TYPE_CHECKING:
    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import ColorInterp

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
    "_nodata_equal",
    "_snap_out",
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
        x2, y2 = t @ (self.width, self.height)
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
        )


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
