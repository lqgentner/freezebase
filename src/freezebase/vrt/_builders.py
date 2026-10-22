"""Builders that assemble a VRT from rasters and other VRTs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window

from freezebase.vrt._grid import (
    RasterInfo,
    _assert_extent_aligned,
    _assert_grid_aligned,
    _validate_tiles,
    raster_info,
)
from freezebase.vrt._model import VRT, Band, Source, SourceProperties

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rasterio.coords import BoundingBox
    from rasterio.enums import ColorInterp

    from freezebase.vrt._model import AnyPath

_ALMOST_EQUAL_PRECISION = 1e-6


def _band(info: RasterInfo, index: int) -> Band:
    """Build the band that reads band ``index`` (0-based) of ``info`` whole."""
    block_height, block_width = info.block_shapes[index]
    props = (
        None
        if isinstance(info.ref, VRT)
        else SourceProperties(
            info.width, info.height, info.dtypes[index], block_width, block_height
        )
    )
    return Band(
        dtype=info.dtypes[index],
        sources=[Source(path=info.ref, band=index + 1, properties=props)],
        nodata=info.nodatavals[index],
        description=info.descriptions[index] or "",
        color_interp=info.colorinterp[index],
        unit=info.units[index] or None,
        metadata=dict(info.band_tags[index]),
    )


def from_raster(path: AnyPath) -> VRT:
    """Build a 1:1 VRT over one raster file.

    Parameters
    ----------
    path : str | Path | UPath
        Raster file.

    Returns
    -------
    VRT
        A VRT with the file's grid, bands, descriptions, colour interpretation, units,
        nodata and metadata.
    """
    info = raster_info(path)
    return VRT(
        crs=info.crs,
        transform=info.transform,
        width=info.width,
        height=info.height,
        bands=[_band(info, i) for i in range(info.count)],
        metadata=dict(info.tags),
    )


def _check_same_grid(infos: Sequence[RasterInfo]) -> None:
    """Raise unless every input shares the first input's CRS, transform and size."""
    first = infos[0]
    for info in infos[1:]:
        if (
            info.crs != first.crs
            or (info.width, info.height) != (first.width, first.height)
            or not info.transform.almost_equals(first.transform, _ALMOST_EQUAL_PRECISION)
        ):
            msg = (
                f"Sources must share one grid (CRS, transform, size); use warp to align {info.ref}"
            )
            raise ValueError(msg)


def _selected_bands(info: RasterInfo, wanted: Sequence[int] | None) -> Sequence[int]:
    """Return the 1-based bands to take from ``info``, validated."""
    if wanted is None:
        return range(1, info.count + 1)
    for band in wanted:
        if not 1 <= band <= info.count:
            msg = f"Source {info.ref} has no band {band} ({info.count} bands)."
            raise ValueError(msg)
    return wanted


def stack(
    sources: Sequence[AnyPath | VRT],
    *,
    bands: Sequence[Sequence[int] | None] | None = None,
    band_names: Sequence[str] | None = None,
    color_interp: Sequence[ColorInterp] | None = None,
) -> VRT:
    """Stack the bands of sources on one grid as successive bands.

    Parameters
    ----------
    sources : Sequence[str | Path | UPath | VRT]
        Rasters or VRTs on an identical grid. A ``VRT`` is nested.
    bands : Sequence[Sequence[int] | None] | None, optional
        1-based bands to take from each source; ``None`` for an entry, or for the argument,
        takes all bands.
    band_names : Sequence[str] | None, optional
        Description of each output band, replacing the source's.
    color_interp : Sequence[ColorInterp] | None, optional
        Colour interpretation of each output band, replacing the source's.

    Returns
    -------
    VRT
        A VRT with one band per selected source band.

    Raises
    ------
    ValueError
        If ``sources`` is empty, the grids differ (use ``warp`` first), a ``bands`` entry
        is out of range, or ``bands``, ``band_names`` or ``color_interp`` do not match.
    """
    if not sources:
        msg = "stack needs at least one source."
        raise ValueError(msg)
    if bands is not None and len(bands) != len(sources):
        msg = f"bands has {len(bands)} entries for {len(sources)} sources."
        raise ValueError(msg)
    infos = [raster_info(source) for source in sources]
    _check_same_grid(infos)
    out = [
        _band(info, band - 1)
        for i, info in enumerate(infos)
        for band in _selected_bands(info, None if bands is None else bands[i])
    ]
    if band_names is not None and len(band_names) != len(out):
        msg = f"band_names has {len(band_names)} entries for {len(out)} bands."
        raise ValueError(msg)
    if color_interp is not None and len(color_interp) != len(out):
        msg = f"color_interp has {len(color_interp)} entries for {len(out)} bands."
        raise ValueError(msg)
    for i, band in enumerate(out):
        if band_names is not None:
            band.description = band_names[i]
        if color_interp is not None:
            band.color_interp = color_interp[i]
    first = infos[0]
    return VRT(
        crs=first.crs,
        transform=first.transform,
        width=first.width,
        height=first.height,
        bands=out,
    )


def _mosaic_source(
    info: RasterInfo, band: int, minx: float, maxy: float, nodata: float | None
) -> Source:
    """Build the source drawing band ``band`` (0-based) of a tile at its place on the grid."""
    block_height, block_width = info.block_shapes[band]
    bounds = info.bounds
    return Source(
        path=info.ref,
        band=band + 1,
        src_window=Window(0, 0, info.width, info.height),
        dst_window=Window(
            round((bounds.left - minx) / info.px),
            round((maxy - bounds.top) / info.py),
            info.width,
            info.height,
        ),
        nodata=nodata,
        properties=SourceProperties(
            info.width, info.height, info.dtypes[band], block_width, block_height
        ),
    )


def mosaic(
    sources: Sequence[AnyPath | VRT],
    *,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
    resolution: Literal["finest"] | tuple[float, float] | None = None,
    resampling: Resampling = Resampling.nearest,  # noqa: ARG001  # used once resolution is set
    nodata: float | None = None,
    band_names: Sequence[str] | None = None,
    color_interp: Sequence[ColorInterp] | None = None,
) -> VRT:
    """Mosaic tiles of any band count on one grid; later sources draw over earlier ones.

    Parameters
    ----------
    sources : Sequence[str | Path | UPath | VRT]
        Tiles sharing a CRS, pixel size, band count, dtypes and NODATA values. A ``VRT``
        is nested. Band *i* of the mosaic reads band *i* of every tile.
    bounds : rasterio.coords.BoundingBox | tuple[float, float, float, float] | None, optional
        Extent as ``(left, bottom, right, top)``, on the tiles' pixel grid. Defaults to the
        union of the tiles.
    resolution : {"finest"} | tuple[float, float] | None, optional
        ``None`` requires one common pixel size. Other values are not implemented yet.
    resampling : Resampling, optional
        Resampling for tiles not on the output grid.
    nodata : float | None, optional
        NODATA of every band and source, replacing the first tile's.
    band_names : Sequence[str] | None, optional
        Description of each band, replacing the first tile's.
    color_interp : Sequence[ColorInterp] | None, optional
        Colour interpretation of each band, replacing the first tile's.

    Returns
    -------
    VRT
        A VRT with the first tile's band descriptions, colour interpretation and NODATA.

    Raises
    ------
    ValueError
        If ``sources`` is empty, the tiles are incompatible or off the pixel grid,
        ``bounds`` is not on the grid, or ``band_names`` or ``color_interp`` do not match
        the band count.
    NotImplementedError
        If ``resolution`` is not ``None``.
    """
    if resolution is not None:
        msg = "mosaic supports resolution=None only."
        raise NotImplementedError(msg)
    if not sources:
        msg = "sources must not be empty."
        raise ValueError(msg)
    infos = [raster_info(source) for source in sources]
    _validate_tiles(infos)
    first = infos[0]
    if band_names is not None and len(band_names) != first.count:
        msg = f"band_names has {len(band_names)} entries for {first.count} bands."
        raise ValueError(msg)
    if color_interp is not None and len(color_interp) != first.count:
        msg = f"color_interp has {len(color_interp)} entries for {first.count} bands."
        raise ValueError(msg)

    px, py = first.px, first.py
    if bounds is None:
        minx = min(t.bounds.left for t in infos)
        maxy = max(t.bounds.top for t in infos)
        maxx = max(t.bounds.right for t in infos)
        miny = min(t.bounds.bottom for t in infos)
    else:
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        _assert_extent_aligned(px, py, left=minx, bottom=miny, right=maxx, top=maxy)
    _assert_grid_aligned(infos, minx, maxy)

    out: list[Band] = []
    for i in range(first.count):
        band_nodata = first.nodatavals[i] if nodata is None else nodata
        band = _band(first, i)
        band.nodata = band_nodata
        band.sources = [_mosaic_source(info, i, minx, maxy, band_nodata) for info in infos]
        if band_names is not None:
            band.description = band_names[i]
        if color_interp is not None:
            band.color_interp = color_interp[i]
        out.append(band)
    return VRT(
        crs=first.crs,
        transform=Affine(px, 0.0, minx, 0.0, -py, maxy),
        width=round((maxx - minx) / px),
        height=round((maxy - miny) / py),
        bands=out,
    )
