"""Builders that assemble a VRT from rasters and other VRTs."""

from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from affine import Affine
from rasterio.enums import Resampling
from rasterio.windows import Window
from upath import UPath

from freezebase.vrt._grid import (
    _ALIGNMENT_TOL,
    _PIXEL_SIZE_TOL,
    RasterInfo,
    _assert_extent_aligned,
    _assert_grid_aligned,
    _label,
    _mosaic_grid,
    _validate_tiles,
    raster_info,
)
from freezebase.vrt._model import VRT, Band, PixelFunction, Source, SourceProperties

if TYPE_CHECKING:
    from collections.abc import Sequence

    from rasterio.coords import BoundingBox
    from rasterio.enums import ColorInterp

    from freezebase.vrt._model import AnyPath

_ALMOST_EQUAL_PRECISION = 1e-6
# The kernels a VRT source can resample with; GDAL rejects the others.
_SOURCE_RESAMPLINGS = frozenset(
    {
        Resampling.nearest,
        Resampling.bilinear,
        Resampling.cubic,
        Resampling.cubic_spline,
        Resampling.lanczos,
        Resampling.average,
        Resampling.mode,
        Resampling.gauss,
        Resampling.rms,
    }
)


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
    for index, info in enumerate(infos[1:], start=1):
        if (
            info.crs != first.crs
            or (info.width, info.height) != (first.width, first.height)
            or not info.transform.almost_equals(first.transform, _ALMOST_EQUAL_PRECISION)
        ):
            msg = (
                "Sources must share one grid (CRS, transform, size); "
                f"use warp to align {_label(info, index)}"
            )
            raise ValueError(msg)


def _selected_bands(
    info: RasterInfo, wanted: Sequence[int] | None, index: int = 0
) -> Sequence[int]:
    """Return the 1-based bands to take from ``info``, validated."""
    if wanted is None:
        return range(1, info.count + 1)
    for band in wanted:
        if not 1 <= band <= info.count:
            msg = f"Source {_label(info, index)} has no band {band} ({info.count} bands)."
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
        for band in _selected_bands(info, None if bands is None else bands[i], i)
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
    info: RasterInfo,
    band: int,
    grid: Affine,
    nodata: float | None,
    resampling: Resampling,
) -> Source | None:
    """Build the source of band ``band`` (0-based) of a tile; ``None`` if it holds no centre."""
    block_height, block_width = info.block_shapes[band]
    tile = info.bounds
    rx, ry = grid.a, -grid.e
    col = (tile.left - grid.c) / rx
    row = (grid.f - tile.top) / ry
    on_grid = (
        abs(info.px - rx) <= _PIXEL_SIZE_TOL
        and abs(info.py - ry) <= _PIXEL_SIZE_TOL
        and abs(col - round(col)) <= _ALIGNMENT_TOL
        and abs(row - round(row)) <= _ALIGNMENT_TOL
    )
    if on_grid:
        src_window = Window(0, 0, info.width, info.height)
        dst_window = Window(round(col), round(row), info.width, info.height)
    else:
        # Only pixels whose centre is in the tile; a centre within the alignment tolerance
        # of an edge is on it, inside on the left and top, outside on the right and bottom.
        col0 = math.ceil(col - 0.5 - _ALIGNMENT_TOL)
        col1 = math.ceil((tile.right - grid.c) / rx - 0.5 - _ALIGNMENT_TOL)
        row0 = math.ceil(row - 0.5 - _ALIGNMENT_TOL)
        row1 = math.ceil((grid.f - tile.bottom) / ry - 0.5 - _ALIGNMENT_TOL)
        if col1 <= col0 or row1 <= row0:
            return None
        dst_window = Window(col0, row0, col1 - col0, row1 - row0)
        # The fractional source window those pixels cover keeps the tile georeferenced.
        src_window = Window(
            (col0 - col) * rx / info.px,
            (row0 - row) * ry / info.py,
            (col1 - col0) * rx / info.px,
            (row1 - row0) * ry / info.py,
        )
    return Source(
        path=info.ref,
        band=band + 1,
        src_window=src_window,
        dst_window=dst_window,
        nodata=nodata,
        properties=SourceProperties(
            info.width, info.height, info.dtypes[band], block_width, block_height
        ),
        resampling=None if on_grid else resampling,
    )


def _output_grid(
    infos: Sequence[RasterInfo],
    bounds: BoundingBox | tuple[float, float, float, float] | None,
    resolution: Literal["finest"] | tuple[float, float] | None,
) -> tuple[Affine, int, int]:
    """Return the transform, width and height of the mosaic grid."""
    extent = (
        None
        if bounds is None
        else (float(bounds[0]), float(bounds[1]), float(bounds[2]), float(bounds[3]))
    )
    if resolution is None:
        px, py = infos[0].px, infos[0].py
        if extent is None:
            minx = min(t.bounds.left for t in infos)
            miny = min(t.bounds.bottom for t in infos)
            maxx = max(t.bounds.right for t in infos)
            maxy = max(t.bounds.top for t in infos)
        else:
            minx, miny, maxx, maxy = extent
            _assert_extent_aligned(px, py, left=minx, bottom=miny, right=maxx, top=maxy)
        _assert_grid_aligned(infos, minx, maxy)
        width, height = round((maxx - minx) / px), round((maxy - miny) / py)
    else:
        if resolution == "finest":
            px, py = min(t.px for t in infos), min(t.py for t in infos)
        else:
            px, py = resolution
        if px <= 0 or py <= 0:
            msg = f"resolution must be positive, got {resolution}."
            raise ValueError(msg)
        minx, maxy, width, height = _mosaic_grid(infos, extent, px, py)
    return Affine(px, 0.0, minx, 0.0, -py, maxy), width, height


def mosaic(
    sources: Sequence[AnyPath | VRT],
    *,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
    resolution: Literal["finest"] | tuple[float, float] | None = None,
    resampling: Resampling = Resampling.nearest,
    nodata: float | None = None,
    band_names: Sequence[str] | None = None,
    color_interp: Sequence[ColorInterp] | None = None,
) -> VRT:
    """Mosaic tiles of any band count on one grid; later sources draw over earlier ones.

    Parameters
    ----------
    sources : Sequence[str | Path | UPath | VRT]
        Tiles sharing a CRS, band count, dtypes and NODATA values. A ``VRT`` is nested.
        Band *i* of the mosaic reads band *i* of every tile.
    bounds : rasterio.coords.BoundingBox | tuple[float, float, float, float] | None, optional
        Extent as ``(left, bottom, right, top)``, a whole number of output pixels; its
        top-left corner anchors the grid. Defaults to the union of the tiles.
    resolution : {"finest"} | tuple[float, float] | None, optional
        Output pixel size. ``None`` requires one common pixel size and every tile on one
        grid; ``"finest"`` takes the smallest pixel size per axis; ``(x, y)`` sets it.
        Without ``bounds``, the grid runs through the origin of the first tile of that
        pixel size, or else the union's top-left corner, and covers every tile.
    resampling : Resampling, optional
        Resampling for tiles not on the output grid, which are placed exactly rather than
        snapped to it and fill only the output pixels whose centre they contain. One of
        nearest, bilinear, cubic, cubic_spline, lanczos, average, rms, mode or gauss.
    nodata : float | None, optional
        NODATA of every band and source, replacing the first tile's.
    band_names : Sequence[str] | None, optional
        Description of each band, replacing the first tile's.
    color_interp : Sequence[ColorInterp] | None, optional
        Colour interpretation of each band, replacing the first tile's.

    Returns
    -------
    VRT
        A VRT with the first tile's band descriptions, colour interpretation and NODATA,
        unless overridden.

    Raises
    ------
    ValueError
        If ``sources`` is empty, the tiles are incompatible, off the pixel grid (with
        ``resolution=None``) or rotated, ``bounds`` is not whole pixels, ``resolution`` is
        not positive, ``resampling`` is not supported, or ``band_names`` or
        ``color_interp`` do not match the band count.
    """
    if not sources:
        msg = "sources must not be empty."
        raise ValueError(msg)
    if resampling not in _SOURCE_RESAMPLINGS:
        msg = f"VRT sources cannot use {resampling.name} resampling."
        raise ValueError(msg)
    infos = [raster_info(source) for source in sources]
    _validate_tiles(infos, same_pixel_size=resolution is None)
    first = infos[0]
    if band_names is not None and len(band_names) != first.count:
        msg = f"band_names has {len(band_names)} entries for {first.count} bands."
        raise ValueError(msg)
    if color_interp is not None and len(color_interp) != first.count:
        msg = f"color_interp has {len(color_interp)} entries for {first.count} bands."
        raise ValueError(msg)

    grid, width, height = _output_grid(infos, bounds, resolution)

    out: list[Band] = []
    for i in range(first.count):
        band_nodata = first.nodatavals[i] if nodata is None else nodata
        band = _band(first, i)
        band.nodata = band_nodata
        band.sources = [
            source
            for info in infos
            if (source := _mosaic_source(info, i, grid, band_nodata, resampling)) is not None
        ]
        if band_names is not None:
            band.description = band_names[i]
        if color_interp is not None:
            band.color_interp = color_interp[i]
        out.append(band)
    return VRT(crs=first.crs, transform=grid, width=width, height=height, bands=out)


def derive(
    sources: AnyPath | VRT | Sequence[AnyPath | VRT],
    function: PixelFunction,
    *,
    dtype: str = "float32",
    nodata: float | None = math.nan,
    description: str = "",
    band: int | None = None,
) -> VRT:
    """Derive bands from rasters or VRTs with a GDAL pixel function.

    Parameters
    ----------
    sources : str | Path | UPath | VRT | Sequence[str | Path | UPath | VRT]
        One input, or several on an identical grid. A ``VRT`` is nested whole, so the
        function sees its mosaicked pixels.
    function : PixelFunction
        Function applied to the sources. ``propagateNoData`` is set to ``"true"`` unless
        its arguments already set it, so a nodata input gives a nodata output.
    dtype : str, optional
        NumPy dtype name of the derived bands.
    nodata : float | None, optional
        NODATA of the derived bands.
    description : str, optional
        Description of every derived band.
    band : int | None, optional
        1-based band read from each input. With several inputs it defaults to 1; with one
        input, it derives only that band instead of every band.

    Returns
    -------
    VRT
        With one input, a derived band per input band; with several, one derived band.
        Each source carries its input's NODATA, and ``transfer_type`` defaults to ``dtype``.

    Raises
    ------
    ValueError
        If ``sources`` is empty, the grids differ (use ``warp`` first), or ``band`` is out
        of range.
    """
    inputs = [sources] if isinstance(sources, (str, Path, UPath, VRT)) else list(sources)
    if not inputs:
        msg = "derive needs at least one source."
        raise ValueError(msg)
    infos = [raster_info(source) for source in inputs]
    _check_same_grid(infos)
    first = infos[0]
    args = {"propagateNoData": "true", **function.args}

    if len(infos) == 1:
        wanted = _selected_bands(first, None if band is None else [band])
        groups = [[(first, b)] for b in wanted]
    else:
        wanted = [band or 1]
        for i, info in enumerate(infos):
            _selected_bands(info, wanted, i)
        groups = [[(info, wanted[0]) for info in infos]]
    out = [
        Band(
            dtype=dtype,
            sources=[_derive_source(info, b) for info, b in group],
            nodata=nodata,
            description=description,
            pixel_function=PixelFunction(function.name, args, function.transfer_type or dtype),
        )
        for group in groups
    ]
    return VRT(
        crs=first.crs,
        transform=first.transform,
        width=first.width,
        height=first.height,
        bands=out,
    )


def _derive_source(info: RasterInfo, band: int) -> Source:
    """Build a whole-raster source of band ``band`` (1-based) carrying its input's nodata."""
    source = _band(info, band - 1).sources[0]
    source.nodata = info.nodatavals[band - 1]
    return source
