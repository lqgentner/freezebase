"""Builders that assemble a VRT from rasters and other VRTs."""

from __future__ import annotations

from typing import TYPE_CHECKING

from freezebase.vrt._grid import RasterInfo, raster_info
from freezebase.vrt._model import VRT, Band, Source, SourceProperties

if TYPE_CHECKING:
    from collections.abc import Sequence

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
