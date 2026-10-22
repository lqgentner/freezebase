"""GDAL Virtual Dataset creation."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rasterio.enums import ColorInterp, Resampling
from upath import UPath

from freezebase.vrt._builders import derive, mosaic, stack
from freezebase.vrt._grid import raster_info
from freezebase.vrt._model import VRT, Band, PixelFunction, Source
from freezebase.vrt._warp import warp

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from affine import Affine
    from rasterio.coords import BoundingBox
    from rasterio.crs import CRS

type AnyPath = str | Path | UPath

_RGB_BANDS = 3


def create_decibel_vrt(
    src_path: AnyPath,
    output_vrt: AnyPath,
    *,
    from_intensity: bool = True,
) -> None:
    """Create a VRT applying a decibel conversion.

    Parameters
    ----------
    src_path : str | Path | UPath
        Linear-scale source raster.
    output_vrt : str | Path | UPath
        Output VRT.
    from_intensity : bool, optional
        Use ``10*log10`` for intensity; otherwise use ``20*log10``.
    """
    derived = derive(src_path, PixelFunction.db(10 if from_intensity else 20), description="dB")
    derived.write(output_vrt)


def _geotransform(transform: Affine) -> str:
    """Format an affine transform as GDAL's geotransform string."""
    return ", ".join(str(v) for v in transform.to_gdal())


def create_rgb_vrt(
    vv_path: AnyPath,
    vh_path: AnyPath,
    output_vrt: AnyPath,
    *,
    decibel_scale: bool = False,
) -> None:
    """Create a 3-band RGB VRT from VV and VH rasters.

    Blue is VV/VH for linear inputs and VV-VH for decibel inputs.

    Parameters
    ----------
    vv_path : str | Path | UPath
        VV raster.
    vh_path : str | Path | UPath
        VH raster.
    output_vrt : str | Path | UPath
        Output VRT.
    decibel_scale : bool, optional
        Whether inputs use decibel scale.

    Raises
    ------
    ValueError
        If the inputs have different grids.
    """
    vv = raster_info(vv_path)
    vh = raster_info(vh_path)

    if (vv.width, vv.height) != (vh.width, vh.height):
        msg = (
            f"VV and VH rasters must have the same size, got "
            f"{vv.width}x{vv.height} and {vh.width}x{vh.height}."
        )
        raise ValueError(msg)
    if vv.crs != vh.crs:
        msg = f"VV and VH rasters must share a CRS, got {vv.crs} and {vh.crs}."
        raise ValueError(msg)
    if vv.transform.to_gdal() != vh.transform.to_gdal():
        msg = (
            f"VV and VH rasters must share a geotransform, got "
            f"'{_geotransform(vv.transform)}' and '{_geotransform(vh.transform)}'."
        )
        raise ValueError(msg)

    names = ("VV_dB", "VH_dB", "VV_dB-VH_dB") if decibel_scale else ("VV", "VH", "VV/VH")
    vrt = stack(
        [vv_path, vh_path],
        band_names=names[:2],
        color_interp=(ColorInterp.red, ColorInterp.green),
    )
    blue = derive(
        [vv_path, vh_path],
        PixelFunction.diff() if decibel_scale else PixelFunction.div(),
        description=names[2],
    ).bands[0]
    blue.color_interp = ColorInterp.blue
    vrt.bands.append(blue)
    vrt.write(output_vrt)


def build_vrt_mosaic(
    files: Sequence[AnyPath],
    dst_vrt: AnyPath,
    *,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
) -> None:
    """Mosaic aligned tiles into a VRT.

    Sources on the VRT's store are referenced relative to it, which keeps the pair
    relocatable.

    Parameters
    ----------
    files : Sequence[str | Path | UPath]
        Source tiles, sharing a CRS, pixel size, band count, dtype and NODATA value.
    dst_vrt : str | Path | UPath
        Output VRT.
    bounds : rasterio.coords.BoundingBox or tuple of float or None, optional
        Extent of the mosaic as ``(left, bottom, right, top)``. Defaults to the
        union of the sources. Pass a fixed extent to place every mosaic of a
        series on one canonical grid, so they can be stacked without alignment.

    Raises
    ------
    ValueError
        If inputs are empty or incompatible, or if ``bounds`` does not lie on
        the sources' pixel grid.
    """
    if not files:
        msg = "files must not be empty"
        raise ValueError(msg)
    mosaic(files, bounds=bounds).write(dst_vrt)


def create_warped_vrt(
    src_path: AnyPath,
    output_vrt: AnyPath,
    *,
    crs: CRS | str,
    resolution: float | tuple[float, float] | None,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
    resampling: Resampling = Resampling.average,
    snap: bool = True,
    dtype: str | None = None,
    nodata: float | None = None,
) -> None:
    """Create a warped VRT reprojecting a raster onto a fixed target grid.

    Heights are shifted between vertical datums when both CRSs are compound or 3D.

    With ``snap``, the grid origin grows outward to a multiple of
    ``resolution``, so warps of several sources land on one common grid.

    Parameters
    ----------
    src_path : str | Path | UPath
        Source raster; may be a VRT.
    output_vrt : str | Path | UPath
        Output VRT, written with an absolute source path so it can live
        anywhere.
    crs : rasterio.crs.CRS or str
        Target CRS.
    resolution : float or tuple of float or None
        Target pixel size, or ``(x, y)`` sizes, in units of ``crs``. ``None`` takes the
        grid ``rasterio.warp.calculate_default_transform`` derives.
    bounds : rasterio.coords.BoundingBox or tuple of float or None, optional
        Target extent as ``(left, bottom, right, top)``, in ``crs``. Defaults to
        the reprojected extent of the source.
    resampling : rasterio.enums.Resampling, optional
        Warp kernel; ``average`` suits downsampling to a coarser target.
    snap : bool, optional
        Snap the derived extent onto the ``resolution`` grid. Ignored when
        ``bounds`` is given.
    dtype : str or None, optional
        Data type of the warped band. Defaults to the source's; promote to
        float so an averaging warp yields a mean, not a rounded count.
    nodata : float or None, optional
        NODATA value of the warped band. Defaults to the source's. Pass
        alongside ``dtype`` when the promoted type can't hold the source's
        sentinel, e.g. a 0 count becoming NaN.

    Raises
    ------
    ValueError
        If ``resolution`` is not positive, ``bounds`` is given and does not lie on the
        ``resolution`` grid, no exact transformation is available (e.g. a datum grid is
        missing), or only one of the source CRS and ``crs`` has a vertical datum.
    """
    warp(
        src_path,
        crs=crs,
        resolution=resolution,
        bounds=bounds,
        resampling=resampling,
        snap=snap,
        dtype=dtype,
        nodata=nodata,
    ).write(output_vrt, relative=False)


def create_rgba_vrt(
    src_path: AnyPath,
    output_vrt: AnyPath,
    *,
    luts: Sequence[Sequence[tuple[float, int]]],
    src_bands: Sequence[int] = (1, 1, 1),
) -> None:
    """Create a 4-band uint8 RGBA VRT colour-mapping a raster through lookup tables.

    Each colour band reads one source band through its own piecewise-linear
    ``LUT``. Alpha is 255 except at NODATA, which is skipped and so stays at
    the band's zero initialisation -- transparent, letting a tiler drop empty
    tiles.

    Parameters
    ----------
    src_path : str | Path | UPath
        Source raster; may be a VRT.
    output_vrt : str | Path | UPath
        Output VRT.
    luts : Sequence[Sequence[tuple[float, int]]]
        One table per colour band: ``(source value, output 0-255)`` pairs, in
        ascending source order.
    src_bands : Sequence[int], optional
        Source band each of R, G and B reads. ``(1, 1, 1)`` colour-maps one
        single-band raster; ``(1, 2, 3)`` maps the three bands of an RGB source.

    Raises
    ------
    ValueError
        If ``luts`` or ``src_bands`` does not hold exactly three entries.
    """
    if len(luts) != _RGB_BANDS or len(src_bands) != _RGB_BANDS:
        msg = (
            f"luts and src_bands must hold {_RGB_BANDS} entries, "
            f"got {len(luts)} and {len(src_bands)}."
        )
        raise ValueError(msg)

    info = raster_info(src_path)
    colors = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
    # Alpha is 255 except at NODATA, which is skipped and so stays transparent.
    tables = [*luts, ((0.0, 255), (1.0, 255))]
    bands = [
        src_bands[0] if color is ColorInterp.alpha else src_bands[i]
        for i, color in enumerate(colors)
    ]
    vrt = VRT(
        crs=info.crs,
        transform=info.transform,
        width=info.width,
        height=info.height,
        bands=[
            Band(
                dtype="uint8",
                sources=[
                    Source(
                        path=info.ref,
                        band=band,
                        nodata=info.nodatavals[band - 1],
                        lut=[(float(value), float(out)) for value, out in table],
                    )
                ],
                color_interp=color,
            )
            for color, table, band in zip(colors, tables, bands, strict=True)
        ],
    )
    vrt.write(output_vrt)
