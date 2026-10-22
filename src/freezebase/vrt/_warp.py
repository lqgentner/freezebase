"""Warping a raster or VRT onto another grid."""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import TYPE_CHECKING, Any, Literal
from xml.etree import ElementTree as ET

from affine import Affine
import pyproj
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.errors import RasterioIOError
from rasterio.io import MemoryFile
import rasterio.shutil
from rasterio.vrt import WarpedVRT
from rasterio.warp import calculate_default_transform, transform_bounds
from upath import UPath

from freezebase.raster import _env_for_path, _to_vsi_uri
from freezebase.vrt._grid import (
    _DENSIFY_PTS,
    _assert_extent_aligned,
    _check_no_colour_table,
    _has_colour_table,
    _label,
    _snap_out,
    raster_info,
)
from freezebase.vrt._model import VRT, _check_nodata_fits
from freezebase.vrt._xml import from_xml

if TYPE_CHECKING:
    from collections.abc import Generator

    from rasterio.coords import BoundingBox
    from rasterio.io import DatasetReader

    from freezebase.vrt._model import AnyPath, SourceRef

# GDAL's approximate transformer error, in source pixels.
_APPROX_TOLERANCE = 0.125
_NO_OPERATION = "Cannot find coordinate operations"
_AXES_WITH_HEIGHT = 3
# Metres per band unit, for the units gdalwarp recognises (case-insensitive).
_BAND_UNIT_TO_METRE = {
    "m": 1.0,
    "meter": 1.0,
    "metre": 1.0,
    "ft": 0.3048,
    "foot": 0.3048,
    "us survey foot": 1200 / 3937,
}

type _Grid = tuple[CRS, Affine, int, int]


def warp(
    source: AnyPath | VRT,
    *,
    crs: CRS | str | None = None,
    src_crs: CRS | str | None = None,
    resolution: float | tuple[float, float] | None = None,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
    like: AnyPath | VRT | None = None,
    resampling: Resampling = Resampling.average,
    snap: bool = True,
    dtype: str | None = None,
    nodata: float | None = None,
    tolerance: float | Literal["auto"] = "auto",
    allow_ballpark: bool = False,
) -> VRT:
    """Warp a raster or VRT onto another grid.

    The pixel type is not promoted: an averaging warp of a ``uint16`` band stays ``uint16``
    and rounds the mean (1.5 becomes 2). Pass ``dtype="float32"`` to keep the fraction.

    Parameters
    ----------
    source : str | Path | UPath | VRT
        Raster to warp; a ``VRT`` is embedded in the warp.
    crs : CRS | str | None, optional
        Target CRS; required unless ``like`` is given.
    src_crs : CRS | str | None, optional
        Overrides the source's CRS, e.g. to add the vertical datum its header lacks.
    resolution : float | tuple[float, float] | None, optional
        Target pixel size, or ``(x, y)`` sizes, in units of ``crs``. ``None`` takes the grid
        ``rasterio.warp.calculate_default_transform`` derives.
    bounds : BoundingBox | tuple[float, float, float, float] | None, optional
        Target extent as ``(left, bottom, right, top)`` in ``crs``, a whole number of pixels.
        Defaults to the reprojected extent of the source. Needs ``resolution``.
    like : str | Path | UPath | VRT | None, optional
        Raster whose CRS, transform and size are copied exactly.
    resampling : Resampling, optional
        Warp kernel; ``average`` suits downsampling to a coarser target.
    snap : bool, optional
        Grow the derived extent outward onto the ``resolution`` grid, so warps of several
        sources share one grid. Applies only with ``resolution`` and without ``bounds``.
    dtype : str | None, optional
        Data type of the warped bands. Defaults to the source's.
    nodata : float | None, optional
        NODATA value of the warped bands. Defaults to the source's, which must then fit in
        ``dtype``. Pass it with ``dtype`` when the source's sentinel should not stay a
        valid-looking value, e.g. ``nan``.
    tolerance : float | {"auto"}, optional
        Error allowed when approximating the transformation, in source pixels; ``0`` is
        exact. ``"auto"`` is 0.125, or 0 when heights are shifted.
    allow_ballpark : bool, optional
        Let PROJ fall back to a ballpark transformation, ignoring datum differences, when
        it lacks a grid or a datum link.

    Returns
    -------
    VRT
        A warped VRT; nothing is written. Heights are shifted between vertical datums, and
        converted between vertical units, when the source CRS and ``crs`` differ and both
        are compound or 3D; the source must then have one band.

    Raises
    ------
    ValueError
        If neither ``crs`` nor ``like`` is given, ``like`` comes with ``crs``,
        ``resolution`` or ``bounds``, ``bounds`` comes without ``resolution``,
        ``resolution`` is not positive, ``bounds`` does not lie on the ``resolution`` grid,
        ``tolerance`` is negative or NaN, the source has no CRS and no ``src_crs`` is given,
        only one of the source CRS and the target CRS has a vertical datum, heights would
        be shifted in a source of more than one band, no exact transformation is available
        and ``allow_ballpark`` is false, the source has a colour table, or the nodata (the
        given one, else the source's) does not fit in the warped dtype.
    RuntimeError
        If GDAL's serialized transformer cannot be made exact or ballpark-free.
    """
    _check_tolerance(tolerance)
    ref = source if isinstance(source, VRT) else _absolute(UPath(source))
    with _opened(ref) as src:
        _check_no_colour_table(
            [_has_colour_table(src, i) for i in src.indexes],
            ref.name if isinstance(ref, UPath) else "the source VRT",
        )
        source_crs = CRS.from_user_input(src_crs) if src_crs is not None else src.crs
        if source_crs is None:
            msg = f"{_label(raster_info(ref), 0)} has no CRS; pass src_crs."
            raise ValueError(msg)
        if crs is not None and like is None:
            # Check before PROJ derives the target grid.
            _check_vertical_bands(
                src, vertical=_vertical_shift(source_crs, CRS.from_user_input(crs))
            )
        dst_crs, transform, width, height = _target_grid(
            src, source_crs, crs=crs, resolution=resolution, bounds=bounds, like=like, snap=snap
        )
        vertical = _vertical_shift(source_crs, dst_crs)
        _check_vertical_bands(src, vertical=vertical)
        tolerance = _resolve_tolerance(tolerance, vertical=vertical)
        extras: dict[str, Any] = (
            _vertical_shift_options(src.units[0], source_crs, dst_crs) if vertical else {}
        )
        # Always explicit: rasterio promotes an integer source to float64 for a NaN nodata.
        extras["dtype"] = dtype or src.dtypes[0]
        for band_nodata in {nodata} if nodata is not None else set(src.nodatavals):
            _check_nodata_fits(extras["dtype"], band_nodata)
        if nodata is not None:
            # The source keeps its own sentinel; only the warped bands change.
            extras["src_nodata"] = src.nodata
            extras["nodata"] = nodata
        with (
            WarpedVRT(
                src,
                src_crs=source_crs,
                crs=dst_crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
                # rasterio fails on an exact transformer; it is unwrapped below instead.
                tolerance=tolerance or _APPROX_TOLERANCE,
                **extras,
            ) as warped,
            MemoryFile(ext=".vrt") as memfile,
        ):
            rasterio.shutil.copy(warped, memfile.name, driver="VRT")
            xml = bytes(memfile.getbuffer()).decode()
    vrt = from_xml(xml)
    if vrt.warp is None:  # pragma: no cover  # GDAL always writes a warped VRT here
        msg = "GDAL did not serialize a warped VRT"
        raise RuntimeError(msg)
    # GDAL's path carries no storage options, and the source VRT is the caller's own.
    vrt.warp.source = ref
    vrt.warp.options_xml = _set_transformer(
        vrt.warp.options_xml,
        exact=tolerance == 0,
        allow_ballpark=allow_ballpark,
        reprojects=source_crs != dst_crs,
    )
    _check_transformation(vrt, source_crs, dst_crs)
    return vrt


def _check_tolerance(tolerance: float | Literal["auto"]) -> None:
    """Raise unless ``tolerance`` is ``"auto"`` or a non-negative number."""
    if tolerance != "auto" and not tolerance >= 0:
        msg = f'tolerance must be "auto" or a non-negative number, got {tolerance}.'
        raise ValueError(msg)


def _resolve_tolerance(tolerance: float | Literal["auto"], *, vertical: bool) -> float:
    """Return the approximation error to warp with, in source pixels."""
    if tolerance == "auto":
        # The approximation's error is capped in pixels, not metres of height.
        return 0.0 if vertical else _APPROX_TOLERANCE
    return float(tolerance)


def _to_pyproj(crs: CRS) -> pyproj.CRS:
    """Return ``crs`` as a pyproj CRS, keeping a third axis that WKT1 would drop."""
    return pyproj.CRS.from_wkt(crs.to_wkt(version="WKT2_2019"))


def _has_vertical(crs: CRS) -> bool:
    """Whether ``crs`` carries heights: a vertical or compound CRS, or a 3D one."""
    proj_crs = _to_pyproj(crs)
    # pyproj's is_vertical also looks into a compound CRS's parts.
    return proj_crs.is_vertical or (
        (proj_crs.is_geographic or proj_crs.is_projected)
        and len(proj_crs.axis_info) == _AXES_WITH_HEIGHT
    )


def _vertical_shift(source_crs: CRS, dst_crs: CRS) -> bool:
    """Whether to shift heights; raise if only one CRS says which datum they refer to."""
    if source_crs == dst_crs:
        return False
    source_vertical, dst_vertical = _has_vertical(source_crs), _has_vertical(dst_crs)
    if source_vertical and not dst_vertical:
        msg = (
            f"The source CRS {_crs_name(source_crs)} has a vertical datum but the target CRS "
            f"{_crs_name(dst_crs)} has none; pass a 3D or compound crs, or a 2D src_crs to keep "
            "heights unconverted."
        )
        raise ValueError(msg)
    if dst_vertical and not source_vertical:
        msg = (
            f"The target CRS {_crs_name(dst_crs)} has a vertical datum but the source CRS "
            f"{_crs_name(source_crs)} has none; pass a 3D or compound src_crs, or a 2D crs."
        )
        raise ValueError(msg)
    return source_vertical


def _check_vertical_bands(src: DatasetReader, *, vertical: bool) -> None:
    """Raise if heights would shift in a multi-band raster, as gdalwarp only shifts one band."""
    if vertical and src.count > 1:
        msg = (
            f"Vertical shifts apply to single-band rasters, and the source has {src.count} "
            "bands; warp the height band alone, or pass a 2D src_crs to keep values unshifted."
        )
        raise ValueError(msg)


def _vertical_shift_options(
    band_unit: str | None, source_crs: CRS, dst_crs: CRS
) -> dict[str, str]:
    """Return the warp options of a vertical shift, with gdalwarp's unit factors."""
    # Metres per unit of each CRS's vertical axis, which is the last one.
    source_axis = _to_pyproj(source_crs).axis_info[-1].unit_conversion_factor
    dst_axis = _to_pyproj(dst_crs).axis_info[-1].unit_conversion_factor
    # gdalwarp lets a recognised band unit override the source axis unit for pixel values.
    source_values = _BAND_UNIT_TO_METRE.get((band_unit or "").lower(), source_axis)
    return {
        "APPLY_VERTICAL_SHIFT": "YES",
        "MULT_FACTOR_VERTICAL_SHIFT": f"{source_values / dst_axis:.18g}",
        "MULT_FACTOR_VERTICAL_SHIFT_PIPELINE": f"{source_axis / dst_axis:.18g}",
    }


def _absolute(path: UPath) -> UPath:
    """Return a local path made absolute, so GDAL records where it is."""
    return path.absolute() if path.protocol == "" else path


@contextmanager
def _opened(ref: SourceRef) -> Generator[DatasetReader]:
    """Open a path by its GDAL path in its environment, or a ``VRT`` inside ``VRT.env()``."""
    if isinstance(ref, VRT):
        with ref.open() as src:
            yield src
        return
    with _env_for_path(ref), rasterio.open(_to_vsi_uri(ref)) as src:
        yield src


def _target_grid(
    src: DatasetReader,
    source_crs: CRS,
    *,
    crs: CRS | str | None,
    resolution: float | tuple[float, float] | None,
    bounds: BoundingBox | tuple[float, float, float, float] | None,
    like: AnyPath | VRT | None,
    snap: bool,
) -> _Grid:
    """Return the CRS, transform, width and height of the warp's target grid."""
    if like is not None:
        if crs is not None or resolution is not None or bounds is not None:
            msg = "like sets the whole grid; do not pass crs, resolution or bounds with it."
            raise ValueError(msg)
        ref = raster_info(like)
        if ref.crs is None:
            msg = f"like ({_label(ref, 0)}) has no CRS."
            raise ValueError(msg)
        return ref.crs, ref.transform, ref.width, ref.height
    if crs is None:
        msg = "Pass crs, or like to copy a reference grid."
        raise ValueError(msg)
    dst_crs = CRS.from_user_input(crs)
    if resolution is None:
        if bounds is not None:
            msg = "bounds needs a resolution."
            raise ValueError(msg)
        transform, width, height = calculate_default_transform(
            source_crs, dst_crs, src.width, src.height, *src.bounds
        )
        return dst_crs, transform, width, height
    rx, ry = resolution if isinstance(resolution, tuple) else (resolution, resolution)
    if rx <= 0 or ry <= 0:
        msg = f"resolution must be positive, got {resolution}."
        raise ValueError(msg)
    if bounds is None:
        extent = transform_bounds(source_crs, dst_crs, *src.bounds, densify_pts=_DENSIFY_PTS)
        left, bottom, right, top = _snap_out(extent, rx, ry) if snap else extent
    else:
        left, bottom, right, top = (float(v) for v in bounds)
        _assert_extent_aligned(rx, ry, left=left, bottom=bottom, right=right, top=top)
    width = round((right - left) / rx)
    height = round((top - bottom) / ry)
    return dst_crs, Affine(rx, 0.0, left, 0.0, -ry, top), width, height


def _set_transformer(
    options_xml: str, *, exact: bool, allow_ballpark: bool, reprojects: bool
) -> str:
    """Make the serialized transformer exact or ballpark-free, as GDAL would write it."""
    # rasterio builds an explicit grid's transformer without options, and never exact.
    # The options come from GDAL's own serialization.
    options = ET.fromstring(options_xml)  # noqa: S314
    transformer = options.find("Transformer")
    if transformer is None:
        msg = "GDAL wrote no <Transformer> in the warp options"
        raise RuntimeError(msg)
    if exact and (approx := transformer.find("ApproxTransformer")) is not None:
        base = approx.find("BaseTransformer/*")
        if base is None:
            msg = "Cannot make the warp exact: <ApproxTransformer> holds no base transformer"
            raise RuntimeError(msg)
        transformer.remove(approx)
        transformer.append(base)
    if not allow_ballpark:
        reprojections = list(options.iter("ReprojectionTransformer"))
        if reprojects and not reprojections:
            msg = "Cannot refuse ballpark transformations: GDAL wrote no <ReprojectionTransformer>"
            raise RuntimeError(msg)
        for reprojection in reprojections:
            items = reprojection.find("Options")
            if items is None:
                items = ET.SubElement(reprojection, "Options")
            option = items.find("Option[@key='ALLOW_BALLPARK']")
            if option is None:
                option = ET.SubElement(items, "Option", key="ALLOW_BALLPARK")
            option.text = "NO"
    return ET.tostring(options, encoding="unicode")


def _crs_name(crs: CRS) -> str:
    """Name a CRS by its authority code, else by the name in its WKT."""
    if (authority := crs.to_authority()) is not None:
        return ":".join(authority)
    match = re.search(r'"([^"]*)"', crs.to_wkt())
    return match.group(1) if match else crs.to_wkt()


def _check_transformation(vrt: VRT, source_crs: CRS, dst_crs: CRS) -> None:
    """Open ``vrt`` as GDAL will, raising if PROJ has no transformation it may use."""
    try:
        with vrt.open():
            pass
    except RasterioIOError as err:
        if _NO_OPERATION not in str(err):
            raise
        msg = (
            f"No exact transformation from {_crs_name(source_crs)} to {_crs_name(dst_crs)} "
            "is available. Fetch the PROJ grids it needs (python -m pyproj sync) or pass "
            "allow_ballpark=True."
        )
        raise ValueError(msg) from err
