"""Warping a raster or VRT onto another grid."""

from __future__ import annotations

from contextlib import contextmanager
import re
from typing import TYPE_CHECKING, Any, Literal
from xml.etree import ElementTree as ET

from affine import Affine
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
    _label,
    _snap_out,
    raster_info,
)
from freezebase.vrt._model import VRT
from freezebase.vrt._xml import from_xml

if TYPE_CHECKING:
    from collections.abc import Generator, Mapping

    from rasterio.coords import BoundingBox
    from rasterio.io import DatasetReader

    from freezebase.vrt._model import AnyPath, SourceRef

# GDAL's approximate transformer error, in source pixels.
_APPROX_TOLERANCE = 0.125
_NO_OPERATION = "Cannot find coordinate operations"

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
        NODATA value of the warped bands. Defaults to the source's. Pass it with ``dtype``
        when the source's sentinel should not stay a valid-looking value, e.g. ``nan``.
    tolerance : float | {"auto"}, optional
        Error allowed when approximating the transformation, in source pixels; ``0`` is
        exact. ``"auto"`` is 0.125.
    allow_ballpark : bool, optional
        Let PROJ fall back to a ballpark transformation, ignoring datum differences, when
        it lacks a grid or a datum link.

    Returns
    -------
    VRT
        A warped VRT; nothing is written.

    Raises
    ------
    ValueError
        If neither ``crs`` nor ``like`` is given, ``like`` comes with ``crs``,
        ``resolution`` or ``bounds``, ``bounds`` comes without ``resolution``,
        ``resolution`` is not positive, ``bounds`` does not lie on the ``resolution`` grid,
        ``tolerance`` is negative, the source has no CRS and no ``src_crs`` is given, or no
        exact transformation is available and ``allow_ballpark`` is false.
    """
    tolerance = _resolve_tolerance(tolerance)
    ref = source if isinstance(source, VRT) else _absolute(UPath(source))
    with _opened(ref) as src:
        source_crs = CRS.from_user_input(src_crs) if src_crs is not None else src.crs
        if source_crs is None:
            msg = f"{_label(raster_info(ref), 0)} has no CRS; pass src_crs."
            raise ValueError(msg)
        dst_crs, transform, width, height = _target_grid(
            src, source_crs, crs=crs, resolution=resolution, bounds=bounds, like=like, snap=snap
        )
        extras: dict[str, Any] = {}
        if dtype is not None:
            extras["dtype"] = dtype
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
    vrt = from_xml(xml, storage_options=_storage_options(ref))
    if vrt.warp is None:  # pragma: no cover  # GDAL always writes a warped VRT here
        msg = "GDAL did not serialize a warped VRT"
        raise RuntimeError(msg)
    vrt.warp.options_xml = _set_transformer(
        vrt.warp.options_xml, exact=tolerance == 0, allow_ballpark=allow_ballpark
    )
    _check_transformation(vrt, source_crs, dst_crs)
    return vrt


def _resolve_tolerance(tolerance: float | Literal["auto"]) -> float:
    """Return the approximation error to warp with, in source pixels."""
    if tolerance == "auto":
        return _APPROX_TOLERANCE
    if tolerance < 0:
        msg = f"tolerance must not be negative, got {tolerance}."
        raise ValueError(msg)
    return float(tolerance)


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


def _storage_options(ref: SourceRef) -> Mapping[str, Any] | None:
    """Return the S3 storage options of ``ref``'s files, which GDAL's XML does not carry."""
    files = [ref] if isinstance(ref, UPath) else ref.source_files(follow=False)
    return next((path.storage_options for path in files if path.protocol == "s3"), None)


def _set_transformer(options_xml: str, *, exact: bool, allow_ballpark: bool) -> str:
    """Make the serialized transformer exact or ballpark-free, as GDAL would write it."""
    # rasterio builds an explicit grid's transformer without options, and never exact.
    # The options come from GDAL's own serialization.
    options = ET.fromstring(options_xml)  # noqa: S314
    transformer = options.find("Transformer")
    approx = transformer.find("ApproxTransformer") if transformer is not None else None
    base = approx.find("BaseTransformer/*") if approx is not None else None
    if exact and transformer is not None and approx is not None and base is not None:
        transformer.remove(approx)
        transformer.append(base)
    if not allow_ballpark:
        for reprojection in options.iter("ReprojectionTransformer"):
            items = reprojection.find("Options")
            if items is None:
                items = ET.SubElement(reprojection, "Options")
            ET.SubElement(items, "Option", key="ALLOW_BALLPARK").text = "NO"
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
