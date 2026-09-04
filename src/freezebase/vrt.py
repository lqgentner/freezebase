"""GDAL Virtual Dataset creation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, NamedTuple
from xml.etree import ElementTree as ET

from rasterio.enums import Resampling
import rasterio.shutil
from rasterio.transform import Affine
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from upath import UPath

from freezebase.raster import _env_for_path, _to_vsi_uri, rasterio_open

if TYPE_CHECKING:
    from collections.abc import Sequence
    from pathlib import Path

    from rasterio.coords import BoundingBox
    from rasterio.crs import CRS

type AnyPath = str | Path | UPath

_GDAL_DTYPE_NAMES = {
    "uint8": "Byte",
    "int8": "Int8",
    "uint16": "UInt16",
    "int16": "Int16",
    "uint32": "UInt32",
    "int32": "Int32",
    "uint64": "UInt64",
    "int64": "Int64",
    "float32": "Float32",
    "float64": "Float64",
}

_PIXEL_SIZE_TOL = 1e-6
_ALIGNMENT_TOL = 1e-3
# Reprojected edges curve; four corners underestimate the envelope.
_DENSIFY_PTS = 51
_RGB_BANDS = 3


class _GridMeta(NamedTuple):
    x_size: int
    y_size: int
    geo_transform: str
    projection: str
    crs: CRS
    transform_a: float
    transform_e: float


def _read_grid_meta(path: AnyPath) -> _GridMeta:
    """Read GDAL-compatible raster grid metadata."""
    with rasterio_open(path) as ds:
        return _GridMeta(
            x_size=ds.width,
            y_size=ds.height,
            geo_transform=", ".join(str(v) for v in ds.transform.to_gdal()),
            projection=ds.crs.to_wkt(),
            crs=ds.crs,
            transform_a=ds.transform.a,
            transform_e=ds.transform.e,
        )


def _write_vrt(root: ET.Element, dst: AnyPath) -> None:
    """Serialize a VRT element tree."""
    ET.indent(root)
    xml = ET.tostring(root, encoding="unicode")
    UPath(dst).write_text(xml)


def _source_ref(src: AnyPath, dst_vrt: AnyPath) -> tuple[str, str]:
    """How a VRT should name one of its sources: the text and ``relativeToVRT``.

    A source beside the VRT is referenced by bare filename, for relocatability;
    anything else is referenced by an absolute GDAL path, so the VRT can read
    tiles it does not sit next to.
    """
    # Resolved so a relative input can't masquerade as adjacent, or leak
    # unresolved into the "absolute" branch.
    src, dst_vrt = UPath(src).resolve(), UPath(dst_vrt).resolve()
    if src.protocol == dst_vrt.protocol and str(src.parent) == str(dst_vrt.parent):
        return src.name, "1"
    return _to_vsi_uri(src), "0"


def _add_simple_source(
    band: ET.Element,
    src: AnyPath,
    dst_vrt: AnyPath,
    *,
    source_band: int = 1,
) -> None:
    """Append a SimpleSource, referenced relative to the VRT where possible."""
    text, relative = _source_ref(src, dst_vrt)
    source = ET.SubElement(band, "SimpleSource")
    filename = ET.SubElement(source, "SourceFilename", relativeToVRT=relative)
    filename.text = text
    ET.SubElement(source, "SourceBand").text = str(source_band)


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
    src_path = UPath(src_path)
    meta = _read_grid_meta(src_path)
    fact = 10 if from_intensity else 20

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(meta.x_size),
        rasterYSize=str(meta.y_size),
    )
    ET.SubElement(root, "SRS").text = meta.projection
    ET.SubElement(root, "GeoTransform").text = meta.geo_transform

    band = ET.SubElement(
        root,
        "VRTRasterBand",
        dataType="Float32",
        band="1",
        subClass="VRTDerivedRasterBand",
    )
    ET.SubElement(band, "ColorInterp").text = "Gray"
    ET.SubElement(band, "Description").text = "dB"
    ET.SubElement(band, "NoDataValue").text = "nan"
    ET.SubElement(band, "PixelFunctionType").text = "dB"
    ET.SubElement(band, "PixelFunctionArguments", fact=str(fact))
    ET.SubElement(band, "SourceTransferType").text = "Float32"
    _add_simple_source(band, src_path, output_vrt)

    _write_vrt(root, output_vrt)


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
    vv_path, vh_path = UPath(vv_path), UPath(vh_path)
    vv = _read_grid_meta(vv_path)
    vh = _read_grid_meta(vh_path)

    if (vv.x_size, vv.y_size) != (vh.x_size, vh.y_size):
        msg = (
            f"VV and VH rasters must have the same size, got "
            f"{vv.x_size}x{vv.y_size} and {vh.x_size}x{vh.y_size}."
        )
        raise ValueError(msg)
    if vv.crs != vh.crs:
        msg = f"VV and VH rasters must share a CRS, got {vv.crs} and {vh.crs}."
        raise ValueError(msg)
    if vv.geo_transform != vh.geo_transform:
        msg = (
            f"VV and VH rasters must share a geotransform, got "
            f"'{vv.geo_transform}' and '{vh.geo_transform}'."
        )
        raise ValueError(msg)

    band_desc = ["VV_dB", "VH_dB", "VV_dB-VH_dB"] if decibel_scale else ["VV", "VH", "VV/VH"]
    operator = "diff" if decibel_scale else "div"

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(vv.x_size),
        rasterYSize=str(vv.y_size),
    )
    ET.SubElement(root, "SRS").text = vv.projection
    ET.SubElement(root, "GeoTransform").text = vv.geo_transform

    red = ET.SubElement(root, "VRTRasterBand", dataType="Float32", band="1")
    ET.SubElement(red, "ColorInterp").text = "Red"
    ET.SubElement(red, "Description").text = band_desc[0]
    ET.SubElement(red, "NoDataValue").text = "nan"
    _add_simple_source(red, vv_path, output_vrt)

    green = ET.SubElement(root, "VRTRasterBand", dataType="Float32", band="2")
    ET.SubElement(green, "ColorInterp").text = "Green"
    ET.SubElement(green, "Description").text = band_desc[1]
    ET.SubElement(green, "NoDataValue").text = "nan"
    _add_simple_source(green, vh_path, output_vrt)

    blue = ET.SubElement(
        root,
        "VRTRasterBand",
        dataType="Float32",
        band="3",
        subClass="VRTDerivedRasterBand",
    )
    ET.SubElement(blue, "ColorInterp").text = "Blue"
    ET.SubElement(blue, "Description").text = band_desc[2]
    ET.SubElement(blue, "NoDataValue").text = "nan"
    ET.SubElement(blue, "PixelFunctionType").text = operator
    ET.SubElement(blue, "SourceTransferType").text = "Float32"
    _add_simple_source(blue, vv_path, output_vrt)
    _add_simple_source(blue, vh_path, output_vrt)

    _write_vrt(root, output_vrt)


class _TileInfo(NamedTuple):
    path: UPath
    name: str
    bounds: BoundingBox
    width: int
    height: int
    px: float
    py: float
    shear_x: float
    shear_y: float
    band_count: int
    dtype: str
    nodata: float | None
    description: str
    crs: CRS
    crs_wkt: str


def _read_tile_info(path: UPath) -> _TileInfo:
    with rasterio_open(path) as ds:
        return _TileInfo(
            path=path,
            name=path.name,
            bounds=ds.bounds,
            width=ds.width,
            height=ds.height,
            px=ds.transform.a,
            py=-ds.transform.e,
            shear_x=ds.transform.b,
            shear_y=ds.transform.d,
            band_count=ds.count,
            dtype=ds.dtypes[0],
            nodata=ds.nodata,
            description=ds.descriptions[0] or "",
            crs=ds.crs,
            crs_wkt=ds.crs.to_wkt(),
        )


def _nodata_equal(a: float | None, b: float | None) -> bool:
    """Compare NODATA values, including None and NaN."""
    if a is None or b is None:
        return a is b
    if math.isnan(a) and math.isnan(b):
        return True
    return a == b


def _validate_mosaic_tiles(tiles: list[_TileInfo]) -> None:
    """Validate tiles for an axis-aligned, single-band mosaic.

    Raises
    ------
    ValueError
        If tile grids, data types, NODATA or band counts are incompatible.
    """
    ref = tiles[0]

    if ref.band_count != 1:
        msg = f"Only single-band tiles supported, got {ref.band_count} bands in '{ref.name}'."
        raise ValueError(msg)
    if ref.dtype not in _GDAL_DTYPE_NAMES:
        msg = f"Unsupported tile dtype '{ref.dtype}' in '{ref.name}'."
        raise ValueError(msg)

    for t in tiles:
        _validate_tile(t, ref)


def _validate_tile(t: _TileInfo, ref: _TileInfo) -> None:
    """Validate one tile against the mosaic reference."""
    if t.band_count != 1:
        msg = f"Only single-band tiles supported, got {t.band_count} bands in '{t.name}'."
        raise ValueError(msg)
    if t.crs != ref.crs:
        msg = f"All tiles must share a CRS; '{t.name}' differs from '{ref.name}'."
        raise ValueError(msg)
    if t.dtype != ref.dtype:
        msg = (
            f"All tiles must share a dtype; '{t.name}' is '{t.dtype}', "
            f"'{ref.name}' is '{ref.dtype}'."
        )
        raise ValueError(msg)
    if not _nodata_equal(t.nodata, ref.nodata):
        msg = (
            f"All tiles must share a NODATA value; '{t.name}' is {t.nodata}, "
            f"'{ref.name}' is {ref.nodata}."
        )
        raise ValueError(msg)
    if abs(t.px - ref.px) > _PIXEL_SIZE_TOL or abs(t.py - ref.py) > _PIXEL_SIZE_TOL:
        msg = "All tiles must share the same pixel size to be mosaicked into a VRT"
        raise ValueError(msg)
    if abs(t.shear_x) > _PIXEL_SIZE_TOL or abs(t.shear_y) > _PIXEL_SIZE_TOL:
        msg = f"Rotated/sheared tiles cannot be mosaicked into an axis-aligned VRT: '{t.name}'."
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


def _assert_grid_aligned(tiles: list[_TileInfo], minx: float, maxy: float) -> None:
    """Reject tile origins outside the mosaic pixel grid."""
    px, py = tiles[0].px, tiles[0].py
    for t in tiles:
        col = (t.bounds.left - minx) / px
        row = (maxy - t.bounds.top) / py
        if abs(col - round(col)) > _ALIGNMENT_TOL or abs(row - round(row)) > _ALIGNMENT_TOL:
            msg = (
                f"Tile '{t.name}' is not aligned to the common pixel grid; "
                "its origin does not fall on an integer pixel offset."
            )
            raise ValueError(msg)


def build_vrt_mosaic(
    files: Sequence[AnyPath],
    dst_vrt: AnyPath,
    *,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
) -> None:
    """Mosaic aligned, single-band tiles into a VRT.

    Sources beside the VRT are referenced by bare filename, which keeps the pair
    relocatable; sources elsewhere are referenced by absolute GDAL path.

    Parameters
    ----------
    files : Sequence[str | Path | UPath]
        Source tiles, sharing a CRS, pixel size, dtype and NODATA value.
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

    dst_vrt = UPath(dst_vrt)
    tiles = [_read_tile_info(UPath(f)) for f in files]
    _validate_mosaic_tiles(tiles)

    px, py = tiles[0].px, tiles[0].py
    if bounds is None:
        minx = min(t.bounds.left for t in tiles)
        maxy = max(t.bounds.top for t in tiles)
        maxx = max(t.bounds.right for t in tiles)
        miny = min(t.bounds.bottom for t in tiles)
    else:
        minx, miny, maxx, maxy = (float(v) for v in bounds)
        _assert_extent_aligned(px, py, left=minx, bottom=miny, right=maxx, top=maxy)
    _assert_grid_aligned(tiles, minx, maxy)
    mosaic_w = round((maxx - minx) / px)
    mosaic_h = round((maxy - miny) / py)

    gdal_dtype = _GDAL_DTYPE_NAMES[tiles[0].dtype]
    nodata = tiles[0].nodata

    root = ET.Element("VRTDataset", rasterXSize=str(mosaic_w), rasterYSize=str(mosaic_h))
    ET.SubElement(root, "SRS").text = tiles[0].crs_wkt
    ET.SubElement(root, "GeoTransform").text = f"{minx}, {px}, 0, {maxy}, 0, {-py}"

    band = ET.SubElement(root, "VRTRasterBand", dataType=gdal_dtype, band="1")
    if nodata is not None:
        ET.SubElement(band, "NoDataValue").text = str(nodata)
    ET.SubElement(band, "ColorInterp").text = "Gray"
    ET.SubElement(band, "Description").text = tiles[0].description

    source_tag = "ComplexSource" if nodata is not None else "SimpleSource"
    for t in tiles:
        xoff = round((t.bounds.left - minx) / px)
        yoff = round((maxy - t.bounds.top) / py)
        source = ET.SubElement(band, source_tag)
        text, relative = _source_ref(t.path, dst_vrt)
        filename = ET.SubElement(source, "SourceFilename", relativeToVRT=relative)
        filename.text = text
        ET.SubElement(source, "SourceBand").text = "1"
        ET.SubElement(
            source,
            "SrcRect",
            xOff="0",
            yOff="0",
            xSize=str(t.width),
            ySize=str(t.height),
        )
        ET.SubElement(
            source,
            "DstRect",
            xOff=str(xoff),
            yOff=str(yoff),
            xSize=str(t.width),
            ySize=str(t.height),
        )
        if nodata is not None:
            ET.SubElement(source, "NODATA").text = str(nodata)

    _write_vrt(root, dst_vrt)


def create_warped_vrt(
    src_path: AnyPath,
    output_vrt: AnyPath,
    *,
    crs: CRS | str,
    resolution: float,
    bounds: BoundingBox | tuple[float, float, float, float] | None = None,
    resampling: Resampling = Resampling.average,
    snap: bool = True,
    dtype: str | None = None,
    nodata: float | None = None,
) -> None:
    """Create a warped VRT reprojecting a raster onto a fixed target grid.

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
    resolution : float
        Target pixel size, in units of ``crs``.
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
        If ``resolution`` is not positive, or if ``bounds`` is given and does
        not lie on the ``resolution`` grid.
    """
    if resolution <= 0:
        msg = f"resolution must be positive, got {resolution}."
        raise ValueError(msg)

    src_path, output_vrt = UPath(src_path), UPath(output_vrt)
    with rasterio_open(src_path) as src:
        if bounds is None:
            extent = transform_bounds(src.crs, crs, *src.bounds, densify_pts=_DENSIFY_PTS)
            left, bottom, right, top = _snap_out(extent, resolution) if snap else extent
        else:
            left, bottom, right, top = (float(v) for v in bounds)
            _assert_extent_aligned(
                resolution, resolution, left=left, bottom=bottom, right=right, top=top
            )
        width = round((right - left) / resolution)
        height = round((top - bottom) / resolution)
        transform = Affine(resolution, 0.0, left, 0.0, -resolution, top)
        extra: dict[str, Any] = {}
        if dtype is not None:
            extra["dtype"] = dtype
        if nodata is not None:
            # The source keeps its own sentinel; only the warped band changes.
            extra["src_nodata"] = src.nodata
            extra["nodata"] = nodata
        with (
            WarpedVRT(
                src,
                crs=crs,
                transform=transform,
                width=width,
                height=height,
                resampling=resampling,
                **extra,
            ) as vrt,
            # The source is already open; only the write needs its own env.
            _env_for_path(output_vrt),
        ):
            # Written with the source as an absolute path, so it reopens anywhere.
            rasterio.shutil.copy(vrt, _to_vsi_uri(output_vrt), driver="VRT")


def _snap_out(
    bounds: tuple[float, float, float, float],
    res: float,
) -> tuple[float, float, float, float]:
    """Grow ``bounds`` outward onto the grid of ``res``."""
    left, bottom, right, top = bounds
    return (
        math.floor(left / res) * res,
        math.floor(bottom / res) * res,
        math.ceil(right / res) * res,
        math.ceil(top / res) * res,
    )


def _add_lut(source: ET.Element, lut: Sequence[tuple[float, int]]) -> None:
    """Append a piecewise-linear source value to output value table."""
    ET.SubElement(source, "LUT").text = ",".join(f"{value}:{out}" for value, out in lut)


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

    src_path, output_vrt = UPath(src_path), UPath(output_vrt)
    meta = _read_grid_meta(src_path)
    with rasterio_open(src_path) as ds:
        nodatavals = ds.nodatavals

    root = ET.Element(
        "VRTDataset",
        rasterXSize=str(meta.x_size),
        rasterYSize=str(meta.y_size),
    )
    ET.SubElement(root, "SRS").text = meta.projection
    ET.SubElement(root, "GeoTransform").text = meta.geo_transform

    colors = ("Red", "Green", "Blue", "Alpha")
    for index, color in enumerate(colors):
        band = ET.SubElement(root, "VRTRasterBand", dataType="Byte", band=str(index + 1))
        ET.SubElement(band, "ColorInterp").text = color
        # Bands init to 0, so a skipped NODATA pixel reads as transparent black.
        source = ET.SubElement(band, "ComplexSource")
        text, relative = _source_ref(src_path, output_vrt)
        filename = ET.SubElement(source, "SourceFilename", relativeToVRT=relative)
        filename.text = text
        is_alpha = index == len(colors) - 1
        src_band = src_bands[0] if is_alpha else src_bands[index]
        ET.SubElement(source, "SourceBand").text = str(src_band)
        _add_lut(source, ((0.0, 255), (1.0, 255)) if is_alpha else luts[index])
        nodata = nodatavals[src_band - 1]
        if nodata is not None:
            ET.SubElement(source, "NODATA").text = str(nodata)

    _write_vrt(root, output_vrt)
