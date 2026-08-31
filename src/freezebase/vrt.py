"""GDAL Virtual Dataset creation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, NamedTuple
from xml.etree import ElementTree as ET

from upath import UPath

from freezebase.raster import rasterio_open

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


def _add_simple_source(band: ET.Element, source_name: str, *, source_band: int = 1) -> None:
    """Append a relative SimpleSource."""
    source = ET.SubElement(band, "SimpleSource")
    filename = ET.SubElement(source, "SourceFilename", relativeToVRT="1")
    filename.text = source_name
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
    _add_simple_source(band, src_path.name)

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
    _add_simple_source(red, vv_path.name)

    green = ET.SubElement(root, "VRTRasterBand", dataType="Float32", band="2")
    ET.SubElement(green, "ColorInterp").text = "Green"
    ET.SubElement(green, "Description").text = band_desc[1]
    ET.SubElement(green, "NoDataValue").text = "nan"
    _add_simple_source(green, vh_path.name)

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
    _add_simple_source(blue, vv_path.name)
    _add_simple_source(blue, vh_path.name)

    _write_vrt(root, output_vrt)


class _TileInfo(NamedTuple):
    name: str
    parent: str
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
            name=path.name,
            parent=str(path.parent),
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


def _validate_mosaic_tiles(tiles: list[_TileInfo], dst_parent: str) -> None:
    """Validate tiles for an axis-aligned, single-band mosaic.

    Raises
    ------
    ValueError
        If tile grids, data types, NODATA, bands, or directories are incompatible.
    """
    ref = tiles[0]

    if ref.band_count != 1:
        msg = f"Only single-band tiles supported, got {ref.band_count} bands in '{ref.name}'."
        raise ValueError(msg)
    if ref.dtype not in _GDAL_DTYPE_NAMES:
        msg = f"Unsupported tile dtype '{ref.dtype}' in '{ref.name}'."
        raise ValueError(msg)

    for t in tiles:
        _validate_tile(t, ref, dst_parent)


def _validate_tile(t: _TileInfo, ref: _TileInfo, dst_parent: str) -> None:
    """Validate one tile against the mosaic reference."""
    if t.parent != dst_parent:
        msg = (
            f"Tile '{t.name}' is in '{t.parent}', but tiles must live in the same "
            f"directory as the VRT ('{dst_parent}') to be referenced by bare filename."
        )
        raise ValueError(msg)
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


def build_vrt_mosaic(files: Sequence[AnyPath], dst_vrt: AnyPath) -> None:
    """Mosaic aligned, single-band tiles into a relocatable VRT.

    Parameters
    ----------
    files : Sequence[str | Path | UPath]
        Source tiles in the destination directory.
    dst_vrt : str | Path | UPath
        Output VRT.

    Raises
    ------
    ValueError
        If inputs are empty or incompatible.
    """
    if not files:
        msg = "files must not be empty"
        raise ValueError(msg)

    dst_vrt = UPath(dst_vrt)
    tiles = [_read_tile_info(UPath(f)) for f in files]
    _validate_mosaic_tiles(tiles, str(dst_vrt.parent))

    px, py = tiles[0].px, tiles[0].py
    minx = min(t.bounds.left for t in tiles)
    maxy = max(t.bounds.top for t in tiles)
    maxx = max(t.bounds.right for t in tiles)
    miny = min(t.bounds.bottom for t in tiles)
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
        filename = ET.SubElement(source, "SourceFilename", relativeToVRT="1")
        filename.text = t.name
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
