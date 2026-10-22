"""Serialization of the VRT model to GDAL's VRT XML."""

from __future__ import annotations

import numbers
import re
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import rasterio
from rasterio.enums import ColorInterp, Resampling

from freezebase.vrt._model import _GDAL_DTYPE_NAMES, VRT
from freezebase.vrt._paths import gdal_path

if TYPE_CHECKING:
    from rasterio.windows import Window
    from upath import UPath

    from freezebase.vrt._model import Band, PixelFunction, Source, SourceRef, Warp

_MIN_INLINE_NESTING_GDAL = (3, 11)

# GDAL names that are not the capitalized rasterio member name.
_COLOR_INTERP_NAMES = {
    ColorInterp.Y: "YCbCr_Y",
    ColorInterp.Cb: "YCbCr_Cb",
    ColorInterp.Cr: "YCbCr_Cr",
    ColorInterp.rededge: "RedEdge",
    ColorInterp.nir: "NIR",
    ColorInterp.swir: "SWIR",
    ColorInterp.mwir: "MWIR",
    ColorInterp.lwir: "LWIR",
    ColorInterp.tir: "TIR",
    ColorInterp.other_ir: "OtherIR",
    ColorInterp.sar_ka: "SAR_Ka",
    ColorInterp.sar_k: "SAR_K",
    ColorInterp.sar_ku: "SAR_Ku",
    ColorInterp.sar_x: "SAR_X",
    ColorInterp.sar_c: "SAR_C",
    ColorInterp.sar_s: "SAR_S",
    ColorInterp.sar_l: "SAR_L",
    ColorInterp.sar_p: "SAR_P",
}

_RESAMPLING_NAMES = {Resampling.cubic_spline: "cubicspline"}


def to_element(vrt: VRT, *, relative_to: UPath | None) -> ET.Element:
    """Build the ``<VRTDataset>`` element of ``vrt``."""
    root = ET.Element("VRTDataset", rasterXSize=str(vrt.width), rasterYSize=str(vrt.height))
    if vrt.warp is not None:
        root.set("subClass", "VRTWarpedDataset")
    _add_metadata(root, vrt.metadata)
    if vrt.crs is not None:
        ET.SubElement(root, "SRS").text = vrt.crs.to_wkt()
    ET.SubElement(root, "GeoTransform").text = ", ".join(
        repr(float(v)) for v in vrt.transform.to_gdal()
    )
    for index, band in enumerate(vrt.bands, start=1):
        _add_band(root, band, index, warped=vrt.warp is not None, relative_to=relative_to)
    if vrt.warp is not None:
        _add_warp(root, vrt.warp, relative_to=relative_to)
    return root


def _require_inline_nesting() -> None:
    """Raise unless rasterio's GDAL reads VRTs nested inline in a source."""
    version = rasterio.__gdal_version__
    major, minor = (int(part) for part in re.findall(r"\d+", version)[:2])
    if (major, minor) < _MIN_INLINE_NESTING_GDAL:
        msg = f"Nested VRTs need GDAL >= 3.11; rasterio bundles GDAL {version}."
        raise RuntimeError(msg)


def _gdal_dtype(dtype: str) -> str:
    """Map a NumPy dtype name to GDAL's."""
    try:
        return _GDAL_DTYPE_NAMES[dtype]
    except KeyError:
        msg = f"Unsupported dtype {dtype!r}; expected one of {sorted(_GDAL_DTYPE_NAMES)}."
        raise ValueError(msg) from None


def _number(value: float) -> str:
    """Format an integer as itself and anything else as a float, NaN as ``nan``."""
    if isinstance(value, numbers.Integral):
        return str(int(value))
    return repr(float(value))


def _path_text(path: UPath, *, relative_to: UPath | None) -> tuple[str, str]:
    """Return the GDAL path of ``path`` and its ``relativeToVRT`` flag."""
    text, relative = gdal_path(path, relative_to=relative_to)
    return text, "1" if relative else "0"


def _add_metadata(parent: ET.Element, metadata: dict[str, str]) -> None:
    """Append a ``<Metadata>`` element unless ``metadata`` is empty."""
    if not metadata:
        return
    element = ET.SubElement(parent, "Metadata")
    for key, value in metadata.items():
        ET.SubElement(element, "MDI", key=key).text = value


def _add_band(
    root: ET.Element,
    band: Band,
    index: int,
    *,
    warped: bool,
    relative_to: UPath | None,
) -> None:
    """Append a ``<VRTRasterBand>`` with its sources."""
    element = ET.SubElement(
        root, "VRTRasterBand", dataType=_gdal_dtype(band.dtype), band=str(index)
    )
    if band.pixel_function is not None:
        element.set("subClass", "VRTDerivedRasterBand")
    elif warped:
        element.set("subClass", "VRTWarpedRasterBand")
    _add_metadata(element, band.metadata)
    if band.nodata is not None:
        ET.SubElement(element, "NoDataValue").text = _number(band.nodata)
    if band.description:
        ET.SubElement(element, "Description").text = band.description
    if band.unit is not None:
        ET.SubElement(element, "UnitType").text = band.unit
    ET.SubElement(element, "ColorInterp").text = _COLOR_INTERP_NAMES.get(
        band.color_interp, band.color_interp.name.capitalize()
    )
    if band.pixel_function is not None:
        _add_pixel_function(element, band.pixel_function)
    for source in band.sources:
        _add_source(element, source, relative_to=relative_to)


def _add_pixel_function(band: ET.Element, function: PixelFunction) -> None:
    """Append the pixel function elements of a derived band."""
    ET.SubElement(band, "PixelFunctionType").text = function.name
    if function.args:
        ET.SubElement(
            band,
            "PixelFunctionArguments",
            {key: str(value) for key, value in function.args.items()},
        )
    if function.transfer_type is not None:
        ET.SubElement(band, "SourceTransferType").text = _gdal_dtype(function.transfer_type)


def _add_source(band: ET.Element, source: Source, *, relative_to: UPath | None) -> None:
    """Append a ``SimpleSource`` or ``ComplexSource``."""
    complex_ = source.nodata is not None or source.lut is not None
    element = ET.SubElement(band, "ComplexSource" if complex_ else "SimpleSource")
    if source.resampling is not None:
        element.set(
            "resampling",
            _RESAMPLING_NAMES.get(source.resampling, source.resampling.name),
        )
    _add_source_ref(element, source.path, relative_to=relative_to)
    ET.SubElement(element, "SourceBand").text = str(source.band)
    if (props := source.properties) is not None:
        ET.SubElement(
            element,
            "SourceProperties",
            RasterXSize=str(props.width),
            RasterYSize=str(props.height),
            DataType=_gdal_dtype(props.dtype),
            BlockXSize=str(props.block_width),
            BlockYSize=str(props.block_height),
        )
    if source.src_window is not None:
        _add_window(element, "SrcRect", source.src_window)
    if source.dst_window is not None:
        _add_window(element, "DstRect", source.dst_window)
    if source.nodata is not None:
        ET.SubElement(element, "NODATA").text = _number(source.nodata)
    if source.lut is not None:
        ET.SubElement(element, "LUT").text = ",".join(
            f"{_number(value)}:{_number(output)}" for value, output in source.lut
        )


def _add_source_ref(element: ET.Element, path: SourceRef, *, relative_to: UPath | None) -> None:
    """Append a ``<SourceFilename>``, or the nested VRT inline."""
    if isinstance(path, VRT):
        _require_inline_nesting()
        # GDAL resolves no relative paths inside an inline VRT.
        element.append(to_element(path, relative_to=None))
        return
    text, relative = _path_text(path, relative_to=relative_to)
    ET.SubElement(element, "SourceFilename", relativeToVRT=relative).text = text


def _add_window(element: ET.Element, tag: str, window: Window) -> None:
    """Append a ``SrcRect`` or ``DstRect``."""
    ET.SubElement(
        element,
        tag,
        xOff=_number(window.col_off),
        yOff=_number(window.row_off),
        xSize=_number(window.width),
        ySize=_number(window.height),
    )


def _add_warp(root: ET.Element, warp: Warp, *, relative_to: UPath | None) -> None:
    """Append the block size and warp options, with ``<SourceDataset>`` reinserted."""
    ET.SubElement(root, "BlockXSize").text = str(warp.block_size[0])
    ET.SubElement(root, "BlockYSize").text = str(warp.block_size[1])
    # The options come from GDAL or from a VRT the caller chose to read.
    options = ET.fromstring(warp.options_xml)  # noqa: S314
    # GDAL writes an in-memory warp source as escaped XML text, not as an inline element.
    if isinstance(warp.source, VRT):
        root_source = to_element(warp.source, relative_to=None)
        ET.indent(root_source)
        text, relative = ET.tostring(root_source, encoding="unicode"), "0"
    else:
        text, relative = _path_text(warp.source, relative_to=relative_to)
    source = ET.Element("SourceDataset", relativeToVRT=relative)
    source.text = text
    tags = [child.tag for child in options]
    position = len(tags) - tags[::-1].index("Option") if "Option" in tags else 0
    options.insert(position, source)
    root.append(options)
