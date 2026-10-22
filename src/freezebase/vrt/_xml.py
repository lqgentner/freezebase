"""Serialization of the VRT model to GDAL's VRT XML, and parsing it back."""

from __future__ import annotations

import copy
import numbers
import re
from typing import TYPE_CHECKING, Any
from xml.etree import ElementTree as ET

from affine import Affine
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.windows import Window

from freezebase.vrt._model import (
    _GDAL_DTYPE_NAMES,
    VRT,
    Band,
    PixelFunction,
    Source,
    SourceProperties,
    Warp,
)
from freezebase.vrt._paths import gdal_path, inherited_options, parse_gdal_path

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from upath import UPath

    from freezebase.vrt._model import SourceRef

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

_DTYPES = {gdal: numpy for numpy, gdal in _GDAL_DTYPE_NAMES.items()}
_COLOR_INTERPS = {
    _COLOR_INTERP_NAMES.get(member, member.name.capitalize()).lower(): member
    for member in ColorInterp
}
_RESAMPLINGS = {
    **{member.name: member for member in Resampling},
    **{name: member for member, name in _RESAMPLING_NAMES.items()},
    "near": Resampling.nearest,
}

# The attributes each element may carry; GDALWarpOptions' contents are kept unchecked.
_ATTRIBUTES = {
    "VRTDataset": {"rasterXSize", "rasterYSize", "subClass"},
    "SRS": {"dataAxisToSRSAxisMapping"},
    "MDI": {"key"},
    "VRTRasterBand": {"dataType", "band", "subClass"},
    "SimpleSource": {"resampling"},
    "ComplexSource": {"resampling"},
    "SourceFilename": {"relativeToVRT", "shared"},
    "SourceDataset": {"relativeToVRT"},
    "SourceProperties": {"RasterXSize", "RasterYSize", "DataType", "BlockXSize", "BlockYSize"},
    "SrcRect": {"xOff", "yOff", "xSize", "ySize"},
    "DstRect": {"xOff", "yOff", "xSize", "ySize"},
}
_WARPED_DATASET = "VRTWarpedDataset"
_DERIVED_BAND = "VRTDerivedRasterBand"
_WARPED_BAND = "VRTWarpedRasterBand"


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


def from_xml(
    text: str,
    *,
    base: UPath | None = None,
    storage_options: Mapping[str, Any] | None = None,
) -> VRT:
    """Parse VRT XML; relative source paths resolve against the VRT file ``base``."""
    # Python >= 3.12 ships an expat that refuses entity expansion attacks.
    return from_element(ET.fromstring(text), base=base, storage_options=storage_options)  # noqa: S314


def from_element(
    root: ET.Element,
    *,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> VRT:
    """Parse a ``<VRTDataset>`` element into the model."""
    if root.tag != "VRTDataset":
        raise _unsupported(root)
    _check_attributes(root)
    warped = _subclass(root, _WARPED_DATASET) is not None
    vrt = VRT(
        None,
        Affine.identity(),
        int(_attribute(root, "rasterXSize")),
        int(_attribute(root, "rasterYSize")),
    )
    block: dict[str, int] = {}
    options: ET.Element | None = None
    for child in _children(root):
        match child.tag:
            case "Metadata":
                vrt.metadata = _metadata(child)
            case "SRS":
                vrt.crs = CRS.from_user_input(_text(child))
            case "GeoTransform":
                vrt.transform = Affine.from_gdal(*(float(v) for v in _text(child).split(",")))
            case "VRTRasterBand":
                vrt.bands.append(_band(child, base=base, storage_options=storage_options))
            case "BlockXSize" | "BlockYSize" if warped:
                block[child.tag] = int(_text(child))
            case "GDALWarpOptions" if warped:
                options = child
            case _:
                raise _unsupported(child)
    if warped:
        vrt.warp = _warp(options, block, base=base, storage_options=storage_options)
    return vrt


def _unsupported(element: ET.Element) -> ValueError:
    """Return the error for an element the model cannot hold."""
    return ValueError(f"Unsupported VRT element <{element.tag}>")


def _check_attributes(element: ET.Element) -> None:
    """Raise if ``element`` carries an attribute the model cannot hold."""
    for name in element.attrib:
        if name not in _ATTRIBUTES.get(element.tag, set()):
            msg = f"Unsupported attribute {name!r} on <{element.tag}>"
            raise ValueError(msg)


def _children(element: ET.Element) -> Iterator[ET.Element]:
    """Yield the children of ``element``, checking their attributes."""
    for child in element:
        # Any name is a pixel function argument.
        if child.tag != "PixelFunctionArguments":
            _check_attributes(child)
        yield child


def _attribute(element: ET.Element, name: str) -> str:
    """Return a required attribute."""
    value = element.get(name)
    if value is None:
        msg = f"<{element.tag}> lacks the {name!r} attribute"
        raise ValueError(msg)
    return value


def _text(element: ET.Element) -> str:
    """Return the text of ``element``, empty if it has none."""
    return element.text or ""


def _subclass(element: ET.Element, *allowed: str) -> str | None:
    """Return the ``subClass`` of ``element``, raising unless it is one of ``allowed``."""
    subclass = element.get("subClass")
    if subclass is not None and subclass not in allowed:
        msg = f"Unsupported <{element.tag}> subClass {subclass!r}"
        raise ValueError(msg)
    return subclass


def _dtype(name: str) -> str:
    """Map a GDAL data type name to NumPy's."""
    try:
        return _DTYPES[name]
    except KeyError:
        msg = f"Unsupported GDAL data type {name!r}; expected one of {sorted(_DTYPES)}."
        raise ValueError(msg) from None


def _lookup[T](table: Mapping[str, T], name: str, what: str) -> T:
    """Look ``name`` up case-insensitively."""
    try:
        return table[name.strip().lower()]
    except KeyError:
        msg = f"Unsupported {what} {name!r}"
        raise ValueError(msg) from None


def _metadata(element: ET.Element) -> dict[str, str]:
    """Parse a ``<Metadata>`` element."""
    items: dict[str, str] = {}
    for item in _children(element):
        if item.tag != "MDI":
            raise _unsupported(item)
        items[_attribute(item, "key")] = _text(item)
    return items


def _band(  # noqa: C901, PLR0912  # one case per element, as in the writer
    element: ET.Element,
    *,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> Band:
    """Parse a ``<VRTRasterBand>``."""
    subclass = _subclass(element, _DERIVED_BAND, _WARPED_BAND)
    derived = subclass == _DERIVED_BAND
    band = Band(_dtype(_attribute(element, "dataType")))
    function: dict[str, Any] = {}
    for child in _children(element):
        match child.tag:
            case "Metadata":
                band.metadata = _metadata(child)
            case "NoDataValue":
                band.nodata = float(_text(child))
            case "Description":
                band.description = _text(child)
            case "UnitType":
                band.unit = _text(child)
            case "ColorInterp":
                band.color_interp = _lookup(_COLOR_INTERPS, _text(child), "colour interpretation")
            case "PixelFunctionType" if derived:
                function["name"] = _text(child)
            case "PixelFunctionArguments" if derived:
                function["args"] = dict(child.attrib)
            case "SourceTransferType" if derived:
                function["transfer_type"] = _dtype(_text(child))
            case "SimpleSource" | "ComplexSource" if subclass != _WARPED_BAND:
                band.sources.append(_source(child, base=base, storage_options=storage_options))
            case _:
                raise _unsupported(child)
    if derived:
        if "name" not in function:
            msg = f"<{_DERIVED_BAND}> lacks <PixelFunctionType>"
            raise ValueError(msg)
        band.pixel_function = PixelFunction(**function)
    return band


def _source(  # noqa: C901  # one case per element, as in the writer
    element: ET.Element,
    *,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> Source:
    """Parse a ``<SimpleSource>`` or ``<ComplexSource>``."""
    complex_ = element.tag == "ComplexSource"
    path: SourceRef | None = None
    fields: dict[str, Any] = {}
    if (resampling := element.get("resampling")) is not None:
        fields["resampling"] = _lookup(_RESAMPLINGS, resampling, "resampling")
    for child in _children(element):
        match child.tag:
            case "SourceFilename":
                path = _source_ref(child, base=base, storage_options=storage_options)
            case "VRTDataset":
                # Inline VRTs hold absolute paths but inherit the S3 options.
                path = from_element(
                    child, base=None, storage_options=inherited_options(base, storage_options)
                )
            case "SourceBand":
                fields["band"] = int(_text(child))
            case "SourceProperties":
                fields["properties"] = SourceProperties(
                    int(_attribute(child, "RasterXSize")),
                    int(_attribute(child, "RasterYSize")),
                    _dtype(_attribute(child, "DataType")),
                    int(_attribute(child, "BlockXSize")),
                    int(_attribute(child, "BlockYSize")),
                )
            case "SrcRect":
                fields["src_window"] = _window(child)
            case "DstRect":
                fields["dst_window"] = _window(child)
            case "NODATA" if complex_:
                fields["nodata"] = float(_text(child))
            case "LUT" if complex_:
                fields["lut"] = [
                    (float(value), float(output))
                    for value, output in (pair.split(":") for pair in _text(child).split(","))
                ]
            case _:
                raise _unsupported(child)
    if path is None:
        msg = f"<{element.tag}> lacks <SourceFilename>"
        raise ValueError(msg)
    return Source(path, **fields)


def _source_ref(
    element: ET.Element,
    *,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> SourceRef:
    """Parse a ``<SourceFilename>`` or ``<SourceDataset>``: a path, or a VRT as escaped XML."""
    text = _text(element)
    if text.lstrip().startswith("<VRTDataset"):
        # GDAL names an in-memory VRT by its XML, which has no location to resolve against.
        return from_xml(text, storage_options=inherited_options(base, storage_options))
    return parse_gdal_path(
        text,
        relative=element.get("relativeToVRT") == "1",
        base=base,
        storage_options=storage_options,
    )


def _window(element: ET.Element) -> Window:
    """Parse a ``<SrcRect>`` or ``<DstRect>``."""
    return Window(
        float(_attribute(element, "xOff")),
        float(_attribute(element, "yOff")),
        float(_attribute(element, "xSize")),
        float(_attribute(element, "ySize")),
    )


def _warp(
    options: ET.Element | None,
    block: Mapping[str, int],
    *,
    base: UPath | None,
    storage_options: Mapping[str, Any] | None,
) -> Warp:
    """Parse the warp from ``<GDALWarpOptions>`` and the dataset's block size."""
    if options is None:
        msg = "A warped VRT needs <GDALWarpOptions>"
        raise ValueError(msg)
    for tag in ("BlockXSize", "BlockYSize"):
        if tag not in block:
            msg = f"A warped VRT needs <{tag}>"
            raise ValueError(msg)
    options = copy.deepcopy(options)
    source = options.find("SourceDataset")
    if source is None:
        msg = "<GDALWarpOptions> lacks <SourceDataset>"
        raise ValueError(msg)
    _check_attributes(source)
    path = _source_ref(source, base=base, storage_options=storage_options)
    options.remove(source)
    # Indentation depends on where the options sat; drop it so equal warps compare equal.
    for node in options.iter():
        if node.text is not None and not node.text.strip():
            node.text = None
        if node.tail is not None and not node.tail.strip():
            node.tail = None
    xml = ET.tostring(options, encoding="unicode")
    return Warp(path, xml, (block["BlockXSize"], block["BlockYSize"]))
