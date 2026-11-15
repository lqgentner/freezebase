"""The VRT model: mutable dataclasses mirroring a GDAL Virtual Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
import math
from typing import TYPE_CHECKING, ClassVar, Literal, Self, overload
from xml.etree import ElementTree as ET

import numpy as np
from rasterio.coords import BoundingBox
from rasterio.enums import ColorInterp, Resampling

from freezebase._deprecation import deprecated_alias
from freezebase.vrt import _summary

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from contextlib import AbstractContextManager
    from pathlib import Path
    from typing import Any

    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.io import DatasetReader
    from rasterio.windows import Window
    from upath import UPath

type AnyPath = str | Path | UPath
type SourceRef = UPath | VRT

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
    # rasterio's names; NumPy has no complex integer types.
    "complex_int16": "CInt16",
    "complex_int32": "CInt32",
    "complex64": "CFloat32",
    "complex128": "CFloat64",
}


# The real types, which NumPy can describe.
_NUMPY_DTYPES = frozenset(name for name in _GDAL_DTYPE_NAMES if not name.startswith("complex"))


def _nodata_equal(a: float | None, b: float | None) -> bool:
    """Compare nodata values, NaN equal to NaN."""
    if a is None or b is None:
        return a is b
    return a == b or (math.isnan(a) and math.isnan(b))


def _check_nodata_fits(dtype: str, nodata: float | None) -> None:
    """Raise if ``nodata`` cannot be stored in ``dtype``, so GDAL would drop or alter it."""
    if nodata is None or dtype not in _NUMPY_DTYPES:
        return
    kind = np.dtype(dtype)
    if kind.kind in "iu":
        info = np.iinfo(kind)
        fits = math.isfinite(nodata) and nodata == int(nodata) and info.min <= nodata <= info.max
    else:
        fits = not math.isfinite(nodata) or abs(nodata) <= float(np.finfo(kind).max)
    if not fits:
        msg = (
            f"nodata {nodata:g} does not fit in {dtype}; pass a nodata {dtype} can hold, "
            "or a dtype that holds it."
        )
        raise ValueError(msg)


def _fields_equal(a: Source | Band, b: Source | Band) -> bool:
    """Compare every dataclass field, nodata NaN-aware."""
    return all(
        _nodata_equal(a.nodata, b.nodata)
        if item.name == "nodata"
        else getattr(a, item.name) == getattr(b, item.name)
        for item in fields(a)
    )


def _ref_repr(ref: SourceRef) -> str:
    """Represent a source path, or a nested VRT on one line."""
    return _summary.safe_short(ref) if isinstance(ref, VRT) else repr(ref)


@dataclass
class PixelFunction:
    """A GDAL built-in pixel function applied by a derived band.

    Parameters
    ----------
    name : str
        GDAL pixel function name, such as ``"dB"`` or ``"div"``.
    args : dict[str, str | int | float], optional
        Pixel function arguments; values are stored as strings.
    transfer_type : str | None, optional
        NumPy dtype name the sources are read as before the function runs.
    """

    name: str
    args: dict[str, str | int | float] = field(default_factory=dict)
    transfer_type: str | None = None

    def __post_init__(self) -> None:
        """Store argument values as strings, as they read back from XML."""
        self.args = {key: str(value) for key, value in self.args.items()}

    @classmethod
    def db(cls, fact: Literal[10, 20] = 10) -> Self:
        """Convert to decibels.

        Parameters
        ----------
        fact : {10, 20}, optional
            ``10`` for intensity, ``20`` for amplitude.

        Returns
        -------
        PixelFunction
            ``fact * log10(x)``.
        """
        return cls("dB", {"fact": fact})

    @classmethod
    def div(cls) -> Self:
        """Divide the first source by the second.

        Returns
        -------
        PixelFunction
            The ``div`` pixel function.
        """
        return cls("div")

    @classmethod
    def diff(cls) -> Self:
        """Subtract the second source from the first.

        Returns
        -------
        PixelFunction
            The ``diff`` pixel function.
        """
        return cls("diff")

    @classmethod
    def scale(cls) -> Self:
        """Apply the source's scale and offset.

        Returns
        -------
        PixelFunction
            The ``scale`` pixel function.
        """
        return cls("scale")

    @classmethod
    def minimum(cls) -> Self:
        """Take the per-pixel minimum of the sources.

        Returns
        -------
        PixelFunction
            The ``min`` pixel function.
        """
        return cls("min")

    @classmethod
    def maximum(cls) -> Self:
        """Take the per-pixel maximum of the sources.

        Returns
        -------
        PixelFunction
            The ``max`` pixel function.
        """
        return cls("max")

    @classmethod
    def mean(cls) -> Self:
        """Take the per-pixel mean of the sources.

        Returns
        -------
        PixelFunction
            The ``mean`` pixel function.
        """
        return cls("mean")


@dataclass
class SourceProperties:
    """Header of a source raster, as ``gdalbuildvrt`` records it.

    Parameters
    ----------
    width, height : int
        Raster size in pixels.
    dtype : str
        NumPy dtype name.
    block_width, block_height : int
        Block size in pixels.
    """

    width: int
    height: int
    dtype: str
    block_width: int
    block_height: int


@dataclass
class Source:
    """One band of a file or nested VRT drawn into a VRT band.

    Parameters
    ----------
    path : UPath | VRT
        Absolute path of the source file, or a VRT nested inline.
    band : int, optional
        1-based band of the source to read.
    src_window : Window | None, optional
        Region of the source to read. GDAL maps the whole source onto the whole band only
        when both windows are ``None``.
    dst_window : Window | None, optional
        Region of the VRT band to write to.
    nodata : float | None, optional
        Source value treated as nodata; setting it writes a ``ComplexSource``.
    lut : list[tuple[float, float]] | None, optional
        ``(input, output)`` pairs mapping source values; setting it writes a ``ComplexSource``.
    resampling : Resampling | None, optional
        Resampling used when the windows differ in size.
    properties : SourceProperties | None, optional
        Header of the source raster.
    """

    path: SourceRef
    band: int = 1
    src_window: Window | None = None
    dst_window: Window | None = None
    nodata: float | None = None
    lut: list[tuple[float, float]] | None = None
    resampling: Resampling | None = None
    properties: SourceProperties | None = None

    def __eq__(self, other: object) -> bool:
        """Compare all fields, treating NaN nodata as equal.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            Whether ``other`` is a ``Source`` with equal fields.
        """
        if not isinstance(other, Source):
            return NotImplemented
        return _fields_equal(self, other)

    def __repr__(self) -> str:
        """Show the path, the band and the fields that are set, a nested VRT on one line.

        Returns
        -------
        str
            ``Source(path=..., band=..., ...)``.
        """
        parts = [f"path={_ref_repr(self.path)}", f"band={self.band}"]
        parts += [
            f"{item.name}={value!r}"
            for item in fields(self)
            if item.name not in {"path", "band"}
            and (value := getattr(self, item.name)) is not None
        ]
        return f"Source({', '.join(parts)})"

    # Mutable, so unhashable, as a plain dataclass is.
    __hash__: ClassVar[None] = None


@dataclass
class Band:
    """One VRT band.

    Parameters
    ----------
    dtype : str
        NumPy dtype name.
    sources : list[Source], optional
        Sources in draw order: a later source wins where they overlap.
    nodata : float | None, optional
        Band nodata value.
    description : str, optional
        Band description.
    color_interp : ColorInterp, optional
        Colour interpretation.
    unit : str | None, optional
        Unit of the pixel values.
    metadata : dict[str, str], optional
        Band metadata items.
    pixel_function : PixelFunction | None, optional
        Pixel function combining the sources; setting it makes a derived band.
    scale, offset : float | None, optional
        Scale and offset turning pixel values into physical values; ``None`` writes none.
    """

    dtype: str
    sources: list[Source] = field(default_factory=list)
    nodata: float | None = None
    description: str = ""
    color_interp: ColorInterp = ColorInterp.gray
    unit: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    pixel_function: PixelFunction | None = None
    scale: float | None = None
    offset: float | None = None

    def __eq__(self, other: object) -> bool:
        """Compare all fields, treating NaN nodata as equal.

        Parameters
        ----------
        other : object
            Object to compare with.

        Returns
        -------
        bool
            Whether ``other`` is a ``Band`` with equal fields.
        """
        if not isinstance(other, Band):
            return NotImplemented
        return _fields_equal(self, other)

    # Mutable, so unhashable, as a plain dataclass is.
    __hash__: ClassVar[None] = None


@dataclass
class Warp:
    """The warp of a warped VRT.

    Parameters
    ----------
    source : UPath | VRT
        Dataset being warped.
    options_xml : str
        ``<GDALWarpOptions>`` element without its ``<SourceDataset>`` child.
    block_size : tuple[int, int]
        Block width and height of the warped dataset.
    """

    source: SourceRef
    options_xml: str
    block_size: tuple[int, int]

    def __repr__(self) -> str:
        """Show a nested VRT on one line and only the length of the options.

        Returns
        -------
        str
            ``Warp(source=..., options_xml=<N chars>, block_size=...)``.
        """
        return (
            f"Warp(source={_ref_repr(self.source)}, "
            f"options_xml=<{len(self.options_xml)} chars>, block_size={self.block_size!r})"
        )

    @property
    def vertical_shift(self) -> bool:
        """Whether the warp applies a vertical shift.

        Returns
        -------
        bool
            True when the warp options set ``APPLY_VERTICAL_SHIFT=YES``.
        """
        # The options come from GDAL or from a VRT the caller chose to read.
        options = ET.fromstring(self.options_xml)  # noqa: S314
        return any(
            (option.text or "").strip().upper() == "YES"
            for option in options.iterfind("Option[@name='APPLY_VERTICAL_SHIFT']")
        )


@dataclass
class VRT:
    """A GDAL Virtual Dataset.

    Parameters
    ----------
    crs : CRS | None
        Coordinate reference system.
    transform : Affine
        Pixel-to-CRS affine transform.
    width, height : int
        Raster size in pixels.
    bands : list[Band], optional
        Bands in order.
    metadata : dict[str, str], optional
        Dataset metadata items.
    warp : Warp | None, optional
        Set for a warped VRT, whose bands then carry no sources.
    metadata_domains : dict[str, dict[str, str]], optional
        Dataset metadata items of named domains, such as ``IMAGE_STRUCTURE``, by domain.
    """

    crs: CRS | None
    transform: Affine
    width: int
    height: int
    bands: list[Band] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    warp: Warp | None = None
    metadata_domains: dict[str, dict[str, str]] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Summarize the grid, the bands and the steps that build the VRT, without I/O.

        Returns
        -------
        str
            A header line, one line per band, and a tree of the steps and their sources.
        """
        return _summary.text_repr(self)

    def _repr_html_(self) -> str:
        """Render the summary as HTML for notebooks."""
        return _summary.html_repr(self)

    @property
    def bounds(self) -> BoundingBox:
        """Extent in units of the CRS, as rasterio reports it for the opened VRT.

        Returns
        -------
        rasterio.coords.BoundingBox
            ``(left, bottom, right, top)``; for a rotated grid, the envelope of its corners.
            For a south-up grid, ``bottom`` is above ``top``, as in rasterio.
        """
        a, b, c, d, e, f = self.transform[:6]
        if b == d == 0:
            return BoundingBox(c, f + e * self.height, c + a * self.width, f)
        corners = [(0, 0), (0, self.height), (self.width, self.height), (self.width, 0)]
        xs = [a * col + b * row + c for col, row in corners]
        ys = [d * col + e * row + f for col, row in corners]
        return BoundingBox(min(xs), min(ys), max(xs), max(ys))

    @property
    def res(self) -> tuple[float, float]:
        """Pixel width and height in units of the CRS.

        Returns
        -------
        tuple[float, float]
            ``(x, y)`` pixel size, positive.
        """
        a, b, _, d, e, _ = self.transform[:6]
        return math.hypot(a, d), math.hypot(b, e)

    @property
    def shape(self) -> tuple[int, int]:
        """Raster size in pixels.

        Returns
        -------
        tuple[int, int]
            ``(height, width)``.
        """
        return self.height, self.width

    @property
    def count(self) -> int:
        """Number of bands.

        Returns
        -------
        int
            Length of ``bands``.
        """
        return len(self.bands)

    @property
    def dtypes(self) -> tuple[str, ...]:
        """Data type of each band.

        Returns
        -------
        tuple[str, ...]
            NumPy dtype names, in band order.
        """
        return tuple(band.dtype for band in self.bands)

    @property
    def nodatavals(self) -> tuple[float | None, ...]:
        """Nodata value of each band.

        Returns
        -------
        tuple[float | None, ...]
            Nodata values in band order; ``None`` for a band without one.
        """
        return tuple(band.nodata for band in self.bands)

    def reproject(
        self,
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
        """Reproject this VRT onto another grid.

        Same as ``freezebase.vrt.reproject`` with this VRT as the source.

        Parameters
        ----------
        crs : CRS | str | None, optional
            Target CRS; required unless ``like`` is given.
        src_crs : CRS | str | None, optional
            Overrides this VRT's CRS.
        resolution : float | tuple[float, float] | None, optional
            Target pixel size, or ``(x, y)`` sizes, in units of ``crs``.
        bounds : BoundingBox | tuple[float, float, float, float] | None, optional
            Target extent as ``(left, bottom, right, top)`` in ``crs``. Needs ``resolution``.
        like : str | Path | UPath | VRT | None, optional
            Raster whose CRS, transform and size are copied exactly.
        resampling : Resampling, optional
            Warp kernel.
        snap : bool, optional
            Grow the derived extent outward onto the ``resolution`` grid.
        dtype : str | None, optional
            Data type of the warped bands. Defaults to this VRT's.
        nodata : float | None, optional
            NODATA value of the warped bands. Defaults to this VRT's.
        tolerance : float | {"auto"}, optional
            Error allowed when approximating the transformation, in source pixels.
        allow_ballpark : bool, optional
            Let PROJ fall back to a ballpark transformation.

        Returns
        -------
        VRT
            A warped VRT nesting a copy of this one; nothing is written.

        Raises
        ------
        ValueError
            As ``freezebase.vrt.reproject``.
        RuntimeError
            As ``freezebase.vrt.reproject``.
        """
        from freezebase.vrt._warp import reproject  # noqa: PLC0415  # _warp imports this module

        return reproject(
            self,
            crs=crs,
            src_crs=src_crs,
            resolution=resolution,
            bounds=bounds,
            like=like,
            resampling=resampling,
            snap=snap,
            dtype=dtype,
            nodata=nodata,
            tolerance=tolerance,
            allow_ballpark=allow_ballpark,
        )

    def derive(
        self,
        function: PixelFunction,
        *,
        dtype: str = "float32",
        nodata: float | None = math.nan,
        description: str = "",
        band: int | None = None,
    ) -> VRT:
        """Derive bands from this VRT with a GDAL pixel function.

        Same as ``freezebase.vrt.derive`` with this VRT as the only source.

        Parameters
        ----------
        function : PixelFunction
            Function applied to each band.
        dtype : str, optional
            NumPy dtype name of the derived bands.
        nodata : float | None, optional
            NODATA of the derived bands.
        description : str, optional
            Description of every derived band.
        band : int | None, optional
            1-based band to derive; ``None`` derives every band.

        Returns
        -------
        VRT
            A derived band per selected band, nesting a copy of this VRT.

        Raises
        ------
        ValueError
            As ``freezebase.vrt.derive``.
        """
        from freezebase.vrt._builders import derive  # noqa: PLC0415  # _builders imports this module

        return derive(
            self, function, dtype=dtype, nodata=nodata, description=description, band=band
        )

    def select(self, bands: Sequence[int] | np.ndarray) -> VRT:
        """Select bands of this VRT, in any order.

        Parameters
        ----------
        bands : Sequence[int] | numpy.ndarray
            1-based bands to keep, in output order; a band may repeat.

        Returns
        -------
        VRT
            The selected bands, copied, with this VRT's dataset metadata.

        Raises
        ------
        ValueError
            If ``bands`` is empty, holds a non-integer or names a band this VRT lacks.
        """
        from freezebase.vrt._steps import select  # noqa: PLC0415  # _steps imports this module

        return select(self, bands)

    def set(
        self,
        *,
        band_names: list[str] | None = None,
        color_interp: list[ColorInterp] | None = None,
        units: list[str] | None = None,
        tags: Mapping[str, str] | None = None,
        band_tags: Sequence[Mapping[str, str]] | None = None,
    ) -> VRT:
        """Return a copy of this VRT with new metadata; the pixels are unchanged.

        Parameters
        ----------
        band_names : list[str] | None, optional
            Band descriptions, one per band.
        color_interp : list[ColorInterp] | None, optional
            Per-band color interpretation.
        units : list[str] | None, optional
            Band unit strings, one per band.
        tags : Mapping[str, str] | None, optional
            Dataset-level tags, added to the existing ones.
        band_tags : Sequence[Mapping[str, str]] | None, optional
            Per-band tags, one mapping per band, added to the existing ones.

        Returns
        -------
        VRT
            The copy; ``None`` arguments leave that metadata as it is.

        Raises
        ------
        ValueError
            If a per-band sequence does not hold one entry per band.
        """
        from freezebase.vrt._steps import set_metadata  # noqa: PLC0415  # _steps imports this module

        return set_metadata(
            self,
            band_names=band_names,
            color_interp=color_interp,
            units=units,
            tags=tags,
            band_tags=band_tags,
        )

    def clip_box(self, minx: float, miny: float, maxx: float, maxy: float) -> VRT:
        """Cut out the pixels inside a box, without resampling.

        The box edges must lie on this VRT's pixel grid, so the result keeps whole pixels;
        its transform and size are those of the ``rasterio.windows`` window over the box.
        Unlike rioxarray's ``clip_box``, the box must lie inside the raster and is not
        snapped to the pixels it touches.

        Parameters
        ----------
        minx, miny, maxx, maxy : float
            Box edges in units of the CRS.

        Returns
        -------
        VRT
            A VRT reading the box from a copy of each band, with this VRT's band and
            dataset metadata.

        Raises
        ------
        ValueError
            If the grid is rotated or not north-up, or the box is empty, not finite, off
            the pixel grid or extends beyond this VRT.
        """
        from freezebase.vrt._steps import clip_box  # noqa: PLC0415  # _steps imports this module

        return clip_box(self, minx, miny, maxx, maxy)

    def colorize(
        self,
        colors: Sequence[tuple[float, tuple[int, ...]]] | str | Callable[[np.ndarray], Any],
        *,
        vmin: float | None = None,
        vmax: float | None = None,
        band: int = 1,
    ) -> VRT:
        """Render one band as an RGBA image through a color ramp.

        Colors are interpolated linearly between stops; values below the first stop or
        above the last take its color. Stop values compare with the stored pixel values,
        before any scale and offset. Pixels equal to the band's nodata and NaN pixels are
        transparent (all four bands 0); other pixels take the stop's alpha, 255 for an RGB
        stop. Pixels no source covers, such as gaps in a mosaic, read as 0 unless the band
        has a nodata value, and are colored as 0.

        Parameters
        ----------
        colors : Sequence[tuple[float, tuple[int, ...]]] | str | Colormap
            ``(value, (r, g, b))`` or ``(value, (r, g, b, a))`` stops with 0-255 channels,
            in strictly ascending value order; or a matplotlib colormap or its name, which
            needs matplotlib. A colormap of fewer than 256 colors, such as ``"tab10"``,
            gives each color an equal, unblended share of ``vmin`` to ``vmax``.
        vmin, vmax : float | None, optional
            Values the ends of a colormap map to; required for a colormap, not allowed
            with stops.
        band : int, optional
            1-based band to render.

        Returns
        -------
        VRT
            A four-band ``uint8`` RGBA VRT reading a copy of the band.

        Raises
        ------
        ValueError
            If ``band`` is out of range, there are fewer than two stops, the stops are not
            ascending or a color is not 3 or 4 integers from 0 to 255, ``vmin`` and ``vmax``
            are missing, not finite or not ascending for a colormap, or given with stops,
            or matplotlib has no colormap of that name.
        ImportError
            If ``colors`` names a colormap and matplotlib is not installed.
        """
        from freezebase.vrt._steps import colorize  # noqa: PLC0415  # _steps imports this module

        return colorize(self, colors, vmin=vmin, vmax=vmax, band=band)

    def to_xml(self, *, relative_to: UPath | None = None) -> str:
        """Serialize to VRT XML.

        Parameters
        ----------
        relative_to : UPath | None, optional
            Path of the VRT file that source paths are written relative to; ``None`` writes
            absolute paths.

        Returns
        -------
        str
            The ``<VRTDataset>`` document.

        Raises
        ------
        ValueError
            If a band or source dtype has no GDAL equivalent, a ``metadata_domains`` key
            is empty, or a band of a warped VRT has sources or a pixel function.
        RuntimeError
            If a source is a nested VRT and rasterio's GDAL is older than 3.11.
        """
        from freezebase.vrt._xml import to_element  # noqa: PLC0415  # _xml imports this module

        root = to_element(self, relative_to=relative_to)
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")

    def to_vrt(self, path: AnyPath, *, relative: bool = True) -> UPath:
        """Write the VRT as a ``.vrt`` file at ``path``.

        Sources on the same store as ``path`` (local, or the same S3 bucket) are written
        relative to it; others as absolute GDAL paths. Local writes are atomic.

        Parameters
        ----------
        path : str | Path | UPath
            Destination, local (a path or ``file://``) or ``s3://``.
        relative : bool, optional
            ``False`` writes every source path absolute.

        Returns
        -------
        UPath
            The path written.

        Raises
        ------
        ValueError
            If ``path`` is neither local nor ``s3://``, or as ``to_xml``.
        RuntimeError
            If a source is a nested VRT and rasterio's GDAL is older than 3.11.
        """
        from freezebase.vrt._io import write  # noqa: PLC0415  # _io imports raster, like _xml

        return write(self, path, relative=relative)

    write = deprecated_alias(to_vrt, "VRT.write", "VRT.to_vrt")

    @overload
    def to_cog(
        self,
        path: AnyPath,
        profile: dict[str, Any] | None = ...,
        *,
        band_names: list[str] | None = ...,
        color_interp: list[ColorInterp] | None = ...,
        tags: Mapping[str, str] | None = ...,
        band_tags: Sequence[Mapping[str, str]] | None = ...,
        units: list[str] | None = ...,
        checksum: Literal[False] = ...,
    ) -> None: ...
    @overload
    def to_cog(
        self,
        path: AnyPath,
        profile: dict[str, Any] | None = ...,
        *,
        band_names: list[str] | None = ...,
        color_interp: list[ColorInterp] | None = ...,
        tags: Mapping[str, str] | None = ...,
        band_tags: Sequence[Mapping[str, str]] | None = ...,
        units: list[str] | None = ...,
        checksum: Literal[True],
    ) -> str: ...
    @overload
    def to_cog(
        self,
        path: AnyPath,
        profile: dict[str, Any] | None = ...,
        *,
        band_names: list[str] | None = ...,
        color_interp: list[ColorInterp] | None = ...,
        tags: Mapping[str, str] | None = ...,
        band_tags: Sequence[Mapping[str, str]] | None = ...,
        units: list[str] | None = ...,
        checksum: bool,
    ) -> str | None: ...
    def to_cog(
        self,
        path: AnyPath,
        profile: dict[str, Any] | None = None,
        *,
        band_names: list[str] | None = None,
        color_interp: list[ColorInterp] | None = None,
        tags: Mapping[str, str] | None = None,
        band_tags: Sequence[Mapping[str, str]] | None = None,
        units: list[str] | None = None,
        checksum: bool = False,
    ) -> str | None:
        """Write the pixels as a Cloud Optimized GeoTIFF.

        Same as ``freezebase.raster.write_cog`` with this VRT as the data.

        Parameters
        ----------
        path : str | Path | UPath
            Destination COG (local or S3).
        profile : dict[str, Any] | None, optional
            ``dtype``, ``nodata`` and native GDAL ``COG`` creation options, as for
            ``write_cog``.
        band_names : list[str] | None, optional
            Band descriptions, one per band.
        color_interp : list[ColorInterp] | None, optional
            Per-band color interpretation.
        tags : Mapping[str, str] | None, optional
            Dataset-level tags.
        band_tags : Sequence[Mapping[str, str]] | None, optional
            Per-band tags, one mapping per band.
        units : list[str] | None, optional
            Band unit strings, one per band.
        checksum : bool, optional
            Hash the encoded file and return the digest.

        Returns
        -------
        str | None
            The SHA-256 digest as a multihash (``"1220"`` + hex) when ``checksum`` is
            ``True``, otherwise ``None``.

        Raises
        ------
        ValueError
            As ``write_cog``.
        RasterWriteError
            If the write fails.
        """
        from freezebase.raster import write_cog  # noqa: PLC0415  # raster imports freezebase.vrt

        return write_cog(
            self,
            path,
            profile,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
            checksum=checksum,
        )

    def to_raster(
        self,
        path: AnyPath,
        profile: dict[str, Any] | None = None,
        *,
        driver: str = "GTiff",
        band_names: list[str] | None = None,
        color_interp: list[ColorInterp] | None = None,
        tags: Mapping[str, str] | None = None,
        band_tags: Sequence[Mapping[str, str]] | None = None,
        units: list[str] | None = None,
    ) -> None:
        """Write the pixels as a raster file in any format GDAL can copy to.

        The result is a single file, without a ``.aux.xml`` sidecar, so what the format
        cannot hold, such as a PNG's CRS and transform, is dropped.

        Parameters
        ----------
        path : str | Path | UPath
            Destination (local or S3).
        profile : dict[str, Any] | None, optional
            ``dtype``, ``nodata`` and GDAL creation options of ``driver``. ``dtype`` and
            ``nodata`` apply as for ``to_cog``; the keys describing the dataset (``crs``,
            ``transform``, ``count``, ``width``, ``height``) are ignored. For ``GTiff``,
            the creation options default to ``RASTERIO_PROFILE_DEFAULTS``; for another
            driver, the GTiff-only ``tiled``, ``blockxsize``, ``blockysize`` and
            ``interleave`` are dropped. A ``driver`` key must match ``driver``.
        driver : str, optional
            GDAL driver name, such as ``"GTiff"`` or ``"PNG"``. Use ``to_cog`` for a COG.
        band_names : list[str] | None, optional
            Band descriptions, one per band.
        color_interp : list[ColorInterp] | None, optional
            Per-band color interpretation.
        tags : Mapping[str, str] | None, optional
            Dataset-level tags.
        band_tags : Sequence[Mapping[str, str]] | None, optional
            Per-band tags, one mapping per band.
        units : list[str] | None, optional
            Band unit strings, one per band.

        Raises
        ------
        ValueError
            If ``driver`` is ``COG``, ``profile`` sets another driver, a per-band sequence
            has the wrong length, the dtype is unknown, a nodata value does not fit the
            dtype, or the bands differ in dtype or nodata without ``profile`` setting it.
        RasterWriteError
            If the write fails, also when ``driver`` cannot hold the bands, such as
            ``float32`` in a PNG, writes several files, or is GeoTIFF and cannot store the
            CRS, such as a projected CRS with ellipsoidal heights.
        """
        from freezebase.raster import _write_raster_from_vrt  # noqa: PLC0415  # raster imports freezebase.vrt

        _write_raster_from_vrt(
            self,
            path,
            profile,
            driver=driver,
            band_names=band_names,
            color_interp=color_interp,
            tags=tags,
            band_tags=band_tags,
            units=units,
        )

    @property
    def sources(self) -> list[SourceRef]:
        """Direct sources in document order, then the warp source.

        Duplicates are removed: files by path, nested VRTs by identity. No I/O.

        Returns
        -------
        list[UPath | VRT]
            The direct sources.
        """
        from freezebase.vrt._io import sources  # noqa: PLC0415  # _io imports raster, like _xml

        return sources(self)

    def source_files(
        self, *, follow: bool = True, storage_options: Mapping[str, Any] | None = None
    ) -> list[UPath]:
        """Return the leaf files the VRT reads, recursing into nested VRTs.

        Followed VRT files are scanned for ``<SourceFilename>`` and ``<SourceDataset>``
        only, so they may hold elements ``read`` refuses; sources of any other kind, and
        paths on GDAL virtual file systems other than ``/vsis3/`` and ``/vsicurl/``, are
        not listed.

        Parameters
        ----------
        follow : bool, optional
            Read ``.vrt`` file sources and recurse into them; ``False`` lists them as files.
        storage_options : Mapping[str, Any] | None, optional
            Options for reading followed VRT files and their S3 sources, as for ``read``.
            ``None`` keeps each path's own, and a followed file without any inherits those
            of the VRT that references it.

        Returns
        -------
        list[UPath]
            Files in first-seen order, without duplicates.

        Raises
        ------
        ValueError
            If a followed VRT file names a relative source path inside an in-memory VRT.
        ImportError
            If a followed VRT file is ``http(s)://`` and aiohttp is not installed.
        """
        from freezebase.vrt._io import source_files  # noqa: PLC0415  # _io imports raster

        return source_files(self, follow=follow, storage_options=storage_options)

    def env(
        self, *, storage_options: Mapping[str, Any] | None = None
    ) -> AbstractContextManager[None]:
        """Return the rasterio environment that can read every source.

        Local sources need a plain environment, S3 sources the credentialed S3 environment,
        public HTTP(S) sources the HTTP one. Local sources mix with either. Callers using
        thread pools enter it in the main thread, because credentials set in a worker thread
        don't reach GDAL's reader threads.

        Parameters
        ----------
        storage_options : Mapping[str, Any] | None, optional
            Options for the VRT files followed to find the sources, as for
            ``source_files``.

        Returns
        -------
        AbstractContextManager[None]
            Context manager that applies the environment while active.

        Raises
        ------
        ValueError
            If S3 sources have different storage options, S3 is mixed with HTTP(S), or a
            source uses another protocol, or as ``source_files``.
        ImportError
            As ``source_files``.
        """
        from freezebase.vrt._io import env  # noqa: PLC0415  # _io imports raster, like _xml

        return env(self, storage_options=storage_options)

    def open(
        self, *, storage_options: Mapping[str, Any] | None = None, **kwargs: Any
    ) -> AbstractContextManager[DatasetReader]:
        """Open the VRT from memory, without writing a file.

        Parameters
        ----------
        storage_options : Mapping[str, Any] | None, optional
            Passed to ``env()``.
        **kwargs : Any
            Passed to ``rasterio.open``.

        Returns
        -------
        AbstractContextManager[DatasetReader]
            Context manager yielding the dataset inside `env()`.

        Raises
        ------
        ValueError
            As `env()`, or if a dtype has no GDAL equivalent.
        ImportError
            As `env()`.
        """
        from freezebase.vrt._io import open_vrt  # noqa: PLC0415  # _io imports raster, like _xml

        return open_vrt(self, storage_options=storage_options, **kwargs)
