"""The VRT model: mutable dataclasses mirroring a GDAL Virtual Dataset."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Self
from xml.etree import ElementTree as ET

from rasterio.enums import ColorInterp

if TYPE_CHECKING:
    from pathlib import Path

    from affine import Affine
    from rasterio.crs import CRS
    from rasterio.enums import Resampling
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
}


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
    """

    dtype: str
    sources: list[Source] = field(default_factory=list)
    nodata: float | None = None
    description: str = ""
    color_interp: ColorInterp = ColorInterp.gray
    unit: str | None = None
    metadata: dict[str, str] = field(default_factory=dict)
    pixel_function: PixelFunction | None = None


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
    """

    crs: CRS | None
    transform: Affine
    width: int
    height: int
    bands: list[Band] = field(default_factory=list)
    metadata: dict[str, str] = field(default_factory=dict)
    warp: Warp | None = None

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
            If a band or source dtype has no GDAL equivalent.
        RuntimeError
            If a source is a nested VRT and rasterio's GDAL is older than 3.11.
        """
        from freezebase.vrt._xml import to_element  # noqa: PLC0415  # _xml imports this module

        root = to_element(self, relative_to=relative_to)
        ET.indent(root)
        return ET.tostring(root, encoding="unicode")

    def write(self, path: AnyPath, *, relative: bool = True) -> UPath:
        """Write the VRT to ``path``.

        Sources on the same store as ``path`` (local, or the same S3 bucket) are written
        relative to it; others as absolute GDAL paths. Local writes are atomic.

        Parameters
        ----------
        path : str | Path | UPath
            Destination, local or ``s3://``.
        relative : bool, optional
            ``False`` writes every source path absolute.

        Returns
        -------
        UPath
            The path written.

        Raises
        ------
        ValueError
            If a band or source dtype has no GDAL equivalent.
        RuntimeError
            If a source is a nested VRT and rasterio's GDAL is older than 3.11.
        """
        from freezebase.vrt._io import write  # noqa: PLC0415  # _io imports raster, like _xml

        return write(self, path, relative=relative)
