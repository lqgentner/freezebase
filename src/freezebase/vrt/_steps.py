"""One-input steps that derive a new VRT from an existing one."""

from __future__ import annotations

import copy
import importlib
import itertools
import math
import numbers
import operator
from typing import TYPE_CHECKING, Any

import numpy as np
from rasterio.enums import ColorInterp
import rasterio.windows
from rasterio.windows import Window

from freezebase.raster import _apply_vrt_metadata
from freezebase.vrt._grid import _ALIGNMENT_TOL
from freezebase.vrt._model import VRT, Band, Source

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

type ColorStop = tuple[float, tuple[int, ...]]

_RGBA = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
_OPAQUE = 255
# Samples taken from a continuous colormap, as many as matplotlib's default lookup table.
_COLORMAP_SAMPLES = 256


def _check_band(vrt: VRT, band: int) -> None:
    """Raise unless ``band`` (1-based) exists in ``vrt``."""
    if not 1 <= band <= vrt.count:
        msg = f"VRT has no band {band} ({vrt.count} bands)."
        raise ValueError(msg)


def _views(vrt: VRT, bands: Sequence[int]) -> list[tuple[VRT, int]]:
    """Return, per 1-based band, a copy of what it reads and its band number in that copy.

    Each view holds only its own band, so a chain of steps nests each band once rather
    than every band once per output band. A warp covers all bands at once, so the bands
    of a warped VRT share one whole copy.
    """
    if vrt.warp is not None:
        whole = copy.deepcopy(vrt)
        return [(whole, band) for band in bands]
    return [
        (VRT(vrt.crs, vrt.transform, vrt.width, vrt.height, [copy.deepcopy(vrt.bands[b - 1])]), 1)
        for b in bands
    ]


def _reading(band: Band, view: VRT, view_band: int, window: Window | None) -> Band:
    """Build a band with the metadata of ``band`` that reads ``view_band`` of ``view``."""
    return Band(
        dtype=band.dtype,
        sources=[
            Source(
                path=view,
                band=view_band,
                src_window=window,
                dst_window=None if window is None else Window(0, 0, window.width, window.height),
            )
        ],
        nodata=band.nodata,
        description=band.description,
        color_interp=band.color_interp,
        unit=band.unit,
        metadata=dict(band.metadata),
        # The source is read raw, so the band keeps the scale and offset that apply to it.
        scale=band.scale,
        offset=band.offset,
    )


def select(vrt: VRT, bands: Sequence[int] | np.ndarray) -> VRT:
    """Return the bands ``bands`` of ``vrt``, in that order, as ``VRT.select`` documents."""
    if len(bands) == 0:
        msg = "bands must not be empty."
        raise ValueError(msg)
    try:
        wanted = [operator.index(band) for band in bands]
    except TypeError:
        msg = f"bands must hold integers, got {list(bands)}."
        raise ValueError(msg) from None
    for band in wanted:
        _check_band(vrt, band)
    if vrt.warp is None:
        out = [copy.deepcopy(vrt.bands[band - 1]) for band in wanted]
    else:
        out = [
            _reading(vrt.bands[band - 1], view, view_band, None)
            for band, (view, view_band) in zip(wanted, _views(vrt, wanted), strict=True)
        ]
    return VRT(vrt.crs, vrt.transform, vrt.width, vrt.height, out, metadata=dict(vrt.metadata))


def set_metadata(
    vrt: VRT,
    *,
    band_names: list[str] | None,
    color_interp: list[ColorInterp] | None,
    units: list[str] | None,
    tags: Mapping[str, str] | None,
    band_tags: Sequence[Mapping[str, str]] | None,
) -> VRT:
    """Return a copy of ``vrt`` with the metadata set, as ``VRT.set`` documents."""
    out = copy.deepcopy(vrt)
    _apply_vrt_metadata(
        out,
        band_names=band_names,
        color_interp=color_interp,
        tags=tags,
        band_tags=band_tags,
        units=units,
    )
    return out


def _grid_offset(value: float, name: str) -> int:
    """Round a pixel offset that must be whole, or raise."""
    if abs(value - round(value)) > _ALIGNMENT_TOL:
        msg = f"{name} is not on the pixel grid; clip_box keeps whole pixels only."
        raise ValueError(msg)
    return round(value)


def clip_box(vrt: VRT, minx: float, miny: float, maxx: float, maxy: float) -> VRT:
    """Return the pixels of ``vrt`` inside a box, as ``VRT.clip_box`` documents."""
    a, b, c, d, e, f = vrt.transform[:6]
    if b != 0 or d != 0 or a <= 0 or e >= 0:
        msg = "clip_box needs a north-up grid without rotation; reproject the VRT first."
        raise ValueError(msg)
    box = (minx, miny, maxx, maxy)
    if not all(math.isfinite(edge) for edge in box):
        msg = f"The box {box} has an edge that is not finite."
        raise ValueError(msg)
    if minx >= maxx or miny >= maxy:
        msg = f"The box {box} is empty."
        raise ValueError(msg)
    col0 = _grid_offset((minx - c) / a, "minx")
    col1 = _grid_offset((maxx - c) / a, "maxx")
    row0 = _grid_offset((maxy - f) / e, "maxy")
    row1 = _grid_offset((miny - f) / e, "miny")
    if col0 < 0 or row0 < 0 or col1 > vrt.width or row1 > vrt.height:
        msg = f"The box {box} extends beyond the VRT's bounds {tuple(vrt.bounds)}."
        raise ValueError(msg)
    window = Window(col0, row0, col1 - col0, row1 - row0)
    return VRT(
        crs=vrt.crs,
        transform=rasterio.windows.transform(window, vrt.transform),
        width=col1 - col0,
        height=row1 - row0,
        bands=[
            _reading(band, view, view_band, window)
            for band, (view, view_band) in zip(
                vrt.bands, _views(vrt, range(1, vrt.count + 1)), strict=True
            )
        ],
        metadata=dict(vrt.metadata),
    )


def _sample(colormap: Callable[[np.ndarray], Any], positions: np.ndarray) -> list[tuple[int, ...]]:
    """Return the 0-255 RGBA colors of ``colormap`` at ``positions`` from 0 to 1."""
    rgba = np.asarray(colormap(positions), dtype=float)
    channels = np.rint(np.clip(rgba, 0.0, 1.0) * 255).astype(int)
    return [tuple(int(channel) for channel in color) for color in channels]


def _colormap_stops(
    colormap: str | Callable[[np.ndarray], Any], vmin: float | None, vmax: float | None
) -> list[ColorStop]:
    """Turn a matplotlib colormap, or its name, into stops from ``vmin`` to ``vmax``."""
    if vmin is None or vmax is None:
        msg = "A colormap needs vmin and vmax."
        raise ValueError(msg)
    if not (math.isfinite(vmin) and math.isfinite(vmax) and vmin < vmax):
        msg = f"vmin must be below vmax, both finite; got {vmin} and {vmax}."
        raise ValueError(msg)
    if isinstance(colormap, str):
        try:
            # Imported by name: matplotlib is optional, so type checkers may not find it.
            mpl = importlib.import_module("matplotlib")
        except ImportError as e:
            msg = "Colormap names need matplotlib; install it, or pass color stops."
            raise ImportError(msg) from e
        if colormap not in mpl.colormaps:
            msg = f"matplotlib has no colormap named {colormap!r}."
            raise ValueError(msg)
        colormap = mpl.colormaps[colormap]
    count = getattr(colormap, "N", _COLORMAP_SAMPLES)
    if count >= _COLORMAP_SAMPLES:
        values = np.linspace(vmin, vmax, _COLORMAP_SAMPLES)
        colors = _sample(colormap, np.linspace(0.0, 1.0, _COLORMAP_SAMPLES))
        return [(float(v), color) for v, color in zip(values, colors, strict=True)]
    # Few colors, as in a listed colormap: one flat step each, so no two classes blend.
    # Matplotlib gives the value at an edge the color above it.
    edges = np.linspace(vmin, vmax, count + 1)
    colors = _sample(colormap, (np.arange(count) + 0.5) / count)
    stops: list[ColorStop] = []
    for k, color in enumerate(colors):
        end = edges[k + 1] if k == count - 1 else np.nextafter(edges[k + 1], -np.inf)
        stops += [(float(edges[k]), color), (float(end), color)]
    return stops


def _check_stops(stops: Sequence[ColorStop]) -> None:
    """Raise unless the stops are at least two, ascending, with 0-255 RGB(A) colors."""
    if len(stops) < 2:  # noqa: PLR2004  # a ramp needs two ends
        msg = f"colors needs at least two stops, got {len(stops)}."
        raise ValueError(msg)
    for value, color in stops:
        if len(color) not in {3, 4} or not all(
            isinstance(c, numbers.Integral) and 0 <= c <= _OPAQUE for c in color
        ):
            msg = f"The color of stop {value} must be 3 or 4 integers from 0 to 255, got {color}."
            raise ValueError(msg)
    values = [value for value, _ in stops]
    if not all(math.isfinite(v) for v in values) or any(
        later <= earlier for earlier, later in itertools.pairwise(values)
    ):
        msg = f"Stop values must be finite and strictly ascending, got {values}."
        raise ValueError(msg)


def colorize(
    vrt: VRT,
    colors: Sequence[ColorStop] | str | Callable[[np.ndarray], Any],
    *,
    vmin: float | None,
    vmax: float | None,
    band: int,
) -> VRT:
    """Render one band of ``vrt`` as RGBA, as ``VRT.colorize`` documents."""
    _check_band(vrt, band)
    if isinstance(colors, str) or callable(colors):
        stops = _colormap_stops(colors, vmin, vmax)
    else:
        if vmin is not None or vmax is not None:
            msg = "vmin and vmax apply to a colormap; color stops carry their own values."
            raise ValueError(msg)
        stops = list(colors)
        _check_stops(stops)
    source = vrt.bands[band - 1]
    # GDAL maps NaN through a leading NaN entry only; integer pixels are never NaN.
    head = [(math.nan, 0.0)] if source.dtype.startswith("float") else []
    tables = [
        head
        + [
            (float(value), float(color[i] if i < len(color) else _OPAQUE))
            for value, color in stops
        ]
        for i in range(4)
    ]
    ((view, view_band),) = _views(vrt, [band])
    return VRT(
        crs=vrt.crs,
        transform=vrt.transform,
        width=vrt.width,
        height=vrt.height,
        # Pixels equal to the nodata are skipped, so all four bands keep their initial 0.
        bands=[
            Band(
                dtype="uint8",
                sources=[Source(path=view, band=view_band, nodata=source.nodata, lut=table)],
                color_interp=interp,
            )
            for interp, table in zip(_RGBA, tables, strict=True)
        ],
    )
