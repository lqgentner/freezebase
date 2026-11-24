"""Footprints of a VRT and its source files in degrees, and the SVG maps of its HTML repr."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
import html
from importlib import resources
import math
from typing import TYPE_CHECKING

import pyproj
from upath import UPath

if TYPE_CHECKING:
    from collections.abc import Callable

    from affine import Affine
    from rasterio.crs import CRS

    from freezebase.vrt._model import VRT

type Point = tuple[float, float]
type Ring = tuple[Point, ...]
type Window = tuple[float, float, float, float]

# Outlines of more source files cost more than they show.
MAX_OUTLINES = 64
# Bounds the nested VRTs walked to find source files.
MAX_VISITS = 40
_EDGE_POINTS = 8
# Semi-major axes of Earth's ellipsoids and spheres lie in this range, in meters.
_EARTH = (6.3e6, 6.4e6)
_ASPECT = 1.26
_STEPS = ("0.0001", "0.0002", "0.0005", "0.001", "0.002", "0.005", "0.01", "0.02", "0.05")
_STEPS += ("0.1", "0.2", "0.25", "0.5", "1", "2", "5", "10", "15", "30", "45", "90")
_MAX_LINES = 4
# Width of a label character, in font sizes.
_CHAR = 0.62
TO = " \N{EN DASH} "

LAND = resources.files("freezebase.vrt").joinpath("_land.txt").read_text(encoding="utf-8")
LAND = LAND.strip()


@dataclass(frozen=True)
class Outline:
    """A source file's outline in degrees, with the path shown on hover."""

    title: str
    ring: Ring


@dataclass(frozen=True)
class Footprint:
    """The VRT's outline in degrees, the outlines of its source files and how many it left out."""

    ring: Ring
    sources: tuple[Outline, ...] = ()
    more_sources: int = 0
    plain_degrees: bool = False


@dataclass(frozen=True)
class _Converter:
    """Converts coordinates of one CRS to degrees east and north on its own datum."""

    transformer: pyproj.Transformer
    to_degrees: float
    meridian: float
    # Whether the CRS already is in degrees east of Greenwich and north.
    plain: bool


class _Degrees:
    """Builds outlines in degrees, keeping one converter per CRS."""

    def __init__(self) -> None:
        self.converters: dict[str, _Converter | None] = {}

    def converter(self, crs: CRS) -> _Converter | None:
        """Return the converter of ``crs``, or ``None`` if it has no geographic datum."""
        wkt = crs.to_wkt()
        if wkt not in self.converters:
            self.converters[wkt] = _converter(wkt)
        return self.converters[wkt]

    def ring(self, crs: CRS | None, transform: Affine, window: Window, points: int) -> Ring | None:
        """Outline a pixel window of a grid, or ``None`` if it has no place on the globe."""
        if crs is None or (converter := self.converter(crs)) is None:
            return None
        col, row, width, height = window
        corners = [
            (col, row),
            (col + width, row),
            (col + width, row + height),
            (col, row + height),
        ]
        pixels = [
            (c0 + (c1 - c0) * step / points, r0 + (r1 - r0) * step / points)
            for (c0, r0), (c1, r1) in zip(corners, corners[1:] + corners[:1], strict=True)
            for step in range(points)
        ]
        a, b, c, d, e, f = transform[:6]
        xs = [a * x + b * y + c for x, y in pixels]
        ys = [d * x + e * y + f for x, y in pixels]
        lons, lats = converter.transformer.transform(xs, ys)
        ring = _centered(
            tuple(
                (lon * converter.to_degrees + converter.meridian, lat * converter.to_degrees)
                for lon, lat in zip(lons, lats, strict=True)
            )
        )
        return ring if _drawable(ring) else None


def _converter(wkt: str) -> _Converter | None:
    """Convert to the CRS's own longitude and latitude: a conversion, which needs no grid."""
    crs = pyproj.CRS.from_wkt(wkt)
    geodetic = crs.geodetic_crs
    # A rotated-pole CRS is derived from the geographic CRS whose degrees the map needs.
    while geodetic is not None and geodetic.is_derived and geodetic.source_crs is not None:
        geodetic = geodetic.source_crs
    if geodetic is None or not geodetic.is_geographic:
        return None
    meridian, ellipsoid = geodetic.prime_meridian, geodetic.ellipsoid
    if meridian is None or ellipsoid is None:
        return None
    if not _EARTH[0] <= ellipsoid.semi_major_metre <= _EARTH[1]:
        return None  # another body, which the world map would misplace
    to_degrees = math.degrees(geodetic.axis_info[0].unit_conversion_factor)
    # pyproj types the factor as str, though it is a float.
    offset = math.degrees(meridian.longitude * float(meridian.unit_conversion_factor))
    plain = crs.is_geographic and not crs.is_derived and to_degrees == 1 and offset == 0
    return _Converter(
        pyproj.Transformer.from_crs(crs, geodetic, always_xy=True), to_degrees, offset, plain
    )


def _centered(ring: Ring) -> Ring:
    """Shift longitudes by whole turns so the ring centers within ±180°; center a global one."""
    lons = [lon for lon, _ in ring]
    if not all(math.isfinite(lon) for lon in lons):
        return ring
    low, high = min(lons), max(lons)
    center = (low + high) / 2
    if high - low >= 360 - 1e-6:
        return tuple((min(max(lon - center, -180), 180), lat) for lon, lat in ring)
    shift = -360 * math.floor((center + 180) / 360)
    return tuple((lon + shift, lat) for lon, lat in ring)


def _drawable(ring: Ring) -> bool:
    """Whether a ring is finite, within ±180°, and neither crosses the antimeridian nor a pole."""
    if not all(
        math.isfinite(lon) and math.isfinite(lat) and abs(lat) <= 90 and abs(lon) <= 180  # noqa: PLR2004
        for lon, lat in ring
    ):
        return False
    pairs = zip(ring, ring[1:] + ring[:1], strict=True)
    return all(abs(lon1 - lon0) <= 180 for (lon0, _), (lon1, _) in pairs)  # noqa: PLR2004


def footprint(vrt: VRT, title: Callable[[UPath], str]) -> Footprint | None:
    """Outline ``vrt`` and the files it places, or ``None`` if it has no place on the globe."""
    degrees = _Degrees()
    ring = degrees.ring(vrt.crs, vrt.transform, (0, 0, vrt.width, vrt.height), _EDGE_POINTS)
    if ring is None or vrt.crs is None:
        return None
    converter = degrees.converter(vrt.crs)
    outlines, more = _source_outlines(vrt, degrees, title)
    return Footprint(ring, tuple(outlines), more, converter is not None and converter.plain)


def _source_outlines(
    vrt: VRT, degrees: _Degrees, title: Callable[[UPath], str]
) -> tuple[list[Outline], int]:
    """Outline the files placed by the VRTs in ``vrt``; a warp records no source size."""
    outlines = _Outlines(degrees, title)
    seen: set[int] = set()
    queue: list[VRT] = [vrt]
    while queue and len(seen) < MAX_VISITS:
        node = queue.pop(0)
        if id(node) in seen:
            continue
        seen.add(id(node))
        if node.warp is not None:
            if not isinstance(node.warp.source, UPath):
                queue.append(node.warp.source)
            continue
        for band in node.bands:
            for source in band.sources:
                if isinstance(source.path, UPath):
                    window = source.dst_window
                    placed = None
                    if window is not None:
                        placed = (window.col_off, window.row_off, window.width, window.height)
                    outlines.add(node, source.path, placed)
                else:
                    queue.append(source.path)
    return outlines.found, outlines.more


class _Outlines:
    """Collects the outlines of distinct source files, up to MAX_OUTLINES."""

    def __init__(self, degrees: _Degrees, title: Callable[[UPath], str]) -> None:
        self.degrees = degrees
        self.title = title
        self.found: list[Outline] = []
        self.keys: set[tuple[str, str, tuple[float, ...], Window]] = set()
        self.more = 0

    def add(self, node: VRT, path: UPath, window: Window | None) -> None:
        """Outline ``path`` where ``node`` places it, the whole grid without a window."""
        grid = (0, 0, node.width, node.height) if window is None else window
        # By place, so equal copies of a nested VRT outline each file once.
        crs = "" if node.crs is None else node.crs.to_wkt()
        key = (str(path), crs, tuple(node.transform[:6]), grid)
        if key in self.keys:
            return
        self.keys.add(key)
        if len(self.found) == MAX_OUTLINES:
            self.more += 1
            return
        ring = self.degrees.ring(node.crs, node.transform, grid, 1)
        if ring is not None:
            self.found.append(Outline(self.title(path), ring))


def degree_range(values: list[float], positive: str, negative: str) -> str:
    """Format the range of ``values`` in degrees with hemisphere letters."""
    low, high = min(values), max(values)
    span = high - low
    decimals = 6 if span <= 0 else max(2, min(6, 2 - math.floor(math.log10(span))))
    ends = (_degrees(value, decimals, positive, negative) for value in (low, high))
    return TO.join(ends)


def _degrees(value: float, decimals: int, positive: str, negative: str, gap: str = " ") -> str:
    """Format one angle in degrees with its hemisphere letter."""
    value = round(value, decimals) + 0.0
    letter = "" if value == 0 else f"{gap}{positive if value > 0 else negative}"
    return f"{abs(value):.{decimals}f}°{letter}"


def world_svg(fp: Footprint) -> str:
    """Draw a world map marking where the footprint lies."""
    lons = [lon + 180 for lon, _ in fp.ring]
    lats = [90 - lat for _, lat in fp.ring]
    cx, cy = (min(lons) + max(lons)) / 2, (min(lats) + max(lats)) / 2
    width, height = max(max(lons) - min(lons), 6), max(max(lats) - min(lats), 5)
    return (
        '<svg class="fzb-vrt-world" width="280" height="140" viewBox="0 0 360 180" '
        'role="img" aria-label="Location on a world map">'
        f'<path class="fzb-vrt-land" d="{LAND}"/>'
        f'<path class="fzb-vrt-cross" d="M{cx:.1f} 0V180M0 {cy:.1f}H360"/>'
        f'<rect class="fzb-vrt-spot" x="{cx - width / 2:.1f}" y="{cy - height / 2:.1f}" '
        f'width="{width:.1f}" height="{height:.1f}"/></svg>'
    )


def _nice_step(span: float) -> Decimal:
    """Return the smallest grid step that draws at most a few lines across ``span``."""
    for text in _STEPS:
        if span / float(text) <= _MAX_LINES:
            return Decimal(text)
    return Decimal(_STEPS[-1])


def _expand(box: list[float]) -> list[float]:
    """Grow ``[left, top, width, height]`` around its center to the plot's aspect ratio."""
    left, top, width, height = box
    if width < height * _ASPECT:
        left -= (height * _ASPECT - width) / 2
        width = height * _ASPECT
    else:
        top -= (width / _ASPECT - height) / 2
        height = width / _ASPECT
    return [left, top, width, height]


def zoom_svg(fp: Footprint) -> str:
    """Draw the footprint and its source files on a graticule, north up."""
    rings = [fp.ring, *(outline.ring for outline in fp.sources)]
    lats = [lat for ring in rings for _, lat in ring]
    k = max(math.cos(math.radians((min(lats) + max(lats)) / 2)), 0.05)
    xs = [lon * k for ring in rings for lon, _ in ring]
    ys = [-lat for lat in lats]
    size = max(max(xs) - min(xs), max(ys) - min(ys), 1e-6)
    pad = size * 0.1
    box = _expand(
        [min(xs) - pad, min(ys) - pad, max(xs) - min(xs) + 2 * pad, max(ys) - min(ys) + 2 * pad]
    )
    font = box[3] * 0.05
    lat_step = _nice_step(box[3])
    widest = max(abs(box[1]), abs(box[1] + box[3]))
    chars = len(_degrees(widest, _places(lat_step), "N", "S", gap=""))
    # Room for the latitude labels on the left and the longitude labels below.
    gutter, margin = font * (_CHAR * chars + 0.5), font * 1.6
    box = _expand([box[0] - gutter, box[1], box[2] + gutter, box[3] + margin])
    left, top, width, height = box
    decimals = max(0, 3 - math.floor(math.log10(height)))

    def num(value: float) -> str:
        """Format a length to the plot's precision."""
        return f"{value:.{decimals}f}"

    # Coordinates count from the plot's corner: SVG draws in single precision, which
    # degrees of a small footprint would exhaust.
    def at_x(x: float) -> str:
        """Format an x coordinate from the plot's left edge."""
        return num(x - left)

    def at_y(y: float) -> str:
        """Format a y coordinate from the plot's top edge."""
        return num(y - top)

    def points(ring: Ring) -> str:
        """Format a ring as SVG points."""
        return " ".join(f"{at_x(lon * k)},{at_y(-lat)}" for lon, lat in ring)

    lines: list[str] = []
    labels: list[str] = []
    lon_step = _nice_step(width / k)
    lon = math.ceil(left / k / float(lon_step))
    while lon * float(lon_step) * k <= left + width:
        x = lon * float(lon_step) * k
        lines.append(f"M{at_x(x)} {num(-height / 2)}V{num(height * 1.5)}")
        text = _degrees(float(lon * lon_step), _places(lon_step), "E", "W", gap="")
        if left + gutter < x and x + font * (_CHAR * len(text) + 0.2) < left + width:
            base = num(height - font * 0.45)
            labels.append(f'<text x="{at_x(x + font * 0.2)}" y="{base}">{text}</text>')
        lon += 1
    lat = math.ceil(-(top + height) / float(lat_step))
    while -lat * float(lat_step) >= top:
        y = -lat * float(lat_step)
        lines.append(f"M{num(-width / 2)} {at_y(y)}H{num(width * 1.5)}")
        if top + font < y < top + height - margin:
            text = _degrees(float(lat * lat_step), _places(lat_step), "N", "S", gap="")
            labels.append(f'<text x="{num(font * 0.2)}" y="{at_y(y - font * 0.3)}">{text}</text>')
        lat += 1
    # One opacity for the group, so the shared edges of adjacent files draw no darker.
    sources = "".join(
        f'<polygon points="{points(outline.ring)}"><title>{html.escape(outline.title)}</title>'
        "</polygon>"
        for outline in fp.sources
    )
    return (
        f'<svg class="fzb-vrt-zoom" viewBox="0 0 {num(width)} {num(height)}" '
        'role="img" aria-label="Footprint of the VRT and its source files">'
        f'<path class="fzb-vrt-graticule" d="{"".join(lines)}"/>'
        f'<g class="fzb-vrt-src">{sources}</g>'
        f'<polygon class="fzb-vrt-out" points="{points(fp.ring)}"/>'
        f'<g class="fzb-vrt-labels" font-size="{num(font)}">{"".join(labels)}</g></svg>'
    )


def _places(step: Decimal) -> int:
    """Return how many decimals a multiple of ``step`` needs."""
    exponent = step.normalize().as_tuple().exponent
    return max(0, -exponent) if isinstance(exponent, int) else 0
