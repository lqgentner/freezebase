# A standalone script, not a package module.
# ruff: noqa: INP001
"""Write the world land outline the VRT HTML repr draws, from Natural Earth's 110 m land.

Download https://naciscdn.org/naturalearth/110m/physical/ne_110m_land.zip (public domain),
then run ``uv run python scripts/land_outline.py ne_110m_land.zip``.
"""

from __future__ import annotations

from itertools import pairwise
import math
from pathlib import Path
import struct
import sys
import zipfile

OUTPUT = Path(__file__).parents[1] / "src" / "freezebase" / "vrt" / "_land.txt"
# In degrees: about one pixel of the 280 px wide map.
TOLERANCE = 1.0
MIN_AREA = 1.0
_POLYGON = 5
_HEADER = 100

type Point = tuple[float, float]


def rings(shp: bytes) -> list[list[Point]]:
    """Read the rings of the polygons in a shapefile's main file."""
    found: list[list[Point]] = []
    offset = _HEADER
    while offset < len(shp):
        (length,) = struct.unpack(">i", shp[offset + 4 : offset + 8])
        content = shp[offset + 8 : offset + 8 + 2 * length]
        offset += 8 + 2 * length
        if struct.unpack("<i", content[:4])[0] != _POLYGON:
            continue
        parts, points = struct.unpack("<2i", content[36:44])
        starts = [*struct.unpack(f"<{parts}i", content[44 : 44 + 4 * parts]), points]
        base = 44 + 4 * parts
        coords = struct.unpack(f"<{2 * points}d", content[base : base + 16 * points])
        pairs = list(zip(coords[::2], coords[1::2], strict=True))
        found += [pairs[start:end] for start, end in pairwise(starts)]
    return found


def simplify(points: list[Point], tolerance: float) -> list[Point]:
    """Simplify a line with the Douglas-Peucker algorithm."""
    keep = {0, len(points) - 1}
    stack = [(0, len(points) - 1)]
    while stack:
        first, last = stack.pop()
        (x0, y0), (x1, y1) = points[first], points[last]
        norm = math.hypot(x1 - x0, y1 - y0)
        best, index = 0.0, 0
        for i in range(first + 1, last):
            x, y = points[i]
            if norm:
                distance = abs((x1 - x0) * (y0 - y) - (x0 - x) * (y1 - y0)) / norm
            else:
                distance = math.hypot(x - x0, y - y0)
            if distance > best:
                best, index = distance, i
        if best > tolerance:
            keep.add(index)
            stack += [(first, index), (index, last)]
    return [points[i] for i in sorted(keep)]


def area(points: list[tuple[int, int]]) -> float:
    """Return the unsigned area of a ring."""
    pairs = zip(points, points[1:] + points[:1], strict=True)
    return abs(sum(x0 * y1 - x1 * y0 for (x0, y0), (x1, y1) in pairs)) / 2


def path(all_rings: list[list[Point]]) -> str:
    """Return an SVG path of the rings in a 360 by 180 frame, north up."""
    parts: list[str] = []
    for ring in all_rings:
        frame: list[tuple[int, int]] = []
        for lon, lat in simplify(ring, TOLERANCE):
            point = (round(lon + 180), round(90 - lat))
            if not frame or frame[-1] != point:
                frame.append(point)
        if len(frame) > 1 and frame[0] == frame[-1]:
            frame.pop()
        if len(frame) >= 3 and area(frame) >= MIN_AREA:  # noqa: PLR2004  # a polygon
            parts.append("M" + "L".join(f"{x} {y}" for x, y in frame) + "Z")
    return "".join(parts)


def main() -> None:
    """Write the outline from the zip given on the command line."""
    with zipfile.ZipFile(sys.argv[1]) as archive:
        name = next(item for item in archive.namelist() if item.endswith(".shp"))
        outline = path(rings(archive.read(name)))
    OUTPUT.write_text(outline + "\n", encoding="utf-8")
    print(f"wrote {len(outline)} characters to {OUTPUT}")


if __name__ == "__main__":
    main()
