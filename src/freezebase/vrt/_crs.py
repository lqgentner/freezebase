"""Short CRS labels, and parsing of CRSs serialized in VRT XML."""

from __future__ import annotations

from functools import lru_cache
import re

import pyproj
from rasterio.crs import CRS
from rasterio.errors import CRSError

_WKT = re.compile(r"[A-Z][A-Z0-9_]*\s*\[.*\]", re.DOTALL)
_EPSG = re.compile(r"EPSG:(\d+)", re.IGNORECASE)
_NAME = re.compile(r'"([^"]*)"')
_AXES_WITH_HEIGHT = 3


def crs_label(crs: CRS) -> str:
    """Name a CRS by its authority code, else by the name in its WKT."""
    return _wkt_label(crs.to_wkt())


@lru_cache(maxsize=256)
def _wkt_label(wkt: str) -> str:
    """Label a CRS given as WKT; cached, as identifying a CRS without a code is slow."""
    if (authority := CRS.from_wkt(wkt).to_authority()) is not None:
        return ":".join(authority)
    name = wkt_name(wkt)
    # A 3D CRS shares its name with the 2D one; its third axis is the ellipsoidal height.
    if (
        len(pyproj.CRS.from_wkt(wkt).axis_info) == _AXES_WITH_HEIGHT
        and "height" not in name.lower()
    ):
        return f"{name} + ellipsoidal height"
    return name


def wkt_name(wkt: str) -> str:
    """Return the name in a WKT, else the WKT."""
    match = _NAME.search(wkt)
    return match.group(1) if match else wkt


def parse_srs(text: str) -> CRS | None:
    """Parse a WKT or EPSG code; ``None`` for anything else, or if it does not parse."""
    # Not CRS.from_user_input: GDAL reads any other text as a file name or URL.
    text = text.strip()
    try:
        if _WKT.fullmatch(text):
            return CRS.from_wkt(text)
        if match := _EPSG.fullmatch(text):
            return CRS.from_epsg(int(match.group(1)))
    except CRSError:
        return None
    return None
