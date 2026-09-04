"""MGRS grid generation and manipulation."""

from __future__ import annotations

from copy import copy
import re
from typing import TYPE_CHECKING, Any, cast, overload, override

from affine import Affine
import geopandas as gpd
from geopandas import GeoSeries
import mgrs
import numpy as np
from odc.geo.geobox import GeoBox
import pandas as pd
from pyproj import CRS, Transformer
import shapely
from shapely import box, union_all
from shapely.geometry.base import BaseGeometry

if TYPE_CHECKING:
    from collections.abc import Iterator
    from typing import Any, Literal, Self

    from numpy.typing import ArrayLike, NDArray
    from odc.geo.geobox import SomeShape

    type GeoArray = NDArray[np.object_]

    Hemisphere = Literal["N", "S"]

WGS84 = CRS.from_epsg(4326)

_MIN_UTM_ZONE = 1
_MAX_UTM_ZONE = 60
_VALID_HEMISPHERES = ("N", "S")


def _validate_hemisphere(hemisphere: ArrayLike) -> None:
    """Raise ``ValueError`` unless every hemisphere value is exactly ``'N'`` or ``'S'``."""
    arr = np.asarray(hemisphere)
    valid = np.isin(arr, _VALID_HEMISPHERES)
    if not valid.all():
        bad = np.unique(arr[~valid]).tolist()
        msg = f"Hemisphere must be one of {_VALID_HEMISPHERES}, got {bad}."
        raise ValueError(msg)


def _validate_zone(zone: ArrayLike) -> None:
    """Raise ``ValueError`` unless every UTM zone is in the range 1-60."""
    arr = np.asarray(zone)
    out_of_range = (arr < _MIN_UTM_ZONE) | (arr > _MAX_UTM_ZONE)
    if out_of_range.any():
        bad = np.unique(arr[out_of_range]).tolist()
        msg = f"UTM zone must be in [{_MIN_UTM_ZONE}, {_MAX_UTM_ZONE}], got {bad}."
        raise ValueError(msg)


@overload
def transform_shapely(
    geom: GeoArray,
    crs_from: CRS,
    crs_to: CRS,
) -> GeoArray: ...


@overload
def transform_shapely(
    geom: BaseGeometry,
    crs_from: CRS,
    crs_to: CRS,
) -> BaseGeometry: ...


def transform_shapely(
    geom: GeoArray | BaseGeometry,
    crs_from: CRS,
    crs_to: CRS,
) -> GeoArray | BaseGeometry:
    """Transform Shapely geometries between CRSs.

    Parameters
    ----------
    geom : BaseGeometry | GeoArray
        Geometry or geometry array.
    crs_from : CRS
        Source CRS.
    crs_to : CRS
        Target CRS.

    Returns
    -------
    BaseGeometry | GeoArray
        Transformed geometry or array.
    """
    transformer = Transformer.from_crs(crs_from=crs_from, crs_to=crs_to, always_xy=True)
    # `interleaved=False` selects the (x, y) -> (x, y) transformer overload, which
    # shapely's stubs do not model.
    # pyrefly: ignore [no-matching-overload]
    return shapely.transform(geom, transformer.transform, interleaved=False)


_SOUTHERN_BANDS: frozenset[str] = frozenset("CDEFGHJKLM")

_ZONE_31 = 31
_ZONE_32 = 32
_ZONE_33 = 33
_ZONE_34 = 34
_ZONE_35 = 35
_ZONE_36 = 36
_ZONE_37 = 37

_NORTHERN_EXCEPTION_ZONES = {
    _ZONE_31,
    _ZONE_32,
    _ZONE_33,
    _ZONE_34,
    _ZONE_35,
    _ZONE_36,
    _ZONE_37,
}


def utms_to_epsgs(zone: ArrayLike, hemisphere: ArrayLike) -> NDArray[np.integer]:
    """Convert UTM zones and hemispheres to EPSG codes.

    Inputs follow NumPy broadcasting rules.

    Returns
    -------
    NDArray[np.integer]
        EPSG codes (326xx for north, 327xx for south).

    Raises
    ------
    ValueError
        If any zone is outside 1-60 or any hemisphere is not ``'N'``/``'S'``.
    """
    zone_arr = np.asarray(zone)
    hemisphere_arr = np.asarray(hemisphere)
    _validate_zone(zone_arr)
    _validate_hemisphere(hemisphere_arr)
    is_south = hemisphere_arr == "S"
    base = np.where(is_south, 32700, 32600)
    return np.asarray(base + zone_arr)


def utm_to_crs(zone: int, hemisphere: Hemisphere) -> CRS:
    """Return the CRS for a UTM zone and hemisphere.

    Returns
    -------
    CRS
        UTM CRS.

    Raises
    ------
    ValueError
        If ``zone`` is outside 1-60 or ``hemisphere`` is not ``'N'``/``'S'``.
    """
    _validate_zone(zone)
    _validate_hemisphere(hemisphere)
    base = 32700 if hemisphere == "S" else 32600
    epsg = base + zone
    return CRS.from_epsg(epsg)


# Zone, band, 100 km square, then paired precision digits.
_MGRS_RE = re.compile(
    r"^(?P<zone>\d{1,2})(?P<band>[C-HJ-NP-X])(?P<square>[A-HJ-NP-Z]{2})(?P<digits>\d*)$",
    re.IGNORECASE,
)
_MGRS_10KM_DIGITS = 2


def _parse_mgrs(mgrs_code: str) -> re.Match[str]:
    """Parse an MGRS reference, raising ``ValueError`` on malformed input."""
    match = _MGRS_RE.match(mgrs_code.strip())
    if match is None:
        msg = f"Malformed MGRS reference: {mgrs_code!r}"
        raise ValueError(msg)
    digits = match.group("digits")
    if len(digits) % 2 != 0:
        msg = f"MGRS reference {mgrs_code!r} has an odd number of location digits."
        raise ValueError(msg)
    return match


def mgrs_to_crs(mgrs_code: str) -> CRS:
    """Return the UTM CRS for an MGRS reference.

    Parameters
    ----------
    mgrs_code : str
        MGRS reference such as ``"32TMT64"``.

    Returns
    -------
    CRS
        UTM CRS.

    Raises
    ------
    ValueError
        If the reference is malformed or the zone is outside 1-60.
    """
    match = _parse_mgrs(mgrs_code)
    zone = int(match.group("zone"))
    _validate_zone(zone)
    band = match.group("band").upper()
    epsg = (32700 if band in _SOUTHERN_BANDS else 32600) + zone
    return CRS.from_epsg(epsg)


class MGRSGeoBox(GeoBox):
    """A 10 x 10 km MGRS square backed by ``odc.geo.GeoBox``.

    Parameters
    ----------
    mgrs_code : str
        MGRS reference.
    shape : SomeShape
        Raster dimensions.
    affine : Affine
        Pixel-to-CRS transform.
    crs : CRS
        Cell CRS.
    """

    def __init__(
        self,
        mgrs_code: str,
        shape: SomeShape,
        affine: Affine,
        crs: CRS,
    ) -> None:
        self.mgrs_code = mgrs_code
        super().__init__(shape, affine, crs)

    @classmethod
    def from_mgrs(cls, mgrs_code: str, resolution: float = 10.0) -> Self:
        """Create a grid square from a 10 km MGRS reference.

        Parameters
        ----------
        mgrs_code : str
            MGRS reference.
        resolution : float
            Pixel size in metres.

        Returns
        -------
        MGRSGeoBox
            Grid square.

        Raises
        ------
        ValueError
            If ``resolution`` is not positive, does not evenly divide the 10 km
            grid square, or ``mgrs_code`` is not a 10 km reference (two location
            digits, e.g. ``"32TMT64"``).
        """
        if resolution <= 0:
            msg = f"resolution must be positive, got {resolution}."
            raise ValueError(msg)
        if not (_10KM_SIZE / resolution).is_integer():
            msg = f"resolution {resolution} must evenly divide the 10 km cell size."
            raise ValueError(msg)
        match = _parse_mgrs(mgrs_code)
        if len(match.group("digits")) != _MGRS_10KM_DIGITS:
            msg = (
                f"{mgrs_code!r} is not a 10 km MGRS reference "
                f"(expected {_MGRS_10KM_DIGITS} location digits)."
            )
            raise ValueError(msg)

        converter = mgrs.MGRS()
        zone, hemisphere, easting, northing = converter.MGRSToUTM(mgrs_code)
        # ``mgrs`` types the hemisphere as a plain ``str``; only 'N'/'S' occur,
        # and ``utm_to_crs`` rejects anything else at runtime.
        crs = utm_to_crs(zone, cast("Hemisphere", hemisphere))

        n_pixels = int(_10KM_SIZE / resolution)
        nw_northing = northing + _10KM_SIZE
        affine = Affine(resolution, 0, easting, 0, -resolution, nw_northing)

        return cls(mgrs_code, (n_pixels, n_pixels), affine, crs)

    @override
    def __repr__(self) -> str:
        """Return a string representation of the MGRSGeoBox."""
        return f"MGRSGeoBox({self.mgrs_code!r})"


class UTMZones:
    """All UTM zones in WGS84, including Norway and Svalbard exceptions."""

    def __init__(self) -> None:
        self.gdf = self._build_zones()

    def get_zone_geometry(self, zone: int, hemisphere: Hemisphere) -> BaseGeometry:
        """Return the WGS84 geometry of a single UTM zone.

        Raises
        ------
        ValueError
            If ``zone``/``hemisphere`` are invalid or name no known zone.
        """
        _validate_zone(zone)
        _validate_hemisphere(hemisphere)
        gdf = self.gdf
        matches = gdf[(gdf.zone == zone) & (gdf.hemisphere == hemisphere)]
        if matches.empty:
            msg = f"No UTM zone found for zone={zone!r}, hemisphere={hemisphere!r}."
            raise ValueError(msg)
        return matches.iloc[0].geometry

    def find_intersecting(self, geometry: GeoArray | GeoSeries | BaseGeometry) -> gpd.GeoDataFrame:
        """Return UTM zones intersecting one or more geometries."""
        result = self.gdf.sindex.query(geometry, predicate="intersects")
        hits = result[1] if result.ndim == 2 else result  # noqa: PLR2004
        mask = pd.Series(data=False, index=self.gdf.index)
        mask.iloc[hits] = True

        return self.gdf[mask.to_numpy()]

    def _build_zones(self) -> gpd.GeoDataFrame:
        """Generate a GeoDataFrame containing all UTM zones."""
        parts: list[dict[str, Any]] = []

        for zone in range(1, 61):
            west = (zone - 1) * 6 - 180
            east = west + 6

            parts.append(
                {
                    "name": f"{zone}S",
                    "zone": zone,
                    "hemisphere": "S",
                    "epsg": utms_to_epsgs(zone, "S"),
                    "geometry": box(west, -80, east, 0),
                },
            )

            if zone in _NORTHERN_EXCEPTION_ZONES:
                geom = self._build_north_exception(zone, west, east)
            else:
                geom = box(west, 0, east, 84)

            parts.append(
                {
                    "name": f"{zone}N",
                    "zone": zone,
                    "hemisphere": "N",
                    "epsg": utms_to_epsgs(zone, "N"),
                    "geometry": geom,
                },
            )

        return gpd.GeoDataFrame(parts, crs=WGS84)

    @classmethod
    def _build_north_exception(cls, zone: int, west: float, east: float) -> BaseGeometry:
        """Build a northern zone with Norway and Svalbard exceptions.

        Parameters
        ----------
        zone : int
            UTM zone in the range 31-37.
        west : float
            Standard western longitude.
        east : float
            Standard eastern longitude.

        Returns
        -------
        Polygon
            Zone geometry in WGS84.
        """
        parts = [box(west, 0, east, 56)]

        if zone == _ZONE_31:
            parts.append(box(west, 56, 3, 64))
        elif zone == _ZONE_32:
            parts.append(box(3, 56, east, 64))
        else:
            parts.append(box(west, 56, east, 64))

        parts.append(box(west, 64, east, 72))

        if zone in (_ZONE_32, _ZONE_34, _ZONE_36):
            pass
        elif zone == _ZONE_31:
            parts.append(box(west, 72, 9, 84))
        elif zone == _ZONE_33:
            parts.append(box(9, 72, 21, 84))
        elif zone == _ZONE_35:
            parts.append(box(21, 72, 33, 84))
        elif zone == _ZONE_37:
            parts.append(box(33, 72, east, 84))

        return union_all(parts)


_100KM_SIZE = 100_000
_10KM_SIZE = 10_000
_FALSE_EASTING = 500_000
_FALSE_NORTHING_SOUTH = 10_000_000
_MGRS_PRECISION_10KM = 1


class MGRSGrid:
    """The 10 km MGRS squares intersecting an area of interest.

    Adjacent UTM zones may overlap because each zone is gridded independently.

    Parameters
    ----------
    filter_geometry : GeoSeries | GeoArray | BaseGeometry
        Area of interest in WGS84. Intersecting squares are included whole.
    resolution : float, default 10.0
        Pixel size in metres. Must divide 10 km evenly.

    Raises
    ------
    ValueError
        If ``filter_geometry`` carries a CRS other than WGS84.

    """

    def __init__(
        self,
        filter_geometry: GeoSeries | GeoArray | BaseGeometry,
        resolution: float = 10.0,
    ) -> None:
        self._filter_tree = self._strtree_from_geometry(filter_geometry)
        self.resolution = resolution

        self._fine_template = self._make_subdivision_template()
        self.utm_zones = UTMZones()

        self.northings: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self.eastings: NDArray[np.int32] = np.empty(0, dtype=np.int32)
        self.zones: NDArray[np.uint8] = np.empty(0, dtype=np.uint8)
        self.hemispheres: NDArray[np.str_] = np.empty(0, dtype=np.str_)
        self.geometries: NDArray[np.object_] = np.empty(0, dtype=np.object_)

        self._materialize()

    @staticmethod
    def _strtree_from_geometry(
        geometry: gpd.GeoDataFrame | GeoSeries | GeoArray | BaseGeometry,
    ) -> shapely.STRtree:
        """Create a new STRtree spatial index from a geometry."""
        if hasattr(geometry, "crs") and geometry.crs != WGS84:
            name = geometry.crs.name if isinstance(geometry.crs, CRS) else str(geometry.crs)
            msg = f"`filter_geometry must be '{WGS84.name}', got '{name}'"
            raise ValueError(msg)
        if isinstance(geometry, gpd.GeoDataFrame):
            geoms = geometry.geometry.to_numpy()
        elif isinstance(geometry, GeoSeries):
            geoms = geometry.to_numpy()
        elif isinstance(geometry, BaseGeometry):
            geoms = np.array([geometry])
        elif isinstance(geometry, np.ndarray):
            geoms = geometry
        else:
            geoms = np.asarray(geometry, dtype=object)
        # STRtree ignores None but not empty geometries.
        non_empty = geoms.copy()
        non_empty[shapely.is_empty(non_empty)] = None
        return shapely.STRtree(non_empty)

    def _make_subdivision_template(self) -> NDArray[np.int32]:
        """Create a template of 10 km cells anchored at origin (0, 0).

        Returns
        -------
        NDArray[np.int32]
            Array of shape (100, 4) with columns ``[x_min, y_min, x_max, y_max]``
            for each 10 km cell within a single 100 km cell.
        """
        n = int(_100KM_SIZE / _10KM_SIZE)
        offsets = np.arange(n, dtype=np.int32) * _10KM_SIZE
        xx, yy = np.meshgrid(offsets, offsets)
        sw_x = xx.ravel()
        sw_y = yy.ravel()
        return np.column_stack([sw_x, sw_y, sw_x + _10KM_SIZE, sw_y + _10KM_SIZE])

    def _intersects_any(self, geoms: GeoArray) -> NDArray[np.bool_]:
        """Return a boolean mask of geometries that intersect any filter geometry."""
        hits = self._filter_tree.query(geoms, predicate="intersects")[0]
        mask = np.zeros(len(geoms), dtype=np.bool_)
        mask[hits] = True
        return mask

    def _materialize(self) -> None:
        filter_geom = self._filter_tree.geometries
        matched_utm_zones = self.utm_zones.find_intersecting(filter_geom)
        for _, zone_row in matched_utm_zones.iterrows():
            zone: int = zone_row["zone"]
            hemisphere: Hemisphere = zone_row["hemisphere"]
            self._materialize_zone(zone, hemisphere)

        mgrs_codes = self._compute_mgrs_codes(
            self.eastings,
            self.northings,
            zones=self.zones,
            hemispheres=self.hemispheres,
        )
        idxs = np.argsort(mgrs_codes)
        self.mgrs_codes = mgrs_codes[idxs]
        self.eastings = self.eastings[idxs]
        self.northings = self.northings[idxs]
        self.zones = self.zones[idxs]
        self.hemispheres = self.hemispheres[idxs]
        self.geometries = self.geometries[idxs]

    def _materialize_zone(self, zone: int, hemisphere: Hemisphere) -> None:
        utm_crs = utm_to_crs(zone, hemisphere)
        zone_geom_wgs = self.utm_zones.get_zone_geometry(zone, hemisphere)
        coarse_x, coarse_y = self._build_coarse_coords(zone, hemisphere)
        coarse_boxes_utm = shapely.box(
            coarse_x,
            coarse_y,
            coarse_x + _100KM_SIZE,
            coarse_y + _100KM_SIZE,
        )
        coarse_boxes_wgs = transform_shapely(coarse_boxes_utm, utm_crs, WGS84)
        intersects_geom = self._intersects_any(coarse_boxes_wgs)
        intersects_zone = shapely.intersects(coarse_boxes_wgs, zone_geom_wgs)
        mask = intersects_geom & intersects_zone
        coarse_x = coarse_x[mask]
        coarse_y = coarse_y[mask]

        fine_x, fine_y = self._subdivide_coarse_coords(coarse_x, coarse_y)
        fine_boxes_utm = shapely.box(
            fine_x,
            fine_y,
            fine_x + _10KM_SIZE,
            fine_y + _10KM_SIZE,
        )
        fine_boxes_wgs = transform_shapely(fine_boxes_utm, utm_crs, WGS84)
        intersects_geom = self._intersects_any(fine_boxes_wgs)
        intersects_zone = shapely.intersects(fine_boxes_wgs, zone_geom_wgs)
        mask = intersects_geom & intersects_zone

        n = len(mask[mask])
        self.eastings = np.concat([self.eastings, fine_x[mask]])
        self.northings = np.concat([self.northings, fine_y[mask]])
        self.zones = np.concat([self.zones, np.full(n, zone, dtype=np.uint8)])
        self.hemispheres = np.concat([self.hemispheres, np.full(n, hemisphere, dtype=np.str_)])
        self.geometries = np.concat([self.geometries, fine_boxes_wgs[mask]])

    def _build_coarse_coords(
        self,
        zone: int,
        hemisphere: Hemisphere,
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Build 100 km-spaced coordinates in local UTM space.

        Parameters
        ----------
        zone : int
            UTM zone.
        hemisphere : Hemisphere
            UTM hemisphere.

        Returns
        -------
        tuple[NDArray[np.int32], NDArray[np.int32]]
            100 km-spaced eastings and northings.
        """
        cell_size = _100KM_SIZE
        zone_geom_wgs = self.utm_zones.get_zone_geometry(zone=zone, hemisphere=hemisphere)
        utm_crs = utm_to_crs(zone, hemisphere)
        zone_geom_utm = transform_shapely(zone_geom_wgs, WGS84, utm_crs)
        minx, miny, maxx, maxy = zone_geom_utm.bounds

        false_northing = _FALSE_NORTHING_SOUTH if hemisphere == "S" else 0

        easting_start = _FALSE_EASTING + np.floor((minx - _FALSE_EASTING) / cell_size) * cell_size
        northing_start = false_northing + np.floor((miny - false_northing) / cell_size) * cell_size
        easting_end = _FALSE_EASTING + np.ceil((maxx - _FALSE_EASTING) / cell_size) * cell_size
        northing_end = false_northing + np.ceil((maxy - false_northing) / cell_size) * cell_size

        easting_coords = np.arange(easting_start, easting_end, cell_size, dtype=np.int32)
        northing_coords = np.arange(northing_start, northing_end, cell_size, dtype=np.int32)

        easting_grid, northing_grid = np.meshgrid(easting_coords, northing_coords)

        return easting_grid.ravel(), northing_grid.ravel()

    def _subdivide_coarse_coords(
        self,
        eastings: NDArray[np.int32],
        northings: NDArray[np.int32],
    ) -> tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Subdivide 100 km coordinates into a 10 km grid.

        Returns
        -------
        tuple[NDArray[np.int32], NDArray[np.int32]]
            10 km-spaced eastings and northings.
        """
        tpl = self._fine_template.T
        x = eastings[..., np.newaxis]
        y = northings[..., np.newaxis]

        fine_x = (x + tpl[0]).ravel()
        fine_y = (y + tpl[1]).ravel()

        return fine_x, fine_y

    def _compute_mgrs_codes(
        self,
        eastings: NDArray[np.int32],
        northings: NDArray[np.int32],
        *,
        zones: NDArray[np.uint8],
        hemispheres: NDArray[np.str_],
        precision: int = _MGRS_PRECISION_10KM,
    ) -> NDArray[np.str_]:
        """Compute MGRS references from UTM coordinates.

        Parameters
        ----------
        eastings : NDArray[np.int32]
            UTM eastings.
        northings : NDArray[np.int32]
            UTM northings.
        zones : NDArray[np.uint8]
            UTM zone numbers.
        hemispheres : NDArray[np.str_]
            Hemisphere letters.
        precision : int
            MGRS precision level.

        Returns
        -------
        NDArray[np.str_]
            MGRS references.
        """
        self.utm_to_mgrs = np.vectorize(
            mgrs.MGRS().UTMToMGRS,
            otypes=[str],
            excluded={"MGRSPrecision"},
        )

        return self.utm_to_mgrs(
            zones,
            hemispheres,
            eastings,
            northings,
            MGRSPrecision=precision,
        )

    def to_geodataframe(
        self,
    ) -> gpd.GeoDataFrame:
        """Return the grid squares in WGS84.

        Returns
        -------
        gpd.GeoDataFrame
            One row per square with its MGRS and UTM metadata.
        """
        return gpd.GeoDataFrame(
            {
                "mgrs_code": self.mgrs_codes,
                "zone": self.zones,
                "hemisphere": self.hemispheres,
                "easting": self.eastings,
                "northing": self.northings,
                "epsg": utms_to_epsgs(self.zones, self.hemispheres),
                "geometry": self.geometries,
            },
            crs=WGS84,
        )

    def __len__(self) -> int:
        """Return the number of grid squares."""
        return len(self.eastings)

    @overload
    def __getitem__(self, index: int) -> MGRSGeoBox: ...
    @overload
    def __getitem__(self, index: slice) -> Self: ...

    def __getitem__(self, index: int | slice) -> MGRSGeoBox | Self:
        """Return an MGRSGeoBox by index or a sliced MGRSGrid."""
        if isinstance(index, slice):
            new = copy(self)
            new.mgrs_codes = self.mgrs_codes[index]
            new.eastings = self.eastings[index]
            new.northings = self.northings[index]
            new.zones = self.zones[index]
            new.hemispheres = self.hemispheres[index]
            new.geometries = self.geometries[index]
            return new
        return self._entry_to_geobox(index)

    def __iter__(self) -> Iterator[MGRSGeoBox]:
        """Iterate over grid squares."""
        for idx in range(len(self)):
            yield self._entry_to_geobox(idx)

    def _entry_to_geobox(self, index: int) -> MGRSGeoBox:
        """Convert one entry to an MGRSGeoBox."""
        resolution = self.resolution
        mgrs_code = str(self.mgrs_codes[index])
        # `zones` holds numpy scalars; `utm_to_crs` is annotated for plain ints.
        zone = int(self.zones[index])
        hemisphere = self.hemispheres[index]
        crs = utm_to_crs(zone, hemisphere)
        n_pixels = int(_10KM_SIZE / resolution)
        sw_easting = self.eastings[index]
        nw_northing = self.northings[index] + _10KM_SIZE
        affine = Affine(resolution, 0, sw_easting, 0, -resolution, nw_northing)
        return MGRSGeoBox(
            mgrs_code=mgrs_code,
            shape=(n_pixels, n_pixels),
            affine=affine,
            crs=crs,
        )
