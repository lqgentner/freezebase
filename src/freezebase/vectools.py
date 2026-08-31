"""Vector geometry helpers."""

from pathlib import Path
from secrets import token_hex

import geopandas as gpd
from shapely import (
    GeometryCollection,
    LineString,
    MultiLineString,
    MultiPoint,
    MultiPolygon,
    Point,
    Polygon,
)
from shapely.geometry.base import BaseGeometry


def save_and_read_parquet(gdf: gpd.GeoDataFrame, out_path: str | Path) -> gpd.GeoDataFrame:
    """Atomically save a GeoDataFrame as GeoParquet and read it back.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        Data to save.
    out_path : str or Path
        Output path.

    Returns
    -------
    gpd.GeoDataFrame
        Saved data read from disk.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.{token_hex(8)}.tmp")
    try:
        gdf.to_parquet(tmp_path, engine="pyarrow")
        tmp_path.replace(out_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return gpd.read_parquet(out_path)


def drop_z_if_zero(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Drop the Z axis when every Z coordinate is zero.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometries to inspect.

    Returns
    -------
    geopandas.GeoDataFrame
        Copied GeoDataFrame, possibly forced to 2D.

    Raises
    ------
    ValueError
        If the geometry does not have a Z axis.
    TypeError
        If the geometry type is unknown.

    """
    gdf_copy = gdf.copy()
    if is_z_axis_zero(gdf):
        gdf_copy.geometry = gdf_copy.force_2d()
    return gdf_copy


def is_z_axis_zero(gdf: gpd.GeoDataFrame) -> bool:
    """Return whether every Z coordinate is zero.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometries to inspect.

    Returns
    -------
    bool
        Whether all Z values are zero.

    Raises
    ------
    ValueError
        If the geometry does not have a Z axis.
    TypeError
        If the geometry type is unknown.

    """
    z_values = gdf.geometry.map(_extract_z_values)
    return all(all(z == 0 for z in z_list) for z_list in z_values)


def _extract_z_values(geom: BaseGeometry) -> list[float]:
    """Return all Z values from a geometry.

    Parameters
    ----------
    geom : shapely.geometry.base.BaseGeometry
        Geometry to inspect.

    Returns
    -------
    list[float]
        Z values.

    Raises
    ------
    ValueError
        If the geometry does not have a Z axis.
    TypeError
        If the geometry type is unknown.

    """
    if geom.is_empty:
        return []
    if not geom.has_z:
        msg = "Geometry has no Z axis"
        raise ValueError(msg)
    match geom:
        case Point():
            z = [geom.z]
        case LineString():
            z = [coord[2] for coord in geom.coords]
        case Polygon():
            z = _polygon_z_values(geom)
        case MultiPoint():
            z = [point.z for point in geom.geoms]
        case MultiLineString():
            z = [coord[2] for line in geom.geoms for coord in line.coords]
        case MultiPolygon():
            z = [value for poly in geom.geoms for value in _polygon_z_values(poly)]
        case GeometryCollection():
            z = _collection_z_values(geom)
        case _:
            msg = f"Unsupported geometry type '{type(geom).__name__}'."
            raise TypeError(msg)
    return z


def _polygon_z_values(geom: Polygon) -> list[float]:
    """Extract Z values from a polygon's exterior and every interior ring."""
    z = [coord[2] for coord in geom.exterior.coords]
    for ring in geom.interiors:
        z.extend(coord[2] for coord in ring.coords)
    return z


def _collection_z_values(geom: GeometryCollection) -> list[float]:
    """Extract Z values from 3D collection members."""
    z: list[float] = []
    for sub_geom in geom.geoms:
        if sub_geom.is_empty or not sub_geom.has_z:
            continue
        z.extend(_extract_z_values(sub_geom))
    return z


def simplify_multipolygons(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Convert single-part MultiPolygons to Polygons.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Geometries to simplify.

    Returns
    -------
    geopandas.GeoDataFrame
        Simplified copy.
    """
    gdf_copy = gdf.copy()
    gdf_copy.geometry = [
        geom.geoms[0] if isinstance(geom, MultiPolygon) and len(geom.geoms) == 1 else geom
        for geom in gdf_copy.geometry
    ]
    return gdf_copy
