"""Tests for freezebase.vectools module."""

from pathlib import Path
import tempfile

import geopandas as gpd
import pyarrow.parquet as pq
import pytest
import shapely

from freezebase.vectools import (
    drop_z_if_zero,
    is_z_axis_zero,
    simplify_multipolygons,
    write_parquet,
)


@pytest.fixture
def polygon_3d() -> shapely.Polygon:
    """Create a sample 3d polygon."""
    return shapely.Polygon([(0, 0, 1), (1, 0, 2), (1, 1, 3), (0, 1, 4), (0, 0, 1)])


@pytest.fixture
def sample_gdf() -> gpd.GeoDataFrame:
    """Create a sample GeoDataFrame."""
    point = shapely.Point(1, 2)
    line = shapely.LineString([(0, 0, 1), (1, 1, 2), (2, 2, 3)])
    poly_2d = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 0)])
    poly_3d = shapely.Polygon([(0, 0, 1), (1, 0, 2), (1, 1, 3), (0, 0, 1)])
    multipolygon = shapely.MultiPolygon([poly_2d, poly_3d])
    geomcoll = shapely.GeometryCollection([point, line, poly_3d])

    return gpd.GeoDataFrame(
        {
            "name": ["A", "B", "C", "D", "E", "F"],
            "geometry": [point, line, poly_2d, poly_3d, multipolygon, geomcoll],
        },
        crs="EPSG:4326",
    )


class TestWriteParquet:
    """Test write_parquet function."""

    def test_write_roundtrip(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """The written file reads back as the original GeoDataFrame."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test.parquet"
            write_parquet(sample_gdf, out_path)
            assert sample_gdf.equals(gpd.read_parquet(out_path))

    def test_write_creates_directory(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Missing parent directories are created."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "nested" / "dir" / "test.parquet"
            write_parquet(sample_gdf, out_path)
            assert sample_gdf.equals(gpd.read_parquet(out_path))

    def test_write_with_string_path(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """A string path is accepted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = str(Path(temp_dir) / "test.parquet")
            write_parquet(sample_gdf, out_path)
            assert sample_gdf.equals(gpd.read_parquet(out_path))

    def test_no_leftover_staging_file(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """The atomic staging temp file must not remain after a successful write."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test.parquet"
            write_parquet(sample_gdf, out_path)
            assert not list(Path(temp_dir).glob(".*.tmp"))

    def test_failed_write_leaves_no_file(
        self, sample_gdf: gpd.GeoDataFrame, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A write that fails midway leaves neither the output nor the staging file."""

        def fail(_self: gpd.GeoDataFrame, path: Path, **_kwargs: object) -> None:
            Path(path).write_bytes(b"PAR1 truncated")
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(gpd.GeoDataFrame, "to_parquet", fail)
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test.parquet"
            with pytest.raises(OSError, match="disk full"):
                write_parquet(sample_gdf, out_path)
            assert not list(Path(temp_dir).iterdir())

    def test_writes_zstd_row_groups(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """The file is zstd-compressed and split into row groups of the given size."""
        with tempfile.TemporaryDirectory() as temp_dir:
            out_path = Path(temp_dir) / "test.parquet"
            write_parquet(sample_gdf, out_path, compression_level=3, row_group_size=4)
            metadata = pq.ParquetFile(out_path).metadata
            assert metadata.num_row_groups == 2
            assert metadata.row_group(0).column(0).compression == "ZSTD"


class TestIsZAxisZeros:
    """Test is_z_axis_zero function."""

    def test_all_z_zero(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when all Z coordinates are zero."""
        gdf_allzero = sample_gdf.copy()
        gdf_allzero.geometry = gdf_allzero.force_2d().force_3d()
        assert gdf_allzero.has_z.all()

        assert is_z_axis_zero(gdf_allzero) is True

    def test_some_z_nonzero(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when some Z coordinates are non-zero."""
        gdf_nonzero = sample_gdf.copy()
        gdf_nonzero.geometry = gdf_nonzero.force_3d()
        assert gdf_nonzero.has_z.all()

        assert is_z_axis_zero(gdf_nonzero) is False

    def test_no_z(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when not all geometries have z axes."""
        assert sample_gdf.has_z.any()
        assert not sample_gdf.has_z.all()

        with pytest.raises(ValueError, match="Geometry has no Z axis"):
            is_z_axis_zero(sample_gdf)


class TestDropZIfZero:
    """Test drop_z_if_zero function."""

    def test_all_z_zero(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when all Z coordinates are zero."""
        gdf_allzero = sample_gdf.copy()
        gdf_allzero.geometry = gdf_allzero.force_2d().force_3d()
        assert gdf_allzero.has_z.all()

        gdf_copy = gdf_allzero.copy()

        result = drop_z_if_zero(gdf_allzero)
        assert not result.has_z.all()

        # Assert that the data is not modified inplace
        assert gdf_allzero.equals(gdf_copy)

    def test_some_z_nonzero(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when some Z coordinates are non-zero."""
        gdf_nonzero = sample_gdf.copy()
        gdf_nonzero.geometry = gdf_nonzero.force_3d()
        assert gdf_nonzero.has_z.all()

        result = drop_z_if_zero(gdf_nonzero)
        assert result.equals(gdf_nonzero)

    def test_no_z(self, sample_gdf: gpd.GeoDataFrame) -> None:
        """Test when not all geometries have z axes."""
        assert sample_gdf.has_z.any()
        assert not sample_gdf.has_z.all()

        with pytest.raises(ValueError, match="Geometry has no Z axis"):
            is_z_axis_zero(sample_gdf)


class TestZInteriorRings:
    """Interior-ring Z coordinates must not be silently discarded (FC-06)."""

    def test_polygon_hole_z_is_inspected(self) -> None:
        shell = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 0)]
        hole = [(2, 2, 5), (4, 2, 5), (4, 4, 5), (2, 4, 5), (2, 2, 5)]
        poly = shapely.Polygon(shell, [hole])
        gdf = gpd.GeoDataFrame({"geometry": [poly]}, crs="EPSG:4326")

        # Exterior Z is all zero, but the hole carries a real Z value.
        assert is_z_axis_zero(gdf) is False

        result = drop_z_if_zero(gdf)
        assert result.has_z.all()  # meaningful Z must be preserved

    def test_multipolygon_hole_z_is_inspected(self) -> None:
        shell = [(0, 0, 0), (10, 0, 0), (10, 10, 0), (0, 10, 0), (0, 0, 0)]
        hole = [(2, 2, 7), (4, 2, 7), (4, 4, 7), (2, 4, 7), (2, 2, 7)]
        poly = shapely.Polygon(shell, [hole])
        gdf = gpd.GeoDataFrame({"geometry": [shapely.MultiPolygon([poly])]}, crs="EPSG:4326")

        assert is_z_axis_zero(gdf) is False

    def test_mixed_dimensional_collection_does_not_raise(self) -> None:
        # A collection mixing a 2D point with a 3D line must not abort the
        # traversal; only the members with Z contribute.
        point_2d = shapely.Point(1, 2)
        line_3d = shapely.LineString([(0, 0, 3), (1, 1, 3)])
        coll = shapely.GeometryCollection([point_2d, line_3d])
        gdf = gpd.GeoDataFrame({"geometry": [coll]}, crs="EPSG:4326")

        assert is_z_axis_zero(gdf) is False


def test_simplify_multipolygons() -> None:
    """Test simplify_multipolygons function."""
    polygon = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0, 1), (0, 0)])
    multi_single = shapely.MultiPolygon([polygon])
    multi_multi = shapely.MultiPolygon([polygon, polygon])
    point = shapely.Point(1, 2)

    gdf = gpd.GeoDataFrame(
        {
            "name": ["A", "B", "C"],
            "geometry": [multi_single, multi_multi, point],
        },
    )
    gdf_copy = gdf.copy()

    result = simplify_multipolygons(gdf)

    # Check that MultiPolygon with single polygon was converted to Polygon
    assert result.geometry.iloc[0].equals(polygon)

    # Check that a MultiPolygon with multiple polygons was not changed
    assert result.geometry.iloc[1].equals(multi_multi)

    # Check that a different geometry was not changed
    assert result.geometry.iloc[2].equals(point)

    # Check that the original input was not modified
    assert gdf.equals(gdf_copy)
