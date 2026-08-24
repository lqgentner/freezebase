"""Tests for freezebase.raster."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pyproj import CRS
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from upath import UPath

from freezebase.raster import (
    COG_PROFILE,
    RASTERIO_PROFILE_DEFAULTS,
    _env_for_path,
    _to_vsi_uri,
    build_rasterio_profile,
    get_epsg_string,
    get_utm_zone_string,
    group_tiffs_by_crs,
    merge_tiffs,
    rewrite_tiff,
    utm_zone_to_crs,
    write_cog,
)

WIDTH, HEIGHT = 8, 6

# A projected CRS that pyproj can build but cannot map back to an EPSG code.
NON_EPSG_CRS = CRS.from_proj4("+proj=laea +lat_0=52 +lon_0=10 +datum=WGS84")


def assert_is_cog(path: Path) -> None:
    """Lightweight COG check via plain rasterio (no rio_cogeo dependency)."""
    with rasterio.open(path) as ds:
        assert ds.driver == "GTiff"
        assert ds.profile["tiled"] is True
        assert ds.tags(ns="IMAGE_STRUCTURE").get("LAYOUT") == "COG"


def make_tiff(
    path: Path,
    *,
    fill: float = 1.0,
    origin: tuple[float, float] = (500000, 5200000),
    crs: str | CRS | None = "EPSG:32632",
    pixel_size: float = 10,
    description: str | None = "VV",
) -> None:
    data = np.full((HEIGHT, WIDTH), fill, dtype=np.float32)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(origin[0], origin[1], pixel_size, pixel_size),
        nodata=np.nan,
    ) as dst:
        dst.write(data, 1)
        if description is not None:
            dst.set_band_description(1, description)


class TestCrsHelpers:
    def test_get_utm_zone_string_valid(self) -> None:
        assert get_utm_zone_string("EPSG:32601") == "01N"
        assert get_utm_zone_string("EPSG:32732") == "32S"

    def test_get_utm_zone_string_invalid_raises_valueerror(self) -> None:
        # Previously raised UnboundLocalError referencing an unassigned `crs`.
        with pytest.raises(ValueError, match="Invalid `projparams`"):
            get_utm_zone_string("not-a-crs")

    def test_get_epsg_string_valid(self) -> None:
        assert get_epsg_string("EPSG:4326") == "EPSG:4326"

    def test_get_epsg_string_invalid_raises_valueerror(self) -> None:
        with pytest.raises(ValueError, match="Invalid `projparams`"):
            get_epsg_string("not-a-crs")

    def test_utm_zone_to_crs_valid(self) -> None:
        assert utm_zone_to_crs("32N").to_epsg() == 32632
        assert utm_zone_to_crs("01S").to_epsg() == 32701

    def test_utm_zone_to_crs_rejects_bad_hemisphere(self) -> None:
        # Previously returned EPSG:32732 for the invalid hemisphere 'X'.
        with pytest.raises(ValueError, match="hemisphere"):
            utm_zone_to_crs("32X")

    def test_utm_zone_to_crs_rejects_bad_zone(self) -> None:
        with pytest.raises(ValueError, match="UTM zone"):
            utm_zone_to_crs("99N")

    def test_utm_zone_to_crs_rejects_non_integer_zone(self) -> None:
        with pytest.raises(ValueError, match="not an integer"):
            utm_zone_to_crs("abN")

    def test_get_utm_zone_string_rejects_valid_but_non_utm_crs(self) -> None:
        # Parses fine but has no UTM zone; `group_tiffs_by_crs` needs this error.
        with pytest.raises(ValueError, match="Could not extract CRS identifier"):
            get_utm_zone_string("EPSG:4326")

    def test_get_epsg_string_rejects_crs_without_epsg_code(self) -> None:
        with pytest.raises(ValueError, match="Could not extract EPSG code"):
            get_epsg_string(NON_EPSG_CRS)


class TestPathProtocolSupport:
    def test_to_vsi_uri_passes_local_paths_through(self, tmp_path: Path) -> None:
        assert _to_vsi_uri(tmp_path / "a.tif") == str(tmp_path / "a.tif")

    def test_to_vsi_uri_rejects_unsupported_protocol(self) -> None:
        # Must fail rather than reach GDAL as a local path.
        with pytest.raises(ValueError, match="Unsupported protocol 'memory'"):
            _to_vsi_uri(UPath("memory://bucket/a.tif"))

    def test_env_for_path_accepts_local_paths(self, tmp_path: Path) -> None:
        with _env_for_path(tmp_path / "a.tif"):
            pass  # entering the plain rasterio Env must not raise

    def test_env_for_path_rejects_unsupported_protocol(self) -> None:
        with (
            pytest.raises(ValueError, match="Unsupported protocol 'memory'"),
            _env_for_path(UPath("memory://bucket/a.tif")),
        ):
            pass  # pragma: no cover -- the context manager must not open


class TestBuildRasterioProfile:
    def test_returns_defaults_when_given_nothing(self) -> None:
        assert build_rasterio_profile() == RASTERIO_PROFILE_DEFAULTS

    def test_does_not_mutate_the_module_defaults(self) -> None:
        build_rasterio_profile({"compress": "lzw"})
        assert RASTERIO_PROFILE_DEFAULTS["compress"] == "deflate"

    def test_later_profiles_win(self) -> None:
        profile = build_rasterio_profile({"compress": "lzw"}, {"compress": "zstd"})
        assert profile["compress"] == "zstd"

    def test_none_entries_are_skipped(self) -> None:
        profile = build_rasterio_profile(None, {"blockxsize": 256}, None)
        assert profile["blockxsize"] == 256
        assert profile["driver"] == "GTiff"


class TestGroupTiffsByCrs:
    def test_rejects_empty_input(self) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            group_tiffs_by_crs([])

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="Source file not found"):
            group_tiffs_by_crs([tmp_path / "absent.tif"])

    def test_file_without_crs_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "nocrs.tif"
        make_tiff(path, crs=None)

        with pytest.raises(TypeError, match="has no CRS"):
            group_tiffs_by_crs([path])

    def test_groups_utm_and_non_utm_separately(self, tmp_path: Path) -> None:
        utm32 = tmp_path / "utm32.tif"
        utm33 = tmp_path / "utm33.tif"
        wgs84 = tmp_path / "wgs84.tif"
        make_tiff(utm32, crs="EPSG:32632")
        make_tiff(utm33, crs="EPSG:32633")
        make_tiff(wgs84, crs="EPSG:4326", origin=(8.0, 47.0), pixel_size=0.001)

        groups = group_tiffs_by_crs([utm32, utm33, wgs84])

        assert {k: [p.name for p in v] for k, v in groups.items()} == {
            "UTM32N": ["utm32.tif"],
            "UTM33N": ["utm33.tif"],
            "EPSG4326": ["wgs84.tif"],
        }

    def test_same_crs_files_share_one_group_in_input_order(self, tmp_path: Path) -> None:
        first = tmp_path / "first.tif"
        second = tmp_path / "second.tif"
        make_tiff(first)
        make_tiff(second)

        groups = group_tiffs_by_crs([first, second])

        assert list(groups) == ["UTM32N"]
        assert [p.name for p in groups["UTM32N"]] == ["first.tif", "second.tif"]

    def test_zone_number_is_zero_padded(self, tmp_path: Path) -> None:
        path = tmp_path / "utm01.tif"
        make_tiff(path, crs="EPSG:32601")

        # 'UTM1N' would sort and group differently from 'UTM01N'.
        assert list(group_tiffs_by_crs([path])) == ["UTM01N"]

    def test_southern_hemisphere_zone(self, tmp_path: Path) -> None:
        path = tmp_path / "utm32s.tif"
        make_tiff(path, crs="EPSG:32732")

        assert list(group_tiffs_by_crs([path])) == ["UTM32S"]

    def test_accepts_string_paths(self, tmp_path: Path) -> None:
        path = tmp_path / "utm32.tif"
        make_tiff(path)

        groups = group_tiffs_by_crs([str(path)])

        assert isinstance(groups["UTM32N"][0], UPath)


class TestRewriteTiffLocal:
    def test_copies_between_paths_by_default(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(src, dst, profile=COG_PROFILE)

        # Source is preserved by default (move=False).
        assert src.exists()
        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VV",)
            assert (ds.read(1) == 1.0).all()

    def test_move_deletes_source(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(src, dst, profile=COG_PROFILE, move=True)

        assert not src.exists()
        assert_is_cog(dst)

    def test_preserves_existing_destination_on_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src, fill=1.0)
        make_tiff(dst, fill=9.0)

        with pytest.raises(RuntimeError):
            rewrite_tiff(src, dst, profile={"driver": "NotARealDriver"})

        # The pre-existing destination must survive the failed rewrite intact,
        # and the source must not be deleted.
        assert src.exists()
        assert not list(tmp_path.glob(f".{dst.name}.*.tmp"))
        with rasterio.open(dst) as ds:
            assert (ds.read(1) == 9.0).all()

    def test_rewrites_in_place_to_cog(self, tmp_path: Path) -> None:
        path = tmp_path / "tile.tif"
        make_tiff(path)

        rewrite_tiff(path, path, profile=COG_PROFILE)

        assert path.exists()
        assert not (tmp_path / f".{path.name}.tmp").exists()
        assert_is_cog(path)
        with rasterio.open(path) as ds:
            assert ds.descriptions == ("VV",)
            assert (ds.read(1) == 1.0).all()

    def test_in_place_preserves_band_names_override(self, tmp_path: Path) -> None:
        path = tmp_path / "tile.tif"
        make_tiff(path)

        rewrite_tiff(path, path, profile=COG_PROFILE, band_names=["renamed"])

        with rasterio.open(path) as ds:
            assert ds.descriptions == ("renamed",)

    def test_color_interp_override_is_injected(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(src, dst, profile=COG_PROFILE, color_interp=[ColorInterp.gray])

        with rasterio.open(dst) as ds:
            assert ds.colorinterp == (ColorInterp.gray,)
            # Injecting metadata must not cost the source's band description.
            assert ds.descriptions == ("VV",)

    def test_defaults_to_a_tiled_deflate_gtiff(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        # GTiff keeps the defaults that a non-GTiff driver would have stripped.
        rewrite_tiff(src, dst)

        with rasterio.open(dst) as ds:
            assert ds.driver == "GTiff"
            assert ds.profile["tiled"] is True
            assert ds.profile["compress"] == "deflate"
            assert ds.block_shapes == [(512, 512)]
            assert (ds.read(1) == 1.0).all()

    def test_no_leftover_temp_file_on_failure(self, tmp_path: Path) -> None:
        path = tmp_path / "tile.tif"
        make_tiff(path)

        with pytest.raises(RuntimeError):
            rewrite_tiff(path, path, profile={"driver": "NotARealDriver"})

        assert not list(tmp_path.glob(f".{path.name}.*.tmp"))
        # original file must still be intact -- the failed write never
        # touched it, since it happens on a separate temp file locally.
        with rasterio.open(path) as ds:
            assert (ds.read(1) == 1.0).all()


class TestMergeTiffs:
    def test_merges_same_crs_tiles(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_tiff(a, fill=1.0, origin=(500000, 5200000))
        make_tiff(b, fill=2.0, origin=(500000 + WIDTH * 10, 5200000))
        dst = tmp_path / "merged.tif"

        merge_tiffs([a, b], dst)

        with rasterio.open(dst) as ds:
            assert ds.width == WIDTH * 2
            assert ds.descriptions == ("VV",)

    def test_rejects_mismatched_crs(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_tiff(a, crs="EPSG:32632")
        make_tiff(b, crs="EPSG:32633")
        dst = tmp_path / "merged.tif"

        with pytest.raises(ValueError, match="share a CRS"):
            merge_tiffs([a, b], dst)

    def test_handles_missing_descriptions(self, tmp_path: Path) -> None:
        # A source without band descriptions must not crash metadata injection.
        a = tmp_path / "a.tif"
        make_tiff(a, description=None)
        dst = tmp_path / "merged.tif"

        merge_tiffs([a], dst)

        with rasterio.open(dst) as ds:
            assert ds.descriptions == (None,)

    def test_rejects_empty_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="cannot be empty"):
            merge_tiffs([], tmp_path / "merged.tif")

    def test_missing_source_raises_before_any_write(self, tmp_path: Path) -> None:
        present = tmp_path / "a.tif"
        make_tiff(present)
        dst = tmp_path / "merged.tif"

        with pytest.raises(FileNotFoundError, match="Source file not found"):
            merge_tiffs([present, tmp_path / "absent.tif"], dst)

        assert not dst.exists()

    def test_merge_failure_is_wrapped_in_runtime_error(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tif"
        make_tiff(a)
        dst = tmp_path / "merged.tif"

        with pytest.raises(RuntimeError, match="Failed to merge GeoTIFFs"):
            merge_tiffs([a], dst, method="not-a-real-method")

        # A failed merge must not leave a half-written destination behind.
        assert not dst.exists()

    def test_preserves_band_names_from_first_source(self, tmp_path: Path) -> None:
        a = tmp_path / "a.tif"
        b = tmp_path / "b.tif"
        make_tiff(a, description="VH")
        make_tiff(b, origin=(500000 + WIDTH * 10, 5200000), description="VH")
        dst = tmp_path / "merged.tif"

        merge_tiffs([a, b], dst)

        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VH",)


class TestWriteCog:
    def test_writes_valid_cog(self, tmp_path: Path) -> None:
        dst = tmp_path / "out.tif"
        data = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
        profile = {
            "dtype": "float32",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
            "nodata": np.nan,
        }

        write_cog(data, dst, profile, band_names=["VH"])

        assert_is_cog(dst)
        assert not (tmp_path / f".{dst.name}.tmp").exists()
        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VH",)
            assert (ds.read(1) == 3.0).all()

    def test_no_leftover_temp_file_or_dst_on_failure(self, tmp_path: Path) -> None:
        dst = tmp_path / "out.tif"
        data = np.full((HEIGHT, WIDTH), 3.0, dtype=np.float32)
        profile = {
            "dtype": "float32",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
            "nodata": np.nan,
        }

        with pytest.raises(RuntimeError):
            write_cog(data, dst, {**profile, "dtype": "not-a-real-dtype"})

        assert not dst.exists()
        assert not (tmp_path / f".{dst.name}.tmp").exists()

    def test_writes_two_dimensional_array_as_single_band(self, tmp_path: Path) -> None:
        dst = tmp_path / "out2d.tif"
        # A bare (height, width) array must be written to band 1.
        data = np.full((HEIGHT, WIDTH), 5.0, dtype=np.float32)

        write_cog(
            data,
            dst,
            {
                "dtype": "float32",
                "count": 1,
                "width": WIDTH,
                "height": HEIGHT,
                "crs": "EPSG:32632",
                "transform": from_origin(500000, 5200000, 10, 10),
                "nodata": np.nan,
            },
            band_names=["VH"],
            color_interp=[ColorInterp.gray],
        )

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.count == 1
            assert ds.descriptions == ("VH",)
            assert ds.colorinterp == (ColorInterp.gray,)
            assert (ds.read(1) == 5.0).all()

    def test_writes_without_band_metadata(self, tmp_path: Path) -> None:
        dst = tmp_path / "bare.tif"
        data = np.full((HEIGHT, WIDTH), 7.0, dtype=np.float32)

        write_cog(
            data,
            dst,
            {
                "dtype": "float32",
                "count": 1,
                "width": WIDTH,
                "height": HEIGHT,
                "crs": "EPSG:32632",
                "transform": from_origin(500000, 5200000, 10, 10),
                "nodata": np.nan,
            },
        )

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.descriptions == (None,)
            assert (ds.read(1) == 7.0).all()

    def test_writes_three_dimensional_array_with_per_band_metadata(self, tmp_path: Path) -> None:
        dst = tmp_path / "out3d.tif"
        data = np.stack(
            [
                np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32),
                np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32),
            ],
        )

        write_cog(
            data,
            dst,
            {
                "dtype": "float32",
                "count": 2,
                "width": WIDTH,
                "height": HEIGHT,
                "crs": "EPSG:32632",
                "transform": from_origin(500000, 5200000, 10, 10),
                "nodata": np.nan,
            },
            band_names=["VV", "VH"],
        )

        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VV", "VH")
            assert (ds.read(1) == 1.0).all()
            assert (ds.read(2) == 2.0).all()
