"""Tests for freezebase.raster."""

from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any, assert_type

import numpy as np
from pyproj import CRS
import pytest
import rasterio
from rasterio.enums import ColorInterp
import rasterio.shutil
from rasterio.transform import from_origin
from upath import UPath

from freezebase import raster
from freezebase.raster import (
    COG_PROFILE,
    GDAL_HTTP_OPTIONS,
    RASTERIO_PROFILE_DEFAULTS,
    _env_for_path,
    _rewrite_via_memory,
    _to_vsi_uri,
    build_rasterio_profile,
    get_epsg_string,
    get_utm_zone_string,
    group_tiffs_by_crs,
    http_env,
    rasterio_open,
    rewrite_tiff,
    utm_zone_to_crs,
    write_cog,
)
from freezebase.s3 import GDAL_S3_OPTIONS
from freezebase.vrt import (
    VRT,
    PixelFunction,
    create_decibel_vrt,
    create_warped_vrt,
    derive,
    from_raster,
    mosaic,
    stack,
    warp,
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


class TestHttpSources:
    def test_https_maps_to_vsicurl(self) -> None:
        assert _to_vsi_uri("https://example.com/a.tif") == "/vsicurl/https://example.com/a.tif"
        assert _to_vsi_uri("http://example.com/a.tif") == "/vsicurl/http://example.com/a.tif"

    def test_https_env_sets_http_options(self) -> None:
        with _env_for_path("https://example.com/a.tif"):
            assert rasterio.env.getenv()["GDAL_HTTP_TIMEOUT"] == "30"

    def test_http_env_overrides(self) -> None:
        with http_env(GDAL_HTTP_TIMEOUT="5"):
            assert rasterio.env.getenv()["GDAL_HTTP_TIMEOUT"] == "5"

    def test_s3_options_share_the_retry_settings(self) -> None:
        for key in ("GDAL_HTTP_MAX_RETRY", "GDAL_HTTP_RETRY_DELAY", "GDAL_HTTP_RETRY_CODES"):
            assert GDAL_S3_OPTIONS[key] == GDAL_HTTP_OPTIONS[key]

    def test_unsupported_protocol_still_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported protocol 'ftp'"):
            _to_vsi_uri("ftp://example.com/a.tif")


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

        # A description of a different length grows GDAL_METADATA, so writing it
        # into the finished COG would relocate the tag and cost the layout.
        assert_is_cog(path)
        with rasterio.open(path) as ds:
            assert ds.descriptions == ("renamed",)

    def test_color_interp_override_is_injected(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(src, dst, profile=COG_PROFILE, color_interp=[ColorInterp.gray])

        # Color interpretation sits in fixed TIFF tags, so unlike the others it
        # can be written into the finished COG without disturbing the layout.
        assert_is_cog(dst)
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


class TestRewriteTiffVrt:
    def test_rewrite_a_vrt_to_gtiff(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_tiff(a, fill=1.0)
        make_tiff(b, fill=2.0, origin=(500000 + WIDTH * 10, 5200000))
        dst = tmp_path / "dst.tif"

        rewrite_tiff(mosaic([a, b]), dst, {"driver": "GTiff"})

        with rasterio.open(dst) as ds:
            assert ds.width == WIDTH * 2
            assert ds.descriptions == ("VV",)
            assert ds.read(1)[0, 0] == 1.0
            assert ds.read(1)[0, -1] == 2.0
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tif", "b.tif", "dst.tif"]

    def test_rewrite_a_vrt_to_cog_keeps_the_model_untouched(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        make_tiff(src)
        vrt = from_raster(src)

        rewrite_tiff(vrt, dst, COG_PROFILE, band_names=["HH"], units=["dB"], tags={"B": "2"})

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("HH",)
            assert ds.units == ("dB",)
            assert ds.tags()["B"] == "2"
        assert vrt.bands[0].description == "VV"
        assert "B" not in vrt.metadata

    def test_rewrite_vrt_with_move_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)

        with pytest.raises(ValueError, match="move needs a source file"):
            rewrite_tiff(from_raster(src), tmp_path / "dst.tif", move=True)

        assert not (tmp_path / "dst.tif").exists()

    def test_rewrite_vrt_failure_keeps_the_destination(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        make_tiff(src)
        make_tiff(dst, fill=9.0)

        with pytest.raises(RuntimeError, match="Failed to rewrite"):
            rewrite_tiff(from_raster(src), dst, {"driver": "NotARealDriver"})

        assert not list(tmp_path.glob(f".{dst.name}.*.tmp"))
        with rasterio.open(dst) as ds:
            assert (ds.read(1) == 9.0).all()

    def test_rewrite_vrt_band_length_error_is_a_value_error(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)

        with pytest.raises(ValueError, match="one entry per band"):
            rewrite_tiff(from_raster(src), tmp_path / "dst.tif", units=["a", "b"])

        assert not (tmp_path / "dst.tif").exists()

    def test_rewrite_vrt_ignores_profile_dtype_and_nodata(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(from_raster(src), dst, {"dtype": "uint8", "nodata": 7})

        with rasterio.open(dst) as ds:
            assert ds.dtypes == ("float32",)
            assert np.isnan(ds.nodata)

    def test_rewrite_vrt_merges_band_tags_and_replaces_units_and_color_interp(
        self, tmp_path: Path
    ) -> None:
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        make_tiff(src)
        with rasterio.open(src, "r+") as ds:
            ds.update_tags(1, A="1")
            ds.set_band_unit(1, "m")

        rewrite_tiff(
            from_raster(src),
            dst,
            band_tags=[{"B": "2"}],
            units=["dB"],
            color_interp=[ColorInterp.red],
        )

        with rasterio.open(dst) as ds:
            assert ds.tags(1) == {"A": "1", "B": "2"}
            assert ds.units == ("dB",)
            assert ds.colorinterp == (ColorInterp.red,)

    def test_rewrite_vrt_to_s3_failure_leaves_nothing(
        self,
        tmp_path: Path,
        s3_uploads: dict[str, bytes],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        monkeypatch.setattr(tempfile, "tempdir", str(scratch))

        with pytest.raises(RuntimeError, match="Failed to rewrite"):
            rewrite_tiff(
                from_raster(src), UPath("s3://bucket/out/x.tif"), {"driver": "NotADriver"}
            )

        assert not s3_uploads
        assert not list(scratch.iterdir())

    def test_rewrite_merges_tags(self, tmp_path: Path) -> None:
        src, dst = tmp_path / "src.tif", tmp_path / "dst.tif"
        make_tiff(src)
        with rasterio.open(src, "r+") as ds:
            ds.update_tags(A="1")

        rewrite_tiff(from_raster(src), dst, tags={"B": "2"})

        with rasterio.open(dst) as ds:
            assert ds.tags()["A"] == "1"
            assert ds.tags()["B"] == "2"

    def test_merge_tiffs_is_gone(self) -> None:
        assert not hasattr(raster, "merge_tiffs")


def make_typed_tiff(path: Path, values: list[float], dtype: str, nodata: float) -> None:
    """Write a one-row, one-band raster of ``values`` on the shared test grid."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=len(values),
        height=1,
        count=1,
        dtype=dtype,
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=nodata,
    ) as ds:
        ds.write(np.array([values], dtype), 1)


class TestMixedBands:
    """A raster file holds one dtype and one nodata, so mixed bands are never coerced."""

    def test_mixed_dtypes_raise_without_a_profile_dtype(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_typed_tiff(a, [1, 2, 3], "uint8", 0)
        make_typed_tiff(b, [0.25, 1000.5, -7.0], "float32", np.nan)

        with pytest.raises(ValueError, match="pass dtype in profile"):
            write_cog(stack([a, b]), tmp_path / "o.tif")

        assert not (tmp_path / "o.tif").exists()

    def test_mixed_dtypes_write_with_a_profile_dtype(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_typed_tiff(a, [1, 2, 3], "uint8", 0)
        make_typed_tiff(b, [0.25, 1000.5, -7.0], "float32", np.nan)

        write_cog(stack([a, b]), tmp_path / "o.tif", {"dtype": "float32", "nodata": -9999})

        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("float32", "float32")
            assert ds.read(2).tolist() == [[0.25, 1000.5, -7.0]]
            assert ds.read(1).tolist() == [[1, 2, 3]]

    def test_mixed_nodata_raise_without_a_profile_nodata(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_typed_tiff(a, [0, 5, 6], "uint8", 0)
        make_typed_tiff(b, [255, 0, 7], "uint8", 255)

        with pytest.raises(ValueError, match="pass nodata in profile"):
            write_cog(stack([a, b]), tmp_path / "o.tif")

    def test_mixed_nodata_remap_per_band_with_a_profile_nodata(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_typed_tiff(a, [0, 5, 6], "uint8", 0)
        make_typed_tiff(b, [255, 0, 7], "uint8", 255)

        write_cog(stack([a, b]), tmp_path / "o.tif", {"nodata": 254})

        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.nodatavals == (254, 254)
            assert ds.read(1).tolist() == [[254, 5, 6]]
            assert ds.read(2).tolist() == [[254, 0, 7]]

    def test_rewrite_tiff_refuses_mixed_bands_too(self, tmp_path: Path) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_typed_tiff(a, [0, 5, 6], "uint8", 0)
        make_typed_tiff(b, [255, 0, 7], "uint8", 255)

        with pytest.raises(ValueError, match="pass nodata in profile"):
            rewrite_tiff(stack([a, b]), tmp_path / "o.tif", {"driver": "GTiff"})
        with pytest.raises(ValueError, match="pass nodata in profile"):
            rewrite_tiff(stack([a, b]), tmp_path / "o.tif", COG_PROFILE)

    def test_a_non_local_upath_is_a_raster_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Non-local UPaths have no __fspath__, so they must not be taken for arrays.
        src = tmp_path / "src.tif"
        make_tiff(src)
        remote = UPath("s3://bucket/src.tif")
        seen: list[object] = []

        def fake_from_raster(path: object) -> VRT:
            seen.append(path)
            return from_raster(src)

        monkeypatch.setattr("freezebase.vrt.from_raster", fake_from_raster)

        write_cog(remote, tmp_path / "o.tif")

        assert seen == [remote]
        assert_is_cog(tmp_path / "o.tif")


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

    def test_checksum_defaults_to_none(self, tmp_path: Path) -> None:
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

        digest = write_cog(data, dst, profile)

        assert_type(digest, None)
        assert digest is None

    def test_checksum_matches_the_written_bytes(self, tmp_path: Path) -> None:
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

        digest = write_cog(data, dst, profile, band_names=["VH"], checksum=True)

        assert_type(digest, str)
        assert digest.startswith("1220")
        assert digest == "1220" + hashlib.sha256(dst.read_bytes()).hexdigest()
        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VH",)
            assert (ds.read(1) == 3.0).all()

    def test_checksum_still_writes_atomically_on_failure(self, tmp_path: Path) -> None:
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
            write_cog(data, dst, {**profile, "dtype": "not-a-real-dtype"}, checksum=True)

        assert not dst.exists()
        assert not (tmp_path / f".{dst.name}.tmp").exists()


class TestMetadataInjection:
    """Dataset tags, band tags and band units, on both write paths.

    They live in the GDAL_METADATA TIFF tag, which is written last and can cost
    a COG its header-first layout, so every case asserts ``LAYOUT: COG`` too.
    """

    profile = {  # noqa: RUF012
        "dtype": "float32",
        "count": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 5200000, 10, 10),
        "nodata": np.nan,
    }

    def test_write_cog_retains_tags_units_and_cog_layout(self, tmp_path: Path) -> None:
        dst = tmp_path / "tagged.tif"
        data = np.stack(
            [
                np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32),
                np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32),
            ],
        )

        write_cog(
            data,
            dst,
            self.profile,
            band_names=["VV", "VH"],
            tags={"PRODUCT_TYPE": "RTC_LRW", "BACKSCATTER_CONVENTION": "Power"},
            band_tags=[{"POLARIZATION": "VV"}, {"POLARIZATION": "VH"}],
            units=["natural", "natural"],
        )

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.tags()["PRODUCT_TYPE"] == "RTC_LRW"
            assert ds.tags()["BACKSCATTER_CONVENTION"] == "Power"
            assert ds.tags(1) == {"POLARIZATION": "VV"}
            assert ds.tags(2) == {"POLARIZATION": "VH"}
            assert ds.units == ("natural", "natural")
            assert ds.descriptions == ("VV", "VH")

    def test_rewrite_to_cog_retains_tags_units_and_cog_layout(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(
            src,
            dst,
            profile=COG_PROFILE,
            tags={"PRODUCT_TYPE": "RTC"},
            band_tags=[{"POLARIZATION": "VV"}],
            units=["natural"],
        )

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.tags()["PRODUCT_TYPE"] == "RTC"
            assert ds.tags(1) == {"POLARIZATION": "VV"}
            assert ds.units == ("natural",)
            # The extra staging hop must not cost the source's band description.
            assert ds.descriptions == ("VV",)
        # The intermediate GTiff stage is cleaned up along with the temp.
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.tif", "src.tif"]

    def test_rewrite_in_place_to_cog_retains_tags(self, tmp_path: Path) -> None:
        path = tmp_path / "tile.tif"
        make_tiff(path)

        rewrite_tiff(path, path, profile=COG_PROFILE, tags={"A": "1"}, units=["natural"])

        assert_is_cog(path)
        with rasterio.open(path) as ds:
            assert ds.tags()["A"] == "1"
            assert ds.units == ("natural",)

    def test_rewrite_merges_tags_onto_the_source_ones(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)
        with rasterio.open(src, "r+") as ds:
            ds.update_tags(KEPT="from-source", OVERWRITTEN="from-source")

        rewrite_tiff(src, dst, profile=COG_PROFILE, tags={"OVERWRITTEN": "from-caller"})

        with rasterio.open(dst) as ds:
            assert ds.tags()["KEPT"] == "from-source"
            assert ds.tags()["OVERWRITTEN"] == "from-caller"

    def test_rewrite_to_gtiff_retains_tags_without_staging(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        rewrite_tiff(src, dst, tags={"A": "1"}, band_tags=[{"B": "2"}], units=["dB"])

        with rasterio.open(dst) as ds:
            assert ds.driver == "GTiff"
            assert ds.tags()["A"] == "1"
            assert ds.tags(1) == {"B": "2"}
            assert ds.units == ("dB",)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["dst.tif", "src.tif"]

    def test_rewrite_leaves_no_staging_file_on_failure(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)

        # A unit for a band that does not exist fails during injection, after
        # the staging copy has already been written.
        with pytest.raises(RuntimeError):
            rewrite_tiff(src, dst, profile=COG_PROFILE, tags={"A": "1"}, units=["natural", "dB"])

        assert not dst.exists()
        assert sorted(p.name for p in tmp_path.iterdir()) == ["src.tif"]

    def test_memory_stage_retains_tags_units_and_cog_layout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # The S3 destination path, exercised locally: the metadata goes into the
        # in-memory GTiff and the COG driver writes the layout around it.
        src = tmp_path / "src.tif"
        dst = tmp_path / "dst.tif"
        make_tiff(src)
        dst_profile = {k: v for k, v in COG_PROFILE.items() if k != "driver"}

        # The upload is the only S3-specific step; land its bytes locally.
        def land(src: bytes | Path, _dst: UPath) -> None:
            dst.write_bytes(src if isinstance(src, bytes) else src.read_bytes())

        monkeypatch.setattr(raster, "_upload", land)

        _rewrite_via_memory(
            UPath(src),
            UPath("s3://bucket/dst.tif"),
            driver="COG",
            dst_profile=dst_profile,
            band_names=None,
            color_interp=None,
            tags={"A": "1"},
            band_tags=[{"B": "2"}],
            units=["natural"],
        )

        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.tags()["A"] == "1"
            assert ds.tags(1) == {"B": "2"}
            assert ds.units == ("natural",)


class TestPerBandLengths:
    """Every per-band sequence must hold exactly one entry per band.

    A short sequence used to be applied to the leading bands and leave the rest
    untouched, which nothing downstream can detect: an unset unit reads back as
    ``None``, exactly like one written empty on purpose.
    """

    profile = {  # noqa: RUF012
        "dtype": "float32",
        "count": 2,
        "width": WIDTH,
        "height": HEIGHT,
        "crs": "EPSG:32632",
        "transform": from_origin(500000, 5200000, 10, 10),
        "nodata": np.nan,
    }

    def _data(self) -> np.ndarray:
        return np.stack(
            [
                np.full((HEIGHT, WIDTH), 1.0, dtype=np.float32),
                np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32),
            ],
        )

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"units": ["natural"]}, "units"),
            ({"band_names": ["VV"]}, "band_names"),
            ({"band_tags": [{"A": "1"}]}, "band_tags"),
            ({"color_interp": [ColorInterp.gray]}, "color_interp"),
        ],
    )
    def test_write_cog_rejects_a_short_sequence(
        self,
        tmp_path: Path,
        kwargs: dict[str, Any],
        expected: str,
    ) -> None:
        with pytest.raises(RuntimeError, match=f"{expected} must hold one entry per band"):
            write_cog(self._data(), tmp_path / "out.tif", self.profile, **kwargs)

    def test_write_cog_rejects_a_long_sequence(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError, match="one entry per band"):
            write_cog(
                self._data(),
                tmp_path / "out.tif",
                self.profile,
                units=["natural", "natural", "dB"],
            )

    def test_rewrite_rejects_a_short_sequence(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        write_cog(self._data(), src, self.profile, band_names=["VV", "VH"])

        with pytest.raises(RuntimeError, match="units must hold one entry per band"):
            rewrite_tiff(src, tmp_path / "dst.tif", units=["natural"])

    def test_a_rejected_write_leaves_nothing_behind(self, tmp_path: Path) -> None:
        with pytest.raises(RuntimeError):
            write_cog(self._data(), tmp_path / "out.tif", self.profile, units=["natural"])

        assert list(tmp_path.iterdir()) == []


class TestCogCreationOptions:
    """COG-driver settings are applied to the final copy."""

    profile = {  # noqa: RUF012
        "dtype": "float32",
        "nodata": np.nan,
        "count": 1,
        "width": WIDTH,
        "height": HEIGHT,
        "crs": CRS.from_epsg(32632),
        "transform": from_origin(500000, 5000000, 10, 10),
    }

    @staticmethod
    def _data() -> np.ndarray:
        rng = np.random.default_rng(0)
        return rng.random((HEIGHT, WIDTH), dtype="float32")

    def test_write_cog_honours_the_requested_codec(self, tmp_path: Path) -> None:
        """The bug this whole change exists for: settings were silently dropped."""
        dst = tmp_path / "lerc.tif"
        write_cog(
            self._data(), dst, {**self.profile, "compress": "lerc_zstd", "max_z_error": 1e-3}
        )

        with rasterio.open(dst) as src:
            structure = src.tags(ns="IMAGE_STRUCTURE")
        assert structure["COMPRESSION"] == "LERC_ZSTD"
        assert float(structure["MAX_Z_ERROR"]) == pytest.approx(1e-3)

    def test_write_cog_still_defaults_to_deflate(self, tmp_path: Path) -> None:
        dst = tmp_path / "default.tif"
        write_cog(self._data(), dst, self.profile)

        with rasterio.open(dst) as src:
            assert src.tags(ns="IMAGE_STRUCTURE")["COMPRESSION"] == "DEFLATE"

    def test_write_cog_forwards_native_cog_options(self, tmp_path: Path) -> None:
        dst = tmp_path / "custom.tif"
        write_cog(self._data(), dst, {**self.profile, "blocksize": 128, "overviews": "NONE"})

        with rasterio.open(dst) as src:
            assert src.block_shapes == [(128, 128)]
            assert src.overviews(1) == []

    def test_write_cog_respects_a_lossy_bound(self, tmp_path: Path) -> None:
        data = self._data()
        dst = tmp_path / "bounded.tif"
        write_cog(data, dst, {**self.profile, "compress": "lerc", "max_z_error": 1e-3})

        with rasterio.open(dst) as src:
            back = src.read(1)
        # LERC's float32 reconstruction can land slightly over the nominal
        # bound, so allow headroom rather than asserting an exact guarantee.
        assert np.abs(back - data).max() <= 2e-3

    def test_rewrite_tiff_accepts_gtiff_spelling(self, tmp_path: Path) -> None:
        src = tmp_path / "src.tif"
        write_cog(self._data(), src, self.profile)
        dst = tmp_path / "dst.tif"
        rewrite_tiff(src, dst, profile={"compress": "zstd", "zstd_level": 1, "predictor": 3})

        with rasterio.open(dst) as out:
            structure = out.tags(ns="IMAGE_STRUCTURE")
        assert structure["COMPRESSION"] == "ZSTD"
        assert structure["PREDICTOR"] == "3"


def _landed(uploads: dict[str, bytes], key: str, tmp_path: Path) -> Path:
    """Write an uploaded object's bytes to a local file to inspect it."""
    local = tmp_path / "landed" / key.rsplit("/", 1)[-1]
    local.parent.mkdir(exist_ok=True)
    local.write_bytes(uploads[key])
    return local


class TestS3WritesUploadThroughS3fs:
    """Every S3 write encodes locally and uploads once, never through /vsis3/."""

    dst = "s3://bucket/out/x.tif"

    def test_write_cog_without_checksum(
        self, tmp_path: Path, s3_uploads: dict[str, bytes]
    ) -> None:
        profile = {
            "dtype": "float32",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
        }
        write_cog(np.ones((HEIGHT, WIDTH), np.float32), UPath(self.dst), profile, tags={"A": "1"})
        landed = _landed(s3_uploads, self.dst, tmp_path)
        assert_is_cog(landed)
        with rasterio.open(landed) as ds:
            assert ds.tags()["A"] == "1"

    def test_write_cog_checksum_is_of_the_uploaded_bytes(
        self,
        s3_uploads: dict[str, bytes],
    ) -> None:
        profile = {
            "dtype": "float32",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
        }
        digest = write_cog(
            np.ones((HEIGHT, WIDTH), np.float32), UPath(self.dst), profile, checksum=True
        )
        assert digest == "1220" + hashlib.sha256(s3_uploads[self.dst]).hexdigest()

    def test_rewrite_tiff_moves_only_after_the_upload(
        self,
        tmp_path: Path,
        s3_uploads: dict[str, bytes],
    ) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)
        rewrite_tiff(src, UPath(self.dst), tags={"A": "1"}, move=True)
        with rasterio.open(_landed(s3_uploads, self.dst, tmp_path)) as ds:
            assert ds.tags()["A"] == "1"
        assert not src.exists()

    def test_rasterio_open_uploads_on_a_clean_close(
        self,
        tmp_path: Path,
        s3_uploads: dict[str, bytes],
    ) -> None:
        profile = {
            "driver": "GTiff",
            "dtype": "uint8",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
        }
        with rasterio_open(UPath(self.dst), "w", **profile) as ds:
            ds.write(np.full((1, HEIGHT, WIDTH), 7, np.uint8))
        with rasterio.open(_landed(s3_uploads, self.dst, tmp_path)) as ds:
            assert int(ds.read(1)[0, 0]) == 7

        other = "s3://bucket/out/failed.tif"

        def fail_mid_write() -> None:
            with rasterio_open(UPath(other), "w", **profile):
                msg = "boom"
                raise RuntimeError(msg)

        with pytest.raises(RuntimeError, match="boom"):
            fail_mid_write()
        assert other not in s3_uploads

    def test_rewrite_a_vrt(self, tmp_path: Path, s3_uploads: dict[str, bytes]) -> None:
        a, b = tmp_path / "a.tif", tmp_path / "b.tif"
        make_tiff(a, fill=1)
        make_tiff(b, fill=2, origin=(500000 + WIDTH * 10, 5200000))
        rewrite_tiff(mosaic([a, b]), UPath(self.dst), COG_PROFILE)
        with rasterio.open(_landed(s3_uploads, self.dst, tmp_path)) as ds:
            assert ds.width == 2 * WIDTH
            assert ds.descriptions == ("VV",)

    def test_create_warped_vrt(self, tmp_path: Path, s3_uploads: dict[str, bytes]) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)
        dst = "s3://bucket/out/x.vrt"
        create_warped_vrt(src, UPath(dst), crs="EPSG:32632", resolution=20)
        xml = s3_uploads[dst].decode()
        assert "<VRTDataset" in xml
        assert str(src) in xml

    def test_xml_vrt(self, tmp_path: Path, s3_uploads: dict[str, bytes]) -> None:
        src = tmp_path / "src.tif"
        make_tiff(src)
        dst = "s3://bucket/out/db.vrt"
        create_decibel_vrt(src, UPath(dst))
        assert s3_uploads[dst].decode().startswith("<VRTDataset")

    def test_a_failed_upload_raises_and_keeps_the_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def refuse(_src: bytes | Path, _dst: UPath) -> None:
            msg = "AccessDenied"
            raise PermissionError(msg)

        monkeypatch.setattr(raster, "_upload", refuse)
        src = tmp_path / "src.tif"
        make_tiff(src)
        with pytest.raises(RuntimeError, match="AccessDenied"):
            rewrite_tiff(src, UPath(self.dst), move=True)
        assert src.exists()


def make_int_tiff(
    path: Path, values: list[int], *, dtype: str = "uint16", nodata: int = 0
) -> None:
    """Write a one-row raster holding ``values``."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=len(values),
        height=1,
        count=1,
        dtype=dtype,
        crs="EPSG:32632",
        transform=from_origin(500000, 5200000, 10, 10),
        nodata=nodata,
    ) as dst:
        dst.write(np.array([values], dtype=dtype), 1)


class TestWriteCogFromVrt:
    def test_streams_a_mosaic(self, tmp_path: Path) -> None:
        left, right = tmp_path / "l.tif", tmp_path / "r.tif"
        make_tiff(left, fill=1.0)
        make_tiff(right, fill=2.0, origin=(500000 + WIDTH * 10, 5200000))
        vrt = mosaic([left, right])
        dst = tmp_path / "o.tif"
        write_cog(vrt, dst)
        assert_is_cog(dst)
        with rasterio.open(dst) as ds, vrt.open() as expected:
            np.testing.assert_array_equal(ds.read(), expected.read())
        assert not list(tmp_path.glob(".*.tmp"))

    def test_accepts_a_raster_path(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src, fill=3.0)
        dst = tmp_path / "o.tif"
        write_cog(src, dst)
        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.read(1)[0, 0] == 3.0

    def test_keeps_dtype_and_nodata_by_default(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_int_tiff(src, [0, 5], dtype="uint16", nodata=0)
        write_cog(src, tmp_path / "o.tif")
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("uint16",)
            assert ds.nodata == 0

    def test_metadata_lands_in_the_header(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        dst = tmp_path / "o.tif"
        write_cog(
            _vrt_of(src),
            dst,
            band_names=["VH"],
            tags={"A": "1"},
            band_tags=[{"B": "2"}],
            units=["dB"],
            color_interp=[ColorInterp.alpha],
        )
        assert_is_cog(dst)
        with rasterio.open(dst) as ds:
            assert ds.descriptions == ("VH",)
            assert ds.tags()["A"] == "1"
            assert ds.tags(1)["B"] == "2"
            assert ds.units == ("dB",)
            assert ds.colorinterp == (ColorInterp.alpha,)

    def test_metadata_does_not_touch_the_input_vrt(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        vrt = _vrt_of(src)
        write_cog(vrt, tmp_path / "o.tif", {"nodata": -1.0}, band_names=["X"], tags={"A": "1"})
        assert vrt == _vrt_of(src)

    def test_band_metadata_length_is_checked(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        with pytest.raises(ValueError, match="band_names"):
            write_cog(src, tmp_path / "o.tif", band_names=["a", "b"])

    def test_nodata_change_remaps_pixels(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_int_tiff(src, [0, 5], dtype="uint16", nodata=0)
        write_cog(src, tmp_path / "o.tif", {"nodata": 65535})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.read(1).tolist() == [[65535, 5]]
            assert ds.nodata == 65535

    def test_cast_keeps_scale_and_offset(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_int_tiff(src, [0, 5], dtype="uint8", nodata=0)
        with rasterio.open(src, "r+") as ds:
            ds.scales = (0.5,)
            ds.offsets = (2.0,)
        write_cog(src, tmp_path / "o.tif", {"dtype": "uint16"})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("uint16",)
            assert (ds.scales, ds.offsets) == ((0.5,), (2.0,))

    def test_cast_with_nan_nodata_needs_an_integer_nodata(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        with pytest.raises(ValueError, match="casting to uint8 needs a nodata"):
            write_cog(src, tmp_path / "o.tif", {"dtype": "uint8"})
        assert not (tmp_path / "o.tif").exists()

    def test_cast_with_a_fitting_nodata(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        write_cog(src, tmp_path / "o.tif", {"dtype": np.dtype("uint8"), "nodata": 255})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("uint8",)
            assert ds.nodata == 255

    def test_nodata_that_does_not_fit_the_dtype_raises(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_int_tiff(src, [0, 5], dtype="uint8", nodata=0)
        with pytest.raises(ValueError, match="does not fit"):
            write_cog(src, tmp_path / "o.tif", {"nodata": 65535})

    def test_creation_options_come_from_the_profile(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        write_cog(src, tmp_path / "o.tif", {"compress": "lzw", "blocksize": 256})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.compression.value == "LZW"
            assert ds.block_shapes == [(256, 256)]

    def test_checksum_matches_the_file(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        dst = tmp_path / "o.tif"
        digest = write_cog(src, dst, checksum=True)
        assert digest == "1220" + hashlib.sha256(dst.read_bytes()).hexdigest()

    def test_failure_removes_the_temp_file(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        real_copy = rasterio.shutil.copy

        def copy_then_fail(*args: Any, **kwargs: Any) -> None:
            real_copy(*args, **kwargs)
            assert list(tmp_path.glob(".o.tif.*.tmp"))
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(rasterio.shutil, "copy", copy_then_fail)
        with pytest.raises(RuntimeError, match="disk full"):
            write_cog(src, tmp_path / "o.tif")
        assert not list(tmp_path.glob(".*.tmp"))
        assert not (tmp_path / "o.tif").exists()

    def test_s3_destination_uploads_once(
        self, tmp_path: Path, s3_uploads: dict[str, bytes]
    ) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        digest = write_cog(src, UPath("s3://bucket/out/x.tif"), checksum=True)
        assert list(s3_uploads) == ["s3://bucket/out/x.tif"]
        landed = _landed(s3_uploads, "s3://bucket/out/x.tif", tmp_path)
        assert_is_cog(landed)
        assert digest == "1220" + hashlib.sha256(landed.read_bytes()).hexdigest()

    def test_s3_destination_without_checksum(
        self, tmp_path: Path, s3_uploads: dict[str, bytes]
    ) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        assert write_cog(src, UPath("s3://bucket/out/x.tif")) is None
        assert list(s3_uploads) == ["s3://bucket/out/x.tif"]

    def test_ndarray_still_requires_a_profile(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="profile is required"):
            write_cog(np.zeros((2, 2), np.float32), tmp_path / "o.tif")


class TestWriteCogNodataRemap:
    """A nodata change moves the old nodata pixels to the new value, for every model shape."""

    def test_mosaic_with_a_native_tile_nodata(self, tmp_path: Path) -> None:
        tile = tmp_path / "a.tif"
        make_int_tiff(tile, [0, 5, 7, 9], dtype="int16", nodata=0)
        m = mosaic([tile], nodata=-9999)
        write_cog(m, tmp_path / "o.tif", {"nodata": -1})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.read(1).tolist() == [[-1, 5, 7, 9]]
            assert ds.nodata == -1

    @pytest.mark.parametrize(
        ("profile", "expected", "nodata"),
        [
            ({"nodata": 65535}, [[65535, 5], [6, 7]], 65535),
            ({"dtype": "float32", "nodata": float("nan")}, [[float("nan"), 5], [6, 7]], "nan"),
        ],
    )
    def test_warped_vrt(
        self, tmp_path: Path, profile: dict[str, Any], expected: list[list[float]], nodata: Any
    ) -> None:
        src = tmp_path / "b.tif"
        with rasterio.open(
            src,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:32632",
            transform=from_origin(500000, 5200000, 10, 10),
            nodata=0,
        ) as dst:
            dst.write(np.array([[0, 5], [6, 7]], np.uint16), 1)
        w = warp(src, crs="EPSG:32632", resolution=10)
        write_cog(w, tmp_path / "o.tif", profile)
        with rasterio.open(tmp_path / "o.tif") as ds:
            np.testing.assert_array_equal(ds.read(1), np.array(expected))
            np.testing.assert_array_equal(ds.nodata, float(nodata))

    def test_derived_band(self, tmp_path: Path) -> None:
        src = tmp_path / "c.tif"
        with rasterio.open(
            src,
            "w",
            driver="GTiff",
            width=4,
            height=1,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=from_origin(500000, 5200000, 10, 10),
            nodata=0,
        ) as dst:
            dst.write(np.array([[0, 10, 100, 1]], np.float32), 1)
        d = derive(src, PixelFunction.db())
        write_cog(d, tmp_path / "o.tif", {"nodata": -9999.0})
        with rasterio.open(tmp_path / "o.tif") as ds, d.open() as ref:
            out, expected = ds.read(1), ref.read(1)
            assert ds.nodata == -9999.0
            assert out[0, 0] == -9999.0
            np.testing.assert_allclose(out[0, 1:], expected[0, 1:])

    def test_none_keeps_the_nodata(self, tmp_path: Path) -> None:
        src = tmp_path / "f.tif"
        make_tiff(src)
        with rasterio.open(src, "r+") as ds:
            ds.write(np.array([[np.nan, 1.5] + [1.0] * (WIDTH - 2)] * HEIGHT, np.float32), 1)
        write_cog(src, tmp_path / "o.tif", {"nodata": None})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert np.isnan(ds.nodata)
            assert np.isnan(ds.read(1)[0, 0])

    def test_dtype_spellings_are_normalised(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        write_cog(src, tmp_path / "o.tif", {"dtype": "f8"})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("float64",)

    def test_array_likes_are_written(self, tmp_path: Path) -> None:
        class ArrayLike:
            def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
                return np.ones((HEIGHT, WIDTH), np.float32)

        profile = {
            "dtype": "float32",
            "count": 1,
            "width": WIDTH,
            "height": HEIGHT,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
        }
        write_cog(ArrayLike(), tmp_path / "o.tif", profile)  # pyrefly: ignore[bad-argument-type]  # duck-typed input
        assert_is_cog(tmp_path / "o.tif")


class TestWriteCogInputs:
    def test_masked_array_is_filled_with_nodata(self, tmp_path: Path) -> None:
        profile = {
            "dtype": "int16",
            "count": 1,
            "width": 3,
            "height": 1,
            "crs": "EPSG:32632",
            "transform": from_origin(500000, 5200000, 10, 10),
            "nodata": -1,
        }
        data = np.ma.masked_array([[5, 6, 7]], mask=[[True, False, False]], dtype="int16")
        write_cog(data, tmp_path / "o.tif", profile)
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.read(1).tolist() == [[-1, 6, 7]]

    def test_path_like_object_takes_the_path_route(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        write_cog(PurePosixPath(src), tmp_path / "o.tif")  # pyrefly: ignore[bad-argument-type]  # any os.PathLike works at runtime
        assert_is_cog(tmp_path / "o.tif")

    def test_unknown_dtype_raises_value_error(self, tmp_path: Path) -> None:
        src = tmp_path / "s.tif"
        make_tiff(src)
        with pytest.raises(ValueError, match="Unsupported dtype"):
            write_cog(src, tmp_path / "o.tif", {"dtype": "foo"})

    def test_wrap_failure_is_a_write_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        src = tmp_path / "s.tif"
        make_int_tiff(src, [0, 5])

        def fail(*_args: Any, **_kwargs: Any) -> None:
            msg = "cannot open"
            raise rasterio.errors.RasterioIOError(msg)

        monkeypatch.setattr("freezebase.vrt.stack", fail)
        with pytest.raises(RuntimeError, match="Failed to write COG: cannot open"):
            write_cog(src, tmp_path / "o.tif", {"nodata": 9})


def _vrt_of(path: Path) -> VRT:
    """Build a 1:1 VRT over a raster."""
    return from_raster(path)
