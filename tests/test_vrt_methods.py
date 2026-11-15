"""Tests for the chaining methods and writers on ``VRT``."""

from __future__ import annotations

import hashlib
import inspect
from typing import TYPE_CHECKING, Any, assert_type

import numpy as np
import pytest
import rasterio
import rasterio.errors
import rasterio.shutil
from upath import UPath

from freezebase import vrt
from freezebase.errors import RasterWriteError
from freezebase.raster import _encode, build_rasterio_profile, write_cog
from freezebase.vrt import VRT, PixelFunction, from_raster, mosaic
from tests.rasters import make_tiff

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

ORIGIN = (500000.0, 5200000.0)
WIDTH = 6


def _tiles(tmp_path: Path) -> list[Path]:
    """Write two side-by-side float32 tiles."""
    left = make_tiff(tmp_path / "l.tif", fill=1.0)
    right = make_tiff(tmp_path / "r.tif", fill=2.0, origin=(ORIGIN[0] + WIDTH * 10, ORIGIN[1]))
    return [left, right]


def _uint16(tmp_path: Path) -> VRT:
    """Return a VRT over a uint16 raster whose first pixel is its nodata, 0."""
    path = make_tiff(tmp_path / "u.tif", dtype="uint16", nodata=0)
    return from_raster(path)


def _read(path: Path | str) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


def _parameters(func: Callable[..., Any], skip: int) -> list[tuple[str, Any, str, Any]]:
    """Describe the parameters after the first ``skip``, NaN defaults included."""
    return [
        (p.name, p.kind, repr(p.default), p.annotation)
        for p in list(inspect.signature(func).parameters.values())[skip:]
    ]


@pytest.mark.parametrize(
    ("method", "function", "skip"),
    [
        # self versus the source.
        (VRT.reproject, vrt.reproject, 1),
        (VRT.derive, vrt.derive, 1),
        # self and path versus data and dst_file.
        (VRT.to_cog, write_cog, 2),
    ],
    ids=["reproject", "derive", "to_cog"],
)
def test_methods_match_their_functions(
    method: Callable[..., Any], function: Callable[..., Any], skip: int
) -> None:
    assert _parameters(method, skip) == _parameters(function, skip)


def test_to_raster_metadata_keywords_match_to_cog() -> None:
    names = {"band_names", "color_interp", "tags", "band_tags", "units"}
    to_raster = [p for p in _parameters(VRT.to_raster, 0) if p[0] in names]
    to_cog = [p for p in _parameters(VRT.to_cog, 0) if p[0] in names]
    assert len(to_raster) == len(names)
    assert to_raster == to_cog


class TestChaining:
    def test_reproject_equals_the_function(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        kwargs: dict[str, Any] = {"crs": "EPSG:3857", "resolution": 20.0, "dtype": "float64"}
        assert source.reproject(**kwargs) == vrt.reproject(source, **kwargs)

    def test_derive_equals_the_function(self, tmp_path: Path) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", count=2, fill=100.0))
        kwargs: dict[str, Any] = {"nodata": -1.0, "description": "dB", "band": 2}
        assert source.derive(PixelFunction.db(), **kwargs) == vrt.derive(
            source, PixelFunction.db(), **kwargs
        )

    def test_a_chain_reads_like_the_nested_calls(self, tmp_path: Path) -> None:
        tiles = _tiles(tmp_path)
        chained = (
            mosaic(tiles).reproject(crs="EPSG:3857", resolution=20.0).derive(PixelFunction.db())
        )
        nested = vrt.derive(
            vrt.reproject(mosaic(tiles), crs="EPSG:3857", resolution=20.0), PixelFunction.db()
        )
        assert chained == nested
        with chained.open() as ds:
            values = ds.read(1)
        # The tiles hold 1 and 2: 0 dB and 3 dB.
        assert np.nanmin(values) == 0
        assert np.nanmax(values) == pytest.approx(10 * np.log10(2), rel=1e-6)


class TestToVrt:
    def test_writes_and_returns_the_path(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        out = source.to_vrt(tmp_path / "m.vrt")
        assert out == UPath(tmp_path / "m.vrt")
        assert vrt.read(out) == source

    def test_write_is_a_deprecated_alias(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        with pytest.warns(DeprecationWarning, match=r"VRT\.write .*use VRT\.to_vrt"):
            out = source.write(tmp_path / "m.vrt", relative=False)
        assert out.read_text() == source.to_vrt(tmp_path / "n.vrt", relative=False).read_text()


class TestToCog:
    def test_equals_write_cog(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        result = source.to_cog(tmp_path / "a.tif", {"dtype": "uint8", "nodata": 255})
        assert_type(result, None)
        write_cog(source, tmp_path / "b.tif", {"dtype": "uint8", "nodata": 255})
        assert (tmp_path / "a.tif").read_bytes() == (tmp_path / "b.tif").read_bytes()

    def test_checksum_is_of_the_file(self, tmp_path: Path) -> None:
        dst = tmp_path / "o.tif"
        digest = mosaic(_tiles(tmp_path)).to_cog(dst, checksum=True, band_names=["VV"])
        assert_type(digest, str)
        assert digest == "1220" + hashlib.sha256(dst.read_bytes()).hexdigest()
        with rasterio.open(dst) as ds:
            assert ds.driver == "GTiff"
            assert ds.descriptions == ("VV",)


class TestToRaster:
    def test_writes_a_tiled_gtiff_by_default(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        dst = tmp_path / "o.tif"
        source.to_raster(dst, band_names=["VV"], tags={"A": "1"}, units=["dB"])
        with rasterio.open(dst) as ds, source.open() as expected:
            np.testing.assert_array_equal(ds.read(), expected.read())
            assert ds.driver == "GTiff"
            assert ds.compression is not None
            assert ds.compression.value == "DEFLATE"
            assert ds.profile["tiled"]
            assert ds.descriptions == ("VV",)
            assert ds.tags()["A"] == "1"
            assert ds.units == ("dB",)
        assert sorted(p.name for p in tmp_path.iterdir()) == ["l.tif", "o.tif", "r.tif"]

    def test_casts_and_remaps_nodata(self, tmp_path: Path) -> None:
        source = _uint16(tmp_path)
        source.to_raster(tmp_path / "o.tif", {"dtype": "int32", "nodata": -1})
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.dtypes == ("int32",)
            assert ds.nodata == -1
            values = ds.read(1)
        expected = _read(tmp_path / "u.tif")[0].astype("int32")
        expected[expected == 0] = -1
        np.testing.assert_array_equal(values, expected)

    def test_nodata_that_does_not_fit_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="does not fit"):
            _uint16(tmp_path).to_raster(tmp_path / "o.tif", {"nodata": -1})

    def test_accepts_build_rasterio_profile(self, tmp_path: Path) -> None:
        profile = build_rasterio_profile({"compress": "lzw"})
        _uint16(tmp_path).to_raster(tmp_path / "o.tif", profile, driver="gtiff")
        with rasterio.open(tmp_path / "o.tif") as ds:
            assert ds.compression is not None
            assert ds.compression.value == "LZW"

    def test_profile_driver_must_match(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="profile sets driver 'GTiff' but driver is 'PNG'"):
            _uint16(tmp_path).to_raster(tmp_path / "o.png", build_rasterio_profile(), driver="PNG")
        assert not (tmp_path / "o.png").exists()

    @pytest.mark.parametrize("driver", ["COG", "cog"])
    def test_cog_driver_points_to_to_cog(self, tmp_path: Path, driver: str) -> None:
        with pytest.raises(ValueError, match="use to_cog"):
            _uint16(tmp_path).to_raster(
                tmp_path / "o.tif", {"target_srs": "EPSG:3857"}, driver=driver
            )
        assert not (tmp_path / "o.tif").exists()

    def test_png_gets_no_gtiff_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", dtype="uint8", nodata=0))
        real_copy = rasterio.shutil.copy
        passed: dict[str, Any] = {}

        def record(*args: Any, **kwargs: Any) -> None:
            passed.update(kwargs)
            real_copy(*args, **kwargs)

        monkeypatch.setattr(rasterio.shutil, "copy", record)
        profile = build_rasterio_profile({"driver": "PNG", "crs": "EPSG:4326", "count": 3})
        source.to_raster(tmp_path / "q.png", profile, driver="PNG")
        assert passed == {"driver": "PNG"}
        assert "not support" not in caplog.text

    def test_missing_directory_names_the_destination(self, tmp_path: Path) -> None:
        dst = tmp_path / "nope" / "q.png"
        source = from_raster(make_tiff(tmp_path / "a.tif", dtype="uint8", nodata=0))
        with pytest.raises(RasterWriteError, match=f"directory of {dst} does not exist"):
            source.to_raster(dst, driver="PNG")
        with pytest.raises(RasterWriteError, match=f"directory of {dst} does not exist"):
            source.to_cog(dst)

    def test_png_with_its_sidecar(self, tmp_path: Path) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", dtype="uint8", nodata=0))
        dst = tmp_path / "q.png"
        source.to_raster(dst, driver="PNG")
        with rasterio.open(dst) as ds:
            assert ds.driver == "PNG"
            # The CRS lives in GDAL's .aux.xml sidecar, which lands beside the PNG.
            assert ds.crs == source.crs
            np.testing.assert_array_equal(ds.read(), _read(tmp_path / "a.tif"))
        assert sorted(p.name for p in tmp_path.iterdir()) == ["a.tif", "q.png", "q.png.aux.xml"]

    def test_envi_keeps_its_header(self, tmp_path: Path) -> None:
        source = _uint16(tmp_path)
        source.to_raster(tmp_path / "e.img", driver="ENVI")
        assert {"e.img", "e.hdr"} <= {p.name for p in tmp_path.iterdir()}
        with rasterio.open(tmp_path / "e.img") as ds:
            assert ds.driver == "ENVI"
            np.testing.assert_array_equal(ds.read(), _read(tmp_path / "u.tif"))
        assert not list(tmp_path.glob(".*"))

    def test_png_of_float32_is_a_write_error(self, tmp_path: Path) -> None:
        source = mosaic(_tiles(tmp_path))
        with pytest.raises(RasterWriteError, match=r"Failed to write raster: .*Float32"):
            source.to_raster(tmp_path / "q.png", driver="PNG")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["l.tif", "r.tif"]

    def test_failure_keeps_the_destination(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        source = _uint16(tmp_path)
        dst = tmp_path / "o.tif"
        dst.write_bytes(b"old")
        real_copy = rasterio.shutil.copy

        def copy_then_fail(*args: Any, **kwargs: Any) -> None:
            real_copy(*args, **kwargs)
            assert list(tmp_path.glob(".o.tif.*.tmp"))
            msg = "disk full"
            raise OSError(msg)

        monkeypatch.setattr(rasterio.shutil, "copy", copy_then_fail)
        with pytest.raises(RasterWriteError, match="disk full"):
            source.to_raster(dst)
        assert dst.read_bytes() == b"old"
        assert not list(tmp_path.glob(".*"))

    def test_wrap_failure_is_a_write_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_args: Any, **_kwargs: Any) -> None:
            msg = "cannot open"
            raise rasterio.errors.RasterioIOError(msg)

        monkeypatch.setattr("freezebase.vrt._builders._stack", fail)
        with pytest.raises(RasterWriteError, match="Failed to write raster: cannot open"):
            _uint16(tmp_path).to_raster(tmp_path / "o.tif", {"nodata": 9})

    def test_unsupported_protocol_is_a_write_error(self, tmp_path: Path) -> None:
        with pytest.raises(RasterWriteError, match="Unsupported protocol 'memory'"):
            _uint16(tmp_path).to_raster("memory://bucket/o.tif")

    def test_s3_uploads_the_raster_after_its_sidecar(
        self, tmp_path: Path, s3_uploads: dict[str, bytes]
    ) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", dtype="uint8", nodata=0))
        source.to_raster("s3://bucket/q.png", driver="PNG")
        assert list(s3_uploads) == ["s3://bucket/q.png.aux.xml", "s3://bucket/q.png"]
        assert s3_uploads["s3://bucket/q.png"].startswith(b"\x89PNG")

    def test_checksum_with_a_sidecar_is_of_the_raster(self, tmp_path: Path) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", dtype="uint8", nodata=0))
        dst = tmp_path / "q.png"
        digest = _encode(source.to_xml(), UPath(dst), {"driver": "PNG"}, checksum=True)
        assert digest == "1220" + hashlib.sha256(dst.read_bytes()).hexdigest()
        assert (tmp_path / "q.png.aux.xml").exists()

    def test_s3_gtiff_is_one_upload(self, tmp_path: Path, s3_uploads: dict[str, bytes]) -> None:
        _uint16(tmp_path).to_raster("s3://bucket/o.tif")
        assert list(s3_uploads) == ["s3://bucket/o.tif"]
