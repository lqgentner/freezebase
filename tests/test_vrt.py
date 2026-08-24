"""Tests for freezebase.vrt VRT generation."""

from __future__ import annotations

from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import rasterio
from rasterio.transform import Affine, from_origin

from freezebase.vrt import _nodata_equal, build_vrt_mosaic, create_decibel_vrt, create_rgb_vrt

WIDTH, HEIGHT = 6, 4


def make_tiff(
    path: Path,
    *,
    origin: tuple[float, float] = (500000, 5200000),
    fill: float | None = None,
    dtype: str = "float32",
    count: int = 1,
    nodata: float | None = np.nan,
    transform: Affine | None = None,
    crs: str = "EPSG:32632",
) -> None:
    rng = np.random.default_rng(0)
    if fill is not None:
        data = np.full((HEIGHT, WIDTH), fill, dtype=dtype)
    else:
        data = rng.random((HEIGHT, WIDTH)).astype(dtype)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=WIDTH,
        height=HEIGHT,
        count=count,
        dtype=dtype,
        crs=crs,
        transform=transform if transform is not None else from_origin(*origin, 10, 10),
        nodata=nodata,
    ) as dst:
        for band in range(1, count + 1):
            dst.write(data, band)
        dst.set_band_description(1, "VV")


@pytest.fixture
def vv_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "vv.tif"
    make_tiff(path)
    return path


@pytest.fixture
def vh_tiff(tmp_path: Path) -> Path:
    path = tmp_path / "vh.tif"
    make_tiff(path)
    return path


class TestDecibelVrt:
    def test_creates_valid_vrt(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path)

        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        assert root.get("rasterXSize") == str(WIDTH)
        assert root.get("rasterYSize") == str(HEIGHT)
        band = root.find("VRTRasterBand")
        assert band is not None
        assert band.findtext("PixelFunctionType") == "dB"
        assert band.find("PixelFunctionArguments").get("fact") == "10"

        # GDAL must be able to open the derived VRT
        with rasterio.open(vrt_path) as src:
            assert src.width == WIDTH
            assert src.crs.to_epsg() == 32632

    def test_amplitude_uses_factor_20(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path, from_intensity=False)

        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        assert root.find("VRTRasterBand/PixelFunctionArguments").get("fact") == "20"

    def test_source_referenced_relative(self, vv_tiff: Path, tmp_path: Path) -> None:
        vrt_path = tmp_path / "vv_db.vrt"

        create_decibel_vrt(vv_tiff, vrt_path)

        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        source = root.find("VRTRasterBand/SimpleSource/SourceFilename")
        assert source.text == "vv.tif"
        assert source.get("relativeToVRT") == "1"


class TestRgbVrt:
    def test_linear_scale_band_layout(
        self,
        vv_tiff: Path,
        vh_tiff: Path,
        tmp_path: Path,
    ) -> None:
        vrt_path = tmp_path / "rgb.vrt"

        create_rgb_vrt(vv_tiff, vh_tiff, vrt_path)

        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        bands = root.findall("VRTRasterBand")
        assert [b.findtext("Description") for b in bands] == ["VV", "VH", "VV/VH"]
        assert bands[2].findtext("PixelFunctionType") == "div"

        with rasterio.open(vrt_path) as src:
            assert src.count == 3

    def test_decibel_scale_band_layout(
        self,
        vv_tiff: Path,
        vh_tiff: Path,
        tmp_path: Path,
    ) -> None:
        vrt_path = tmp_path / "rgb.vrt"

        create_rgb_vrt(vv_tiff, vh_tiff, vrt_path, decibel_scale=True)

        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        bands = root.findall("VRTRasterBand")
        assert [b.findtext("Description") for b in bands] == ["VV_dB", "VH_dB", "VV_dB-VH_dB"]
        assert bands[2].findtext("PixelFunctionType") == "diff"

    def test_rejects_mismatched_dimensions(self, vv_tiff: Path, tmp_path: Path) -> None:
        vh_path = tmp_path / "vh_big.tif"
        with rasterio.open(
            vh_path,
            "w",
            driver="GTiff",
            width=WIDTH + 1,
            height=HEIGHT,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=from_origin(500000, 5200000, 10, 10),
            nodata=np.nan,
        ) as dst:
            dst.write(np.zeros((HEIGHT, WIDTH + 1), dtype=np.float32), 1)

        with pytest.raises(ValueError, match="same size"):
            create_rgb_vrt(vv_tiff, vh_path, tmp_path / "rgb.vrt")

    def test_rejects_mismatched_crs(self, vv_tiff: Path, tmp_path: Path) -> None:
        vh_path = tmp_path / "vh_crs.tif"
        make_tiff(vh_path)
        with rasterio.open(
            vh_path,
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="float32",
            crs="EPSG:32633",
            transform=from_origin(500000, 5200000, 10, 10),
            nodata=np.nan,
        ) as dst:
            dst.write(np.zeros((HEIGHT, WIDTH), dtype=np.float32), 1)

        with pytest.raises(ValueError, match="CRS"):
            create_rgb_vrt(vv_tiff, vh_path, tmp_path / "rgb.vrt")

    def test_rejects_mismatched_geotransform(self, vv_tiff: Path, tmp_path: Path) -> None:
        # Same size and CRS, different origin: the bands cover different ground.
        vh_path = tmp_path / "vh_shifted.tif"
        make_tiff(vh_path, origin=(600000, 5300000))

        with pytest.raises(ValueError, match="geotransform"):
            create_rgb_vrt(vv_tiff, vh_path, tmp_path / "rgb.vrt")


class TestVrtXmlSafety:
    def test_special_characters_in_filename_produce_valid_xml(self, tmp_path: Path) -> None:
        # A literal '&' in the filename would break naive string interpolation.
        src = tmp_path / "a&b<test>.tif"
        make_tiff(src)
        vrt_path = tmp_path / "db.vrt"

        create_decibel_vrt(src, vrt_path)

        # Must parse as well-formed XML with the exact (unescaped) filename.
        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        source = root.find("VRTRasterBand/SimpleSource/SourceFilename")
        assert source is not None
        assert source.text == "a&b<test>.tif"


class TestBuildVrtMosaic:
    def test_mosaics_two_adjacent_tiles(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, origin=(500000, 5200000), fill=1.0)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0)
        vrt_path = tmp_path / "mosaic.vrt"

        build_vrt_mosaic([tile_a, tile_b], vrt_path)

        with rasterio.open(vrt_path) as ds:
            assert ds.width == WIDTH * 2
            assert ds.height == HEIGHT
            arr = ds.read(1)
            assert (arr[:, :WIDTH] == 1.0).all()
            assert (arr[:, WIDTH:] == 2.0).all()
            assert ds.descriptions == ("VV",)

    def test_survives_directory_move(self, tmp_path: Path) -> None:
        src_dir = tmp_path / "src"
        src_dir.mkdir()
        tile_a = src_dir / "a.tif"
        tile_b = src_dir / "b.tif"
        make_tiff(tile_a, origin=(500000, 5200000), fill=1.0)
        make_tiff(tile_b, origin=(500000, 5200000 - HEIGHT * 10), fill=2.0)
        vrt_path = src_dir / "mosaic.vrt"
        build_vrt_mosaic([tile_a, tile_b], vrt_path)

        moved_dir = tmp_path / "moved"
        src_dir.rename(moved_dir)

        with rasterio.open(moved_dir / "mosaic.vrt") as ds:
            arr = ds.read(1)
            assert (arr[:HEIGHT, :] == 1.0).all()
            assert (arr[HEIGHT:, :] == 2.0).all()

    def test_rejects_empty_input(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            build_vrt_mosaic([], tmp_path / "mosaic.vrt")

    def test_rejects_mismatched_pixel_size(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0)
        with rasterio.open(
            tile_b,
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="float32",
            crs="EPSG:32632",
            transform=from_origin(500000, 5200000, 20, 20),
            nodata=np.nan,
        ) as dst:
            dst.write(np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32), 1)

        with pytest.raises(ValueError, match="pixel size"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_mismatched_crs(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0)
        with rasterio.open(
            tile_b,
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="float32",
            crs="EPSG:32633",
            transform=from_origin(500000 + WIDTH * 10, 5200000, 10, 10),
            nodata=np.nan,
        ) as dst:
            dst.write(np.full((HEIGHT, WIDTH), 2.0, dtype=np.float32), 1)

        with pytest.raises(ValueError, match="CRS"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_mismatched_dtype(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0)
        with rasterio.open(
            tile_b,
            "w",
            driver="GTiff",
            width=WIDTH,
            height=HEIGHT,
            count=1,
            dtype="int16",
            crs="EPSG:32632",
            transform=from_origin(500000 + WIDTH * 10, 5200000, 10, 10),
            nodata=0,
        ) as dst:
            dst.write(np.full((HEIGHT, WIDTH), 2, dtype=np.int16), 1)

        with pytest.raises(ValueError, match="dtype"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_tile_outside_vrt_directory(self, tmp_path: Path) -> None:
        other = tmp_path / "other"
        other.mkdir()
        tile_a = tmp_path / "a.tif"
        tile_b = other / "b.tif"
        make_tiff(tile_a, fill=1.0)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0)

        with pytest.raises(ValueError, match="same"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_off_grid_tile(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, origin=(500000, 5200000), fill=1.0)
        # Shift by half a pixel so the tile does not fall on the common grid.
        make_tiff(tile_b, origin=(500000 + WIDTH * 10 + 5, 5200000), fill=2.0)

        with pytest.raises(ValueError, match="aligned"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_multiband_reference_tile(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0, count=2)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0, count=2)

        with pytest.raises(ValueError, match="single-band"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_multiband_non_reference_tile(self, tmp_path: Path) -> None:
        # A later multi-band tile hits the per-tile check, not the up-front one.
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0, count=2)

        with pytest.raises(ValueError, match="single-band"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_unsupported_dtype(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        make_tiff(tile_a, fill=1.0, dtype="complex64", nodata=None)

        # Complex dtypes have no GDAL name; better than emitting a broken VRT.
        with pytest.raises(ValueError, match="Unsupported tile dtype"):
            build_vrt_mosaic([tile_a], tmp_path / "mosaic.vrt")

    def test_rejects_mismatched_nodata(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0, nodata=np.nan)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0, nodata=0.0)

        # Matching CRS, dtype and pixel size is not enough; seams need NODATA too.
        with pytest.raises(ValueError, match="NODATA"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_rejects_sheared_tile(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0)
        make_tiff(
            tile_b,
            fill=2.0,
            transform=Affine(10, 2, 500000 + WIDTH * 10, 0, -10, 5200000),
        )

        with pytest.raises(ValueError, match="Rotated/sheared"):
            build_vrt_mosaic([tile_a, tile_b], tmp_path / "mosaic.vrt")

    def test_accepts_tiles_without_nodata(self, tmp_path: Path) -> None:
        tile_a = tmp_path / "a.tif"
        tile_b = tmp_path / "b.tif"
        make_tiff(tile_a, fill=1.0, nodata=None)
        make_tiff(tile_b, origin=(500000 + WIDTH * 10, 5200000), fill=2.0, nodata=None)
        vrt_path = tmp_path / "mosaic.vrt"

        build_vrt_mosaic([tile_a, tile_b], vrt_path)

        # Without a NODATA value the sources are plain SimpleSource entries.
        root = ET.parse(vrt_path).getroot()  # noqa: S314 -- parsing our own just-written fixture
        band = root.find("VRTRasterBand")
        assert band is not None
        assert band.find("NoDataValue") is None
        assert len(band.findall("SimpleSource")) == 2
        assert band.findall("ComplexSource") == []

        with rasterio.open(vrt_path) as ds:
            assert ds.width == WIDTH * 2


class TestNodataEqual:
    def test_both_none_is_equal(self) -> None:
        assert _nodata_equal(None, None)

    def test_none_against_value_is_not_equal(self) -> None:
        assert not _nodata_equal(None, 0.0)
        assert not _nodata_equal(0.0, None)

    def test_nan_matches_nan(self) -> None:
        # `nan == nan` is False, so NODATA comparison needs the special case.
        assert _nodata_equal(float("nan"), float("nan"))

    def test_nan_against_value_is_not_equal(self) -> None:
        assert not _nodata_equal(float("nan"), 0.0)

    def test_plain_values_compare_by_value(self) -> None:
        assert _nodata_equal(0.0, 0.0)
        assert not _nodata_equal(0.0, -9999.0)
