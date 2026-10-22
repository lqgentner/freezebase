"""Tests for the freezebase.vrt model and its XML serialization."""

from __future__ import annotations

from dataclasses import replace
import math
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp, Resampling
from rasterio.io import MemoryFile
import rasterio.shutil
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from upath import UPath

from freezebase.vrt import VRT, Band, PixelFunction, Source, SourceProperties, Warp
from freezebase.vrt._paths import gdal_path
from tests.rasters import make_tiff

DB_FLOAT32 = replace(PixelFunction.db(), transfer_type="float32")


def parse(xml: str | bytes) -> ET.Element:
    """Parse VRT XML written by the code under test or by GDAL."""
    return ET.fromstring(xml)  # noqa: S314 -- parsing our own serialization


def find(element: ET.Element, path: str) -> ET.Element:
    """Return the descendant at ``path``, failing the test if it is absent."""
    found = element.find(path)
    assert found is not None, f"no element at {path!r}"
    return found


def vrt_over(path: Path, **band_kwargs: Any) -> VRT:
    """Build a one-band VRT on the grid of ``path`` with one source reading its band 1."""
    with rasterio.open(path) as ds:
        crs, transform, width, height, dtype = (
            ds.crs,
            ds.transform,
            ds.width,
            ds.height,
            ds.dtypes[0],
        )
    source = Source(UPath(path))
    band = Band(dtype, [source], **band_kwargs)
    return VRT(crs, transform, width, height, [band])


class TestToXml:
    def test_simple_source_reads_back_the_tile(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        v = vrt_over(tile)
        assert find(parse(v.to_xml()), "VRTRasterBand/SimpleSource") is not None
        with rasterio.open(v.to_xml()) as ds, rasterio.open(tile) as src:
            assert ds.dtypes == ("float32",)
            np.testing.assert_array_equal(ds.read(1), src.read(1))

    def test_complex_source_masks_its_nodata(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", dtype="uint16", nodata=None)
        v = vrt_over(tile, nodata=0.0)
        v.bands[0].sources[0].nodata = 0.0
        root = parse(v.to_xml())
        assert find(root, "VRTRasterBand/ComplexSource/NODATA").text == "0.0"
        assert root.find("VRTRasterBand/SimpleSource") is None
        with rasterio.open(v.to_xml()) as ds:
            masked = ds.read(1, masked=True)
        assert masked.mask[0, 0]
        assert not masked.mask[0, 1]

    def test_lut_maps_values(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", fill=5.0)
        v = vrt_over(tile)
        v.bands[0].dtype = "uint8"
        v.bands[0].sources[0].lut = [(0.0, 0.0), (10.0, 255.0)]
        root = parse(v.to_xml())
        assert find(root, "VRTRasterBand/ComplexSource/LUT").text == "0.0:0.0,10.0:255.0"
        with rasterio.open(v.to_xml()) as ds:
            assert ds.dtypes == ("uint8",)
            assert set(np.unique(ds.read(1))) <= {127, 128}

    def test_derived_band_applies_db(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", fill=100.0)
        v = vrt_over(tile, pixel_function=DB_FLOAT32)
        band = find(parse(v.to_xml()), "VRTRasterBand")
        assert band.get("subClass") == "VRTDerivedRasterBand"
        assert find(band, "PixelFunctionType").text == "dB"
        assert find(band, "PixelFunctionArguments").attrib == {"fact": "10"}
        assert find(band, "SourceTransferType").text == "Float32"
        with rasterio.open(v.to_xml()) as ds:
            np.testing.assert_allclose(ds.read(1), 20.0)

    def test_metadata_description_unit_colorinterp(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        v = vrt_over(
            tile,
            description="vv",
            unit="dB",
            color_interp=ColorInterp.red,
            metadata={"B": "2"},
        )
        v.metadata = {"A": "1"}
        with rasterio.open(v.to_xml()) as ds:
            assert ds.tags()["A"] == "1"
            assert ds.tags(1)["B"] == "2"
            assert ds.descriptions == ("vv",)
            assert ds.units == ("dB",)
            assert ds.colorinterp == (ColorInterp.red,)

    def test_every_color_interp_round_trips(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        v = vrt_over(tile)
        members = list(ColorInterp)
        v.bands = [Band("float32", [Source(UPath(tile))], color_interp=ci) for ci in members]
        with rasterio.open(v.to_xml()) as ds:
            assert ds.colorinterp == tuple(members)

    def test_band_nodata_nan_is_written_as_nan(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"), nodata=math.nan)
        assert find(parse(v.to_xml()), "VRTRasterBand/NoDataValue").text == "nan"
        with rasterio.open(v.to_xml()) as ds:
            assert math.isnan(ds.nodata)

    def test_source_properties_and_resampling(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        v = vrt_over(tile)
        source = v.bands[0].sources[0]
        source.properties = SourceProperties(6, 4, "float32", 6, 4)
        source.resampling = Resampling.cubic_spline
        # GDAL reads nothing when only one of the two windows is given.
        source.src_window = source.dst_window = Window(0, 0, 6, 4)
        element = find(parse(v.to_xml()), "VRTRasterBand/SimpleSource")
        assert [child.tag for child in element] == [
            "SourceFilename",
            "SourceBand",
            "SourceProperties",
            "SrcRect",
            "DstRect",
        ]
        assert element.get("resampling") == "cubicspline"
        assert find(element, "SourceProperties").attrib == {
            "RasterXSize": "6",
            "RasterYSize": "4",
            "DataType": "Float32",
            "BlockXSize": "6",
            "BlockYSize": "4",
        }
        with rasterio.open(v.to_xml()) as ds, rasterio.open(tile) as src:
            np.testing.assert_array_equal(ds.read(1), src.read(1))

    def test_nested_vrt_source_is_inline(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=10.0)
        b = make_tiff(tmp_path / "b.tif", origin=(500060.0, 5200000.0), fill=100.0)
        whole = Window(0, 0, 6, 4)
        inner = VRT(
            CRS.from_epsg(32632),
            from_origin(500000.0, 5200000.0, 10.0, 10.0),
            12,
            4,
            [
                Band(
                    "float32",
                    [
                        Source(UPath(a), src_window=whole, dst_window=Window(0, 0, 6, 4)),
                        Source(UPath(b), src_window=whole, dst_window=Window(6, 0, 6, 4)),
                    ],
                    nodata=math.nan,
                ),
            ],
        )
        outer = VRT(
            inner.crs,
            inner.transform,
            12,
            4,
            [Band("float32", [Source(inner)], pixel_function=DB_FLOAT32)],
        )
        root = parse(outer.to_xml())
        assert find(root, "VRTRasterBand/SimpleSource/VRTDataset") is not None
        assert root.find("VRTRasterBand/SimpleSource/SourceFilename") is None
        with rasterio.open(outer.to_xml()) as ds:
            data = ds.read(1)
        np.testing.assert_allclose(data[:, :6], 10.0)
        np.testing.assert_allclose(data[:, 6:], 20.0)

    def test_nesting_requires_gdal_3_11(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inner = vrt_over(make_tiff(tmp_path / "a.tif"))
        outer = VRT(inner.crs, inner.transform, 6, 4, [Band("float32", [Source(inner)])])
        monkeypatch.setattr(rasterio, "__gdal_version__", "3.10.3")
        with pytest.raises(RuntimeError, match=r"3\.11"):
            outer.to_xml()

    def test_geographic_crs_keeps_lon_lat_axis_order(self, tmp_path: Path) -> None:
        tile = make_tiff(
            tmp_path / "a.tif", crs="EPSG:4326", origin=(7.0, 47.0), res=(0.001, 0.001)
        )
        v = vrt_over(tile)
        with rasterio.open(v.to_xml()) as ds:
            assert ds.bounds.left == pytest.approx(7.0)
            assert ds.bounds.top == pytest.approx(47.0)

    def test_float_windows_are_written_exactly(self) -> None:
        source = Source(UPath("/data/a.tif"), dst_window=Window(0.5, 0, 7.5, 4))
        v = VRT(None, from_origin(0.0, 0.0, 1.0, 1.0), 8, 4, [Band("float32", [source])])
        root = parse(v.to_xml())
        assert root.find("SRS") is None
        assert find(root, "VRTRasterBand/SimpleSource/DstRect").attrib == {
            "xOff": "0.5",
            "yOff": "0",
            "xSize": "7.5",
            "ySize": "4",
        }

    def test_dataset_children_are_in_order(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"))
        v.metadata = {"A": "1"}
        root = parse(v.to_xml())
        assert [child.tag for child in root] == [
            "Metadata",
            "SRS",
            "GeoTransform",
            "VRTRasterBand",
        ]
        assert find(root, "GeoTransform").text == "500000.0, 10.0, 0.0, 5200000.0, 0.0, -10.0"

    def test_unsupported_dtype_raises(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"))
        v.bands[0].dtype = "complex64"
        with pytest.raises(ValueError, match="complex64"):
            v.to_xml()


def warped_template(tile: Path) -> tuple[str, tuple[int, int], VRT, np.ndarray]:
    """Return GDAL's warp options, block size, grid and pixels for ``tile`` warped to WGS 84."""
    with (
        rasterio.open(tile) as src,
        WarpedVRT(src, crs="EPSG:4326") as warped,
        MemoryFile(ext=".vrt") as mem,
    ):
        expected = warped.read(1)
        rasterio.shutil.copy(warped, mem.name, driver="VRT")
        root = parse(mem.read())
        grid = VRT(
            warped.crs,
            warped.transform,
            warped.width,
            warped.height,
            [Band("float32", nodata=math.nan)],
        )
    options = find(root, "GDALWarpOptions")
    options.remove(find(options, "SourceDataset"))
    block = (int(find(root, "BlockXSize").text or 0), int(find(root, "BlockYSize").text or 0))
    return ET.tostring(options, encoding="unicode"), block, grid, expected


class TestWarp:
    def test_file_source_reads_like_warped_vrt(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        options_xml, block, v, expected = warped_template(tile)
        v.warp = Warp(UPath(tile), options_xml, block)
        root = parse(v.to_xml())
        assert root.get("subClass") == "VRTWarpedDataset"
        assert find(root, "VRTRasterBand").get("subClass") == "VRTWarpedRasterBand"
        assert [child.tag for child in root][-3:] == [
            "BlockXSize",
            "BlockYSize",
            "GDALWarpOptions",
        ]
        options = [child.tag for child in find(root, "GDALWarpOptions")]
        last_option = len(options) - 1 - options[::-1].index("Option")
        assert options[last_option + 1] == "SourceDataset"
        assert find(root, "GDALWarpOptions/SourceDataset").text == str(tile)
        with rasterio.open(v.to_xml()) as ds:
            np.testing.assert_array_equal(ds.read(1), expected)

    def test_vrt_source_is_written_as_xml_text(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        options_xml, block, v, expected = warped_template(tile)
        v.warp = Warp(vrt_over(tile), options_xml, block)
        source = find(parse(v.to_xml()), "GDALWarpOptions/SourceDataset")
        assert source.get("relativeToVRT") == "0"
        assert list(source) == []
        assert (source.text or "").startswith("<VRTDataset")
        with rasterio.open(v.to_xml()) as ds:
            np.testing.assert_array_equal(ds.read(1), expected)

    def test_options_without_option_elements_take_source_first(self) -> None:
        v = VRT(None, from_origin(0.0, 0.0, 1.0, 1.0), 2, 2, [Band("float32")])
        v.warp = Warp(UPath("/data/a.tif"), "<GDALWarpOptions><A /></GDALWarpOptions>", (2, 2))
        options = find(parse(v.to_xml()), "GDALWarpOptions")
        assert [child.tag for child in options] == ["SourceDataset", "A"]

    @pytest.mark.parametrize(("text", "expected"), [("YES", True), ("NO", False)])
    def test_vertical_shift(self, text: str, expected: bool) -> None:
        options = (
            '<GDALWarpOptions><Option name="INIT_DEST">0</Option>'
            f'<Option name="APPLY_VERTICAL_SHIFT">{text}</Option></GDALWarpOptions>'
        )
        assert Warp(UPath("/data/a.tif"), options, (256, 256)).vertical_shift is expected

    def test_no_vertical_shift_option(self) -> None:
        warp = Warp(UPath("/data/a.tif"), "<GDALWarpOptions />", (256, 256))
        assert not warp.vertical_shift


class TestPixelFunction:
    def test_args_are_strings(self) -> None:
        assert PixelFunction("scale", {"a": 1, "b": 0.5}).args == {"a": "1", "b": "0.5"}
        assert PixelFunction("dB", {"fact": 10}) == PixelFunction("dB", {"fact": "10"})

    @pytest.mark.parametrize(
        ("function", "name", "args"),
        [
            (PixelFunction.db(), "dB", {"fact": "10"}),
            (PixelFunction.db(20), "dB", {"fact": "20"}),
            (PixelFunction.div(), "div", {}),
            (PixelFunction.diff(), "diff", {}),
            (PixelFunction.scale(), "scale", {}),
            (PixelFunction.minimum(), "min", {}),
            (PixelFunction.maximum(), "max", {}),
            (PixelFunction.mean(), "mean", {}),
        ],
    )
    def test_constructors(self, function: PixelFunction, name: str, args: dict[str, str]) -> None:
        assert (function.name, function.args, function.transfer_type) == (name, args, None)

    def test_derived_band_without_args_or_transfer_type(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=6.0)
        b = make_tiff(tmp_path / "b.tif", fill=2.0)
        v = vrt_over(a, pixel_function=PixelFunction.div())
        v.bands[0].sources.append(Source(UPath(b)))
        band = find(parse(v.to_xml()), "VRTRasterBand")
        assert band.find("PixelFunctionArguments") is None
        assert band.find("SourceTransferType") is None
        with rasterio.open(v.to_xml()) as ds:
            np.testing.assert_allclose(ds.read(1), 3.0)


class TestWrite:
    """Path policy and ``VRT.write``."""

    def test_source_beside_the_vrt_is_a_bare_name(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        out = vrt_over(tile).write(tmp_path / "m.vrt")
        assert out == UPath(tmp_path / "m.vrt")
        name = find(parse(out.read_text()), ".//SourceFilename")
        assert (name.text, name.get("relativeToVRT")) == ("a.tif", "1")

    def test_source_in_a_sibling_directory_is_dotdot_relative(self, tmp_path: Path) -> None:
        (tmp_path / "tiles").mkdir()
        (tmp_path / "vrts").mkdir()
        tile = make_tiff(tmp_path / "tiles" / "a.tif")
        vrt_over(tile).write(tmp_path / "vrts" / "m.vrt")
        name = find(parse((tmp_path / "vrts" / "m.vrt").read_text()), ".//SourceFilename")
        assert (name.text, name.get("relativeToVRT")) == ("../tiles/a.tif", "1")
        moved = tmp_path.parent / f"{tmp_path.name}_moved"
        tmp_path.rename(moved)
        with rasterio.open(moved / "vrts" / "m.vrt") as ds:
            assert ds.read(1).shape == (4, 6)

    def test_relative_false_writes_absolute(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        vrt_over(tile).write(tmp_path / "m.vrt", relative=False)
        name = find(parse((tmp_path / "m.vrt").read_text()), ".//SourceFilename")
        assert (name.text, name.get("relativeToVRT")) == (str(tile.resolve()), "0")

    def test_s3_source_same_bucket_is_relative(self) -> None:
        got = gdal_path(UPath("s3://b/t/a.tif"), relative_to=UPath("s3://b/v/m.vrt"))
        assert got == ("../t/a.tif", True)

    def test_s3_source_other_bucket_is_absolute(self) -> None:
        got = gdal_path(UPath("s3://c/a.tif"), relative_to=UPath("s3://b/m.vrt"))
        assert got == ("/vsis3/c/a.tif", False)

    def test_local_source_under_s3_vrt_is_absolute(self, tmp_path: Path) -> None:
        tile = tmp_path / "a.tif"
        got = gdal_path(UPath(tile), relative_to=UPath("s3://b/m.vrt"))
        assert got == (str(tile.resolve()), False)

    def test_write_is_atomic_on_failure(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        dst = tmp_path / "m.vrt"
        dst.write_text("old")

        def fail_replace(_self: Any, _target: Any) -> None:
            msg = "boom"
            raise OSError(msg)

        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(Path, "replace", fail_replace)
            with pytest.raises(OSError, match="boom"):
                vrt_over(tile).write(dst)
        assert dst.read_text() == "old"
        assert list(tmp_path.glob("*.tmp")) == []

    def test_written_bytes_are_utf8(self, tmp_path: Path) -> None:
        tiles = tmp_path / "ü"
        tiles.mkdir()
        vrt = vrt_over(make_tiff(tiles / "t.tif"))
        out = vrt.write(tmp_path / "m.vrt", relative=False)
        # Previously written with the locale encoding, unlike the S3 upload.
        assert out.read_bytes() == vrt.to_xml().encode("utf-8")
        assert "ü".encode() in out.read_bytes()

    def test_write_to_s3_uploads_once(self, tmp_path: Path, s3_uploads: dict[str, bytes]) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        out = vrt_over(tile).write("s3://bucket/v/m.vrt")
        assert out == UPath("s3://bucket/v/m.vrt")
        assert list(s3_uploads) == ["s3://bucket/v/m.vrt"]
        name = find(parse(s3_uploads["s3://bucket/v/m.vrt"]), ".//SourceFilename")
        assert name.get("relativeToVRT") == "0"

    def test_special_characters_round_trip(self, tmp_path: Path) -> None:
        tiles = tmp_path / "a b&<ü"
        tiles.mkdir()
        (tmp_path / "v").mkdir()
        tile = make_tiff(tiles / "t.tif")
        inner = vrt_over(tile)
        outer = replace(
            inner,
            bands=[
                replace(inner.bands[0], sources=[replace(inner.bands[0].sources[0], path=inner)])
            ],
        )
        for vrt, name in ((inner, "plain.vrt"), (outer, "nested.vrt")):
            out = vrt.write(tmp_path / "v" / name)
            with rasterio.open(out.path) as ds:
                assert ds.read(1).shape == (4, 6)
