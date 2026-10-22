"""Tests for the freezebase.vrt model and its XML serialization."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import replace
import math
from pathlib import Path
import shutil
from typing import Any
from xml.etree import ElementTree as ET
from xml.sax.saxutils import escape

from affine import Affine
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

from freezebase.vrt import (
    VRT,
    Band,
    PixelFunction,
    Source,
    SourceProperties,
    Warp,
    _io,
    create_warped_vrt,
    mosaic,
    read,
    warp,
)
from freezebase.vrt._paths import gdal_path, parse_gdal_path
from freezebase.vrt._xml import from_xml
from tests.rasters import make_tiff

DB_FLOAT32 = replace(PixelFunction.db(), transfer_type="float32")
DATA = Path(__file__).parent / "data" / "vrt"
UTM_GRID = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0)


def parse(xml: str | bytes) -> ET.Element:
    """Parse VRT XML written by the code under test or by GDAL."""
    return ET.fromstring(xml)  # noqa: S314 -- parsing our own serialization


def find(element: ET.Element, path: str) -> ET.Element:
    """Return the descendant at ``path``, failing the test if it is absent."""
    found = element.find(path)
    assert found is not None, f"no element at {path!r}"
    return found


def vrt_over(path: Path | UPath, **band_kwargs: Any) -> VRT:
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

    def test_metadata_domains(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"))
        v.metadata = {"A": "1"}
        v.metadata_domains = {"IMAGE_STRUCTURE": {"INTERLEAVE": "PIXEL"}, "x": {"B": "2"}}
        root = parse(v.to_xml())
        assert [m.get("domain") for m in root.iter("Metadata")] == [None, "IMAGE_STRUCTURE", "x"]
        with rasterio.open(v.to_xml()) as ds:
            assert ds.tags() == {"A": "1"}
            assert ds.tags(ns="x") == {"B": "2"}
        assert from_xml(v.to_xml()) == v

    def test_empty_metadata_domain_raises(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"))
        v.metadata_domains = {"": {"A": "1"}}
        with pytest.raises(ValueError, match="domain"):
            v.to_xml()

    def test_missing_color_interp_round_trips(self, tmp_path: Path) -> None:
        """GDAL leaves out ColorInterp on bands 2..n of a multi-band warp; so does the model."""
        tile = make_tiff(tmp_path / "rgb.tif", count=3)
        gdal_xml = gdal_warped_xml(str(tile))
        with rasterio.open(gdal_xml) as ds:
            expected = ds.colorinterp
        v = from_xml(gdal_xml)
        assert [band.color_interp for band in v.bands] == [
            ColorInterp.gray,
            ColorInterp.undefined,
            ColorInterp.undefined,
        ]

        def color_interps(xml: str) -> list[str | None]:
            return [band.findtext("ColorInterp") for band in parse(xml).iter("VRTRasterBand")]

        assert color_interps(v.to_xml()) == color_interps(gdal_xml) == ["Gray", None, None]
        assert from_xml(v.to_xml()).to_xml() == v.to_xml()
        with v.open() as ds:
            assert ds.colorinterp == expected

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

    def test_vrt_in_a_symlinked_directory_opens(self, tmp_path: Path) -> None:
        (tmp_path / "scratch").mkdir()
        (tmp_path / "proj" / "tiles").mkdir(parents=True)
        (tmp_path / "proj" / "out").symlink_to(tmp_path / "scratch")
        tile = make_tiff(tmp_path / "proj" / "tiles" / "a.tif")
        out = vrt_over(tile).write(tmp_path / "proj" / "out" / "m.vrt")
        # Previously relative to the symlink's target, which GDAL does not resolve.
        name = find(parse(out.read_text()), ".//SourceFilename")
        assert (name.text, name.get("relativeToVRT")) == ("../tiles/a.tif", "1")
        with rasterio.open(out.path) as ds:
            assert ds.read(1).shape == (4, 6)
        again = read(out).write(tmp_path / "proj" / "out" / "again.vrt")
        with rasterio.open(again.path) as ds:
            assert ds.read(1).shape == (4, 6)

    def test_source_on_another_drive_is_absolute(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def other_drive(_path: str, _start: str) -> str:
            msg = "path is on mount 'D:', start on mount 'C:'"
            raise ValueError(msg)

        monkeypatch.setattr("freezebase.vrt._paths.os.path.relpath", other_drive)
        tile = tmp_path / "a.tif"
        assert gdal_path(UPath(tile), relative_to=UPath(tmp_path / "m.vrt")) == (
            str(tile),
            False,
        )

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


def gdal_warped_xml(source: str) -> str:
    """Return the VRT XML GDAL writes for ``source`` (a path or VRT XML) warped to WGS 84."""
    with (
        rasterio.open(source) as src,
        WarpedVRT(src, crs="EPSG:4326") as warped,
        MemoryFile(ext=".vrt") as mem,
    ):
        rasterio.shutil.copy(warped, mem.name, driver="VRT")
        return mem.read().decode()


def one_band(band: str = "", *, dataset: str = "", attributes: str = "") -> str:
    """Return a one-band VRT document with ``band`` as the band's children."""
    return (
        '<VRTDataset rasterXSize="6" rasterYSize="4">'
        f"{dataset}"
        f'<VRTRasterBand dataType="Float32" band="1"{attributes}>{band}</VRTRasterBand>'
        "</VRTDataset>"
    )


def source_paths(vrt: VRT) -> list[UPath]:
    """Return the file paths of the first band's sources."""
    paths = [source.path for source in vrt.bands[0].sources]
    assert all(isinstance(path, UPath) for path in paths)
    return [path for path in paths if isinstance(path, UPath)]


WARPED = (
    '<VRTDataset rasterXSize="6" rasterYSize="4" subClass="VRTWarpedDataset">'
    '<VRTRasterBand dataType="Float32" band="1" subClass="VRTWarpedRasterBand" />'
    "{}</VRTDataset>"
)
FILE_SOURCE = '<SourceFilename relativeToVRT="0">/data/a.tif</SourceFilename>'


class TestRead:
    """``vrt.read`` and the parser behind it."""

    def test_round_trip_every_feature(self, tmp_path: Path) -> None:
        (tmp_path / "tiles").mkdir()
        (tmp_path / "v").mkdir()
        a = make_tiff(tmp_path / "tiles" / "a.tif")
        b = make_tiff(tmp_path / "tiles" / "b.tif", origin=(500060.0, 5200000.0))
        whole = Window(0, 0, 6, 4)
        mosaic = Band(
            "float32",
            [
                Source(
                    UPath(a),
                    src_window=whole,
                    dst_window=whole,
                    nodata=math.nan,
                    resampling=Resampling.cubic_spline,
                    properties=SourceProperties(6, 4, "float32", 6, 4),
                ),
                Source(
                    UPath(b),
                    src_window=whole,
                    dst_window=Window(6, 0, 6, 4),
                    lut=[(0.0, 0.0), (10.0, 255.0)],
                ),
            ],
            nodata=math.nan,
            description="vv",
            color_interp=ColorInterp.red,
            unit="dB",
            metadata={"B": "2"},
            scale=0.01,
            offset=5.0,
        )
        # A warp as GDAL writes it, whose source is an in-memory VRT.
        warped = from_xml(gdal_warped_xml(vrt_over(a).to_xml()))
        derived = Band(
            "float32",
            [Source(vrt_over(b)), Source(warped)],
            pixel_function=DB_FLOAT32,
        )
        v = VRT(CRS.from_epsg(32632), UTM_GRID, 12, 4, [mosaic, derived], metadata={"A": "1"})
        out = v.write(tmp_path / "v" / "m.vrt")
        assert find(parse(out.read_text()), ".//SourceFilename").text == "../tiles/a.tif"
        assert read(out) == v

    def test_scale_and_offset_reach_gdal(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"), scale=0.5, offset=-1.0)
        band = find(parse(v.to_xml()), "VRTRasterBand")
        assert (find(band, "Scale").text, find(band, "Offset").text) == ("0.5", "-1.0")
        with v.open() as ds:
            assert (ds.scales, ds.offsets) == ((0.5,), (-1.0,))
        assert vrt_over(make_tiff(tmp_path / "b.tif")).bands[0].scale is None
        assert "<Scale>" not in vrt_over(make_tiff(tmp_path / "c.tif")).to_xml()

    def test_round_trip_warp_with_relative_source(self, tmp_path: Path) -> None:
        (tmp_path / "v").mkdir()
        tile = make_tiff(tmp_path / "a.tif")
        v = from_xml(gdal_warped_xml(str(tile)))
        assert v.warp is not None
        assert v.warp.source == UPath(tile)
        assert "SourceDataset" not in v.warp.options_xml
        assert v.bands == [Band("float32", nodata=math.nan)]
        out = v.write(tmp_path / "v" / "w.vrt")
        source = find(parse(out.read_text()), "GDALWarpOptions/SourceDataset")
        assert (source.text, source.get("relativeToVRT")) == ("../a.tif", "1")
        assert read(out) == v
        with (
            rasterio.open(out.path) as ds,
            rasterio.open(tile) as src,
            WarpedVRT(src, crs="EPSG:4326") as expected,
        ):
            np.testing.assert_array_equal(ds.read(1), expected.read(1))

    def test_warp_of_in_memory_vrt_parses_its_escaped_source(self, tmp_path: Path) -> None:
        inner = vrt_over(make_tiff(tmp_path / "a.tif"))
        v = from_xml(gdal_warped_xml(inner.to_xml()))
        assert v.warp is not None
        assert v.warp.source == inner

    def test_round_trip_keeps_s3_storage_options(self, s3_uploads: dict[str, bytes]) -> None:
        sources = [Source(UPath("s3://b/t/a.tif")), Source(UPath("s3://c/a.tif"))]
        v = VRT(CRS.from_epsg(32632), UTM_GRID, 6, 4, [Band("float32", sources)])
        v.write(UPath("s3://b/v/m.vrt", anon=False, key="k"))
        xml = s3_uploads["s3://b/v/m.vrt"].decode()
        assert [name.text for name in parse(xml).iter("SourceFilename")] == [
            "../t/a.tif",
            "/vsis3/c/a.tif",
        ]
        paths = source_paths(from_xml(xml, base=UPath("s3://b/v/m.vrt", key="k")))
        assert [str(path) for path in paths] == ["s3://b/t/a.tif", "s3://c/a.tif"]
        assert all(path.protocol == "s3" for path in paths)
        assert [path.storage_options["key"] for path in paths] == ["k", "k"]

    def test_storage_options_override_inheritance(self, tmp_path: Path) -> None:
        sources = [Source(UPath("s3://b/t/a.tif")), Source(UPath("s3://c/a.tif"))]
        v = VRT(None, UTM_GRID, 6, 4, [Band("float32", sources)])
        local = read(v.write(tmp_path / "m.vrt"), storage_options={"profile": "p"})
        assert [path.storage_options for path in source_paths(local)] == [{"profile": "p"}] * 2
        on_s3 = from_xml(
            v.to_xml(relative_to=UPath("s3://b/v/m.vrt")),
            base=UPath("s3://b/v/m.vrt", key="k"),
            storage_options={"profile": "p"},
        )
        # The relative source keeps the VRT path's options; the other bucket takes the override.
        assert [path.storage_options for path in source_paths(on_s3)] == [
            {"key": "k"},
            {"profile": "p"},
        ]

    @pytest.mark.parametrize("path", ["s3://b/v/m.vrt", UPath("s3://b/v/m.vrt", key="k")])
    def test_s3_vrt_options_are_replaced_by_storage_options(
        self, monkeypatch: pytest.MonkeyPatch, path: str | UPath
    ) -> None:
        sources = [Source(UPath("s3://b/t/a.tif")), Source(UPath("s3://c/a.tif"))]
        xml = VRT(None, UTM_GRID, 6, 4, [Band("float32", sources)])
        read_with: list[dict[str, Any]] = []

        def fake_read_text(self: UPath, *_args: Any, **_kwargs: Any) -> str:
            read_with.append(dict(self.storage_options))
            return xml.to_xml(relative_to=UPath("s3://b/v/m.vrt"))

        monkeypatch.setattr(type(UPath("s3://b/v/m.vrt")), "read_text", fake_read_text)
        v = read(path, storage_options={"profile": "p"})
        assert read_with == [{"profile": "p"}]
        assert [str(p) for p in source_paths(v)] == ["s3://b/t/a.tif", "s3://c/a.tif"]
        assert [p.storage_options for p in source_paths(v)] == [{"profile": "p"}] * 2

    def test_reads_freezebase_0_8_mosaic(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", origin=(500060.0, 5200000.0))
        fixture = shutil.copy(DATA / "legacy_mosaic_0_8.vrt", tmp_path / "m.vrt")
        v = read(fixture)
        assert (v.crs, v.transform, v.width, v.height) == (CRS.from_epsg(32632), UTM_GRID, 12, 4)
        (band,) = v.bands
        assert band.nodata is not None
        assert math.isnan(band.nodata)
        assert band.description == ""
        assert source_paths(v) == [UPath(a), UPath(b)]
        assert [s.dst_window for s in band.sources] == [Window(0, 0, 6, 4), Window(6, 0, 6, 4)]
        assert all(s.nodata is not None and math.isnan(s.nodata) for s in band.sources)
        assert len(parse(v.to_xml()).findall(".//ComplexSource")) == 2
        with rasterio.open(fixture) as old, rasterio.open(v.to_xml()) as new:
            np.testing.assert_array_equal(new.read(1), old.read(1))

    def test_reads_freezebase_0_8_decibel(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=100.0)
        fixture = shutil.copy(DATA / "legacy_decibel_0_8.vrt", tmp_path / "db.vrt")
        v = read(fixture)
        (band,) = v.bands
        assert band.pixel_function == DB_FLOAT32
        assert band.description == "dB"
        assert band.sources == [Source(UPath(a))]
        with rasterio.open(v.to_xml()) as ds:
            np.testing.assert_allclose(ds.read(1), 20.0)

    def test_reads_gdalbuildvrt_output(self, tmp_path: Path) -> None:
        make_tiff(tmp_path / "a.tif")
        make_tiff(tmp_path / "b.tif", origin=(500060.0, 5200000.0))
        fixture = shutil.copy(DATA / "gdalbuildvrt.vrt", tmp_path / "m.vrt")
        v = read(fixture)
        assert (v.crs, v.transform) == (CRS.from_epsg(32632), UTM_GRID)
        assert [s.properties for s in v.bands[0].sources] == [
            SourceProperties(6, 4, "float32", 6, 4)
        ] * 2
        with rasterio.open(fixture) as old, rasterio.open(v.to_xml()) as new:
            np.testing.assert_array_equal(new.read(1), old.read(1))

    def test_unknown_element_raises(self) -> None:
        with pytest.raises(ValueError, match="<OverviewList>"):
            from_xml(one_band(dataset="<OverviewList>2 4</OverviewList>"))

    def test_escaped_xml_source_parses_as_nested(self) -> None:
        inner = VRT(None, UTM_GRID, 6, 4, [Band("float32", [Source(UPath("/data/a.tif"))])])
        xml = one_band(
            f'<SimpleSource><SourceFilename relativeToVRT="0">{escape(inner.to_xml())}'
            "</SourceFilename><SourceBand>1</SourceBand></SimpleSource>"
        )
        assert from_xml(xml).bands[0].sources == [Source(inner)]

    def test_unsupported_vsi_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="Unsupported GDAL path"):
            parse_gdal_path("/vsizip/a.zip/t.tif", relative=False, base=None, storage_options=None)

    def test_color_interp_is_case_insensitive(self) -> None:
        v = from_xml(one_band("<ColorInterp>ycbcr_y</ColorInterp>"))
        assert v.bands[0].color_interp == ColorInterp.Y

    @pytest.mark.parametrize(
        ("text", "relative", "base", "expected"),
        [
            ("/vsicurl/https://ex.com/a.tif", False, None, UPath("https://ex.com/a.tif")),
            ("/data/a.tif", False, None, UPath("/data/a.tif")),
            ("../t/a.tif", True, UPath("/data/v/m.vrt"), UPath("/data/t/a.tif")),
            ("a.tif", True, UPath("https://ex.com/d/m.vrt"), UPath("https://ex.com/d/a.tif")),
        ],
    )
    def test_parse_gdal_path(
        self, text: str, relative: bool, base: UPath | None, expected: UPath
    ) -> None:
        got = parse_gdal_path(text, relative=relative, base=base, storage_options=None)
        assert got == expected

    @pytest.mark.parametrize(
        ("xml", "match"),
        [
            ("<Foo />", "<Foo>"),
            ('<VRTDataset rasterYSize="4" />', "rasterXSize"),
            (one_band(attributes=' blockXSize="128"'), "blockXSize"),
            (one_band('<Metadata domain="x" />'), "domain"),
            (one_band(dataset="<Metadata><Foo /></Metadata>"), "<Foo>"),
            (one_band("<ColorTable />"), "<ColorTable>"),
            (one_band("<ColorInterp>Sepia</ColorInterp>"), "Sepia"),
            (one_band().replace("Float32", "CFloat32"), "CFloat32"),
            (one_band("<PixelFunctionType>dB</PixelFunctionType>"), "<PixelFunctionType>"),
            (one_band(attributes=' subClass="VRTDerivedRasterBand"'), "PixelFunctionType"),
            (one_band(attributes=' subClass="VRTSourcedRasterBand"'), "VRTSourcedRasterBand"),
            (
                one_band("<SimpleSource><SourceBand>1</SourceBand></SimpleSource>"),
                "SourceFilename",
            ),
            (
                one_band(f"<SimpleSource>{FILE_SOURCE}<NODATA>0</NODATA></SimpleSource>"),
                "<NODATA>",
            ),
            (one_band(f'<SimpleSource resampling="blur">{FILE_SOURCE}</SimpleSource>'), "blur"),
            (
                one_band(
                    '<SimpleSource><SourceFilename relativeToVRT="1">a.tif</SourceFilename>'
                    "</SimpleSource>"
                ),
                "relative",
            ),
            (one_band("<BlockXSize>2</BlockXSize>"), "<BlockXSize>"),
            (one_band(dataset="<GDALWarpOptions />"), "<GDALWarpOptions>"),
            (WARPED.format(""), "GDALWarpOptions"),
            (WARPED.format("<GDALWarpOptions />"), "BlockXSize"),
            (
                WARPED.format(
                    "<BlockXSize>2</BlockXSize><BlockYSize>2</BlockYSize><GDALWarpOptions />"
                ),
                "SourceDataset",
            ),
            (one_band().replace("<VRTDataset", '<VRTDataset subClass="Foo"'), "Foo"),
        ],
    )
    def test_malformed_raises(self, xml: str, match: str) -> None:
        with pytest.raises(ValueError, match=match):
            from_xml(xml)


class TestNodataEquality:
    """``Band`` and ``Source`` compare NaN nodata as equal."""

    def test_nan_nodata_is_equal(self) -> None:
        path = UPath("/data/a.tif")
        assert Source(path, nodata=math.nan) == Source(path, nodata=float("nan"))
        assert Band("float32", nodata=math.nan) == Band("float32", nodata=float("nan"))

    @pytest.mark.parametrize(("a", "b"), [(math.nan, None), (math.nan, 0.0), (0.0, 1.0)])
    def test_other_nodata_differs(self, a: float | None, b: float | None) -> None:
        path = UPath("/data/a.tif")
        assert Source(path, nodata=a) != Source(path, nodata=b)
        assert Band("float32", nodata=a) != Band("float32", nodata=b)

    def test_other_fields_still_count(self) -> None:
        assert Source(UPath("/a.tif"), nodata=math.nan) != Source(UPath("/b.tif"), nodata=math.nan)
        assert Band("float32") != Band("uint8")
        assert Band("float32") != "float32"
        assert Source(UPath("/a.tif")) != "/a.tif"


class TestSourcesAndEnv:
    def test_sources_are_ordered_and_unique(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        b = UPath(make_tiff(tmp_path / "b.tif"))
        inner = vrt_over(a)
        bands = [
            Band("float32", [Source(a), Source(inner), Source(b)]),
            Band("float32", [Source(UPath(str(b))), Source(inner), Source(a)]),
        ]
        v = VRT(inner.crs, inner.transform, inner.width, inner.height, bands)
        assert v.sources == [a, inner, b]
        assert v.sources[1] is inner

    def test_warp_source_is_included(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        v = vrt_over(a)
        v.bands[0].sources = []
        v.warp = Warp(a, "<GDALWarpOptions/>", (64, 64))
        assert v.sources == [a]

    def test_source_files_follow_vrt_files(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        b = UPath(make_tiff(tmp_path / "b.tif"))
        warped = vrt_over(a)
        warped.bands[0].sources.append(Source(b))
        warped_path = warped.write(tmp_path / "warped.vrt")
        outer = vrt_over(warped_path)
        outer.bands[0].sources.append(Source(a))
        assert outer.source_files() == [a, b]

    def test_source_files_recurse_into_nested_vrts(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        b = UPath(make_tiff(tmp_path / "b.tif"))
        inner = vrt_over(b)
        outer = vrt_over(a)
        outer.bands[0].sources.append(Source(inner))
        assert outer.source_files() == [a, b]

    def test_source_files_without_follow_stop_at_vrt_files(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        warped_path = vrt_over(a).write(tmp_path / "warped.vrt")
        outer = vrt_over(warped_path)
        assert outer.source_files(follow=False) == [warped_path]

    def test_source_files_pass_storage_options_to_followed_vrts(self, tmp_path: Path) -> None:
        tile = UPath("s3://b/t.tif")
        inner = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(tile)])])
        outer = VRT(
            None, UTM_GRID, 1, 1, [Band("float32", [Source(inner.write(tmp_path / "w.vrt"))])]
        )
        found = outer.source_files(storage_options={"key": "k"})
        assert [(str(p), dict(p.storage_options)) for p in found] == [
            ("s3://b/t.tif", {"key": "k"})
        ]

    def test_env_local_is_plain(self, tmp_path: Path) -> None:
        v = vrt_over(make_tiff(tmp_path / "a.tif"))
        with v.env():
            assert not any(key.startswith("AWS_") for key in rasterio.env.getenv())

    def test_env_without_sources_is_plain(self) -> None:
        with VRT(None, UTM_GRID, 1, 1).env():
            pass

    def test_env_http_uses_http_env(self) -> None:
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(UPath("https://h/a.tif"))])])
        with v.env():
            assert rasterio.env.getenv()["GDAL_DISABLE_READDIR_ON_OPEN"] == "EMPTY_DIR"

    def test_env_s3_uses_s3_env(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        calls: list[UPath] = []

        @contextmanager
        def recorder(path: UPath) -> Iterator[None]:
            calls.append(path)
            yield

        monkeypatch.setattr("freezebase.s3.s3_env", recorder)
        first = UPath("s3://a/x.tif", key="k")
        sources = [Source(UPath(make_tiff(tmp_path / "l.tif"))), Source(first)]
        sources.append(Source(UPath("s3://b/y.tif", key="k")))
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", sources)])
        with v.env():
            pass
        assert calls == [first]

    def test_env_rejects_mixed_credentials(self) -> None:
        sources = [
            Source(UPath("s3://a/x.tif", key="k1")),
            Source(UPath("s3://b/y.tif", key="k2")),
        ]
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", sources)])
        with pytest.raises(ValueError, match="different credentials") as info:
            v.env()
        assert "s3://a/x.tif" in str(info.value)
        assert "s3://b/y.tif" in str(info.value)

    def test_env_rejects_s3_mixed_with_http(self) -> None:
        sources = [Source(UPath("s3://a/x.tif", key="k1")), Source(UPath("https://h/y.tif"))]
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", sources)])
        with pytest.raises(ValueError, match="different credentials"):
            v.env()

    def test_env_rejects_unsupported_protocol(self) -> None:
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(UPath("memory://x.tif"))])])
        with pytest.raises(ValueError, match="Unsupported protocol"):
            v.env()

    def test_open_reads_without_a_file(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        v = vrt_over(tile)
        with v.open() as ds, rasterio.open(tile) as src:
            np.testing.assert_array_equal(ds.read(1), src.read(1))


FOREIGN_VRT = """<VRTDataset rasterXSize="6" rasterYSize="4">
  <SRS>EPSG:32632</SRS>
  <GeoTransform>500000, 10, 0, 5200000, 0, -10</GeoTransform>
  <VRTRasterBand dataType="Float32" band="1">
    <ComplexSource>
      <SourceFilename relativeToVRT="1">a.tif</SourceFilename>
      <SourceBand>1</SourceBand>
      <UseMaskBand>true</UseMaskBand>
    </ComplexSource>
  </VRTRasterBand>
  <OverviewList resampling="average">2</OverviewList>
</VRTDataset>"""


def _one_source_vrt(text: str, *, relative: bool = False) -> str:
    """Return a one-band VRT whose single source is ``text``."""
    flag = "1" if relative else "0"
    return (
        '<VRTDataset rasterXSize="1" rasterYSize="1"><VRTRasterBand dataType="Float32" band="1">'
        f'<SimpleSource><SourceFilename relativeToVRT="{flag}">{escape(text)}</SourceFilename>'
        "</SimpleSource></VRTRasterBand></VRTDataset>"
    )


@pytest.fixture
def s3_env_calls(monkeypatch: pytest.MonkeyPatch) -> list[UPath]:
    """Record the paths ``s3_env`` is entered with, without touching S3."""
    calls: list[UPath] = []

    @contextmanager
    def recorder(path: UPath) -> Iterator[None]:
        calls.append(path)
        yield

    monkeypatch.setattr("freezebase.s3.s3_env", recorder)
    return calls


class TestDiscovery:
    """Source discovery follows VRT files GDAL reads, even ones ``read`` refuses."""

    def test_env_takes_storage_options_through_local_vrts(
        self, tmp_path: Path, s3_env_calls: list[UPath]
    ) -> None:
        opts = {"profile": "research", "endpoint_url": "https://s3.example"}
        tile = UPath("s3://bucket/tiles/t0.tif", **opts)
        warped = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(tile)])]).write(
            tmp_path / "warped.vrt"
        )
        final = read(vrt_over(warped).write(tmp_path / "final.vrt"), storage_options=opts)
        # Previously env() read the local warped.vrt without options: a bare s3:// path.
        with final.env(storage_options=opts):
            pass
        assert [(str(p), dict(p.storage_options)) for p in s3_env_calls] == [
            ("s3://bucket/tiles/t0.tif", opts)
        ]

    def test_open_passes_storage_options_to_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        seen: list[Any] = []
        real = _io.env

        def spy(vrt: VRT, *, storage_options: Any = None) -> Any:
            seen.append(storage_options)
            return real(vrt, storage_options=storage_options)

        monkeypatch.setattr("freezebase.vrt._io.env", spy)
        with vrt_over(make_tiff(tmp_path / "a.tif")).open(storage_options={"key": "k"}) as ds:
            assert ds.count == 1
        assert seen == [{"key": "k"}]

    def test_followed_vrt_inherits_the_referencing_vrts_options(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, s3_env_calls: list[UPath]
    ) -> None:
        local = tmp_path / "local.vrt"
        local.write_text(_one_source_vrt("/vsis3/tiles/t0.tif"))
        mid = _one_source_vrt(str(local))
        real = UPath.read_text

        def fake_read(path: UPath, *args: Any, **kwargs: Any) -> str:
            return mid if path.protocol == "s3" else real(path, *args, **kwargs)

        monkeypatch.setattr(UPath, "read_text", fake_read)
        top = VRT(
            None, UTM_GRID, 1, 1, [Band("float32", [Source(UPath("s3://b/mid.vrt", key="k"))])]
        )
        assert [dict(p.storage_options) for p in top.source_files()] == [{"key": "k"}]
        with top.env():
            pass
        assert [dict(path.storage_options) for path in s3_env_calls] == [{"key": "k"}]

    def test_foreign_elements_do_not_stop_discovery(self, tmp_path: Path) -> None:
        tile = UPath(make_tiff(tmp_path / "a.tif"))
        foreign = tmp_path / "foreign.vrt"
        foreign.write_text(FOREIGN_VRT)
        with pytest.raises(ValueError, match="UseMaskBand"):
            read(foreign)
        # Previously discovery used the strict parser and raised as read() does.
        assert vrt_over(foreign).source_files() == [tile]
        with mosaic([foreign]).open() as ds:
            assert ds.read(1).shape == (4, 6)

    def test_warp_of_a_foreign_vrt(self, tmp_path: Path) -> None:
        make_tiff(tmp_path / "a.tif")
        foreign = tmp_path / "foreign.vrt"
        foreign.write_text(FOREIGN_VRT)
        with warp(foreign, crs="EPSG:3857", resolution=20.0).open() as ds:
            assert ds.count == 1
        create_warped_vrt(foreign, tmp_path / "w.vrt", crs="EPSG:3857", resolution=20.0)

    def test_cycles_end(self, tmp_path: Path) -> None:
        tile = UPath(make_tiff(tmp_path / "t.tif"))
        a, b = tmp_path / "a.vrt", tmp_path / "b.vrt"
        tile_source = '<SimpleSource><SourceFilename relativeToVRT="1">t.tif</SourceFilename>'
        a.write_text(
            _one_source_vrt("b.vrt", relative=True).replace(
                "</VRTRasterBand>", f"{tile_source}</SimpleSource></VRTRasterBand>"
            )
        )
        b.write_text(_one_source_vrt("a.vrt", relative=True))
        top = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(UPath(a))])])
        # Previously a RecursionError.
        assert top.source_files() == [tile]

    def test_follows_escaped_and_warped_sources(self, tmp_path: Path) -> None:
        a = UPath(make_tiff(tmp_path / "a.tif"))
        b = UPath(make_tiff(tmp_path / "b.tif"))
        warped = warp(vrt_over(b), crs="EPSG:3857", resolution=20.0)
        mixed = vrt_over(a)
        mixed.bands[0].sources.append(Source(warped))
        path = warp(mixed.write(tmp_path / "mixed.vrt"), crs="EPSG:3857", resolution=20.0).write(
            tmp_path / "w.vrt"
        )
        assert vrt_over(path).source_files() == [a, b]

    def test_same_path_with_other_options_is_kept(self) -> None:
        first, second = UPath("s3://a/x.tif", key="k1"), UPath("s3://a/x.tif", key="k2")
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(first), Source(second)])])
        # Previously deduplicated by path, hiding the conflict until GDAL read the file.
        assert v.sources == [first, second]
        with pytest.raises(ValueError, match="different credentials"):
            v.env()

    def test_https_vrt_without_aiohttp_raises_clearly(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_module(_name: str) -> None:
            return None

        monkeypatch.setattr("freezebase.vrt._io.find_spec", no_module)
        v = VRT(None, UTM_GRID, 1, 1, [Band("float32", [Source(UPath("https://h/m.vrt"))])])
        with pytest.raises(ImportError, match="aiohttp"):
            v.source_files()
        with pytest.raises(ImportError, match="aiohttp"):
            read("https://h/m.vrt")
