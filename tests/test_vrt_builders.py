"""Tests for the VRT builders."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
from rasterio.transform import Affine

from freezebase.vrt import VRT, Source, from_raster, stack
from freezebase.vrt._grid import raster_info
from tests.rasters import make_tiff

if TYPE_CHECKING:
    from pathlib import Path


def _read(path: Path) -> np.ndarray:
    with rasterio.open(path) as ds:
        return ds.read()


class TestFromRaster:
    def test_copies_everything(self, tmp_path: Path) -> None:
        """A 1:1 VRT matches the file in data and metadata."""
        p = make_tiff(
            tmp_path / "a.tif", dtype="uint16", count=2, nodata=0, descriptions=("a", "b")
        )
        with rasterio.open(p, "r+") as ds:
            ds.units = ("m", "m")
            ds.update_tags(X="1")
            ds.update_tags(1, Y="2")
        vrt = from_raster(p)
        assert [s.path for b in vrt.bands for s in b.sources] == [raster_info(p).ref] * 2
        assert all(
            s.src_window is None and s.dst_window is None for b in vrt.bands for s in b.sources
        )
        with rasterio.open(p) as ref, vrt.open() as ds:
            np.testing.assert_array_equal(ds.read(), ref.read())
            assert ds.crs == ref.crs
            assert ds.transform == ref.transform
            assert ds.dtypes == ref.dtypes
            assert ds.nodatavals == ref.nodatavals
            assert ds.descriptions == ("a", "b")
            assert ds.units == ("m", "m")
            assert ds.tags()["X"] == "1"
            assert ds.tags(1)["Y"] == "2"
            assert ds.colorinterp == ref.colorinterp

    def test_records_source_properties(self, tmp_path: Path) -> None:
        """Each source carries the header of its file."""
        p = make_tiff(tmp_path / "a.tif", size=(6, 4))
        props = from_raster(p).bands[0].sources[0].properties
        assert props is not None
        assert (props.width, props.height, props.dtype) == (6, 4, "float32")


class TestStack:
    def test_two_single_band_files(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=1)
        b = make_tiff(tmp_path / "b.tif", fill=2)
        with stack([a, b]).open() as ds:
            assert ds.count == 2
            np.testing.assert_array_equal(ds.read(2), _read(b)[0])

    def test_selects_bands(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", count=2)
        b = make_tiff(tmp_path / "b.tif", fill=9)
        vrt = stack([a, b], bands=[[2], None])
        assert [s.band for band in vrt.bands for s in band.sources] == [2, 1]
        with vrt.open() as ds:
            assert ds.count == 2
            np.testing.assert_array_equal(ds.read(1), _read(a)[1])

    def test_names_and_color_interp(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif")
        vrt = stack(
            [a, b],
            band_names=["x", "y"],
            color_interp=[ColorInterp.red, ColorInterp.green],
        )
        with vrt.open() as ds:
            assert ds.descriptions == ("x", "y")
            assert ds.colorinterp == (ColorInterp.red, ColorInterp.green)

    def test_rejects_different_grids(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", origin=(500010.0, 5200000.0))
        with pytest.raises(ValueError, match="use warp"):
            stack([a, b])

    def test_rejects_different_crs_and_size(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="use warp"):
            stack([a, make_tiff(tmp_path / "b.tif", crs="EPSG:32633")])
        with pytest.raises(ValueError, match="use warp"):
            stack([a, make_tiff(tmp_path / "c.tif", size=(7, 4))])

    def test_nests_vrt_inputs(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=1)
        b = make_tiff(tmp_path / "b.tif", fill=2)
        inner = stack([a, b])
        outer = stack([inner, a], bands=[[2], None])
        assert isinstance(outer.bands[0].sources[0].path, VRT)
        assert outer.bands[0].sources[0] == Source(path=inner, band=2)
        with outer.open() as ds:
            np.testing.assert_array_equal(ds.read(1), _read(b)[0])
            np.testing.assert_array_equal(ds.read(2), _read(a)[0])

    def test_band_names_length_mismatch_raises(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="band_names"):
            stack([a], band_names=["x", "y"])

    def test_color_interp_length_mismatch_raises(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="color_interp"):
            stack([a], color_interp=[ColorInterp.red, ColorInterp.green])

    def test_bands_argument_errors(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="bands"):
            stack([a], bands=[None, None])
        with pytest.raises(ValueError, match="band 2"):
            stack([a], bands=[[2]])

    def test_empty_sources_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            stack([])


class TestRasterInfo:
    def test_bounds_and_pixel_size(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", transform=Affine(10, 0, 100, 0, -5, 200), size=(3, 2))
        info = raster_info(p)
        assert (info.px, info.py) == (10, 5)
        assert tuple(info.bounds) == (100, 190, 130, 200)

    def test_reads_vrt(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", count=2)
        info = raster_info(from_raster(p))
        assert info.count == 2
        assert isinstance(info.ref, VRT)
