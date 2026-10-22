"""Tests for the VRT builders."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal

import numpy as np
import pytest
import rasterio
from rasterio.coords import disjoint_bounds
from rasterio.enums import ColorInterp, Resampling
from rasterio.transform import Affine
from rasterio.warp import reproject
from rasterio.windows import Window

from freezebase.vrt import VRT, PixelFunction, Source, derive, from_raster, mosaic, stack
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


class TestMosaicAligned:
    def test_two_adjacent_single_band_tiles(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", origin=(500060.0, 5200000.0))
        vrt = mosaic([a, b])
        assert (vrt.width, vrt.height) == (12, 4)
        with vrt.open() as ds:
            data = ds.read(1)
        np.testing.assert_array_equal(data[:, :6], _read(a)[0])
        np.testing.assert_array_equal(data[:, 6:], _read(b)[0])
        source = vrt.bands[0].sources[1]
        assert source.src_window == Window(0, 0, 6, 4)
        assert source.dst_window == Window(6, 0, 6, 4)
        props = source.properties
        assert props is not None
        assert (props.width, props.height, props.dtype) == (6, 4, "float32")

    def test_three_band_tiles_keep_every_band(self, tmp_path: Path) -> None:
        names = ("r", "g", "b")
        a = make_tiff(tmp_path / "a.tif", count=3, descriptions=names)
        b = make_tiff(
            tmp_path / "b.tif",
            count=3,
            origin=(500060.0, 5200000.0),
            descriptions=("x", "y", "z"),
        )
        with mosaic([a, b]).open() as ds:
            assert ds.count == 3
            assert ds.descriptions == names
            np.testing.assert_array_equal(ds.read()[:, :, 6:], _read(b))
            np.testing.assert_array_equal(ds.read()[:, :, :6], _read(a))

    def test_later_source_wins_where_valid(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=1)
        b = make_tiff(tmp_path / "b.tif", fill=2)
        with rasterio.open(b, "r+") as ds:
            data = ds.read(1)
            data[0, 0] = np.nan
            ds.write(data, 1)
        with mosaic([a, b]).open() as ds:
            data = ds.read(1)
        assert data[0, 0] == 1
        assert data[0, 1] == 2

    def test_nodata_override_applies_to_sources(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", nodata=None)
        b = make_tiff(tmp_path / "b.tif", nodata=None, origin=(500060.0, 5200000.0))
        vrt = mosaic([a, b], nodata=0.0)
        assert vrt.bands[0].nodata == 0.0
        assert [s.nodata for s in vrt.bands[0].sources] == [0.0, 0.0]
        assert "<ComplexSource" in vrt.to_xml()
        assert "<NODATA>0" in vrt.to_xml()

    def test_rejects_mismatched_band_count(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", count=2, origin=(500060.0, 5200000.0))
        with pytest.raises(ValueError, match="band"):
            mosaic([a, b])

    def test_rejects_mixed_nodata(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", nodata=None)
        b = make_tiff(tmp_path / "b.tif", nodata=np.nan, origin=(500060.0, 5200000.0))
        with pytest.raises(ValueError, match="NODATA"):
            mosaic([a, b])

    def test_without_nodata_writes_simple_sources(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", nodata=None)
        xml = mosaic([a]).to_xml()
        assert "<SimpleSource" in xml
        assert "ComplexSource" not in xml
        assert "None" not in xml

    def test_vrt_inputs_are_nested(self, tmp_path: Path) -> None:
        a = from_raster(make_tiff(tmp_path / "a.tif", count=2))
        b = make_tiff(tmp_path / "b.tif", count=2, origin=(500060.0, 5200000.0))
        vrt = mosaic([a, b])
        assert vrt.bands[1].sources[0].path is a
        assert vrt.bands[1].sources[0].band == 2
        with vrt.open() as ds:
            np.testing.assert_array_equal(ds.read()[:, :, 6:], _read(b))

    def test_vrt_input_is_named_in_errors(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = from_raster(make_tiff(tmp_path / "b.tif", crs="EPSG:32633"))
        with pytest.raises(ValueError, match="VRT input #1"):
            mosaic([a, b])

    def test_bounds_and_off_grid_errors_as_before(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="whole number"):
            mosaic([a], bounds=(500000, 5199960, 500065, 5200000))
        off = make_tiff(tmp_path / "b.tif", origin=(500065.0, 5200000.0))
        with pytest.raises(ValueError, match="aligned"):
            mosaic([a, off])
        wide = mosaic([a], bounds=(500000, 5199960, 500120, 5200040))
        assert (wide.width, wide.height) == (12, 8)
        assert wide.bands[0].sources[0].dst_window == Window(0, 4, 6, 4)

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            mosaic([])

    def test_names_and_color_interp(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", count=2)
        vrt = mosaic([a], band_names=["p", "q"], color_interp=[ColorInterp.red, ColorInterp.green])
        assert [b.description for b in vrt.bands] == ["p", "q"]
        assert [b.color_interp for b in vrt.bands] == [ColorInterp.red, ColorInterp.green]
        with pytest.raises(ValueError, match="entries"):
            mosaic([a], band_names=["p"])
        with pytest.raises(ValueError, match="entries"):
            mosaic([a], color_interp=[ColorInterp.red])


def _reproject_onto(path: Path, vrt: VRT) -> np.ndarray:
    """Warp band 1 of ``path`` with nearest onto ``vrt``'s grid, NaN outside it."""
    out = np.full((vrt.height, vrt.width), np.nan, dtype="float32")
    with rasterio.open(path) as ds:
        reproject(
            rasterio.band(ds, 1),
            out,
            dst_transform=vrt.transform,
            dst_crs=ds.crs,
            dst_nodata=np.nan,
            resampling=Resampling.nearest,
        )
    return out


class TestMosaicMixedResolution:
    def test_coarse_tile_is_placed_exactly(self, tmp_path: Path) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        # 3 m off the fine grid: rounding would misplace it by 0.3 pixel.
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(500063.0, 5200000.0), res=(30.0, 30.0), size=(4, 2)
        )
        m = mosaic([fine, coarse], resolution="finest")
        with m.open() as ds:
            data = ds.read(1)
        ref = _reproject_onto(coarse, m)
        mask = ~np.isnan(ref) & np.isnan(_reproject_onto(fine, m))
        assert mask.sum() > 0
        np.testing.assert_array_equal(data[mask], ref[mask])
        np.testing.assert_array_equal(data[:4, :6], _read(fine)[0])
        source = m.bands[0].sources[1]
        # Output columns 6-17 have their centres in the tile, from 0.1 source pixel left of it.
        assert source.dst_window == Window(6, 0, 12, 6)
        assert source.src_window is not None
        assert source.src_window.col_off == pytest.approx(-0.1)
        assert (source.src_window.row_off, source.src_window.width) == (0, pytest.approx(4))
        assert source.src_window.height == pytest.approx(2)
        assert source.resampling == Resampling.nearest

    def test_off_grid_tile_draws_only_where_it_holds_the_pixel_centre(
        self, tmp_path: Path
    ) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        # Overlaps the fine tile, covering part of column 4 and row 1 but not their centres.
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(500047.0, 5199983.0), res=(30.0, 30.0), size=(4, 2)
        )
        m = mosaic([fine, coarse], resolution="finest")
        with m.open() as ds:
            data = ds.read(1)
        expected = _reproject_onto(fine, m)
        ref = _reproject_onto(coarse, m)
        expected[~np.isnan(ref)] = ref[~np.isnan(ref)]
        np.testing.assert_array_equal(data, expected)
        # Previously the coarse tile's edge value overwrote these fine pixels.
        np.testing.assert_array_equal(data[1:4, 4], _read(fine)[0, 1:4, 4])
        np.testing.assert_array_equal(data[1, 4:6], _read(fine)[0, 1, 4:6])

    def test_finest_is_taken_per_axis(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", res=(10.0, 20.0))
        b = make_tiff(tmp_path / "b.tif", origin=(500060.0, 5200000.0), res=(20.0, 10.0))
        m = mosaic([a, b], resolution="finest")
        assert (m.transform.a, m.transform.e) == (10.0, -10.0)
        assert (m.width, m.height) == (18, 8)

    def test_anchor_is_the_first_tile_at_target_size(self, tmp_path: Path) -> None:
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(499987.0, 5200013.0), res=(30.0, 30.0), size=(4, 2)
        )
        fine = make_tiff(tmp_path / "fine.tif")
        m = mosaic([coarse, fine], resolution="finest")
        # The union's corner (499987, 5200013) grows outward onto the fine tile's grid.
        assert m.transform == Affine(10.0, 0.0, 499980.0, 0.0, -10.0, 5200020.0)
        assert (m.width, m.height) == (13, 7)
        assert m.bands[0].sources[1].dst_window == Window(2, 2, 6, 4)

    def test_exact_edges_do_not_add_a_pixel(self, tmp_path: Path) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(500060.0, 5200000.0), res=(30.0, 30.0), size=(2, 2)
        )
        m = mosaic([fine, coarse], resolution="finest")
        assert m.transform == Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0)
        assert (m.width, m.height) == (12, 6)
        assert m.bands[0].sources[1].dst_window == Window(6, 0, 6, 6)

    def test_anchor_falls_back_to_union_corner(self, tmp_path: Path) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(499963.0, 5200005.0), res=(30.0, 30.0), size=(4, 2)
        )
        m = mosaic([fine, coarse], resolution=(7.0, 7.0))
        assert m.transform == Affine(7.0, 0.0, 499963.0, 0.0, -7.0, 5200005.0)
        assert (m.width, m.height) == (18, 9)
        assert all(s.resampling == Resampling.nearest for s in m.bands[0].sources)

    def test_explicit_bounds_anchor_the_grid(self, tmp_path: Path) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(500063.0, 5200000.0), res=(30.0, 30.0), size=(4, 2)
        )
        m = mosaic([fine, coarse], bounds=(499990, 5199930, 500200, 5200010), resolution="finest")
        assert m.transform == Affine(10.0, 0.0, 499990.0, 0.0, -10.0, 5200010.0)
        assert (m.width, m.height) == (21, 8)
        assert m.bands[0].sources[0].dst_window == Window(1, 1, 6, 4)
        with pytest.raises(ValueError, match="whole number"):
            mosaic([fine, coarse], bounds=(499990, 5199930, 500205, 5200010), resolution="finest")

    def test_coverage_is_not_lost(self, tmp_path: Path) -> None:
        paths = [
            make_tiff(
                tmp_path / "coarse.tif",
                origin=(499987.0, 5200013.0),
                res=(30.0, 30.0),
                size=(4, 2),
            ),
            make_tiff(tmp_path / "fine.tif"),
            make_tiff(tmp_path / "far.tif", origin=(500101.5, 5199902.5), res=(25.0, 15.0)),
        ]
        m = mosaic(paths, resolution="finest")
        with m.open() as ds:
            out = ds.bounds
        for path in paths:
            tile = raster_info(path).bounds
            assert out.left <= tile.left
            assert out.bottom <= tile.bottom
            assert out.right >= tile.right
            assert out.top >= tile.top
            assert not disjoint_bounds(out, tile)

    def test_on_grid_tiles_are_not_resampled(self, tmp_path: Path) -> None:
        fine = make_tiff(tmp_path / "fine.tif")
        coarse = make_tiff(
            tmp_path / "coarse.tif", origin=(500063.0, 5200000.0), res=(30.0, 30.0), size=(4, 2)
        )
        m = mosaic([fine, coarse], resolution="finest", resampling=Resampling.bilinear)
        fine_source, coarse_source = m.bands[0].sources
        assert fine_source.resampling is None
        assert fine_source.dst_window == Window(0, 0, 6, 4)
        assert coarse_source.resampling == Resampling.bilinear
        assert 'resampling="bilinear"' in m.to_xml()

    def test_same_size_tile_off_the_grid_is_resampled(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", origin=(500065.0, 5200000.0))
        m = mosaic([a, b], resolution="finest")
        source = m.bands[0].sources[1]
        assert source.dst_window == Window(6, 0, 6, 4)
        assert source.src_window == Window(-0.5, 0, 6, 4)
        assert source.resampling == Resampling.nearest

    def test_touching_half_pixel_tiles_leave_no_seam(self, tmp_path: Path) -> None:
        r, x0, y0 = 2.3, 345678.9, 5200000.1
        a = make_tiff(tmp_path / "a.tif", origin=(x0, y0), res=(r, r), fill=1.0)
        # Half a pixel off a's grid, so the pixel centres fall exactly on b's and c's edges.
        b = make_tiff(tmp_path / "b.tif", origin=(x0 + 6.5 * r, y0), res=(r, r), fill=2.0)
        c = make_tiff(tmp_path / "c.tif", origin=(x0 + 12.5 * r, y0), res=(r, r), fill=3.0)
        m = mosaic([a, b, c], resolution=(r, r))
        # Previously float noise moved c's first centre outside it, leaving a NaN column.
        assert [s.dst_window for s in m.bands[0].sources[1:]] == [
            Window(6, 0, 6, 4),
            Window(12, 0, 6, 4),
        ]
        with m.open() as ds:
            data = ds.read(1)
        expected = np.repeat([1.0, 2.0, 3.0], 6)
        np.testing.assert_array_equal(data[:, :18], np.tile(expected, (4, 1)))

    def test_tile_holding_no_pixel_centre_is_dropped(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        # Spans x 500061-500064, short of the pixel centre at 500065.
        sliver = make_tiff(
            tmp_path / "b.tif", origin=(500061.0, 5200000.0), res=(3.0, 10.0), size=(1, 4)
        )
        m = mosaic([a, sliver], resolution=(10.0, 10.0))
        assert (m.width, m.height) == (7, 4)
        assert len(m.bands[0].sources) == 1

    def test_tiles_must_still_match(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        coarse = (30.0, 30.0)
        cases = [
            (make_tiff(tmp_path / "c.tif", res=coarse, crs="EPSG:32633"), "CRS"),
            (make_tiff(tmp_path / "n.tif", res=coarse, count=2), "band count"),
            (make_tiff(tmp_path / "d.tif", res=coarse, dtype="float64"), "dtype"),
            (make_tiff(tmp_path / "z.tif", res=coarse, nodata=0.0), "NODATA"),
        ]
        for b, match in cases:
            with pytest.raises(ValueError, match=match):
                mosaic([a, b], resolution="finest")

    @pytest.mark.parametrize("resampling", [Resampling.max, Resampling.sum, Resampling.q1])
    def test_rejects_resampling_vrt_sources_lack(
        self, tmp_path: Path, resampling: Resampling
    ) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="resampling"):
            mosaic([a], resolution="finest", resampling=resampling)

    def test_accepts_rms_resampling(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        b = make_tiff(tmp_path / "b.tif", origin=(500065.0, 5200000.0))
        m = mosaic([a, b], resolution="finest", resampling=Resampling.rms)
        assert 'resampling="rms"' in m.to_xml()
        with m.open() as ds:
            assert ds.read(1).shape == (4, 13)

    def test_rejects_non_positive_resolution(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif")
        with pytest.raises(ValueError, match="positive"):
            mosaic([a], resolution=(10.0, 0.0))


def _write(
    path: Path, data: np.ndarray, *, nodata: float | None, origin_x: float = 500000.0
) -> Path:
    """Write ``data`` (bands, rows, cols) as a GeoTIFF of its own dtype and nodata."""
    count, height, width = data.shape
    make_tiff(
        path,
        size=(width, height),
        count=count,
        dtype=str(data.dtype),
        nodata=nodata,
        origin=(origin_x, 5200000.0),
    )
    with rasterio.open(path, "r+") as ds:
        ds.write(data)
    return path


class TestDerive:
    def test_db_over_an_in_memory_mosaic_is_not_zero(self, tmp_path: Path) -> None:
        """A dB band over a two-tile band reads the mosaicked pixels."""
        # Previously a dB band over a two-tile band returned 0 everywhere.
        tile_100 = make_tiff(tmp_path / "a.tif", fill=100)
        tile_1000 = make_tiff(tmp_path / "b.tif", fill=1000, origin=(500060.0, 5200000.0))
        m = mosaic([tile_100, tile_1000])
        vrt = derive(m, PixelFunction.db())
        assert isinstance(vrt.bands[0].sources[0].path, VRT)
        with vrt.open() as ds:
            a = ds.read(1)
        assert np.allclose(a, [[20.0] * 6 + [30.0] * 6] * 4)

    def test_nodata_that_does_not_fit_dtype_raises(self, tmp_path: Path) -> None:
        p = _write(tmp_path / "a.tif", np.array([[[0, 1, 2]]], dtype="uint16"), nodata=0)
        # Previously GDAL dropped the NaN nodata, and nodata inputs read as valid zeros.
        with pytest.raises(ValueError, match=r"nodata nan .*int16"):
            derive(p, PixelFunction.scale(), dtype="int16")
        with pytest.raises(ValueError, match=r"nodata 70000 .*int16"):
            derive(p, PixelFunction.scale(), dtype="int16", nodata=70000)
        assert derive(p, PixelFunction.scale(), dtype="int16", nodata=-1).bands[0].nodata == -1

    def test_db_of_nodata_zero_is_nan(self, tmp_path: Path) -> None:
        """A uint16 0 sentinel gives NaN, not -inf."""
        data = np.array([[[0, 10, 100]]], dtype="uint16")
        p = _write(tmp_path / "a.tif", data, nodata=0)
        with derive(p, PixelFunction.db()).open() as ds:
            a = ds.read(1)
        assert np.isnan(a[0, 0])
        assert np.allclose(a[0, 1:], [10.0, 20.0])

    def test_diff_of_integer_nodata_is_nan(self, tmp_path: Path) -> None:
        """An int16 -9999 in either input gives NaN."""
        x = _write(tmp_path / "x.tif", np.array([[[5, -9999, 7, 1]]], dtype="int16"), nodata=-9999)
        y = _write(tmp_path / "y.tif", np.array([[[1, 2, -9999, 1]]], dtype="int16"), nodata=-9999)
        with derive([x, y], PixelFunction.diff()).open() as ds:
            a = ds.read(1)
        np.testing.assert_array_equal(a[0], [4, np.nan, np.nan, 0])

    def test_div_of_integer_nodata_is_nan(self, tmp_path: Path) -> None:
        x = _write(tmp_path / "x.tif", np.array([[[6, -9999, 8]]], dtype="int16"), nodata=-9999)
        y = _write(tmp_path / "y.tif", np.array([[[2, 2, -9999]]], dtype="int16"), nodata=-9999)
        with derive([x, y], PixelFunction.div()).open() as ds:
            a = ds.read(1)
        np.testing.assert_array_equal(a[0], [3, np.nan, np.nan])

    def test_sum_propagates_nodata(self, tmp_path: Path) -> None:
        x = _write(tmp_path / "x.tif", np.array([[[1, -9999, 3]]], dtype="int16"), nodata=-9999)
        y = _write(tmp_path / "y.tif", np.array([[[1, 1, 1]]], dtype="int16"), nodata=-9999)
        vrt = derive([x, y], PixelFunction("sum"))
        assert vrt.bands[0].pixel_function is not None
        assert vrt.bands[0].pixel_function.args["propagateNoData"] == "true"
        with vrt.open() as ds:
            a = ds.read(1)
        np.testing.assert_array_equal(a[0], [2, np.nan, 4])

    def test_one_input_derives_every_band(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", count=2, fill=100)
        with rasterio.open(p, "r+") as ds:
            ds.write(np.full((4, 6), 1000, dtype="float32"), 2)
        vrt = derive(p, PixelFunction.db(), description="power")
        assert [b.sources[0].band for b in vrt.bands] == [1, 2]
        with vrt.open() as ds:
            assert ds.count == 2
            assert ds.descriptions == ("power", "power")
            assert np.allclose(ds.read(1), 20)
            assert np.allclose(ds.read(2), 30)

    def test_one_input_with_band_derives_only_that_band(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", count=2, fill=100)
        vrt = derive(p, PixelFunction.db(), band=2)
        assert len(vrt.bands) == 1
        assert vrt.bands[0].sources[0].band == 2

    def test_several_inputs_read_the_given_band(self, tmp_path: Path) -> None:
        x = make_tiff(tmp_path / "x.tif", count=2, fill=0)
        y = make_tiff(tmp_path / "y.tif", count=2, fill=0)
        with rasterio.open(x, "r+") as ds:
            ds.write(np.full((4, 6), 8, dtype="float32"), 2)
        with rasterio.open(y, "r+") as ds:
            ds.write(np.full((4, 6), 2, dtype="float32"), 2)
        vrt = derive([x, y], PixelFunction.div(), band=2)
        assert len(vrt.bands) == 1
        assert [s.band for s in vrt.bands[0].sources] == [2, 2]
        with vrt.open() as ds:
            assert np.allclose(ds.read(1), 4)

    def test_rejects_different_grids(self, tmp_path: Path) -> None:
        x = make_tiff(tmp_path / "x.tif")
        y = make_tiff(tmp_path / "y.tif", origin=(500060.0, 5200000.0))
        with pytest.raises(ValueError, match="share one grid"):
            derive([x, y], PixelFunction.div())

    def test_rejects_bad_band_and_empty(self, tmp_path: Path) -> None:
        x = make_tiff(tmp_path / "x.tif")
        with pytest.raises(ValueError, match="no band 2"):
            derive([x, x], PixelFunction.div(), band=2)
        with pytest.raises(ValueError, match="at least one"):
            derive([], PixelFunction.div())

    def test_vrt_input_is_named_by_position_in_errors(self, tmp_path: Path) -> None:
        x = make_tiff(tmp_path / "x.tif")
        y = from_raster(make_tiff(tmp_path / "y.tif", origin=(500060.0, 5200000.0)))
        with pytest.raises(ValueError, match="VRT input #1"):
            derive([x, y], PixelFunction.div())

    def test_source_without_nodata_is_simple(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", nodata=None, fill=100)
        vrt = derive(p, PixelFunction.db())
        assert vrt.bands[0].sources[0].nodata is None
        xml = vrt.to_xml()
        assert "<SimpleSource>" in xml
        assert "<NODATA>" not in xml
        with vrt.open() as ds:
            assert np.allclose(ds.read(1), 20)

    def test_caller_propagate_nodata_is_kept(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", fill=100)
        function = PixelFunction("dB", {"propagateNoData": "false"})
        vrt = derive(p, function)
        assert vrt.bands[0].pixel_function is not None
        assert vrt.bands[0].pixel_function.args["propagateNoData"] == "false"
        assert function.args == {"propagateNoData": "false"}

    def test_dtype_and_transfer_type(self, tmp_path: Path) -> None:
        p = make_tiff(tmp_path / "a.tif", dtype="uint16", nodata=0, fill=3)
        vrt = derive(p, PixelFunction.scale(), dtype="float64", nodata=-1.0)
        band = vrt.bands[0]
        assert (band.dtype, band.nodata) == ("float64", -1.0)
        assert band.pixel_function is not None
        assert band.pixel_function.transfer_type == "float64"
        kept = derive(p, PixelFunction("sum", transfer_type="float32")).bands[0].pixel_function
        assert kept is not None
        assert kept.transfer_type == "float32"


class TestSouthUp:
    """South-up rasters (positive pixel height) are refused, not pasted upside down."""

    @staticmethod
    def _south_up(tmp_path: Path) -> Path:
        return make_tiff(tmp_path / "south.tif", transform=Affine(10, 0, 0, 0, 10, 0))

    @pytest.mark.parametrize("resolution", [None, "finest"])
    def test_mosaic_raises(self, tmp_path: Path, resolution: Literal["finest"] | None) -> None:
        # Previously the tile's southern row became the mosaic's top row.
        with pytest.raises(ValueError, match=r"south\.tif.*south-up"):
            mosaic([self._south_up(tmp_path)], resolution=resolution)

    def test_stack_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="south-up"):
            stack([self._south_up(tmp_path)])

    def test_from_raster_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="south-up"):
            from_raster(self._south_up(tmp_path))


def _scaled(path: Path, scale: float = 0.01, offset: float = 5.0, **kwargs: Any) -> Path:
    """Write an int16 raster with a scale and offset."""
    make_tiff(path, dtype="int16", nodata=-1, **kwargs)
    with rasterio.open(path, "r+") as ds:
        ds.scales, ds.offsets = (scale,), (offset,)
    return path


def _paletted(path: Path) -> Path:
    """Write a uint8 raster with a colour table."""
    make_tiff(path, dtype="uint8", nodata=None)
    with rasterio.open(path, "r+") as ds:
        ds.write_colormap(1, {0: (255, 0, 0, 255), 1: (0, 255, 0, 255)})
    return path


class TestScaleAndColourTable:
    """Scale and offset are carried; a colour table, which the model can't hold, raises."""

    def test_builders_carry_scale_and_offset(self, tmp_path: Path) -> None:
        a = _scaled(tmp_path / "a.tif")
        b = _scaled(tmp_path / "b.tif", origin=(500060.0, 5200000.0))
        # Previously dropped: GDAL then read the raw integers as the values.
        for v in (from_raster(a), stack([a]), mosaic([a, b])):
            assert (v.bands[0].scale, v.bands[0].offset) == (0.01, 5.0)
            with v.open() as ds:
                assert (ds.scales, ds.offsets) == ((0.01,), (5.0,))

    def test_unscaled_bands_have_none(self, tmp_path: Path) -> None:
        band = from_raster(make_tiff(tmp_path / "a.tif")).bands[0]
        assert (band.scale, band.offset) == (None, None)

    @pytest.mark.parametrize(("scale", "offset"), [(0.02, 5.0), (0.01, 6.0)])
    def test_mosaic_rejects_differing_scale_or_offset(
        self, tmp_path: Path, scale: float, offset: float
    ) -> None:
        a = _scaled(tmp_path / "a.tif")
        b = _scaled(tmp_path / "b.tif", scale, offset, origin=(500060.0, 5200000.0))
        with pytest.raises(ValueError, match=r"scale and offset.*b\.tif"):
            mosaic([a, b])

    def test_colour_table_raises(self, tmp_path: Path) -> None:
        pal = _paletted(tmp_path / "pal.tif")
        plain = make_tiff(tmp_path / "plain.tif", dtype="uint8", nodata=None)
        # Previously the band kept colour interpretation "palette" without its table.
        for build in (
            lambda: from_raster(pal),
            lambda: stack([plain, pal]),
            lambda: mosaic([plain, pal]),
        ):
            with pytest.raises(ValueError, match=r"pal\.tif.*colour table"):
                build()


def _complex(path: Path, dtype: str = "complex64") -> Path:
    """Write a 2x2 complex raster holding 3+4j, 1, 2j and 10."""
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=2,
        height=2,
        count=1,
        dtype=dtype,
        crs="EPSG:32632",
        transform=Affine(10, 0, 500000, 0, -10, 5200020),
    ) as dst:
        dst.write(np.array([[[3 + 4j, 1 + 0j], [0 + 2j, 10 + 0j]]], dtype=dtype))
    return path


class TestComplex:
    @pytest.mark.parametrize("dtype", ["complex_int16", "complex64", "complex128"])
    def test_from_raster_keeps_complex_values(self, tmp_path: Path, dtype: str) -> None:
        src = _complex(tmp_path / "c.tif", "complex64" if dtype == "complex_int16" else dtype)
        if dtype == "complex_int16":
            with rasterio.open(src) as ds:
                profile = {**ds.profile, "dtype": dtype}
                data = ds.read().astype("complex64")
            with rasterio.open(tmp_path / "ci16.tif", "w", **profile) as ds:
                ds.write(data)
            src = tmp_path / "ci16.tif"
        v = from_raster(src)
        assert v.bands[0].dtype == dtype
        with v.open() as ds:
            np.testing.assert_array_equal(ds.read(1), [[3 + 4j, 1], [2j, 10]])

    def test_derive_takes_the_modulus(self, tmp_path: Path) -> None:
        src = _complex(tmp_path / "c.tif")
        # Previously raised; a float32 transfer type would keep only the real part.
        with derive(src, PixelFunction.db()).open() as ds:
            np.testing.assert_allclose(ds.read(1), 10 * np.log10([[5, 1], [2, 10]]), rtol=1e-6)
        with derive(src, PixelFunction("intensity")).open() as ds:
            np.testing.assert_allclose(ds.read(1), [[25, 1], [4, 100]], rtol=1e-6)

    def test_derive_rejects_functions_without_complex_support(self, tmp_path: Path) -> None:
        src = _complex(tmp_path / "c.tif")
        with pytest.raises(ValueError, match="complex"):
            derive([src, src], PixelFunction.div())
