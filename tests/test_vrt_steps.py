"""Tests for the one-input steps on ``VRT``: select, set, clip_box and colorize."""

from __future__ import annotations

import copy
import sys
from typing import TYPE_CHECKING

from affine import Affine
import numpy as np
import pytest
import rasterio
from rasterio.enums import ColorInterp
import rasterio.windows

from freezebase.vrt import VRT, from_raster, mosaic, stack
from tests.rasters import make_tiff

if TYPE_CHECKING:
    from pathlib import Path

ORIGIN = (500000.0, 5200000.0)
BLACK_TO_ORANGE = [(0.0, (0, 0, 0)), (20.0, (200, 100, 0))]


def _read(source: VRT | Path) -> np.ndarray:
    with source.open() if isinstance(source, VRT) else rasterio.open(source) as ds:
        return ds.read()


def _three_bands(tmp_path: Path) -> VRT:
    """Return a tagged VRT over a 3-band float32 raster of distinct values."""
    path = make_tiff(tmp_path / "rgb.tif", count=3, descriptions=["r", "g", "b"])
    source = from_raster(path)
    source.metadata["SENSOR"] = "S2"
    return source


def _uint16(tmp_path: Path) -> VRT:
    """Return a VRT over a 6x4 uint16 raster holding 0 to 23, with 0 as nodata."""
    return from_raster(make_tiff(tmp_path / "u.tif", dtype="uint16", nodata=0))


class TestSelect:
    def test_picks_and_reorders_bands(self, tmp_path: Path) -> None:
        source = _three_bands(tmp_path)
        before = copy.deepcopy(source)
        picked = source.select([3, 1, 3])
        assert source == before
        assert picked.count == 3
        assert [band.description for band in picked.bands] == ["b", "r", "b"]
        assert picked.metadata == source.metadata
        np.testing.assert_array_equal(_read(picked), _read(source)[[2, 0, 2]])

    def test_bands_equal_those_of_stack(self, tmp_path: Path) -> None:
        source = _three_bands(tmp_path)
        assert source.select([2]).bands == stack([source], bands=[[2]]).bands

    @pytest.mark.parametrize(
        ("bands", "match"), [([], "must not be empty"), ([1, 4], "no band 4")]
    )
    def test_rejects_bad_bands(self, tmp_path: Path, bands: list[int], match: str) -> None:
        with pytest.raises(ValueError, match=match):
            _three_bands(tmp_path).select(bands)


class TestSet:
    def test_sets_metadata_on_a_copy(self, tmp_path: Path) -> None:
        source = _three_bands(tmp_path)
        before = copy.deepcopy(source)
        labeled = source.set(
            band_names=["red", "green", "blue"],
            color_interp=[ColorInterp.red, ColorInterp.green, ColorInterp.blue],
            units=["W", "W", "W"],
            tags={"LEVEL": "2A"},
            band_tags=[{"A": "1"}, {}, {}],
        )
        assert source == before
        with labeled.open() as ds:
            assert ds.descriptions == ("red", "green", "blue")
            assert ds.colorinterp == (ColorInterp.red, ColorInterp.green, ColorInterp.blue)
            assert ds.units == ("W", "W", "W")
            assert ds.tags()["SENSOR"] == "S2"
            assert ds.tags()["LEVEL"] == "2A"
            assert ds.tags(1)["A"] == "1"
            np.testing.assert_array_equal(ds.read(), _read(source))

    def test_none_keeps_the_metadata(self, tmp_path: Path) -> None:
        source = _three_bands(tmp_path)
        assert source.set() == source

    def test_rejects_a_wrong_length(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="one entry per band"):
            _three_bands(tmp_path).set(band_names=["red"])


class TestClipBox:
    @pytest.mark.parametrize(
        "cols_rows",
        [(0, 0, 6, 4), (1, 1, 4, 3), (5, 3, 6, 4), (0, 2, 6, 4)],
        ids=["whole", "inner", "corner-pixel", "bottom-rows"],
    )
    def test_matches_a_rasterio_window(
        self, tmp_path: Path, cols_rows: tuple[int, int, int, int]
    ) -> None:
        path = make_tiff(tmp_path / "a.tif", count=2)
        source = from_raster(path)
        before = copy.deepcopy(source)
        col0, row0, col1, row1 = cols_rows
        box = (
            ORIGIN[0] + col0 * 10,
            ORIGIN[1] - row1 * 10,
            ORIGIN[0] + col1 * 10,
            ORIGIN[1] - row0 * 10,
        )

        clipped = source.clip_box(*box)

        assert source == before
        with rasterio.open(path) as ds, clipped.open() as out:
            window = rasterio.windows.from_bounds(*box, transform=ds.transform)
            assert out.transform == rasterio.windows.transform(window, ds.transform)
            assert (out.height, out.width) == (round(window.height), round(window.width))
            np.testing.assert_array_equal(out.read(), ds.read(window=window))
            np.testing.assert_array_equal(out.nodatavals, ds.nodatavals)

    def test_keeps_band_and_dataset_metadata(self, tmp_path: Path) -> None:
        clipped = _three_bands(tmp_path).clip_box(500010.0, 5199970.0, 500030.0, 5199990.0)
        assert [band.description for band in clipped.bands] == ["r", "g", "b"]
        assert clipped.metadata["SENSOR"] == "S2"
        assert clipped.bounds == (500010.0, 5199970.0, 500030.0, 5199990.0)

    def test_tolerates_rounding_on_the_grid(self, tmp_path: Path) -> None:
        clipped = _uint16(tmp_path).clip_box(500010.0000001, 5199970.0, 500030.0, 5199990.0)
        assert clipped.transform.c == 500010.0
        assert clipped.shape == (2, 2)

    @pytest.mark.parametrize(
        ("box", "match"),
        [
            ((500005.0, 5199970.0, 500030.0, 5199990.0), "minx is not on the pixel grid"),
            ((500010.0, 5199970.0, 500030.0, 5199995.0), "maxy is not on the pixel grid"),
            ((499990.0, 5199970.0, 500030.0, 5199990.0), "extends beyond"),
            ((500010.0, 5199950.0, 500030.0, 5199990.0), "extends beyond"),
            ((500030.0, 5199970.0, 500010.0, 5199990.0), "is empty"),
            ((500010.0, 5199970.0, 500030.0, 5199970.0), "is empty"),
        ],
    )
    def test_rejects_a_bad_box(
        self, tmp_path: Path, box: tuple[float, float, float, float], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _uint16(tmp_path).clip_box(*box)

    @pytest.mark.parametrize(
        "transform",
        [Affine(10, 1, ORIGIN[0], 0, -10, ORIGIN[1]), Affine(10, 0, ORIGIN[0], 0, 10, ORIGIN[1])],
        ids=["rotated", "south-up"],
    )
    def test_rejects_a_rotated_or_south_up_grid(self, tmp_path: Path, transform: Affine) -> None:
        source = _uint16(tmp_path)
        source.transform = transform
        with pytest.raises(ValueError, match="north-up grid without rotation"):
            source.clip_box(500010.0, 5199970.0, 500030.0, 5199990.0)


class TestColorize:
    def test_interpolates_linearly_between_stops(self, tmp_path: Path) -> None:
        source = _uint16(tmp_path)
        before = copy.deepcopy(source)
        rgba = source.colorize(BLACK_TO_ORANGE)
        assert source == before
        with rgba.open() as ds:
            assert ds.count == 4
            assert ds.dtypes == ("uint8",) * 4
            assert ds.colorinterp == (
                ColorInterp.red,
                ColorInterp.green,
                ColorInterp.blue,
                ColorInterp.alpha,
            )
            data = ds.read()
        # Value 10 is halfway to the second stop; 23 lies beyond it and takes its color.
        np.testing.assert_array_equal(data[:, 1, 4], [100, 50, 0, 255])
        np.testing.assert_array_equal(data[:, 3, 5], [200, 100, 0, 255])

    def test_clamps_below_the_first_stop(self, tmp_path: Path) -> None:
        rgba = _uint16(tmp_path).colorize([(5.0, (10, 20, 30)), (6.0, (40, 50, 60))])
        np.testing.assert_array_equal(_read(rgba)[:, 0, 1], [10, 20, 30, 255])

    def test_nodata_is_transparent(self, tmp_path: Path) -> None:
        data = _read(_uint16(tmp_path).colorize(BLACK_TO_ORANGE))
        np.testing.assert_array_equal(data[:, 0, 0], [0, 0, 0, 0])
        assert (data[3].ravel()[1:] == 255).all()

    def test_stop_alpha_is_interpolated(self, tmp_path: Path) -> None:
        stops = [(0.0, (0, 0, 0, 255)), (20.0, (0, 0, 0, 55))]
        alpha = _read(_uint16(tmp_path).colorize(stops))[3]
        assert alpha[1, 4] == 155

    def test_renders_the_chosen_band(self, tmp_path: Path) -> None:
        source = from_raster(make_tiff(tmp_path / "a.tif", count=2, nodata=None))
        red = _read(source.colorize([(0.0, (0, 0, 0)), (255.0, (255, 0, 0))], band=2))[0]
        # Band 2 holds 24 to 47.
        np.testing.assert_array_equal(red, _read(source)[1])

    def test_samples_a_colormap_between_vmin_and_vmax(self, tmp_path: Path) -> None:
        def ramp(x: np.ndarray) -> np.ndarray:
            return np.stack([x, 1 - x, np.zeros_like(x), np.ones_like(x)], axis=-1)

        data = _read(_uint16(tmp_path).colorize(ramp, vmin=0, vmax=20))
        np.testing.assert_array_equal(data[:, 1, 4], [128, 128, 0, 255])
        np.testing.assert_array_equal(data[:, 3, 5], [255, 0, 0, 255])

    def test_matplotlib_colormap_name(self, tmp_path: Path) -> None:
        mpl = pytest.importorskip("matplotlib")
        data = _read(_uint16(tmp_path).colorize("viridis", vmin=1, vmax=23))
        expected = mpl.colormaps["viridis"]([0.0, 1.0], bytes=True)
        np.testing.assert_allclose(data[:, 0, 1], expected[0], atol=1)
        np.testing.assert_allclose(data[:, 3, 5], expected[1], atol=1)

    def test_unknown_colormap_name(self, tmp_path: Path) -> None:
        pytest.importorskip("matplotlib")
        with pytest.raises(ValueError, match="no colormap named 'nope'"):
            _uint16(tmp_path).colorize("nope", vmin=0, vmax=1)

    def test_colormap_name_without_matplotlib(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setitem(sys.modules, "matplotlib", None)
        with pytest.raises(ImportError, match="need matplotlib"):
            _uint16(tmp_path).colorize("viridis", vmin=0, vmax=1)

    @pytest.mark.parametrize(
        ("vmin", "vmax", "match"),
        [
            (None, 1.0, "needs vmin and vmax"),
            (0.0, None, "needs vmin and vmax"),
            (1.0, 1.0, "vmin must be below vmax"),
            (0.0, float("inf"), "vmin must be below vmax"),
        ],
    )
    def test_rejects_bad_colormap_limits(
        self, tmp_path: Path, vmin: float | None, vmax: float | None, match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _uint16(tmp_path).colorize(lambda x: x, vmin=vmin, vmax=vmax)

    @pytest.mark.parametrize(
        ("stops", "match"),
        [
            ([(0.0, (0, 0, 0))], "at least two stops"),
            ([(5.0, (0, 0, 0)), (1.0, (9, 9, 9))], "strictly ascending"),
            ([(1.0, (0, 0, 0)), (1.0, (9, 9, 9))], "strictly ascending"),
            ([(0.0, (0, 0, 0)), (float("nan"), (9, 9, 9))], "finite"),
            ([(0.0, (0, 0)), (1.0, (9, 9, 9))], "3 or 4 values"),
            ([(0.0, (0, 0, 256)), (1.0, (9, 9, 9))], "from 0 to 255"),
        ],
        ids=["one-stop", "descending", "repeated", "nan", "two-channels", "over-255"],
    )
    def test_rejects_bad_stops(
        self, tmp_path: Path, stops: list[tuple[float, tuple[int, ...]]], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            _uint16(tmp_path).colorize(stops)

    def test_rejects_limits_with_stops(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="vmin and vmax apply to a colormap"):
            _uint16(tmp_path).colorize(BLACK_TO_ORANGE, vmin=0)

    def test_rejects_a_missing_band(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="no band 2"):
            _uint16(tmp_path).colorize(BLACK_TO_ORANGE, band=2)


def test_steps_chain_into_a_png(tmp_path: Path) -> None:
    left = make_tiff(tmp_path / "l.tif", fill=1.0)
    right = make_tiff(tmp_path / "r.tif", fill=2.0, origin=(ORIGIN[0] + 60, ORIGIN[1]))
    web = mosaic([left, right]).reproject(crs="EPSG:3857", resolution=20.0)
    left_edge, bottom, right_edge, top = web.bounds
    clipped = web.clip_box(left_edge + 20, bottom + 20, right_edge - 20, top - 20)
    stops = [(1.0, (0, 0, 255)), (2.0, (255, 0, 0))]

    clipped.colorize(stops).to_raster(tmp_path / "q.png", driver="PNG")

    with rasterio.open(tmp_path / "q.png") as ds:
        assert ds.count == 4
        assert (ds.height, ds.width) == (web.height - 2, web.width - 2)
        data = ds.read()
    with clipped.open() as ds:
        values = ds.read(1)
    # Blue where the left tile is, red where the right one is, transparent where neither.
    np.testing.assert_array_equal(data[3] == 255, ~np.isnan(values))
    np.testing.assert_array_equal(data[0][values == 2], 255)
    np.testing.assert_array_equal(data[2][values == 1], 255)
