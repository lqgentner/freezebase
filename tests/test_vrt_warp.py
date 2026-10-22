"""Tests for vrt.warp."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.env import PROJDataFinder
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from upath import UPath

from freezebase.vrt import VRT, from_raster, mosaic, read, warp
from freezebase.vrt._warp import _set_transformer
from tests.rasters import make_tiff

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

DEM_ORIGIN = (2600000.0, 1200000.0)


@pytest.fixture
def no_grids(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Leave PROJ only rasterio's own data, without network, so no grid is reachable."""
    proj_data = PROJDataFinder().search()
    if proj_data is None:
        pytest.skip("rasterio bundles no PROJ data")
    monkeypatch.delenv("PROJ_NETWORK", raising=False)
    with rasterio.Env(PROJ_DATA=proj_data, PROJ_NETWORK="OFF"):
        yield


def _dem(tmp_path: Path) -> Path:
    """Write a DEM of zeros in LV95 without a vertical datum."""
    return make_tiff(
        tmp_path / "dem.tif", origin=DEM_ORIGIN, res=(100.0, 100.0), fill=0.0, crs="EPSG:2056"
    )


class TestGrid:
    def test_reprojects_onto_the_requested_grid(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif", fill=1.0)
        w = warp(src, crs="EPSG:3857", resolution=20.0)
        assert w.warp is not None
        assert w.warp.source == UPath(src)
        with w.open() as ds:
            assert ds.crs.to_epsg() == 3857
            assert ds.transform.a == pytest.approx(20.0)
            assert ds.transform.e == pytest.approx(-20.0)
            # Snapped outward, so the origin sits on the resolution grid.
            assert ds.transform.c % 20.0 == pytest.approx(0.0)
            assert np.nanmax(ds.read(1)) == 1.0

    def test_anisotropic_resolution(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        w = warp(src, crs="EPSG:3857", resolution=(10, 20))
        assert (w.transform.a, w.transform.e) == (10, -20)
        assert w.transform.c % 10 == pytest.approx(0.0)
        assert w.transform.f % 20 == pytest.approx(0.0)

    def test_natural_grid_matches_calculate_default_transform(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        with rasterio.open(src) as ds:
            transform, width, height = calculate_default_transform(
                ds.crs, "EPSG:4326", ds.width, ds.height, *ds.bounds
            )
        w = warp(src, crs="EPSG:4326")
        assert w.transform == transform
        assert (w.width, w.height) == (width, height)

    def test_like_copies_the_grid_exactly(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        like = make_tiff(
            tmp_path / "like.tif", crs="EPSG:3857", origin=(1000003.5, 6000001.25), res=(7, 9)
        )
        with rasterio.open(like) as ds:
            grid = (ds.crs, ds.transform, ds.width, ds.height)
        for ref in (like, from_raster(like)):
            w = warp(src, like=ref)
            assert (w.crs, w.transform, w.width, w.height) == grid

    def test_bounds_fix_the_extent_per_axis(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        bounds = (800000.0, 5800000.0, 800000.0 + 40 * 20.0, 5800000.0 + 30 * 10.0)
        w = warp(src, crs="EPSG:3857", resolution=(20.0, 10.0), bounds=bounds)
        assert (w.width, w.height) == (40, 30)
        assert w.transform == Affine(20.0, 0.0, bounds[0], 0.0, -10.0, bounds[3])

    def test_snap_false_keeps_the_reprojected_extent(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        w = warp(src, crs="EPSG:3857", resolution=20.0, snap=False)
        assert w.transform.c % 20.0 != pytest.approx(0.0)

    @pytest.mark.parametrize("resolution", [0.0, (10.0, -1.0)])
    def test_rejects_non_positive_resolution(
        self, tmp_path: Path, resolution: float | tuple[float, float]
    ) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        with pytest.raises(ValueError, match="resolution must be positive"):
            warp(src, crs="EPSG:3857", resolution=resolution)

    def test_rejects_bounds_off_the_resolution_grid(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        # 20.5 units wide at a 20-unit resolution: not a whole pixel count.
        bounds = (800000.0, 5800000.0, 800000.0 + 20.5, 5800000.0 + 40.0)
        with pytest.raises(ValueError, match="whole number"):
            warp(src, crs="EPSG:3857", resolution=20.0, bounds=bounds)

    def test_bounds_need_a_resolution(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        with pytest.raises(ValueError, match="bounds needs a resolution"):
            warp(src, crs="EPSG:3857", bounds=(0.0, 0.0, 20.0, 20.0))

    def test_needs_crs_or_like(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        with pytest.raises(ValueError, match="crs"):
            warp(src)

    def test_like_excludes_other_grid_arguments(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif")
        with pytest.raises(ValueError, match="like"):
            warp(src, like=src, crs="EPSG:3857")
        with pytest.raises(ValueError, match="like"):
            warp(src, like=src, resolution=20.0)
        with pytest.raises(ValueError, match="like"):
            warp(src, like=src, bounds=(0.0, 0.0, 1.0, 1.0))

    def test_source_without_crs_is_named(self, tmp_path: Path) -> None:
        src = tmp_path / "nocrs.tif"
        with rasterio.open(
            src,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint8",
            transform=Affine(10.0, 0.0, 0.0, 0.0, -10.0, 20.0),
        ) as ds:
            ds.write(np.zeros((1, 2, 2), dtype="uint8"))
        with pytest.raises(ValueError, match=r"nocrs\.tif.*src_crs"):
            warp(src, crs="EPSG:3857", resolution=20.0)
        with pytest.raises(ValueError, match=r"VRT input #0.*src_crs"):
            warp(from_raster(src), crs="EPSG:3857", resolution=20.0)
        with pytest.raises(ValueError, match=r"like \(nocrs\.tif\) has no CRS"):
            warp(make_tiff(tmp_path / "utm.tif"), like=src)

    def test_src_crs_overrides_the_source_crs(self, tmp_path: Path) -> None:
        mislabelled = make_tiff(tmp_path / "a.tif", crs="EPSG:32632")
        labelled = make_tiff(tmp_path / "b.tif", crs="EPSG:32633")
        w = warp(mislabelled, src_crs="EPSG:32633", crs="EPSG:4326")
        assert w.transform == warp(labelled, crs="EPSG:4326").transform


class TestPixels:
    def test_matches_rasterio_reproject(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif", size=(12, 9))
        w = warp(src, crs="EPSG:3857", resolution=7.0, resampling=Resampling.nearest)
        expected = np.full((w.height, w.width), np.nan, dtype="float32")
        with rasterio.open(src) as ds:
            reproject(
                ds.read(1),
                expected,
                src_transform=ds.transform,
                src_crs=ds.crs,
                src_nodata=ds.nodata,
                dst_transform=w.transform,
                dst_crs=w.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
            )
        with w.open() as ds:
            assert np.array_equal(ds.read(1), expected, equal_nan=True)

    def test_integer_average_is_not_promoted(self, tmp_path: Path) -> None:
        src = tmp_path / "counts.tif"
        with rasterio.open(
            src,
            "w",
            driver="GTiff",
            width=2,
            height=2,
            count=1,
            dtype="uint16",
            crs="EPSG:32632",
            transform=Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0),
        ) as ds:
            ds.write(np.array([[[1, 2], [1, 2]]], dtype="uint16"))
        w = warp(src, crs="EPSG:32632", resolution=20.0)
        assert w.bands[0].dtype == "uint16"
        with w.open() as ds:
            assert ds.read(1).tolist() == [[2]]

    def test_dtype_and_nodata_override(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "counts.tif", dtype="uint16", nodata=0)
        w = warp(src, crs="EPSG:32632", resolution=20.0, dtype="float32", nodata=math.nan)
        assert w.bands[0].dtype == "float32"
        assert w.bands[0].nodata is not None
        assert math.isnan(w.bands[0].nodata)
        with w.open() as ds:
            data = ds.read(1)
        # The top-left output pixel averages 0 (the old sentinel, skipped), 1, 6 and 7.
        assert data[0, 0] == pytest.approx(14 / 3)


class TestSources:
    def test_warp_of_an_in_memory_mosaic_round_trips(self, tmp_path: Path) -> None:
        a = make_tiff(tmp_path / "a.tif", fill=1.0)
        b = make_tiff(tmp_path / "b.tif", fill=2.0, origin=(500060.0, 5200000.0))
        w = warp(mosaic([a, b]), crs="EPSG:4326", resolution=1e-4)
        assert w.warp is not None
        assert isinstance(w.warp.source, VRT)
        assert w.warp.source == mosaic([a, b])
        out = w.write(tmp_path / "w.vrt")
        assert read(out) == w
        with w.open() as ds, rasterio.open(out) as written:
            assert set(np.unique(ds.read(1))) >= {1.0, 2.0}
            assert np.array_equal(written.read(1), ds.read(1), equal_nan=True)

    def test_multi_band_source_keeps_every_band(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "rgb.tif", count=3)
        w = warp(src, crs="EPSG:4326", resolution=1e-4)
        assert len(w.bands) == 3
        # GDAL marks a pixel-interleaved source so all bands warp in one pass.
        assert w.metadata_domains == {"IMAGE_STRUCTURE": {"INTERLEAVE": "PIXEL"}}
        assert read(w.write(tmp_path / "w.vrt")) == w
        with w.open() as ds:
            assert ds.count == 3
            assert ds.tags(ns="IMAGE_STRUCTURE")["INTERLEAVE"] == "PIXEL"


class TestOptions:
    def test_written_warp_keeps_its_options(self, tmp_path: Path) -> None:
        w = warp(make_tiff(tmp_path / "utm.tif"), crs="EPSG:4326", resolution=1e-4)
        text = w.write(tmp_path / "w.vrt").read_text()
        # A transformer option: GDAL rejects it as a warp option.
        assert '<Option key="ALLOW_BALLPARK">NO</Option>' in text
        assert '<Option name="ALLOW_BALLPARK">' not in text
        assert "<MaxError>0.125</MaxError>" in text

    def test_allow_ballpark_writes_no_option(self, tmp_path: Path) -> None:
        w = warp(
            make_tiff(tmp_path / "utm.tif"),
            crs="EPSG:4326",
            resolution=1e-4,
            allow_ballpark=True,
        )
        assert w.warp is not None
        assert "ALLOW_BALLPARK" not in w.warp.options_xml

    def test_tolerance_number_overrides_auto(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "utm.tif", size=(12, 9))
        exact = warp(
            src, crs="EPSG:3857", resolution=7.0, resampling=Resampling.nearest, tolerance=0
        )
        assert exact.warp is not None
        assert "ApproxTransformer" not in exact.warp.options_xml
        assert "<GenImgProjTransformer>" in exact.warp.options_xml
        assert read(exact.write(tmp_path / "w.vrt")) == exact
        expected = np.full((exact.height, exact.width), np.nan, dtype="float32")
        with rasterio.open(src) as ds:
            reproject(
                ds.read(1),
                expected,
                src_transform=ds.transform,
                src_crs=ds.crs,
                src_nodata=ds.nodata,
                dst_transform=exact.transform,
                dst_crs=exact.crs,
                dst_nodata=np.nan,
                resampling=Resampling.nearest,
                tolerance=0,
            )
        with exact.open() as ds:
            assert np.array_equal(ds.read(1), expected, equal_nan=True)
        approx = warp(src, crs="EPSG:3857", resolution=7.0, tolerance=0.5)
        assert approx.warp is not None
        assert "<MaxError>0.5</MaxError>" in approx.warp.options_xml

    def test_rejects_negative_tolerance(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="tolerance"):
            warp(make_tiff(tmp_path / "utm.tif"), crs="EPSG:4326", tolerance=-1.0)

    @pytest.mark.usefixtures("no_grids")
    def test_ballpark_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pyproj sync") as info:
            warp(_dem(tmp_path), src_crs="EPSG:2056+5728", crs="EPSG:4979", resolution=1e-3)
        assert str(info.value).startswith(
            "No exact transformation from CH1903+ / LV95 + LN02 height to EPSG:4979"
        )
        assert "allow_ballpark=True" in str(info.value)

    def test_ballpark_option_joins_existing_transformer_options(self) -> None:
        options = (
            "<GDALWarpOptions><Transformer><ReprojectionTransformer>"
            '<Options><Option key="ONLY_BEST">YES</Option></Options>'
            "</ReprojectionTransformer></Transformer></GDALWarpOptions>"
        )
        assert _set_transformer(options, exact=True, allow_ballpark=False) == options.replace(
            "</Options>", '<Option key="ALLOW_BALLPARK">NO</Option></Options>'
        )

    @pytest.mark.usefixtures("no_grids")
    def test_allow_ballpark_opts_out(self, tmp_path: Path) -> None:
        w = warp(
            _dem(tmp_path),
            src_crs="EPSG:2056+5728",
            crs="EPSG:4979",
            resolution=1e-3,
            allow_ballpark=True,
        )
        assert w.crs == CRS.from_epsg(4979)
        with w.open() as ds:
            assert ds.read(1).shape == (w.height, w.width)
