"""Tests for vrt.warp."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING
from xml.etree import ElementTree as ET

import numpy as np
import pyproj
from pyproj import Transformer
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.env import PROJDataFinder
from rasterio.transform import Affine
from rasterio.warp import calculate_default_transform, reproject
from upath import UPath

from freezebase.vrt import VRT, _warp, from_raster, mosaic, read, warp
from freezebase.vrt._warp import _has_vertical, _set_transformer
from tests.rasters import make_tiff

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path

    from numpy.typing import NDArray


# LV95 with LN02 heights in US survey feet; no EPSG code exists for it.
LN02_FTUS = (
    f'COMPOUNDCRS["LV95 + LN02 height (ftUS)",{pyproj.CRS("EPSG:2056").to_wkt()},'
    'VERTCRS["LN02 height (ftUS)",VDATUM["Landesnivellement 1902"],CS[vertical,1],'
    'AXIS["gravity-related height (H)",up,LENGTHUNIT["US survey foot",0.304800609601219]]]]'
)
FTUS = 1200 / 3937


@pytest.fixture
def dem(tmp_path: Path) -> Path:
    """Write a DEM of zeros in LV95 with LN02 heights."""
    return make_tiff(
        tmp_path / "dem_ln02.tif",
        origin=(2620000.0, 1160000.0),
        res=(100.0, 100.0),
        size=(300, 300),
        fill=0.0,
        crs="EPSG:2056+5728",
    )


def _grid_env(proj_grids: Path) -> rasterio.Env:
    """Point GDAL's PROJ at the synced grids, offline."""
    return rasterio.Env(PROJ_DATA=str(proj_grids), PROJ_NETWORK="OFF")


def _pyproj_shift(
    src: str, dst: str, xs: NDArray[np.float64], ys: NDArray[np.float64], z: float = 0.0
) -> NDArray[np.float64]:
    """Return the height pyproj gives height ``z`` at ``(xs, ys)`` in ``src``, in ``dst``."""
    transformer = Transformer.from_crs(
        src, dst, always_xy=True, only_best=True, allow_ballpark=False
    )
    _, _, h = transformer.transform(xs, ys, np.full_like(xs, z), errcheck=True)
    return np.asarray(h)


def _shift_matches_pyproj(
    w: VRT, source: Path, src: str, dst: str, z: float = 0.0
) -> NDArray[np.float32]:
    """Assert each warped pixel holds pyproj's shift at the source pixel it reads; return them."""
    with w.open() as ds:
        got = ds.read(1)
    with rasterio.open(source) as ds:
        src_transform, width, height = ds.transform, ds.width, ds.height
    rows, cols = np.mgrid[0 : w.height, 0 : w.width]
    # Written out, since affine's point operator differs across supported versions.
    xs = w.transform.c + (cols + 0.5) * w.transform.a
    ys = w.transform.f + (rows + 0.5) * w.transform.e
    horizontal = Transformer.from_crs(
        pyproj.CRS(dst).to_2d(), pyproj.CRS(src).to_2d(), always_xy=True
    )
    sx, sy = horizontal.transform(xs, ys)
    src_cols = np.floor((sx - src_transform.c) / src_transform.a).astype(int)
    src_rows = np.floor((sy - src_transform.f) / src_transform.e).astype(int)
    # The outermost source pixels are left out: nearest may pick either side of the edge.
    inside = (src_cols >= 1) & (src_cols < width - 1) & (src_rows >= 1) & (src_rows < height - 1)
    assert inside.sum() > 1000
    cx = src_transform.c + (src_cols[inside] + 0.5) * src_transform.a
    cy = src_transform.f + (src_rows[inside] + 0.5) * src_transform.e
    expected = _pyproj_shift(src, dst, cx, cy, z)
    np.testing.assert_allclose(got[inside], expected, atol=0.02, rtol=0)
    return got[inside]


@pytest.fixture
def no_grids(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Leave PROJ only rasterio's own data, without network, so no grid is reachable."""
    # PROJ remembers a grid lookup for the whole process, so users need grids (EGM2008)
    # that no test with proj_grids needs.
    proj_data = PROJDataFinder().search()
    if proj_data is None:
        pytest.skip("rasterio bundles no PROJ data")
    monkeypatch.delenv("PROJ_NETWORK", raising=False)
    with rasterio.Env(PROJ_DATA=proj_data, PROJ_NETWORK="OFF"):
        yield


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
        expected = warp(labelled, crs="EPSG:4326")
        assert w.transform == expected.transform
        assert w.warp is not None
        source_srs = ET.fromstring(w.warp.options_xml).find(".//SourceSRS")  # noqa: S314
        assert source_srs is not None
        assert CRS.from_user_input(source_srs.text or "") == CRS.from_epsg(32633)
        with w.open() as ds, expected.open() as ref:
            assert np.array_equal(ds.read(1), ref.read(1), equal_nan=True)


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

    def test_scaled_source_is_named_in_the_error(self, tmp_path: Path) -> None:
        src = make_tiff(tmp_path / "scaled.tif")
        with rasterio.open(src, "r+") as ds:
            ds.scales = (2.0,)
        # Previously raised the bare "Unsupported VRT element <Scale>".
        with pytest.raises(ValueError, match=r"scale/offset or colour table.*<Scale>"):
            warp(src, crs="EPSG:4326", resolution=1e-4)

    def test_other_parse_errors_are_not_relabelled(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def reject(_xml: str) -> VRT:
            """Fail as the parser does on an element unrelated to scale or colour."""
            msg = "Unsupported VRT element <Foo>"
            raise ValueError(msg)

        monkeypatch.setattr(_warp, "from_xml", reject)
        with pytest.raises(ValueError, match=r"^Unsupported VRT element <Foo>$"):
            warp(make_tiff(tmp_path / "utm.tif"), crs="EPSG:4326", resolution=1e-4)

    def test_vrt_source_is_the_callers(self, tmp_path: Path) -> None:
        source = mosaic([make_tiff(tmp_path / "a.tif")])
        w = warp(source, crs="EPSG:4326", resolution=1e-4)
        assert w.warp is not None
        assert w.warp.source is source


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

    @pytest.mark.parametrize("tolerance", [-1.0, math.nan])
    def test_rejects_negative_or_nan_tolerance(self, tmp_path: Path, tolerance: float) -> None:
        with pytest.raises(ValueError, match=r"tolerance must be .* a non-negative number"):
            warp(make_tiff(tmp_path / "utm.tif"), crs="EPSG:4326", tolerance=tolerance)

    @pytest.mark.usefixtures("no_grids")
    def test_ballpark_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="pyproj sync") as info:
            warp(
                make_tiff(tmp_path / "utm.tif", fill=0.0),
                src_crs="EPSG:32632+3855",
                crs="EPSG:4979",
                resolution=1e-3,
            )
        assert str(info.value).startswith(
            "No exact transformation from WGS 84 / UTM zone 32N + EGM2008 height to EPSG:4979"
        )
        assert "allow_ballpark=True" in str(info.value)

    def test_ballpark_option_joins_existing_transformer_options(self) -> None:
        options = (
            "<GDALWarpOptions><Transformer><ReprojectionTransformer>"
            '<Options><Option key="ONLY_BEST">YES</Option></Options>'
            "</ReprojectionTransformer></Transformer></GDALWarpOptions>"
        )
        got = _set_transformer(options, exact=True, allow_ballpark=False, reprojects=True)
        assert got == options.replace(
            "</Options>", '<Option key="ALLOW_BALLPARK">NO</Option></Options>'
        )

    def test_existing_ballpark_option_is_replaced(self) -> None:
        options = (
            "<GDALWarpOptions><Transformer><ReprojectionTransformer>"
            '<Options><Option key="ALLOW_BALLPARK">YES</Option></Options>'
            "</ReprojectionTransformer></Transformer></GDALWarpOptions>"
        )
        got = _set_transformer(options, exact=False, allow_ballpark=False, reprojects=True)
        assert got == options.replace(">YES<", ">NO<")

    def test_unwrappable_approx_transformer_raises(self) -> None:
        options = (
            "<GDALWarpOptions><Transformer><ApproxTransformer><MaxError>0.125</MaxError>"
            "</ApproxTransformer></Transformer></GDALWarpOptions>"
        )
        with pytest.raises(RuntimeError, match="exact"):
            _set_transformer(options, exact=True, allow_ballpark=True, reprojects=True)

    def test_missing_reprojection_transformer_raises(self) -> None:
        options = (
            "<GDALWarpOptions><Transformer><GenImgProjTransformer />"
            "</Transformer></GDALWarpOptions>"
        )
        with pytest.raises(RuntimeError, match="ballpark"):
            _set_transformer(options, exact=False, allow_ballpark=False, reprojects=True)
        # Without a reprojection there is nothing to refuse.
        same = _set_transformer(options, exact=False, allow_ballpark=False, reprojects=False)
        assert same == options
        with pytest.raises(RuntimeError, match="<Transformer>"):
            _set_transformer(
                "<GDALWarpOptions />", exact=False, allow_ballpark=True, reprojects=False
            )

    @pytest.mark.usefixtures("no_grids")
    def test_allow_ballpark_opts_out(self, tmp_path: Path) -> None:
        w = warp(
            make_tiff(tmp_path / "utm.tif", fill=0.0),
            src_crs="EPSG:32632+3855",
            crs="EPSG:4979",
            resolution=1e-3,
            allow_ballpark=True,
        )
        assert w.crs == CRS.from_epsg(4979)
        with w.open() as ds:
            data = ds.read(1)
        # The silent zero shift that the default refuses.
        assert np.isfinite(data).any()
        assert np.nanmin(data) == np.nanmax(data) == 0.0


class TestVerticalShift:
    def test_compound_to_3d_gives_ellipsoidal_heights(self, proj_grids: Path, dem: Path) -> None:
        with _grid_env(proj_grids):
            w = warp(dem, crs="EPSG:4979", resampling=Resampling.nearest)
            got = _shift_matches_pyproj(w, dem, "EPSG:2056+5728", "EPSG:4979")
        assert w.warp is not None
        assert w.warp.vertical_shift
        assert 51.0 < got.min() < got.max() < 54.0

    def test_compound_to_compound(self, proj_grids: Path, dem: Path) -> None:
        with _grid_env(proj_grids):
            w = warp(dem, crs="EPSG:2056+5729", resampling=Resampling.nearest)
            got = _shift_matches_pyproj(w, dem, "EPSG:2056+5728", "EPSG:2056+5729")
        assert 0.05 < got.min() < got.max() < 1.0

    def test_3d_to_compound(self, proj_grids: Path, tmp_path: Path) -> None:
        src = make_tiff(
            tmp_path / "ellipsoidal.tif",
            origin=(7.7, 46.6),
            res=(0.001, 0.001),
            size=(300, 200),
            fill=0.0,
            crs="EPSG:4979",
        )
        with _grid_env(proj_grids):
            w = warp(src, crs="EPSG:2056+5728", resampling=Resampling.nearest)
            got = _shift_matches_pyproj(w, src, "EPSG:4979", "EPSG:2056+5728")
        assert -54.0 < got.min() < got.max() < -51.0

    def test_vertical_source_into_2d_target_raises(
        self, dem: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_grid(*_args: object) -> None:
            """Fail if the target grid is derived before the CRS check."""
            raise AssertionError

        monkeypatch.setattr(_warp, "calculate_default_transform", no_grid)
        with pytest.raises(ValueError, match="vertical") as info:
            warp(dem, crs="EPSG:4326")
        assert "pass a 3D or compound crs, or a 2D src_crs" in str(info.value)

    def test_multi_band_shift_raises(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def no_grid(*_args: object) -> None:
            """Fail if the target grid is derived before the band check."""
            raise AssertionError

        two = make_tiff(
            tmp_path / "two.tif", origin=(2620000.0, 1160000.0), count=2, crs="EPSG:2056+5728"
        )
        target = make_tiff(tmp_path / "target.tif", origin=(7.7, 46.6), crs="EPSG:4979")
        # Previously the height shift was added to every band.
        with pytest.raises(ValueError, match="single-band") as info:
            warp(two, like=target)
        assert "2D src_crs" in str(info.value)
        monkeypatch.setattr(_warp, "calculate_default_transform", no_grid)
        with pytest.raises(ValueError, match="single-band"):
            warp(two, crs="EPSG:4979")

    def test_2d_source_into_vertical_target_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="vertical") as info:
            warp(make_tiff(tmp_path / "utm.tif"), crs="EPSG:4979")
        assert "pass a 3D or compound src_crs, or a 2D crs" in str(info.value)

    def test_2d_src_crs_keeps_heights(self, dem: Path) -> None:
        w = warp(dem, src_crs="EPSG:2056", crs="EPSG:4326")
        assert w.warp is not None
        assert not w.warp.vertical_shift
        with w.open() as ds:
            data = ds.read(1)
        assert np.isfinite(data).any()
        assert np.nanmin(data) == np.nanmax(data) == 0.0

    def test_auto_tolerance_is_exact_with_a_shift(self, proj_grids: Path, dem: Path) -> None:
        with _grid_env(proj_grids):
            w = warp(dem, crs="EPSG:4979")
            approx = warp(dem, crs="EPSG:4979", tolerance=0.5)
        assert w.warp is not None
        assert w.warp.vertical_shift
        assert "ApproxTransformer" not in w.to_xml()
        assert "<MaxError>0.5</MaxError>" in approx.to_xml()
        plain = warp(dem, src_crs="EPSG:2056", crs="EPSG:4326")
        assert "<MaxError>0.125</MaxError>" in plain.to_xml()

    def test_written_shift_reopens_with_the_shift(
        self, proj_grids: Path, dem: Path, tmp_path: Path
    ) -> None:
        with _grid_env(proj_grids):
            w = warp(dem, crs="EPSG:4979", resampling=Resampling.nearest)
            text = w.write(tmp_path / "w.vrt").read_text()
            reread = read(tmp_path / "w.vrt")
            with w.open() as ds, reread.open() as written:
                expected = ds.read(1)
                assert np.array_equal(written.read(1), expected, equal_nan=True)
        assert '<Option name="APPLY_VERTICAL_SHIFT">YES</Option>' in text
        assert '<Option key="ALLOW_BALLPARK">NO</Option>' in text
        assert reread == w
        assert np.nanmax(expected) > 50.0

    @pytest.mark.parametrize(
        ("crs", "expected"),
        [
            ("EPSG:2056+5728", True),
            ("EPSG:5728", True),
            ("EPSG:4979", True),
            ("EPSG:4326", False),
            ("EPSG:2056", False),
            ("EPSG:4978", False),
            (LN02_FTUS, True),
            (CRS.from_user_input("EPSG:2056+5728").to_wkt(version="WKT1_GDAL"), True),
            (pyproj.CRS("EPSG:32632").to_3d().to_wkt(), True),
            (
                pyproj.CRS("+proj=longlat +ellps=bessel +towgs84=674.374,15.056,405.346 +type=crs")
                .to_3d()
                .to_wkt(),
                True,
            ),
        ],
        ids=[
            "compound",
            "vertical",
            "geographic-3d",
            "geographic-2d",
            "projected-2d",
            "geocentric",
            "compound-without-code",
            "wkt1-compd-cs",
            "projected-3d",
            "bound-geographic-3d",
        ],
    )
    def test_has_vertical(self, crs: str, expected: bool) -> None:
        assert _has_vertical(CRS.from_user_input(crs)) is expected

    def test_same_crs_applies_no_shift(self, dem: Path) -> None:
        w = warp(dem, crs="EPSG:2056+5728", resolution=200.0)
        assert w.warp is not None
        assert not w.warp.vertical_shift
        assert "<MaxError>0.125</MaxError>" in w.warp.options_xml

    def test_like_with_a_compound_grid(self, proj_grids: Path, dem: Path, tmp_path: Path) -> None:
        like = make_tiff(
            tmp_path / "like.tif",
            origin=(2630000.0, 1150000.0),
            res=(200.0, 200.0),
            size=(50, 50),
            crs="EPSG:2056+5729",
        )
        with _grid_env(proj_grids):
            w = warp(dem, like=like, resampling=Resampling.nearest)
            got = _shift_matches_pyproj(w, dem, "EPSG:2056+5728", "EPSG:2056+5729")
        assert w.warp is not None
        assert w.warp.vertical_shift
        assert 0.05 < got.min() < got.max() < 1.0
        flat = make_tiff(tmp_path / "flat.tif", crs="EPSG:2056")
        with pytest.raises(ValueError, match="2D src_crs"):
            warp(dem, like=flat)


class TestVerticalUnits:
    def test_feet_source_is_converted_to_metres(self, tmp_path: Path) -> None:
        src = make_tiff(
            tmp_path / "ftus.tif",
            origin=(980000.0, 200000.0),
            res=(100.0, 100.0),
            size=(20, 20),
            fill=100.0,
            crs="EPSG:2263+6360",
        )
        # Ballpark makes the geoid shift zero, so only the unit conversion remains.
        w = warp(src, crs="EPSG:4979", allow_ballpark=True)
        reread = read(w.write(tmp_path / "w.vrt"))
        assert reread == w
        for vrt in (w, reread):
            with vrt.open() as ds:
                data = ds.read(1)
            assert np.isfinite(data).any()
            np.testing.assert_allclose(data[np.isfinite(data)], 100 * FTUS, rtol=1e-6)
        options = ET.fromstring(reread.warp.options_xml if reread.warp else "")  # noqa: S314
        factors = {
            option.get("name"): float(option.text or "")
            for option in options.iterfind("Option")
            if (option.get("name") or "").startswith("MULT_FACTOR_VERTICAL_SHIFT")
        }
        assert factors == pytest.approx(
            {"MULT_FACTOR_VERTICAL_SHIFT": FTUS, "MULT_FACTOR_VERTICAL_SHIFT_PIPELINE": FTUS}
        )

    def test_metre_source_into_a_feet_target(self, tmp_path: Path) -> None:
        src = make_tiff(
            tmp_path / "metres.tif",
            origin=(-74.0, 40.8),
            res=(0.001, 0.001),
            size=(20, 20),
            fill=100 * FTUS,
            crs="EPSG:4979",
        )
        w = warp(src, crs="EPSG:2263+6360", allow_ballpark=True)
        with w.open() as ds:
            data = ds.read(1)
        assert np.isfinite(data).any()
        np.testing.assert_allclose(data[np.isfinite(data)], 100.0, rtol=1e-6)

    def test_band_unit_overrides_the_axis_unit(self, tmp_path: Path) -> None:
        src = make_tiff(
            tmp_path / "ftus.tif",
            origin=(980000.0, 200000.0),
            res=(100.0, 100.0),
            size=(20, 20),
            fill=100.0,
            crs="EPSG:2263+6360",
        )
        with rasterio.open(src, "r+") as ds:
            ds.units = ("metre",)
        w = warp(src, crs="EPSG:4979", allow_ballpark=True)
        with w.open() as ds:
            data = ds.read(1)
        np.testing.assert_allclose(data[np.isfinite(data)], 100.0, rtol=1e-6)

    def test_feet_shift_matches_pyproj(self, proj_grids: Path, tmp_path: Path) -> None:
        src = make_tiff(
            tmp_path / "ln02_ftus.tif",
            origin=(2620000.0, 1160000.0),
            res=(100.0, 100.0),
            size=(300, 300),
            fill=100.0,
            crs="EPSG:2056",
        )
        with _grid_env(proj_grids):
            w = warp(src, src_crs=LN02_FTUS, crs="EPSG:4979", resampling=Resampling.nearest)
            got = _shift_matches_pyproj(w, src, LN02_FTUS, "EPSG:4979", z=100.0)
        # 100 ftUS plus the 51-54 m geoid undulation.
        assert 81.0 < got.min() < got.max() < 85.0
