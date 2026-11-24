"""Tests for the text and HTML reprs of the freezebase.vrt model."""

# The reprs print sizes with a multiplication sign.
# ruff: noqa: RUF001

from __future__ import annotations

import copy
import html as htmllib
import http.server
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import ClassVar, NoReturn

from affine import Affine
import fsspec
import pyproj
import pytest
import rasterio
from rasterio.crs import CRS
from rasterio.enums import ColorInterp
import rasterio.windows
from upath import UPath

from freezebase import vrt
from freezebase.vrt import VRT, Band, PixelFunction, Source, Warp, _crs, _map, _model, _summary
from tests.rasters import make_tiff

UTM = CRS.from_epsg(32632)
GRID = Affine(10.0, 0.0, 500000.0, 0.0, -10.0, 5200000.0)


def tiles(directory: Path, count: int) -> list[Path]:
    """Write ``count`` tiles side by side."""
    return [
        make_tiff(directory / f"t{i}.tif", origin=(500000.0 + 60 * i, 5200000.0))
        for i in range(count)
    ]


def latlon_target(directory: Path) -> Path:
    """Write a small EPSG:4326 raster whose grid a reprojection can copy."""
    return make_tiff(
        directory / "target.tif", origin=(9.0, 47.0), res=(0.0001, 0.0001), crs="EPSG:4326"
    )


def model(*sources: Source, crs: CRS | None = UTM) -> VRT:
    """Build a one-band VRT drawing ``sources``, without opening them."""
    return VRT(crs, GRID, 6, 4, [Band("float32", list(sources))])


class TestText:
    def test_single_file(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", descriptions=["elevation"])
        assert repr(vrt.from_raster(tile)) == (
            "VRT  6 × 4 px · 1 band · float32 · EPSG:32632 · 10 m\n"
            'bands  1  "elevation"  float32  nodata=nan\n'
            "steps  1  from_raster  a.tif"
        )

    def test_mosaic(self, tmp_path: Path) -> None:
        assert repr(vrt.mosaic(tiles(tmp_path, 3))) == (
            "VRT  18 × 4 px · 1 band · float32 · EPSG:32632 · 10 m\n"
            "bands  1  float32  nodata=nan\n"
            "steps  1  mosaic  t0.tif, t1.tif, t2.tif"
        )

    def test_reprojected_mosaic(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 3))
        warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
        assert repr(warped) == (
            "VRT  6 × 4 px · 1 band · float32 · EPSG:4326 · 0.0001°\n"
            "bands  1  float32  nodata=nan\n"
            "steps  ┌ 1  mosaic     t0.tif, t1.tif, t2.tif\n"
            "       └ 2  reproject  EPSG:32632 → EPSG:4326"
        )

    def test_reprojected_file(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        warped = vrt.reproject(tile, like=latlon_target(tmp_path))
        assert repr(warped).splitlines()[2:] == [
            "steps  1  reproject  a.tif · EPSG:32632 → EPSG:4326"
        ]

    def test_resampled_without_reprojecting(self, tmp_path: Path) -> None:
        resampled = vrt.reproject(vrt.mosaic(tiles(tmp_path, 2)), crs=UTM, resolution=20)
        assert repr(resampled) == (
            "VRT  6 × 2 px · 1 band · float32 · EPSG:32632 · 20 m\n"
            "bands  1  float32  nodata=nan\n"
            "steps  ┌ 1  mosaic     t0.tif, t1.tif\n"
            "       └ 2  reproject  EPSG:32632"
        )

    def test_reproject_labels_the_src_crs_override(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        target = make_tiff(tmp_path / "target.tif", origin=(-500000.0, 5200000.0))
        warped = vrt.reproject(mosaic, src_crs="EPSG:32633", like=target)
        # The nested mosaic still says EPSG:32632; the warp reads it as EPSG:32633.
        assert repr(warped).splitlines()[3] == "       └ 2  reproject  EPSG:32633 → EPSG:32632"

    def test_derive(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        derived = vrt.derive(tile, PixelFunction.db(), description="VV")
        assert repr(derived) == (
            "VRT  6 × 4 px · 1 band · float32 · EPSG:32632 · 10 m\n"
            'bands  1  "VV"  float32  nodata=nan\n'
            "steps  1  derive  a.tif · dB"
        )

    def test_derive_of_two_files(self, tmp_path: Path) -> None:
        a, b = make_tiff(tmp_path / "a.tif"), make_tiff(tmp_path / "b.tif")
        ratio = vrt.derive([a, b], PixelFunction.div())
        assert repr(ratio).splitlines()[2:] == ["steps  1  derive  a.tif, b.tif · div"]

    def test_stack(self, tmp_path: Path) -> None:
        a, b = (
            make_tiff(tmp_path / "a.tif"),
            make_tiff(tmp_path / "b.tif", dtype="uint16", nodata=0),
        )
        stacked = vrt.stack([a, b], band_names=["VV", "VH"])
        stacked.bands[0].unit = "dB"
        assert repr(stacked) == (
            "VRT  6 × 4 px · 2 bands · float32, uint16 · EPSG:32632 · 10 m\n"
            'bands  1  "VV"  float32  nodata=nan  unit=dB\n'
            '       2  "VH"  uint16   nodata=0\n'
            "steps  1  stack  a.tif, b.tif"
        )

    def test_nested_in_nested(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 3))
        warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
        derived = vrt.derive(warped, PixelFunction.db(), description="dB")
        other = make_tiff(tmp_path / "other.tif", origin=(9.0, 47.0), res=(0.0001, 0.0001))
        other_vrt = vrt.reproject(other, src_crs="EPSG:4326", like=latlon_target(tmp_path))
        stacked = vrt.stack([derived, warped, other_vrt])
        # The reprojected mosaic is read twice but listed once, as step 2.
        assert repr(stacked) == (
            "VRT  6 × 4 px · 3 bands · float32 · EPSG:4326 · 0.0001°\n"
            'bands  1  "dB"  float32  nodata=nan\n'
            "       2        float32  nodata=nan\n"
            "       3        float32  nodata=nan\n"
            "steps  ┌ 1  mosaic     t0.tif, t1.tif, t2.tif\n"
            "       │ 2  reproject  EPSG:32632 → EPSG:4326\n"
            "       └ 3  derive     dB\n"
            "         4  reproject  other.tif · EPSG:4326\n"
            "         5  stack      #3, #2, #4"
        )

    def test_brackets_mark_each_chain(self, tmp_path: Path) -> None:
        first = vrt.derive(vrt.mosaic(tiles(tmp_path, 2)), PixelFunction.db())
        grid = make_tiff(tmp_path / "grid.tif", size=(12, 4))
        other = vrt.reproject(make_tiff(tmp_path / "other.tif"), like=grid)
        second = vrt.derive(other, PixelFunction.db())
        assert repr(vrt.stack([first, second])).splitlines()[3:] == [
            "steps  ┌ 1  mosaic     t0.tif, t1.tif",
            "       └ 2  derive     dB",
            "       ┌ 3  reproject  other.tif · EPSG:32632",
            "       └ 4  derive     dB",
            "         5  stack      #2, #4",
        ]

    def test_equal_steps_are_listed_once(self, tmp_path: Path) -> None:
        derived = vrt.derive(make_tiff(tmp_path / "a.tif"), PixelFunction.db())
        # Equal one-band inputs of the same file read as one derive of two bands.
        assert repr(vrt.stack([derived, copy.deepcopy(derived)])).splitlines()[3:] == [
            "steps  1  derive  a.tif · dB",
        ]

    def test_numbers_are_aligned(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_summary, "MAX_DEPTH", 20)
        chain = vrt.from_raster(make_tiff(tmp_path / "a.tif"))
        for _ in range(9):
            chain = vrt.derive(chain, PixelFunction.scale())
        lines = repr(chain).splitlines()
        assert lines[2] == "steps  ┌  1  from_raster  a.tif"
        assert lines[-1] == "       └ 10  derive       scale"

    def test_one_input_cropped_or_picked_is_a_step(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", count=2)
        derived = vrt.derive(tile, PixelFunction.db())
        # Previously the steps stopped at the derive, as if the crop changed nothing.
        assert repr(vrt.mosaic([derived], resolution=(20, 20))).splitlines()[-2:] == [
            "steps  ┌ 1  derive  a.tif · dB",
            "       └ 2  mosaic",
        ]
        # Reading bands of one input whole is a select, however it was built.
        last = repr(vrt.stack([derived, derived])).splitlines()[-1]
        assert last == "       └ 2  select  bands 1, 2, 1, 2"
        last = repr(vrt.stack([tile], bands=[[2]])).splitlines()[-1]
        assert last == "steps  1  select  a.tif · band 2"
        assert repr(vrt.mosaic([tile])).splitlines()[-1] == "steps  1  mosaic  a.tif"

    def test_equal_inputs_shared_deep_down_are_compared_once(self) -> None:
        # Comparing these with == takes time exponential in the depth.
        node = model(Source(UPath("/d/a.tif")))
        for _ in range(30):
            node = VRT(UTM, GRID, 6, 4, [Band("float32", [Source(node), Source(node)])])
        top = VRT(UTM, GRID, 6, 4, [Band("float32", [Source(node), Source(copy.deepcopy(node))])])
        assert repr(top).splitlines()[-1] == "       └ 6  mosaic"

    def test_a_long_chain_of_views(self) -> None:
        chain = model(Source(UPath("/d/a.tif")))
        for _ in range(3000):
            chain = VRT(UTM, GRID, 6, 4, [Band("float32", [Source(chain)])])
        assert repr(chain).splitlines()[2:] == ["steps  1  from_raster  a.tif"]

    def test_mosaic_of_files_and_a_vrt(self, tmp_path: Path) -> None:
        a, b = tiles(tmp_path, 2)
        mixed = vrt.mosaic([vrt.derive(a, PixelFunction.scale()), b])
        assert repr(mixed).splitlines()[2:] == [
            "steps  1  derive  t0.tif · scale",
            "       2  mosaic  #1, t1.tif",
        ]

    def test_a_plain_view_of_a_vrt_shows_the_vrt(self, tmp_path: Path) -> None:
        derived = vrt.derive(make_tiff(tmp_path / "a.tif"), PixelFunction.db())
        assert repr(vrt.stack([derived])).splitlines()[2:] == ["steps  1  derive  a.tif · dB"]

    def test_without_crs_bands_or_sources(self) -> None:
        empty = VRT(None, Affine(10.0, 0.0, 0.0, 0.0, -20.0, 0.0), 3, 2)
        assert repr(empty) == (
            "VRT  3 × 2 px · 0 bands · no CRS · 10 × 20\nbands  none\nsteps  none"
        )

    def test_a_nested_vrt_without_sources(self) -> None:
        empty = VRT(UTM, GRID, 6, 4, [Band("float32")])
        assert repr(model(Source(empty), Source(UPath("/d/a.tif")))).splitlines()[2:] == [
            "steps  1  empty",
            "       2  mosaic  #1, a.tif",
        ]

    def test_remote_files_keep_their_protocol(self) -> None:
        remote = model(
            Source(UPath("s3://bucket/key/a.tif", anon=True)),
            Source(UPath("https://example.com/data/b.tif")),
        )
        assert repr(remote).splitlines()[2] == "steps  1  mosaic  s3://…/a.tif, https://…/b.tif"

    def test_crs_without_authority_shows_its_name(self) -> None:
        crs = CRS.from_wkt('LOCAL_CS["Lab grid",UNIT["metre",1]]')
        assert repr(model(crs=crs)).splitlines()[0].endswith("· Lab grid · 10 m")

    def test_unreadable_warp_options(self) -> None:
        tile = UPath("/data/a.tif")
        warped = VRT(UTM, GRID, 6, 4, [Band("float32")], warp=Warp(tile, "<oops", (6, 4)))
        assert repr(warped).splitlines()[2] == "steps  1  reproject  a.tif"
        warped.warp = Warp(tile, "<W><SourceSRS>bogus</SourceSRS></W>", (6, 4))
        assert repr(warped).splitlines()[2] == "steps  1  reproject  a.tif · bogus → EPSG:32632"

    def test_long_and_multiline_strings_are_clipped(self) -> None:
        long_name = "x" * 100 + ".tif"
        vrt_model = model(Source(UPath(f"/data/{long_name}")))
        vrt_model.bands[0].description = "first\nsecond"
        lines = repr(vrt_model).splitlines()
        assert lines[1] == 'bands  1  "first second"  float32'
        assert lines[2] == "steps  1  from_raster  " + "x" * 59 + "…"


class TestOneInputSteps:
    """The steps of ``VRT.select``, ``clip_box`` and ``colorize`` (``set`` adds none)."""

    @staticmethod
    def rgb(directory: Path) -> VRT:
        """Return a VRT over a 3-band raster."""
        return vrt.from_raster(make_tiff(directory / "rgb.tif", count=3))

    @staticmethod
    def steps(value: VRT) -> list[str]:
        """Return the step lines of the text repr."""
        lines = repr(value).splitlines()
        first = next(i for i, line in enumerate(lines) if line.startswith("steps"))
        return lines[first:]

    def test_select(self, tmp_path: Path) -> None:
        assert self.steps(self.rgb(tmp_path).select([3, 1])) == [
            "steps  1  select  rgb.tif · bands 3, 1"
        ]

    def test_select_of_a_warped_vrt(self, tmp_path: Path) -> None:
        warped = self.rgb(tmp_path).reproject(crs="EPSG:3857", resolution=20.0)
        assert self.steps(warped.select([3, 2])) == [
            "steps  ┌ 1  from_raster  rgb.tif",
            "       │ 2  reproject    EPSG:32632 → EPSG:3857",
            "       └ 3  select       bands 3, 2",
        ]

    def test_clip_box_reads_its_input_once(self, tmp_path: Path) -> None:
        # Each band reads its own one-band copy, which would otherwise be one step per band.
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        left, bottom, right, top = mosaic.bounds
        assert self.steps(mosaic.clip_box(left + 20, bottom, right - 30, top - 10)) == [
            "steps  ┌ 1  mosaic    t0.tif, t1.tif",
            "       └ 2  clip_box  7 × 3 px at col 2, row 1",
        ]

    def test_clip_box_of_a_warped_vrt(self, tmp_path: Path) -> None:
        warped = self.rgb(tmp_path).reproject(crs="EPSG:3857", resolution=20.0)
        left, bottom, _, top = warped.bounds
        assert self.steps(warped.clip_box(left, bottom, left + 40, top))[-1] == (
            "       └ 3  clip_box     2 × 4 px at col 0, row 0"
        )

    def test_clip_box_of_the_whole_extent_adds_no_step(self, tmp_path: Path) -> None:
        source = self.rgb(tmp_path)
        assert self.steps(source.clip_box(*source.bounds)) == ["steps  1  from_raster  rgb.tif"]

    def test_colorize(self, tmp_path: Path) -> None:
        stops = [(0.0, (0, 0, 0)), (20.0, (200, 100, 0))]
        assert self.steps(self.rgb(tmp_path).colorize(stops, band=2)) == [
            "steps  ┌ 1  select    rgb.tif · band 2",
            "       └ 2  colorize  0 – 20",
        ]

    def test_set_adds_no_step(self, tmp_path: Path) -> None:
        source = self.rgb(tmp_path)
        assert self.steps(source.set(band_names=["r", "g", "b"])) == self.steps(source)

    def test_chain(self, tmp_path: Path) -> None:
        source = self.rgb(tmp_path)
        left, bottom, _, top = source.bounds
        chained = (
            vrt.derive(source, PixelFunction.db())
            .select([2, 1])
            .clip_box(left, bottom, left + 30, top)
            .colorize([(-10.0, (0, 0, 255)), (10.0, (255, 0, 0))])
        )
        # Each step nests only the band the next reads.
        assert self.steps(chained) == [
            "steps  ┌ 1  select    rgb.tif · band 2",
            "       │ 2  derive    dB",
            "       │ 3  clip_box  3 × 4 px at col 0, row 0",
            "       └ 4  colorize  -10 – 10",
        ]

    def test_a_stack_of_one_band_vrts_is_not_joined(self, tmp_path: Path) -> None:
        a, b = (vrt.from_raster(make_tiff(tmp_path / f"{n}.tif")) for n in "ab")
        stacked = vrt.stack([vrt.derive(a, PixelFunction.db()), vrt.derive(b, PixelFunction.db())])
        assert self.steps(stacked)[-1] == "         5  stack        #2, #4"

    def test_steps_survive_a_write_and_read(self, tmp_path: Path) -> None:
        # Previously each band's own copy of the input read back as a stack of equal steps.
        source = self.rgb(tmp_path)
        warped = source.reproject(crs="EPSG:3857", resolution=20.0)
        left, bottom, _, top = warped.bounds
        stops = [(0.0, (0, 0, 0)), (20.0, (200, 100, 0))]
        for built in (
            source.colorize(stops, band=2),
            warped.select([3, 2]),
            warped.clip_box(left, bottom, left + 40, top),
        ):
            path = built.to_vrt(tmp_path / "step.vrt")
            assert self.steps(vrt.read(path))[-1] == self.steps(built)[-1]

    def test_a_clip_beside_its_source_reads_one_step(self, tmp_path: Path) -> None:
        # Previously the rejoined copies lacked the metadata, so the source was listed twice.
        source = self.rgb(tmp_path)
        source.metadata["SENSOR"] = "S2"
        left, bottom, _, top = source.bounds
        assert self.steps(vrt.mosaic([source, source.clip_box(left, bottom, left + 30, top)])) == [
            "steps  ┌ 1  from_raster  rgb.tif",
            "       └ 2  clip_box     3 × 4 px at col 0, row 0",
            "         3  mosaic       #1, #2",
        ]

    @pytest.mark.parametrize(
        ("lut", "detail"),
        [
            ([("x", 1.0), (2.0, 3.0)], "2"),
            ([(float("-inf"), 0.0), (5.0, 1.0)], "5"),
            ([(float("nan"), 0.0)], ""),
        ],
    )
    def test_colorize_ranges_skip_values_that_are_not_finite(
        self, lut: list[tuple[float, float]], detail: str
    ) -> None:
        file = UPath("/d/a.tif")
        rgba = (ColorInterp.red, ColorInterp.green, ColorInterp.blue, ColorInterp.alpha)
        bands = [Band("uint8", [Source(file, lut=lut)], color_interp=c) for c in rgba]
        assert self.steps(VRT(UTM, GRID, 6, 4, bands)) == [
            f"steps  1  colorize  a.tif · {detail}".removesuffix(" · ")
        ]

    def test_lookalikes_keep_the_general_verbs(self) -> None:
        inner = VRT(UTM, GRID, 6, 4, [Band("float32", [Source(UPath("/d/a.tif"))])] * 2)
        moved = VRT(UTM, Affine(10.0, 0.0, 500010.0, 0.0, -10.0, 5200000.0), 6, 4, inner.bands[:1])
        window = rasterio.windows.Window(1, 1, 6, 4)
        whole = rasterio.windows.Window(0, 0, 6, 4)
        # Reading a VRT on another grid resamples it: not a select.
        assert self.steps(model(Source(moved)))[-1] == "       └ 2  stack"
        # A window of the wrong band: not a clip.
        at_window = Affine(10.0, 0.0, 500010.0, 0.0, -10.0, 5199990.0)
        source = Source(inner, band=2, src_window=window, dst_window=whole)
        cut = VRT(UTM, at_window, 6, 4, [Band("float32", [source])])
        assert self.steps(cut)[-1] == "       └ 2  mosaic"
        # A window placed on a grid that is not the window's: not a clip.
        shifted = model(Source(inner, band=1, src_window=window, dst_window=whole))
        assert self.steps(shifted)[-1] == "       └ 2  mosaic"
        # One-band copies on different grids are not joined into one input.
        split = VRT(
            UTM,
            GRID,
            6,
            4,
            [
                Band("float32", [Source(part, src_window=window, dst_window=whole)])
                for part in (moved, VRT(UTM, GRID, 6, 4, inner.bands[:1]))
            ],
        )
        assert self.steps(split)[-1].split()[-3:] == ["stack", "#1,", "#2"]

    def test_html_names_the_steps(self, tmp_path: Path) -> None:
        source = self.rgb(tmp_path)
        html = html_of(
            source.select([2]).clip_box(*source.bounds).colorize([(0, (0, 0, 0)), (9, (9, 9, 9))])
        )
        assert ">select<" in html
        assert ">colorize<" in html


class TestTruncation:
    def test_hundred_sources(self) -> None:
        many = model(*(Source(UPath(f"/data/t{i:03}.tif")) for i in range(100)))
        text = repr(many)
        assert text.splitlines()[2] == (
            "steps  1  mosaic  t000.tif, t001.tif, t002.tif, t003.tif, t004.tif, … +95"
        )
        html = many._repr_html_()
        assert ">t004.tif<" in html
        assert ">t005.tif<" not in html
        assert "… +95" in html

    def test_twenty_bands(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", count=20)
        lines = repr(vrt.from_raster(tile)).splitlines()
        assert lines[0] == "VRT  6 × 4 px · 20 bands · float32 · EPSG:32632 · 10 m"
        assert lines[1] == "bands   1  float32  nodata=nan"
        assert lines[10] == "       10  float32  nodata=nan"
        assert lines[11] == "       … +10 bands"
        html = vrt.from_raster(tile)._repr_html_()
        assert "<summary>Bands <span>(20 · float32)</span></summary>" in html
        assert html.count("<tr>") == 1 + 10 + 1

    def test_nested_vrts_are_capped_per_step(self, tmp_path: Path) -> None:
        derived = [
            vrt.derive(make_tiff(tmp_path / f"a{i}.tif"), PixelFunction.db()) for i in range(7)
        ]
        stacked = vrt.stack(derived)
        lines = repr(stacked).splitlines()
        assert lines[-6:] == [
            "steps  1  derive  a0.tif · dB",
            "       2  derive  a1.tif · dB",
            "       3  derive  a2.tif · dB",
            "       4  derive  a3.tif · dB",
            "       5  derive  a4.tif · dB",
            "       6  stack   #1, #2, #3, #4, #5, … +2",
        ]
        assert "… +2" in stacked._repr_html_()

    def test_depth_is_capped(self, tmp_path: Path) -> None:
        chain = vrt.from_raster(make_tiff(tmp_path / "a.tif"))
        for _ in range(_summary.MAX_DEPTH + 2):
            chain = vrt.derive(chain, PixelFunction.scale())
        lines = repr(chain).splitlines()
        assert len(lines) == 2 + _summary.MAX_DEPTH
        assert lines[2] == "steps  ┌ 1  derive  VRT(6×4, EPSG:32632, 1 band) … · scale"
        assert lines[-1] == "       └ 6  derive  scale"

    def test_step_count_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_summary, "MAX_NODES", 2)
        stacked = vrt.stack(
            [vrt.derive(make_tiff(tmp_path / f"a{i}.tif"), PixelFunction.db()) for i in range(3)]
        )
        assert repr(stacked).splitlines()[-1] == (
            "       3  stack   #1, #2, VRT(6×4, EPSG:32632, 1 band) …"
        )


def html_of(value: VRT) -> str:
    """Return the HTML repr of ``value``."""
    return value._repr_html_()


def outlines(html: str) -> int:
    """Count the source-file outlines in an HTML repr."""
    return html.partition('<g class="fzb-vrt-src">')[2].partition("</g>")[0].count("<polygon")


class TestHtml:
    def test_sections(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        html = html_of(vrt.reproject(mosaic, like=latlon_target(tmp_path)))
        assert '<div class="fzb-vrt-title">VRT</div>' in html
        assert "<details><summary>Bands <span>(1 · float32)</span></summary>" in html
        assert "<details><summary>Steps <span>(2)</span></summary>" in html
        # Collapsed by default.
        assert "<details open" not in html
        assert "<dt>CRS</dt><dd>EPSG:4326<small>WGS 84</small></dd>" in html
        assert "<dt>Size</dt><dd>6 × 4 px</dd>" in html
        assert '<span class="fzb-vrt-verb">reproject</span>' in html
        assert "<span>EPSG:32632 → EPSG:4326</span>" in html
        assert f'title="{tmp_path / "t0.tif"}">t0.tif</span>' in html

    def test_bounds_show_each_axis_with_the_unit(self, tmp_path: Path) -> None:
        html = html_of(vrt.mosaic(tiles(tmp_path, 2)))
        assert (
            '<dt>Bounds</dt><dd><span class="fzb-vrt-ax">x</span>500000 – 500120 m<br>'
            '<span class="fzb-vrt-ax">y</span>5199960 – 5200000 m</dd>'
        ) in html

    def test_geographic_extent_of_a_projected_grid(self, tmp_path: Path) -> None:
        html = html_of(vrt.mosaic(tiles(tmp_path, 2)))
        assert (
            '<dt>Geographic</dt><dd><span class="fzb-vrt-ax">lon</span>9.00000° E – 9.00158° E'
            '<br><span class="fzb-vrt-ax">lat</span>46.953169° N – 46.953529° N</dd>'
        ) in html

    def test_a_lon_lat_grid_has_no_separate_geographic_extent(self, tmp_path: Path) -> None:
        html = html_of(vrt.from_raster(latlon_target(tmp_path)))
        assert "<dt>Bounds</dt><dd>" in html
        assert "Geographic" not in html
        assert 'class="fzb-vrt-zoom"' in html

    def test_maps(self, tmp_path: Path) -> None:
        html = html_of(vrt.mosaic(tiles(tmp_path, 3)))
        assert '<svg class="fzb-vrt-world"' in html
        assert f'<path class="fzb-vrt-land" d="{_map.LAND}"/>' in html
        assert '<svg class="fzb-vrt-zoom"' in html
        assert outlines(html) == 3
        assert f"<title>{tmp_path / 't2.tif'}</title>" in html
        assert '<i class="fzb-vrt-dash"></i>source files</span>' in html
        assert "<text " in html

    def test_outlines_are_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_map, "MAX_OUTLINES", 2)
        html = html_of(vrt.mosaic(tiles(tmp_path, 3)))
        assert outlines(html) == 2
        assert "2 of 3 source files" in html

    def test_a_file_read_by_a_warp_has_no_outline(self, tmp_path: Path) -> None:
        warped = vrt.reproject(make_tiff(tmp_path / "a.tif"), like=latlon_target(tmp_path))
        html = html_of(warped)
        assert 'class="fzb-vrt-zoom"' in html
        assert outlines(html) == 0
        assert "fzb-vrt-dash" not in html.partition("</style>")[2]

    def test_files_behind_a_warp_are_outlined(self, tmp_path: Path) -> None:
        warped = vrt.reproject(vrt.mosaic(tiles(tmp_path, 2)), like=latlon_target(tmp_path))
        assert outlines(html_of(warped)) == 2

    def test_equal_copies_outline_each_file_once(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        assert outlines(html_of(vrt.stack([mosaic, copy.deepcopy(mosaic)]))) == 2

    def test_a_small_footprint_is_drawn_from_the_plot_corner(self) -> None:
        # Single-precision SVG would round degrees of an 8 m footprint into a jagged shape.
        tiny = VRT(UTM, Affine(0.02, 0, 500000, 0, -0.02, 5200000), 400, 400, [Band("float32")])
        html = html_of(tiny)
        view = re.search(r'class="fzb-vrt-zoom" viewBox="0 0 ([\d.]+) ([\d.]+)"', html)
        assert view is not None
        ring = re.search(r'class="fzb-vrt-out" points="([^"]+)"', html)
        assert ring is not None
        corners = [float(value) for point in ring.group(1).split() for value in point.split(",")]
        assert min(corners) > 0
        assert max(corners) < max(float(view.group(1)), float(view.group(2)))

    @pytest.mark.parametrize(
        "crs", [None, CRS.from_wkt('LOCAL_CS["Lab grid",UNIT["metre",1]]')], ids=["none", "local"]
    )
    def test_no_map_without_a_place_on_the_globe(self, crs: CRS | None) -> None:
        html = html_of(model(Source(UPath("/d/a.tif")), crs=crs))
        assert "<svg" not in html
        assert "Geographic" not in html

    def test_rail_joins_steps_that_read_the_step_above(self, tmp_path: Path) -> None:
        first = vrt.derive(vrt.mosaic(tiles(tmp_path, 2)), PixelFunction.db())
        stacked = vrt.stack([first, vrt.from_raster(make_tiff(tmp_path / "o.tif", size=(12, 4)))])
        html = html_of(stacked)
        steps = re.findall(r'<li class="([^"]*)"><span class="fzb-vrt-num">(\d+)', html)
        assert steps == [
            ("fzb-vrt-step fzb-vrt-link", "1"),
            ("fzb-vrt-step", "2"),
            ("fzb-vrt-step", "3"),
            ("fzb-vrt-step fzb-vrt-last", "4"),
        ]
        assert '<span class="fzb-vrt-chip fzb-vrt-ref" title="derive">#2</span>' in html

    def test_grid_notes_show_where_the_grid_changes(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        derived = vrt.derive(vrt.reproject(mosaic, crs=UTM, resolution=20), PixelFunction.db())
        notes = re.findall(r'<div class="fzb-vrt-note">([^<]*)</div>', html_of(derived))
        assert notes == ["12 × 4 px · EPSG:32632 · 10 m", "6 × 2 px · EPSG:32632 · 20 m"]

    def test_escapes_user_strings(self, tmp_path: Path) -> None:
        attack = '<script>alert("x")</script>'
        tile = make_tiff(tmp_path / "a.tif", descriptions=[attack])
        risky = vrt.from_raster(tile)
        risky.bands[0].unit = "<b>m</b>"
        risky.bands[0].sources[0].path = UPath("https://example.com/<img src=x>.tif")
        html = risky._repr_html_()
        assert "<script" not in html
        assert "&lt;script&gt;" in html
        assert "<b>" not in html
        assert "<img" not in html
        # The only URLs are the escaped source path and its display name.
        escaped = (htmllib.escape(str(risky.bands[0].sources[0].path)), "https://…/&lt;img")
        assert all(text in html for text in escaped)
        for text in escaped:
            html = html.replace(text, "")
        assert "http" not in html

    def test_is_self_contained(self, tmp_path: Path) -> None:
        html = vrt.mosaic(tiles(tmp_path, 2))._repr_html_()
        assert "<script" not in html
        assert "http" not in html
        assert not re.search(r"url\(|@import|<link|<img", html)
        # Text follows the page's color; the only fixed colors are the accent's two shades.
        colors = set(re.findall(r"#[0-9a-fA-F]{3,6}\b|rgba?\(|hsla?\(|var\(--[\w-]+", html))
        assert colors == {"#2f7fb0", "#6cbad6", "var(--fzb-vrt-accent"}

    def test_without_bands_or_sources(self) -> None:
        html = VRT(None, GRID, 3, 2)._repr_html_()
        assert "<summary>Bands <span>(0)</span></summary><p" in html
        assert "<summary>Steps <span>(0)</span></summary><p" in html

    def test_a_failing_map_leaves_the_rest(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def fail(*_args: object) -> NoReturn:
            """Fail like a broken PROJ setup."""
            raise pyproj.exceptions.CRSError

        monkeypatch.setattr(_map, "footprint", fail)
        html = html_of(vrt.mosaic(tiles(tmp_path, 2)))
        assert "<svg" not in html
        assert "<summary>Steps <span>(1)</span></summary>" in html

    def test_text_fallback_matches_the_text_repr(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", descriptions=["<elevation>"])
        single = vrt.from_raster(tile)
        fallback = re.search(
            r'<pre class="fzb-vrt-fallback">(.*?)</pre>', single._repr_html_(), re.DOTALL
        )
        assert fallback is not None
        assert htmllib.unescape(fallback.group(1)) == repr(single)


class TestMap:
    def test_another_prime_meridian_and_unit(self) -> None:
        # NTF (Paris) / Lambert zone II: latitudes in grads, longitudes east of Paris.
        paris = VRT(CRS.from_epsg(27572), Affine(10.0, 0.0, 600000.0, 0.0, -10.0, 2200000.0), 6, 4)
        fp = _map.footprint(paris, str)
        assert fp is not None
        lon, lat = fp.ring[0]
        assert lon == pytest.approx(2.33722, abs=1e-4)
        assert lat == pytest.approx(46.8, abs=1e-3)
        assert not fp.plain_degrees

    def test_rotated_pole_is_mapped_in_true_degrees(self) -> None:
        # EURO-CORDEX: rotated longitude and latitude, derived from WGS 84.
        rotated = CRS.from_string(
            "+proj=ob_tran +o_proj=longlat +o_lon_p=0 +o_lat_p=39.25 +lon_0=18 +datum=WGS84"
        )
        cordex = VRT(rotated, Affine(0.11, 0, -28.375, 0, -0.11, 21.835), 424, 412)
        fp = _map.footprint(cordex, str)
        assert fp is not None
        lats = [lat for _, lat in fp.ring]
        assert min(lats) == pytest.approx(21.89, abs=0.01)
        assert max(lats) == pytest.approx(72.57, abs=0.01)
        assert not fp.plain_degrees

    @pytest.mark.parametrize(
        ("west", "width", "lons"), [(0, 360, (-180, 180)), (200, 10, (-160, -150))]
    )
    def test_longitudes_past_180_are_wrapped(
        self, west: float, width: int, lons: tuple[float, float]
    ) -> None:
        grid = VRT(CRS.from_epsg(4326), Affine(1, 0, west, 0, -1, 50), width, 10)
        fp = _map.footprint(grid, str)
        assert fp is not None
        assert (min(lon for lon, _ in fp.ring), max(lon for lon, _ in fp.ring)) == lons

    def test_ferro_meridian(self) -> None:
        # MGI (Ferro): longitudes east of Ferro, 17°40' west of Greenwich.
        ferro = VRT(CRS.from_epsg(4805), Affine(0.1, 0, 30, 0, -0.1, 48), 10, 10)
        fp = _map.footprint(ferro, str)
        assert fp is not None
        assert fp.ring[0] == pytest.approx((12.3333, 48.0), abs=1e-4)

    def test_another_body_has_no_map(self) -> None:
        moon = VRT(CRS.from_string("+proj=longlat +R=1737400 +no_defs"), GRID, 6, 4)
        assert _map.footprint(moon, str) is None

    def test_across_the_antimeridian_there_is_no_map(self) -> None:
        # UTM zone 60 runs past 180° E here.
        wide = VRT(
            CRS.from_epsg(32660), Affine(1000.0, 0.0, 700000.0, 0.0, -1000.0, 100000.0), 200, 100
        )
        assert _map.footprint(wide, str) is None

    @pytest.mark.parametrize(
        ("values", "text"),
        [
            ([-122.5, -121.25], "122.50° W – 121.25° W"),
            ([-0.5, 0.5], "0.50° W – 0.50° E"),
            ([0.0, 0.0], "0.000000° – 0.000000°"),
            ([9.0, 9.0006], "9.000000° E – 9.000600° E"),
        ],
    )
    def test_degree_range(self, values: list[float], text: str) -> None:
        assert _map.degree_range(values, "E", "W") == text


class _NoGrids(http.server.BaseHTTPRequestHandler):
    """Record every request for a PROJ grid and answer that it is missing."""

    hits: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # the name http.server calls
        """Record the request and answer it."""
        self.hits.append(self.path)
        self.send_response(404)
        self.end_headers()


_DHDN_REPR = """
from affine import Affine
from rasterio.crs import CRS
from freezebase.vrt import VRT
# DHDN / 3-degree Gauss-Kruger zone 3; DHDN to WGS 84 uses a grid.
dhdn = VRT(CRS.from_epsg(31467), Affine(10.0, 0, 3500000.0, 0, -10.0, 5400000.0), 6, 4)
assert "Geographic" in dhdn._repr_html_()
"""
_DHDN_TO_WGS84 = """
import pyproj
pyproj.Transformer.from_crs(31467, 4326, always_xy=True).transform(3500000, 5400000)
"""


def test_the_map_makes_no_proj_network_request(tmp_path: Path) -> None:
    server = http.server.HTTPServer(("127.0.0.1", 0), _NoGrids)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    # Fresh processes, since PROJ reads these settings when its context starts.
    env = {
        **os.environ,
        "PROJ_NETWORK": "ON",
        "PROJ_NETWORK_ENDPOINT": f"http://127.0.0.1:{server.server_port}",
        "PROJ_USER_WRITABLE_DIRECTORY": str(tmp_path),
    }
    try:
        subprocess.run([sys.executable, "-c", _DHDN_REPR], env=env, check=True, timeout=60)  # noqa: S603  # this interpreter
        repr_hits = list(_NoGrids.hits)
        # The control: a transformation to WGS 84 asks the stub for the grid. Older PROJ
        # retries the failed download for long, so stop at the first request.
        control = subprocess.Popen([sys.executable, "-c", _DHDN_TO_WGS84], env=env)  # noqa: S603  # this interpreter
        try:
            for _ in range(400):
                if len(_NoGrids.hits) > len(repr_hits) or control.poll() is not None:
                    break
                time.sleep(0.05)
        finally:
            control.kill()
            control.wait()
    finally:
        server.shutdown()
        server.server_close()
    if len(_NoGrids.hits) == len(repr_hits):
        pytest.skip("PROJ found the grid locally, so the stub cannot tell")
    assert repr_hits == []


class TestNestedReprs:
    def test_source_shows_a_nested_vrt_on_one_line(self, tmp_path: Path) -> None:
        nested = vrt.from_raster(make_tiff(tmp_path / "a.tif"))
        assert repr(Source(nested, band=2)) == "Source(path=VRT(6×4, EPSG:32632, 1 band), band=2)"

    def test_source_shows_fields_that_are_set(self) -> None:
        source = Source(UPath("/data/a.tif"), nodata=0.0)
        assert repr(source) == f"Source(path={UPath('/data/a.tif')!r}, band=1, nodata=0.0)"

    def test_warp_shows_a_nested_vrt_on_one_line(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
        assert warped.warp is not None
        text = repr(warped.warp)
        assert text.startswith("Warp(source=VRT(12×4, EPSG:32632, 1 band), options_xml=<")
        assert "\n" not in text

    def test_band_keeps_the_dataclass_repr(self, tmp_path: Path) -> None:
        nested = vrt.from_raster(make_tiff(tmp_path / "a.tif"))
        derived = vrt.derive(nested, PixelFunction.db())
        text = repr(derived.bands[0])
        assert text.startswith("Band(dtype='float32', sources=[Source(path=VRT(6×4, ")
        assert "\n" not in text

    def test_dataclass_does_not_replace_the_custom_reprs(self) -> None:
        for cls in (VRT, Source, Warp):
            assert cls.__repr__.__code__.co_filename == _model.__file__
        assert Band.__repr__.__code__.co_filename != _model.__file__


def test_reprs_do_no_io(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mosaic = vrt.mosaic(tiles(tmp_path, 2))
    warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
    stacked = vrt.stack([warped, vrt.derive(warped, PixelFunction.db())])
    remote = model(
        Source(UPath("s3://bucket/a.tif", anon=True)),
        Source(UPath("https://example.com/b.tif")),
        # A VRT file source, which source_files() would follow.
        Source(UPath(tmp_path / "missing.vrt")),
    )

    def refuse(*_args: object, **_kwargs: object) -> NoReturn:
        """Fail any attempt at I/O."""
        msg = "a repr did I/O"
        raise AssertionError(msg)

    monkeypatch.setattr(rasterio, "open", refuse)
    monkeypatch.setattr(fsspec, "open", refuse)
    monkeypatch.setattr(fsspec, "filesystem", refuse)
    for name in ("open", "read_text", "read_bytes", "exists", "stat"):
        monkeypatch.setattr(UPath, name, refuse)
        monkeypatch.setattr(Path, name, refuse)
    for value in (stacked, remote):
        assert repr(value)
        assert value._repr_html_()
    assert stacked.warp is None
    assert warped.warp is not None
    assert repr(warped.warp)
    assert repr(stacked.bands[1].sources[0])


CUSTOM_TM = CRS.from_wkt(
    'PROJCS["Custom TM",GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,'
    '298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]],'
    'PROJECTION["Transverse_Mercator"],PARAMETER["latitude_of_origin",0],'
    'PARAMETER["central_meridian",9.1],PARAMETER["scale_factor",0.9996],'
    'PARAMETER["false_easting",500000],PARAMETER["false_northing",0],UNIT["metre",1]]'
)


def warped_with(options_xml: str) -> VRT:
    """Build a warped VRT model of a file with the given warp options."""
    return VRT(
        UTM, GRID, 6, 4, [Band("float32")], warp=Warp(UPath("/d/a.tif"), options_xml, (6, 4))
    )


def warp_line(options_xml: str) -> str:
    """Return the step line of a warped VRT with the given warp options."""
    return repr(warped_with(options_xml)).splitlines()[2]


class _Hits(http.server.BaseHTTPRequestHandler):
    """Record every request and answer with an EPSG code."""

    hits: ClassVar[list[str]] = []

    def do_GET(self) -> None:  # the name http.server calls
        """Record the request and answer it."""
        self.hits.append(self.path)
        body = b"EPSG:3035"
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


class TestWarpLabels:
    def test_srs_file_name_is_not_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        (tmp_path / "srs.prj").write_text(CRS.from_epsg(3035).to_wkt())
        monkeypatch.chdir(tmp_path)
        # Previously GDAL read the file and labeled the warp EPSG:3035.
        assert warp_line("<W><SourceSRS>srs.prj</SourceSRS></W>") == (
            "steps  1  reproject  a.tif · srs.prj → EPSG:32632"
        )

    def test_srs_url_is_not_requested(self) -> None:
        server = http.server.HTTPServer(("127.0.0.1", 0), _Hits)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/crs"
            # Previously GDAL fetched the URL.
            assert warp_line(f"<W><SourceSRS>{url}</SourceSRS></W>") == (
                f"steps  1  reproject  a.tif · {url} → EPSG:32632"
            )
        finally:
            server.shutdown()
            server.server_close()
        assert _Hits.hits == []

    def test_epsg_code(self) -> None:
        assert warp_line("<W><SourceSRS> epsg:4326 </SourceSRS></W>") == (
            "steps  1  reproject  a.tif · EPSG:4326 → EPSG:32632"
        )

    def test_unparsable_wkt_shows_its_text(self) -> None:
        assert warp_line("<W><SourceSRS>FOO[1]</SourceSRS></W>") == (
            "steps  1  reproject  a.tif · FOO[1] → EPSG:32632"
        )

    def test_same_crs_shows_one_crs(self) -> None:
        wkt = UTM.to_wkt()
        assert warp_line(
            f"<W><SourceSRS>{wkt}</SourceSRS><TargetSRS>EPSG:32632</TargetSRS></W>"
        ) == ("steps  1  reproject  a.tif · EPSG:32632")
        assert warp_line("<W><SourceSRS>x</SourceSRS><TargetSRS>x</TargetSRS></W>") == (
            "steps  1  reproject  a.tif · x"
        )

    def test_blank_source_srs_shows_one_crs(self) -> None:
        assert warp_line("<W><SourceSRS>  \n </SourceSRS></W>") == (
            "steps  1  reproject  a.tif · EPSG:32632"
        )


class TestLabels:
    def test_3d_crs_without_authority_names_its_height(self) -> None:
        wkt = pyproj.CRS("EPSG:32632").to_3d().to_wkt()
        assert _crs.crs_label(CRS.from_wkt(wkt)) == "WGS 84 / UTM zone 32N + ellipsoidal height"

    def test_custom_crs_is_named_and_identified_once(self) -> None:
        _crs._wkt_label.cache_clear()
        custom = VRT(CUSTOM_TM, GRID, 6, 4, [Band("float32", [Source(UPath("/d/a.tif"))])])
        assert (
            repr(custom).splitlines()[0] == "VRT  6 × 4 px · 1 band · float32 · Custom TM · 10 m"
        )
        repr(custom)
        assert _crs._wkt_label.cache_info().misses == 1

    def test_two_bands_of_one_file(self) -> None:
        a = UPath("/d/a.tif")
        composite = model(Source(a, band=1), Source(a, band=2))
        assert repr(composite).splitlines()[2] == "steps  1  mosaic  a.tif"

    def test_kilometers(self) -> None:
        km = CRS.from_string("+proj=utm +zone=32 +datum=WGS84 +units=km +no_defs")
        grid = VRT(km, Affine(0.01, 0, 500, 0, -0.01, 5200), 6, 4, [Band("float32")])
        assert repr(grid).splitlines()[0].endswith(" · 0.01 km")

    def test_south_up_bounds_run_low_to_high(self) -> None:
        south = VRT(UTM, Affine(10, 0, 500000, 0, 10, 5200000), 6, 4)
        assert '<span class="fzb-vrt-ax">y</span>5200000 – 5200040 m' in html_of(south)

    def test_derive_names_are_clipped(self) -> None:
        name = "evil\nsteps  9  fake" * 5
        evil = VRT(UTM, GRID, 6, 4, [Band("float32", pixel_function=PixelFunction(name))])
        lines = repr(evil).splitlines()
        assert len(lines) == 3
        assert lines[2].endswith("…")

    def test_derived_next_to_plain_bands(self) -> None:
        a, b, c = (Source(UPath(f"/d/{name}.tif")) for name in "abc")
        mixed = VRT(
            UTM,
            GRID,
            6,
            4,
            [Band("float32", [a, b], pixel_function=PixelFunction.div()), Band("float32", [c])],
        )
        assert repr(mixed).splitlines()[3:] == [
            "steps  1  derive  a.tif, b.tif, c.tif · div in 1 of 2 bands"
        ]

    def test_inputs_past_the_caps_are_counted(self) -> None:
        nested = [model(Source(UPath(f"/d/n{i}.tif"))) for i in range(7)]
        mixed = model(*(Source(v) for v in nested), Source(UPath("/d/a.tif")))
        lines = repr(mixed).splitlines()
        assert lines[2] == "steps  1  from_raster  n0.tif"
        assert lines[-1] == "       6  mosaic       #1, #2, #3, #4, #5, a.tif, … +2"
        assert "… +2" in mixed._repr_html_()

    @pytest.mark.parametrize(
        ("nodata", "text"),
        [
            (255, "255"),
            (-9223372036854775808, "-9223372036854775808"),
            (float(-(2**63)), "-9223372036854775808"),
            (-3.4028234663852886e38, "-3.4028234663852886e+38"),
            (0.1, "0.1"),
            (float("inf"), "inf"),
        ],
    )
    def test_nodata_is_exact(self, nodata: float, text: str) -> None:
        band = model(Source(UPath("/d/a.tif"))).bands[0]
        band.nodata = nodata
        row = VRT(UTM, GRID, 6, 4, [band])
        assert repr(row).splitlines()[1] == f"bands  1  float32  nodata={text}"

    def test_files_without_a_name(self) -> None:
        unnamed = model(Source(UPath("/")), Source(UPath("s3://bucket", anon=True)))
        assert repr(unnamed).splitlines()[2] == "steps  1  mosaic  /, s3://bucket/"

    def test_urls_lose_their_query_and_credentials(self) -> None:
        url = "https://user:pw@example.com/data/a.tif?X-Amz-Signature=SECRET#frag"
        presigned = model(Source(UPath(url)))
        assert repr(presigned).splitlines()[2] == "steps  1  from_raster  https://…/a.tif"
        html = presigned._repr_html_()
        assert 'title="https://example.com/data/a.tif"' in html
        assert "SECRET" not in html
        assert "pw@" not in html


class TestInvalidModels:
    def test_falls_back_to_the_short_form(self) -> None:
        broken = VRT(UTM, GRID, 6, 4, [Band(None)])  # pyrefly: ignore[bad-argument-type]  # invalid on purpose
        assert repr(broken) == "VRT(6×4, EPSG:32632, 1 band)"
        assert broken._repr_html_() == "<pre>VRT(6×4, EPSG:32632, 1 band)</pre>"

    def test_falls_back_to_the_object_repr(self) -> None:
        broken = VRT("EPSG:32632", GRID, 6, 4)  # A str, not a CRS.
        assert repr(broken).startswith("<freezebase.vrt._model.VRT object at ")
        assert repr(Source(broken)).startswith("Source(path=<freezebase.vrt._model.VRT object at ")


def test_many_dtypes_are_clipped() -> None:
    dtypes = [
        "uint8",
        "int8",
        "uint16",
        "int16",
        "uint32",
        "int32",
        "float32",
        "float64",
        "complex64",
    ]
    mixed = VRT(UTM, GRID, 6, 4, [Band(dtype) for dtype in dtypes])
    listed = repr(mixed).splitlines()[0].split(" · ")[2]
    assert listed == "uint8, int8, uint16, int16, uint32, int32, float32, float64…"
