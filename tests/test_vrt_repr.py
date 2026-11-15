"""Tests for the text and HTML reprs of the freezebase.vrt model."""

# The reprs print sizes with a multiplication sign.
# ruff: noqa: RUF001

from __future__ import annotations

import html as htmllib
import http.server
from pathlib import Path
import re
import threading
from typing import ClassVar, NoReturn

from affine import Affine
import fsspec
import pyproj
import pytest
import rasterio
from rasterio.crs import CRS
from upath import UPath

from freezebase import vrt
from freezebase.vrt import VRT, Band, PixelFunction, Source, Warp, _crs, _model, _summary
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
            "└─ a.tif"
        )

    def test_mosaic(self, tmp_path: Path) -> None:
        assert repr(vrt.mosaic(tiles(tmp_path, 3))) == (
            "VRT  18 × 4 px · 1 band · float32 · EPSG:32632 · 10 m\n"
            "bands  1  float32  nodata=nan\n"
            "└─ mosaic of 3 files: t0.tif, t1.tif, t2.tif"
        )

    def test_reprojected_mosaic(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 3))
        warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
        assert repr(warped) == (
            "VRT  6 × 4 px · 1 band · float32 · EPSG:4326 · 0.0001°\n"
            "bands  1  float32  nodata=nan\n"
            "└─ reprojected: EPSG:32632 → EPSG:4326\n"
            "   └─ mosaic of 3 files: t0.tif, t1.tif, t2.tif"
        )

    def test_reprojected_file(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        warped = vrt.reproject(tile, like=latlon_target(tmp_path))
        assert repr(warped).splitlines()[2:] == [
            "└─ reprojected: EPSG:32632 → EPSG:4326",
            "   └─ a.tif",
        ]

    def test_resampled_without_reprojecting(self, tmp_path: Path) -> None:
        resampled = vrt.reproject(vrt.mosaic(tiles(tmp_path, 2)), crs=UTM, resolution=20)
        assert repr(resampled) == (
            "VRT  6 × 2 px · 1 band · float32 · EPSG:32632 · 20 m\n"
            "bands  1  float32  nodata=nan\n"
            "└─ resampled: EPSG:32632\n"
            "   └─ mosaic of 2 files: t0.tif, t1.tif"
        )

    def test_reproject_labels_the_src_crs_override(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        target = make_tiff(tmp_path / "target.tif", origin=(-500000.0, 5200000.0))
        warped = vrt.reproject(mosaic, src_crs="EPSG:32633", like=target)
        # The nested mosaic still says EPSG:32632; the warp reads it as EPSG:32633.
        assert repr(warped).splitlines()[2] == "└─ reprojected: EPSG:32633 → EPSG:32632"

    def test_derive(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        derived = vrt.derive(tile, PixelFunction.db(), description="VV")
        assert repr(derived) == (
            "VRT  6 × 4 px · 1 band · float32 · EPSG:32632 · 10 m\n"
            'bands  1  "VV"  float32  nodata=nan\n'
            "└─ derived: dB\n"
            "   └─ a.tif"
        )

    def test_derive_of_two_files(self, tmp_path: Path) -> None:
        a, b = make_tiff(tmp_path / "a.tif"), make_tiff(tmp_path / "b.tif")
        ratio = vrt.derive([a, b], PixelFunction.div())
        assert repr(ratio).splitlines()[2:] == ["└─ derived: div", "   └─ 2 files: a.tif, b.tif"]

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
            "└─ stack of 2 files: a.tif, b.tif"
        )

    def test_nested_in_nested(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 3))
        warped = vrt.reproject(mosaic, like=latlon_target(tmp_path))
        derived = vrt.derive(warped, PixelFunction.db(), description="dB")
        other = make_tiff(tmp_path / "other.tif", origin=(9.0, 47.0), res=(0.0001, 0.0001))
        other_vrt = vrt.reproject(other, src_crs="EPSG:4326", like=latlon_target(tmp_path))
        stacked = vrt.stack([derived, warped, other_vrt])
        assert repr(stacked) == (
            "VRT  6 × 4 px · 3 bands · float32 · EPSG:4326 · 0.0001°\n"
            'bands  1  "dB"  float32  nodata=nan\n'
            "       2        float32  nodata=nan\n"
            "       3        float32  nodata=nan\n"
            "└─ stack of 3 sources\n"
            "   ├─ derived: dB\n"
            "   │  └─ reprojected: EPSG:32632 → EPSG:4326\n"
            "   │     └─ mosaic of 3 files: t0.tif, t1.tif, t2.tif\n"
            "   ├─ reprojected: EPSG:32632 → EPSG:4326\n"
            "   │  └─ mosaic of 3 files: t0.tif, t1.tif, t2.tif\n"
            "   └─ resampled: EPSG:4326\n"
            "      └─ other.tif"
        )

    def test_mosaic_of_files_and_a_vrt(self, tmp_path: Path) -> None:
        a, b = tiles(tmp_path, 2)
        mixed = vrt.mosaic([vrt.derive(a, PixelFunction.scale()), b])
        assert repr(mixed).splitlines()[2:] == [
            "└─ mosaic of 2 sources",
            "   ├─ derived: scale",
            "   │  └─ t0.tif",
            "   └─ t1.tif",
        ]

    def test_a_plain_view_of_a_vrt_shows_the_vrt(self, tmp_path: Path) -> None:
        derived = vrt.derive(make_tiff(tmp_path / "a.tif"), PixelFunction.db())
        assert repr(vrt.stack([derived])).splitlines()[2:] == ["└─ derived: dB", "   └─ a.tif"]

    def test_without_crs_bands_or_sources(self) -> None:
        empty = VRT(None, Affine(10.0, 0.0, 0.0, 0.0, -20.0, 0.0), 3, 2)
        assert repr(empty) == (
            "VRT  3 × 2 px · 0 bands · no CRS · 10 × 20\nbands  none\n└─ no sources"
        )

    def test_remote_files_keep_their_protocol(self) -> None:
        remote = model(
            Source(UPath("s3://bucket/key/a.tif", anon=True)),
            Source(UPath("https://example.com/data/b.tif")),
        )
        assert (
            repr(remote).splitlines()[2] == "└─ mosaic of 2 files: s3://…/a.tif, https://…/b.tif"
        )

    def test_crs_without_authority_shows_its_name(self) -> None:
        crs = CRS.from_wkt('LOCAL_CS["Lab grid",UNIT["metre",1]]')
        assert repr(model(crs=crs)).splitlines()[0].endswith("· Lab grid · 10 m")

    def test_unreadable_warp_options(self) -> None:
        tile = UPath("/data/a.tif")
        warped = VRT(UTM, GRID, 6, 4, [Band("float32")], warp=Warp(tile, "<oops", (6, 4)))
        assert repr(warped).splitlines()[2] == "└─ warped"
        warped.warp = Warp(tile, "<W><SourceSRS>bogus</SourceSRS></W>", (6, 4))
        assert repr(warped).splitlines()[2] == "└─ reprojected: bogus → EPSG:32632"

    def test_long_and_multiline_strings_are_clipped(self) -> None:
        long_name = "x" * 100 + ".tif"
        vrt_model = model(Source(UPath(f"/data/{long_name}")))
        vrt_model.bands[0].description = "first\nsecond"
        lines = repr(vrt_model).splitlines()
        assert lines[1] == 'bands  1  "first second"  float32'
        assert lines[2] == "└─ " + "x" * 59 + "…"


class TestTruncation:
    def test_hundred_sources(self) -> None:
        many = model(*(Source(UPath(f"/data/t{i:03}.tif")) for i in range(100)))
        text = repr(many)
        assert text.splitlines()[2] == (
            "└─ mosaic of 100 files: t000.tif, t001.tif, t002.tif, t003.tif, t004.tif, … +95"
        )
        html = many._repr_html_()
        assert "t004.tif" in html
        assert "t005.tif" not in html
        assert "… +95" in html

    def test_twenty_bands(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", count=20)
        lines = repr(vrt.from_raster(tile)).splitlines()
        assert lines[0] == "VRT  6 × 4 px · 20 bands · float32 · EPSG:32632 · 10 m"
        assert lines[1] == "bands   1  float32  nodata=nan"
        assert lines[10] == "       10  float32  nodata=nan"
        assert lines[11] == "       … +10 bands"
        html = vrt.from_raster(tile)._repr_html_()
        assert "Bands (20)" in html
        assert html.count("<tr>") == 1 + 10 + 1

    def test_nested_vrts_are_capped_per_node(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif")
        derived = [vrt.derive(tile, PixelFunction.db()) for _ in range(7)]
        lines = repr(vrt.stack(derived)).splitlines()
        assert lines[8] == "└─ stack of 7 sources"
        assert lines.count("   ├─ derived: dB") == 5
        assert lines[-1] == "   └─ … +2 more"
        assert "… +2 more" in vrt.stack(derived)._repr_html_()

    def test_depth_is_capped(self, tmp_path: Path) -> None:
        chain = vrt.from_raster(make_tiff(tmp_path / "a.tif"))
        for _ in range(_summary.MAX_DEPTH + 2):
            chain = vrt.derive(chain, PixelFunction.scale())
        lines = repr(chain).splitlines()
        assert len(lines) == 2 + _summary.MAX_DEPTH + 1
        assert lines[-1].endswith("└─ VRT(6×4, EPSG:32632, 1 band) …")

    def test_node_count_is_capped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(_summary, "MAX_NODES", 2)
        tile = make_tiff(tmp_path / "a.tif")
        stacked = vrt.stack([vrt.derive(tile, PixelFunction.db()) for _ in range(3)])
        assert repr(stacked).splitlines()[5:] == [
            "   ├─ derived: dB",
            "   │  └─ a.tif",
            "   ├─ VRT(6×4, EPSG:32632, 1 band) …",
            "   └─ VRT(6×4, EPSG:32632, 1 band) …",
        ]


class TestHtml:
    def test_sections(self, tmp_path: Path) -> None:
        mosaic = vrt.mosaic(tiles(tmp_path, 2))
        html = vrt.reproject(mosaic, like=latlon_target(tmp_path))._repr_html_()
        for section in ("Grid", "Bands (1)", "Pipeline"):
            assert f"<details open><summary>{section}</summary>" in html
        assert "6 × 4 px · 1 band · float32 · EPSG:4326 · 0.0001°" in html
        assert "<dd>EPSG:4326 (WGS 84)</dd>" in html
        assert "reprojected: EPSG:32632 → EPSG:4326" in html
        assert f'title="{tmp_path / "t0.tif"}">t0.tif</span>' in html

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
        # Colors follow the page, so dark themes work.
        assert not re.search(r"#[0-9a-fA-F]{3,6}\b|rgb\(|var\(--", html)

    def test_without_bands_or_sources(self) -> None:
        html = VRT(None, GRID, 3, 2)._repr_html_()
        assert "<summary>Bands (0)</summary><p" in html
        assert ">no sources</span></li>" in html

    def test_text_fallback_matches_the_text_repr(self, tmp_path: Path) -> None:
        tile = make_tiff(tmp_path / "a.tif", descriptions=["<elevation>"])
        single = vrt.from_raster(tile)
        fallback = re.search(
            r'<pre class="fzb-vrt-fallback">(.*?)</pre>', single._repr_html_(), re.DOTALL
        )
        assert fallback is not None
        assert htmllib.unescape(fallback.group(1)) == repr(single)


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
    """Return the pipeline line of a warped VRT with the given warp options."""
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
            "└─ reprojected: srs.prj → EPSG:32632"
        )

    def test_srs_url_is_not_requested(self) -> None:
        server = http.server.HTTPServer(("127.0.0.1", 0), _Hits)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/crs"
            # Previously GDAL fetched the URL.
            assert warp_line(f"<W><SourceSRS>{url}</SourceSRS></W>") == (
                f"└─ reprojected: {url} → EPSG:32632"
            )
        finally:
            server.shutdown()
            server.server_close()
        assert _Hits.hits == []

    def test_epsg_code(self) -> None:
        assert warp_line("<W><SourceSRS> epsg:4326 </SourceSRS></W>") == (
            "└─ reprojected: EPSG:4326 → EPSG:32632"
        )

    def test_unparsable_wkt_shows_its_text(self) -> None:
        assert warp_line("<W><SourceSRS>FOO[1]</SourceSRS></W>") == (
            "└─ reprojected: FOO[1] → EPSG:32632"
        )

    def test_same_crs_is_resampled(self) -> None:
        wkt = UTM.to_wkt()
        assert warp_line(
            f"<W><SourceSRS>{wkt}</SourceSRS><TargetSRS>EPSG:32632</TargetSRS></W>"
        ) == ("└─ resampled: EPSG:32632")
        assert warp_line("<W><SourceSRS>x</SourceSRS><TargetSRS>x</TargetSRS></W>") == (
            "└─ resampled: x"
        )

    def test_blank_source_srs_is_resampled(self) -> None:
        assert warp_line("<W><SourceSRS>  \n </SourceSRS></W>") == "└─ resampled: EPSG:32632"


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
        assert repr(composite).splitlines()[2] == "└─ a.tif"

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
            "└─ derived in 1 of 2 bands: div",
            "   └─ 3 files: a.tif, b.tif, c.tif",
        ]

    def test_more_vrts_come_before_the_files(self) -> None:
        nested = [model(Source(UPath(f"/d/n{i}.tif"))) for i in range(7)]
        mixed = model(*(Source(v) for v in nested), Source(UPath("/d/a.tif")))
        lines = repr(mixed).splitlines()
        assert lines[2] == "└─ mosaic of 8 sources"
        assert lines[-2:] == ["   ├─ … +2 more", "   └─ a.tif"]
        html = mixed._repr_html_()
        assert html.index("… +2 more") < html.index(">a.tif<")

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
        assert repr(unnamed).splitlines()[2] == "└─ mosaic of 2 files: /, s3://bucket/"

    def test_urls_lose_their_query_and_credentials(self) -> None:
        url = "https://user:pw@example.com/data/a.tif?X-Amz-Signature=SECRET#frag"
        presigned = model(Source(UPath(url)))
        assert repr(presigned).splitlines()[2] == "└─ https://…/a.tif"
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
