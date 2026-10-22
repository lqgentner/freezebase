# `freezebase.vrt` VRT Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `freezebase.vrt`'s five fixed-layout functions with an inspectable VRT model. It is built by `mosaic` / `warp` / `stack` / `derive` / `from_raster`, read and written losslessly, and used underneath `write_cog` and `rewrite_tiff`.

**Architecture:**
- `freezebase/vrt.py` becomes a package. The model is mutable dataclasses (`VRT`, `Band`, `Source`, `PixelFunction`, `Warp`), serialized to GDAL VRT XML with `xml.etree` and parsed back.
- Builders read source headers through rasterio and return a `VRT`; nothing touches disk until `VRT.write`.
- `warp` wraps `rasterio.vrt.WarpedVRT`, then parses GDAL's serialized warped VRT into the model.
- `raster.write_cog` / `rewrite_tiff` stream a `VRT` with `rasterio.shutil.copy`.

**Tech Stack:** Python 3.12+, rasterio ≥ 1.5.0 (bundled GDAL ≥ 3.12.1), pyproj, universal-pathlib, pytest, Ruff, Pyrefly (strict).

**Spec:** `docs/superpowers/specs/2026-10-22-vrt-module-design.md`. Read it before any task; this plan argues from it.

## Global Constraints

- **Branches:** see Delivery. Tasks 1–14 go on `feat/vrt-model`, Tasks 15–17 on `feat/raster-on-vrt`.
- **Python and modules:** Python 3.12–3.15. Every module starts with `from __future__ import annotations`. Use `type X = ...`, `collections.abc`, and `X | None`.
- **Checks after every task:**
  - `uv run pytest`, `uv run ruff format`, `uv run ruff check --fix`, `uv run ruff format`, `uv run pyrefly check`. All must be clean.
  - Pyrefly runs in strict mode. Every `cast` or `# pyrefly: ignore[...]` carries a comment saying why.
- **Docstrings and comments:**
  - NumPy docstrings with `Parameters` / `Returns` / `Raises` on public names; no `Examples` / `Notes` sections.
  - Private helpers, fixtures and tests: one-line docstrings.
  - Comments are one line by default and explain why.
- **Dependencies:** no new runtime dependency. No `osgeo` import, no Python pixel functions, no `defusedxml`.
- **Import path:** `freezebase.vrt` stays the import path for every public name.
- **GDAL version:** inline nesting needs GDAL ≥ 3.11. Writing or opening a nested VRT on an older GDAL raises `RuntimeError` naming the version.
- **Public text** (commit messages, CHANGELOG, docs, docstrings): never name the private downstream repository, and don't describe what it builds. Before each commit, `git diff --cached | grep -i glacier` must print nothing.
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
  ```
- **CHANGELOG:** every behaviour change goes under `## Unreleased` (Keep a Changelog). Breaking changes are marked **Breaking:** with before/after. This is a minor release.
- **Coverage:** patch coverage ≥ 90%:
  ```bash
  uv run pytest -q --cov=freezebase --cov-branch --cov-report=term-missing
  ```
- **Tests:** offline, except those marked `integration` (RustFS) and the vertical-shift tests, which skip when the `ch_swisstopo` PROJ grids are absent.
- **Shared constants** (define once, in `vrt/_model.py` unless stated):
  - `_GDAL_DTYPE_NAMES`: the existing table.
  - `_ALIGNMENT_TOL = 1e-3` pixels, `_PIXEL_SIZE_TOL = 1e-6`, `_DENSIFY_PTS = 51`: the existing values, moved to `vrt/_grid.py`.
  - The approximate-transform tolerance is `0.125` pixels.

## Review Focus

1. **Geographic CRS axis order.** A VRT built in `EPSG:4326` must open with longitude as x. GDAL reads an `<SRS>` without `dataAxisToSRSAxisMapping` in traditional GIS order; the writer must not break that. Test in Task 3.
2. **Special characters in paths.** Paths containing spaces, `&`, `<` or non-ASCII letters, both in relative `../` paths and inside an inline nested `<VRTDataset>`. They must round-trip and the VRT must open. Test in Task 4.
3. **A source on another S3 bucket than the VRT.** It is written as absolute `/vsis3/`, and on read it gets the reading path's storage options. Test in Task 5.
4. **A VRT written by freezebase 0.8.** `build_vrt_mosaic` and `create_decibel_vrt` output (bare-filename `ComplexSource`s, no `SourceProperties`) must parse with `vrt.read`. Test in Task 5.
5. **Sources without nodata.** A source with nodata `None` passed to `derive` or `mosaic` writes a `SimpleSource`, not `<NODATA>None</NODATA>`. Mixing it with NaN-nodata sources in one mosaic raises, as today. Tests in Tasks 8 and 10.

---

## File structure

| Path | Responsibility |
|---|---|
| `src/freezebase/vrt/__init__.py` | Public re-exports (model, builders, `read`, legacy wrappers) |
| `src/freezebase/vrt/_model.py` | Dataclasses, `PixelFunction` constructors, the dtype table |
| `src/freezebase/vrt/_paths.py` | Source path ↔ GDAL path, and the relative-path policy |
| `src/freezebase/vrt/_xml.py` | `VRT` ↔ XML element tree, the GDAL version guard |
| `src/freezebase/vrt/_io.py` | `read`, `VRT.write` / `open` / `env` / `source_files` |
| `src/freezebase/vrt/_grid.py` | `RasterInfo`, header reading, validation, alignment and snapping helpers |
| `src/freezebase/vrt/_builders.py` | `from_raster`, `stack`, `mosaic`, `derive` |
| `src/freezebase/vrt/_warp.py` | `warp`, vertical-shift inference, tolerance, ballpark |
| `src/freezebase/vrt/_legacy.py` | `build_vrt_mosaic`, `create_*_vrt`, as wrappers |
| `src/freezebase/raster.py` | HTTPS paths and env, `write_cog(VRT | path)`, `rewrite_tiff(VRT)`; `merge_tiffs` removed |
| `tests/rasters.py` | `make_tiff` test helper shared by the VRT test files |
| `tests/test_vrt_model.py` | Model, XML, paths, read/write, env (Tasks 3–6) |
| `tests/test_vrt_builders.py` | `from_raster`, `stack`, `mosaic`, `derive` (Tasks 7–9) |
| `tests/test_vrt_warp.py` | `warp`, including vertical shifts (Tasks 10–11) |
| `tests/test_vrt.py` | Legacy wrappers (Task 13) |

`VRT`'s methods live on the dataclass in `_model.py` and delegate to `_xml` / `_io` through function-level imports. That keeps the modules acyclic. `raster.py` imports `freezebase.vrt` lazily inside `write_cog` / `rewrite_tiff`, and under `TYPE_CHECKING` for annotations; `vrt` imports `raster` at module level.

---

## Delivery

Two pull requests, both from fresh branches:

1. **`feat/vrt-model`** (from `main`): Tasks 1–14.
2. **`feat/raster-on-vrt`**, created from `feat/vrt-model` after Task 14 (`git switch -c feat/raster-on-vrt`): Tasks 15–17. Rebase it onto `main` once pull request 1 merges.

---

### Task 1: Turn `vrt.py` into a package (pure move)

**Files:**
- Delete: `src/freezebase/vrt.py`
- Create: `src/freezebase/vrt/__init__.py`, `src/freezebase/vrt/_legacy.py`, `tests/rasters.py`
- Modify: `tests/test_vrt.py` (imports), `tests/test_raster.py:910-911` (the `s3_uploads` fixture)
- Add: `docs/superpowers/specs/2026-10-22-vrt-module-design.md`, `docs/superpowers/plans/2026-10-22-vrt-module.md`

**Interfaces:**
- Produces:
  - `freezebase.vrt._legacy` holds today's code unchanged, except that every upload calls `raster._upload(...)` through the module (`from freezebase import raster`), so patching `raster._upload` covers VRT writes too.
  - `tests.rasters.make_tiff(path: Path, *, origin=(500000.0, 5200000.0), res=(10.0, 10.0), size=(6, 4), fill: float | None = None, dtype="float32", count=1, nodata: float | None = nan, crs="EPSG:32632", transform: Affine | None = None, descriptions: Sequence[str] | None = None) -> Path`.
  - With `fill=None`, `make_tiff` writes `np.arange(w*h*count)` reshaped and cast, so every pixel is distinct and a misregistration shows. `size` is `(width, height)`.

- [ ] **Step 1: Create the branch and commit the spec and plan**

```bash
git switch -c feat/vrt-model
git add docs/superpowers
git commit -m "docs: VRT model design and implementation plan"
```

- [ ] **Step 2: Move the code**
  - `git mv src/freezebase/vrt.py src/freezebase/vrt/_legacy.py`.
  - Replace `_upload(...)` calls with `raster._upload(...)`.
  - `vrt/__init__.py` re-exports `build_vrt_mosaic`, `create_decibel_vrt`, `create_rgb_vrt`, `create_rgba_vrt` and `create_warped_vrt`, with a module docstring `"""GDAL Virtual Dataset creation."""`.
  - In `tests/test_vrt.py`, import `_nodata_equal` from `freezebase.vrt._legacy`.
  - In `tests/test_raster.py`, delete `monkeypatch.setattr(vrt, "_upload", record)`, and drop `vrt` from its imports if it becomes unused.
  - Create `tests/rasters.py` with `make_tiff` as specified.

- [ ] **Step 3: Run the whole suite**

Run: `uv run pytest -q`
Expected: the same pass count as on `main`, 0 failures.

- [ ] **Step 4: Checks, then commit**

```bash
git add -A src/freezebase/vrt src/freezebase/vrt.py tests
git commit -m "refactor(vrt): turn vrt.py into a package"
```

---

### Task 2: Public HTTPS sources in `raster`

**Files:**
- Modify: `src/freezebase/raster.py` (`_to_vsi_uri`, `_env_for_path`, new `GDAL_HTTP_OPTIONS`, `http_env`), `src/freezebase/s3.py` (`GDAL_S3_OPTIONS` reuses the retry keys)
- Test: `tests/test_raster.py`

**Interfaces:**
- Produces:
  - `raster.GDAL_HTTP_OPTIONS: Mapping[str, str]` with these keys:
    - `GDAL_HTTP_TIMEOUT="30"`
    - `GDAL_HTTP_CONNECTTIMEOUT="10"`
    - `GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"`
    - `VSI_CACHE="TRUE"`
    - `VSI_CACHE_SIZE="25000000"`
    - `GDAL_HTTP_MAX_RETRY`, `GDAL_HTTP_RETRY_DELAY` and `GDAL_HTTP_RETRY_CODES`, with today's S3 values
  - The retry constants move to `raster.py`; `s3.py` re-imports them, so `freezebase.s3.GDAL_HTTP_MAX_RETRY` etc. stay importable.
  - `raster.http_env(**options: str) -> AbstractContextManager[None]`: a `rasterio.Env` with `GDAL_HTTP_OPTIONS | options` and no credentials.
  - `_to_vsi_uri("https://h/a.tif") == "/vsicurl/https://h/a.tif"`, and the same for `http`.
  - `_env_for_path` has an `"http" | "https"` branch that enters `http_env()`.

- [ ] **Step 1: Write the failing tests** in `tests/test_raster.py`, class `TestHttpSources`:

```python
def test_https_maps_to_vsicurl(self) -> None:
    assert _to_vsi_uri("https://example.com/a.tif") == "/vsicurl/https://example.com/a.tif"
    assert _to_vsi_uri("http://example.com/a.tif") == "/vsicurl/http://example.com/a.tif"

def test_https_env_sets_http_options(self) -> None:
    with _env_for_path("https://example.com/a.tif"):
        assert rasterio.env.getenv()["GDAL_HTTP_TIMEOUT"] == "30"

def test_http_env_overrides(self) -> None:
    with http_env(GDAL_HTTP_TIMEOUT="5"):
        assert rasterio.env.getenv()["GDAL_HTTP_TIMEOUT"] == "5"

def test_s3_options_share_the_retry_settings(self) -> None:
    for key in ("GDAL_HTTP_MAX_RETRY", "GDAL_HTTP_RETRY_DELAY", "GDAL_HTTP_RETRY_CODES"):
        assert GDAL_S3_OPTIONS[key] == GDAL_HTTP_OPTIONS[key]
```

Also update the existing test(s) that expect `ValueError("Unsupported protocol 'https'")` to use `ftp://` instead.

- [ ] **Step 2: Run them; expect failures** (`ImportError` for `http_env`).

Run: `uv run pytest tests/test_raster.py -k "Http or protocol" -q`

- [ ] **Step 3: Implement** as specified in Interfaces. Update `rasterio_open`'s `Raises` docstring to say "other than local, `s3`, `http` or `https`".

- [ ] **Step 4: Run; expect PASS.** Then the whole suite.

- [ ] **Step 5: CHANGELOG and commit.** Under **Added**: "`http://` / `https://` paths in `rasterio_open` and every reader built on it, read through `/vsicurl/` with `GDAL_HTTP_OPTIONS`; `http_env()`." Then:

```bash
git commit -am "feat(raster): read public http(s) rasters"
```

---

### Task 3: The model and its XML serialization

**Files:**
- Create: `src/freezebase/vrt/_model.py`, `src/freezebase/vrt/_xml.py`
- Modify: `src/freezebase/vrt/__init__.py` (export the model)
- Test: `tests/test_vrt_model.py`

**Interfaces:**
- Produces (in `_model.py`, all `@dataclass`, mutable, `field(default_factory=...)` for containers):
  - `type SourceRef = UPath | VRT`
  - `PixelFunction(name: str, args: dict[str, str | int | float] = {}, transfer_type: str | None = None)`. `__post_init__` turns every argument value into `str`, so a parsed VRT compares equal to the one that was written. Classmethods:
    - `db(fact: Literal[10, 20] = 10)` → `PixelFunction("dB", {"fact": fact})`;
    - `div()`, `diff()`, `scale()`;
    - `minimum()` → `"min"`, `maximum()` → `"max"`, `mean()` → `"mean"`.
  - `SourceProperties(width: int, height: int, dtype: str, block_width: int, block_height: int)`
  - `Source(path: SourceRef, band: int = 1, src_window: Window | None = None, dst_window: Window | None = None, nodata: float | None = None, lut: list[tuple[float, float]] | None = None, resampling: Resampling | None = None, properties: SourceProperties | None = None)`
  - `Band(dtype: str, sources: list[Source] = [], nodata: float | None = None, description: str = "", color_interp: ColorInterp = ColorInterp.gray, unit: str | None = None, metadata: dict[str, str] = {}, pixel_function: PixelFunction | None = None)`
  - `Warp(source: SourceRef, options_xml: str, block_size: tuple[int, int])`:
    - `options_xml` is `<GDALWarpOptions>` serialized without its `<SourceDataset>` child;
    - property `vertical_shift -> bool` is true when `options_xml` holds `<Option name="APPLY_VERTICAL_SHIFT">YES</Option>`.
  - `VRT(crs: CRS | None, transform: Affine, width: int, height: int, bands: list[Band] = [], metadata: dict[str, str] = {}, warp: Warp | None = None)`, plus `VRT.to_xml(*, relative_to: UPath | None = None) -> str`. Equality is dataclass equality.
- Produces (in `_xml.py`): `to_element(vrt: VRT, *, relative_to: UPath | None) -> ET.Element`, and `_require_inline_nesting() -> None`.

Serialization rules:
- **Dataset:** `<VRTDataset rasterXSize rasterYSize>`; for a warp, `subClass="VRTWarpedDataset"`. Children in this order:
  1. `<Metadata><MDI key=…>` (only when non-empty)
  2. `<SRS>` (WKT, only when `crs`)
  3. `<GeoTransform>` (`", ".join(repr(v) for v in transform.to_gdal())`)
  4. bands
  5. for a warp: `<BlockXSize>`, `<BlockYSize>`, then `options_xml`, with `<SourceDataset>` reinserted as the first child after its `<Option>`s.
- **Band:** `<VRTRasterBand dataType band>`. Add `subClass="VRTDerivedRasterBand"` with a pixel function, or `subClass="VRTWarpedRasterBand"` under a warp. Children:
  - `Metadata`, `NoDataValue` (`"nan"` for NaN), `Description` (only when non-empty), `UnitType`, `ColorInterp`;
  - for a pixel function: `PixelFunctionType`, `PixelFunctionArguments` (attributes) and `SourceTransferType`;
  - then the sources.
  - `ColorInterp` text is the GDAL name: `ColorInterp.gray` → `"Gray"`, `red` → `"Red"`, `undefined` → `"Undefined"`, and so on.
- **Source tag:** `ComplexSource` if `nodata is not None or lut is not None`, else `SimpleSource`. Add a `resampling="<name>"` attribute when set. Children in this order:
  1. `SourceFilename(relativeToVRT)` or an inline `<VRTDataset>`
  2. `SourceBand`
  3. `SourceProperties(RasterXSize RasterYSize DataType BlockXSize BlockYSize)`
  4. `SrcRect` / `DstRect` (`xOff yOff xSize ySize`, written with `repr` for floats and `str` for ints)
  5. `NODATA`
  6. `LUT` (`"v:o,v:o"`)
- **Nested sources:** a `Source` whose `path` is a `VRT` is written as an inline `<VRTDataset>` element (paths inside absolute). This calls `_require_inline_nesting()`, which raises `RuntimeError(f"Nested VRTs need GDAL >= 3.11; rasterio bundles GDAL {v}.")` when `rasterio.__gdal_version__` is lower.
- **Paths:** written through `_paths.gdal_path`; Task 4 implements it. Until then, write `_to_vsi_uri(path)` with `relativeToVRT="0"`.

- [ ] **Step 1: Write the failing tests** (`tests/test_vrt_model.py`, class `TestToXml`). Each builds a `VRT` by hand over `make_tiff` files and opens `rasterio.open(v.to_xml())`.

```python
def test_simple_source_reads_back_the_tile(self, tmp_path): ...
    # a 6x4 VRT over one tile: read(1) equals the tile; dtypes == ("float32",)
def test_complex_source_masks_its_nodata(self, tmp_path): ...
    # Source(nodata=0.0) over a uint16 tile holding 0 at [0, 0]: tag is ComplexSource and
    # read(1, masked=True).mask[0, 0] with band nodata 0
def test_lut_maps_values(self, tmp_path): ...
    # lut=[(0.0, 0.0), (10.0, 255.0)] over a float tile filled with 5.0 -> uint8 band reads 127 or 128
def test_derived_band_applies_db(self, tmp_path): ...
    # Band(pixel_function=PixelFunction.db(), transfer_type float32) over fill=100.0 -> 20.0
def test_metadata_description_unit_colorinterp(self, tmp_path): ...
    # ds.tags()["A"] == "1"; ds.tags(1)["B"] == "2"; descriptions == ("vv",); units == ("dB",);
    # colorinterp == (ColorInterp.red,)
def test_nested_vrt_source_is_inline(self, tmp_path): ...
    # inner = 2-tile mosaic VRT (built by hand); outer derived dB band with Source(inner);
    # XML has VRTRasterBand/SimpleSource/VRTDataset and no SourceFilename; values are 10*log10(tile values)
def test_nesting_requires_gdal_3_11(self, tmp_path, monkeypatch):
    monkeypatch.setattr(rasterio, "__gdal_version__", "3.10.3")
    with pytest.raises(RuntimeError, match="3.11"):
        outer.to_xml()
def test_geographic_crs_keeps_lon_lat_axis_order(self, tmp_path):
    # Review Focus 1: tile in EPSG:4326 at origin (7.0, 47.0), res 0.001
    with rasterio.open(v.to_xml()) as ds:
        assert ds.bounds.left == pytest.approx(7.0)
        assert ds.bounds.top == pytest.approx(47.0)
def test_float_windows_are_written_exactly(self):
    # dst_window=Window(0.5, 0, 7.5, 4) -> DstRect xOff="0.5" xSize="7.5"
```

- [ ] **Step 2: Run; expect failures** (`ImportError: cannot import name 'VRT'`).

Run: `uv run pytest tests/test_vrt_model.py -q`

- [ ] **Step 3: Implement** `_model.py` and `_xml.py` per the rules above. Export from `vrt/__init__.py`: `VRT`, `Band`, `Source`, `SourceProperties`, `PixelFunction` and `Warp`.

- [ ] **Step 4: Run; expect PASS.**

- [ ] **Step 5: Commit**

```bash
git add src/freezebase/vrt tests/test_vrt_model.py
git commit -m "feat(vrt): VRT model and XML serialization"
```

---

### Task 4: Path policy and `VRT.write`

**Files:**
- Create: `src/freezebase/vrt/_paths.py`, `src/freezebase/vrt/_io.py`
- Modify: `src/freezebase/vrt/_xml.py`, `src/freezebase/vrt/_model.py` (`write`)
- Test: `tests/test_vrt_model.py`

**Interfaces:**
- Produces:
  - `_paths.gdal_path(path: UPath, *, relative_to: UPath | None) -> tuple[str, bool]` returns `(text, relative)`.
    - **Relative** when `relative_to` is given and both paths are on the same store: both local, or both `s3` with the same bucket. The text is `posixpath.relpath(path.path, relative_to.parent.path)`, `../` included. Local paths are resolved first.
    - **Otherwise absolute:** `_to_vsi_uri(path)`; a local path is resolved to an absolute path first.
  - `VRT.write(path: AnyPath, *, relative: bool = True) -> UPath`. Returns the written `UPath`.
    - Calls `to_xml(relative_to=path if relative else None)` and indents with `ET.indent`.
    - Local writes are atomic: `.{name}.{token_hex(8)}.tmp` beside the destination, then `replace`.
    - S3 writes go through `raster._upload(xml.encode(), path)`.

- [ ] **Step 1: Write the failing tests** (class `TestWrite`):

```python
def test_source_beside_the_vrt_is_a_bare_name(self, tmp_path): ...  # text "a.tif", relativeToVRT "1"
def test_source_in_a_sibling_directory_is_dotdot_relative(self, tmp_path): ...
    # tile at tmp/tiles/a.tif, VRT at tmp/vrts/m.vrt -> "../tiles/a.tif"; opens after moving tmp to tmp2
def test_relative_false_writes_absolute(self, tmp_path): ...  # text == str(tile.resolve()), "0"
def test_s3_source_same_bucket_is_relative(self): ...
    # gdal_path(UPath("s3://b/t/a.tif"), relative_to=UPath("s3://b/v/m.vrt")) == ("../t/a.tif", True)
def test_s3_source_other_bucket_is_absolute(self): ...
    # gdal_path(UPath("s3://c/a.tif"), relative_to=UPath("s3://b/m.vrt")) == ("/vsis3/c/a.tif", False)
def test_local_source_under_s3_vrt_is_absolute(self, tmp_path): ...
def test_write_is_atomic_on_failure(self, tmp_path, monkeypatch): ...
    # make ET.tostring raise -> destination unchanged, no *.tmp left
def test_write_to_s3_uploads_once(self, s3_uploads): ...
    # import the s3_uploads fixture into a shared tests/conftest.py (move it from test_raster.py)
def test_special_characters_round_trip(self, tmp_path):
    # Review Focus 2: tile at tmp/"a b&<ü"/t.tif, VRT at tmp/v/m.vrt and a nested inline VRT
    # over it; both open and read the tile
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement** `_paths.gdal_path`, wire it into `_xml.to_element`, and implement `write` in `_io.py`. Move the `s3_uploads` fixture and its helpers (`_S3_PREFIXES`) from `tests/test_raster.py` to `tests/conftest.py`.
- [ ] **Step 4: Run; expect PASS.** Then the whole suite.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): relative source paths and VRT.write"`

---

### Task 5: `vrt.read`

**Files:**
- Modify: `src/freezebase/vrt/_paths.py` (`parse_gdal_path`), `src/freezebase/vrt/_xml.py` (`from_element`), `src/freezebase/vrt/_io.py` (`read`), `src/freezebase/vrt/__init__.py`
- Test: `tests/test_vrt_model.py`, plus fixtures `tests/data/vrt/legacy_mosaic_0_8.vrt`, `legacy_decibel_0_8.vrt` and `gdalbuildvrt.vrt`, all written by hand

**Interfaces:**
- Produces:
  - `read(path: AnyPath, *, storage_options: Mapping[str, Any] | None = None) -> VRT`, exported from `freezebase.vrt`.
  - `from_xml(text: str, *, base: UPath | None = None, storage_options: Mapping[str, Any] | None = None) -> VRT`, private, in `_xml.py`; `warp` uses it in Task 10.
  - `_paths.parse_gdal_path(text: str, *, relative: bool, base: UPath | None, storage_options: Mapping[str, Any] | None) -> UPath`:
    - `/vsis3/b/k` → `UPath("s3://b/k", **opts)`, where `opts` is `storage_options` if given, else `base.storage_options` when `base` is on S3, else `{}`;
    - `/vsicurl/<url>` → `UPath(url)`;
    - relative text → `base.parent / text`, normalized with `posixpath.normpath` and keeping `base`'s storage options;
    - an absolute local path → `UPath(text)`;
    - any other `/vsi*` prefix → `ValueError("Unsupported GDAL path ...")`.

Parser rules:
- Accept exactly the elements and attributes the writer emits, plus:
  - `SRS@dataAxisToSRSAxisMapping` (ignored; Task 3's axis-order test covers it);
  - `SourceFilename@shared`;
  - escaped VRT XML as the text of `SourceFilename` or `SourceDataset`, recognized by the leading `<VRTDataset` and parsed recursively into a `VRT`;
  - `GDALWarpOptions` (kept verbatim minus `SourceDataset`), plus `BlockXSize` / `BlockYSize`.
- Anything else raises `ValueError(f"Unsupported VRT element <{tag}>")`.
- A `ColorInterp` text maps back by name, case-insensitively.
- A missing `SrcRect` / `DstRect` gives `None`.

- [ ] **Step 1: Write the failing tests** (class `TestRead`):

```python
def test_round_trip_every_feature(self, tmp_path):
    # one VRT combining every feature from Task 3, plus a nested VRT
    assert read(v.write(tmp_path / "m.vrt")) == v
def test_round_trip_keeps_s3_storage_options(self, s3_uploads): ...
    # Review Focus 3: VRT on s3://b/v/m.vrt (anon=False, key="k"); sources s3://b/t/a.tif and s3://c/a.tif.
    # Parse the uploaded bytes with from_xml(base=UPath("s3://b/v/m.vrt", key="k")):
    # both sources are s3 UPaths with storage_options["key"] == "k"
def test_storage_options_override_inheritance(self, tmp_path): ...
    # local VRT with a /vsis3/ source; read(..., storage_options={"profile": "p"}) -> source.storage_options["profile"] == "p"
def test_reads_freezebase_0_8_mosaic(self, tmp_path): ...  # Review Focus 4: fixture copied beside two tiles; 2 ComplexSources, NODATA nan
def test_reads_freezebase_0_8_decibel(self, tmp_path): ...  # pixel_function == PixelFunction("dB", {"fact": "10"} or 10)
def test_reads_gdalbuildvrt_output(self, tmp_path): ...  # SourceProperties, shared attr, axis mapping
def test_unknown_element_raises(self):
    with pytest.raises(ValueError, match="<OverviewList>"):
        from_xml(XML_WITH_OVERVIEWLIST)
def test_escaped_xml_source_parses_as_nested(self): ...
def test_unsupported_vsi_prefix_raises(self): ...  # "/vsizip/a.zip/t.tif"
```


- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement.** Write the three fixture files by hand. Each copies the exact structure its producer emits; for the 0.8 fixtures, take the structure from `_legacy.py` before Task 12 changes it.
- [ ] **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): parse VRTs back into the model"`

---

### Task 6: Sources, environments and `VRT.open`

**Files:**
- Modify: `src/freezebase/vrt/_io.py`, `src/freezebase/vrt/_model.py`
- Test: `tests/test_vrt_model.py`

**Interfaces:**
- Produces:
  - `VRT.sources -> list[SourceRef]` (property):
    - the direct sources in document order (band 1's sources first, then band 2's, …), plus `warp.source`;
    - duplicates removed, comparing by `str(path)` for files and identity for VRTs;
    - no I/O.
  - `VRT.source_files(*, follow: bool = True, storage_options: Mapping[str, Any] | None = None) -> list[UPath]`:
    - the leaf files, recursing into nested VRTs and warp sources;
    - with `follow`, a `.vrt` file source is read with `read(path, storage_options=...)` and recursed into;
    - order is preserved and duplicates removed.
  - `VRT.env() -> AbstractContextManager[None]`:
    - every `source_files()` local: `rasterio.Env()`;
    - every S3 path with equal `storage_options`, plus any local paths: `s3_env(first_s3)`;
    - only `http(s)` and local: `http_env()`;
    - S3 with differing storage options, or S3 mixed with HTTP: `ValueError("Sources need different credentials: ...")`, naming two conflicting paths.
  - `VRT.env()`'s docstring states that callers using thread pools enter it in the main thread, because credentials set in a worker thread don't reach GDAL's reader threads.
  - `VRT.open(**kwargs) -> AbstractContextManager[DatasetReader]` enters `env()`, then `rasterio.open(self.to_xml(), **kwargs)`.

- [ ] **Step 1: Write the failing tests** (class `TestSourcesAndEnv`):

```python
def test_sources_are_ordered_and_unique(self, tmp_path): ...
def test_source_files_follow_vrt_files(self, tmp_path): ...
    # local outer.vrt -> warped.vrt (file) -> two tiles: source_files() == [tile_a, tile_b]
def test_source_files_without_follow_stop_at_vrt_files(self, tmp_path): ...
def test_env_local_is_plain(self, tmp_path): ...  # no AWS session: rasterio.env.getenv() has no AWS_ keys
def test_env_s3_uses_s3_env(self, monkeypatch): ...  # patch freezebase.s3.s3_env with a recorder; called once with the first S3 path
def test_env_rejects_mixed_credentials(self): ...  # s3://a (key k1) + s3://b (key k2) -> ValueError match "different credentials"
def test_open_reads_without_a_file(self, tmp_path): ...
```

- [ ] **Step 2: Run; expect failures.** — **Step 3: Implement.** — **Step 4: Run; expect PASS.**
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): source inspection, environments and VRT.open"`

---

### Task 7: Header reading, `from_raster` and `stack`

**Files:**
- Create: `src/freezebase/vrt/_grid.py`, `src/freezebase/vrt/_builders.py`
- Modify: `src/freezebase/vrt/__init__.py`
- Test: `tests/test_vrt_builders.py`

**Interfaces:**
- Produces:
  - `_grid.RasterInfo` (NamedTuple) with fields `ref: SourceRef, crs, transform, width, height, count, dtypes, nodatavals, descriptions, colorinterp, units, band_tags: list[dict[str, str]], tags: dict[str, str], block_shapes`. It provides `px` and `py` (positive pixel sizes) and `bounds`.
  - `_grid.raster_info(src: AnyPath | VRT) -> RasterInfo` opens a path via `rasterio_open` and a `VRT` via `vrt.open()`.
  - `_grid` also holds the moved helpers: `_nodata_equal`, `_assert_extent_aligned(px, py, *, left, bottom, right, top)` and `_snap_out(bounds, res_x, res_y)` (now per axis). Validation errors keep today's messages, naming the file.
  - `from_raster(path: AnyPath) -> VRT`: one `SimpleSource` per band (no windows), with `SourceProperties`. Band dtype, nodata, description, colour interpretation, unit and tags, plus dataset tags, CRS and transform, are all copied.
  - `stack(sources: Sequence[AnyPath | VRT], *, bands: Sequence[Sequence[int] | None] | None = None, band_names: Sequence[str] | None = None, color_interp: Sequence[ColorInterp] | None = None) -> VRT`:
    - all sources must share CRS, transform and size, otherwise `ValueError(... "use warp" ...)`;
    - `bands[i]` selects the bands of source *i* (`None` means all);
    - a `VRT` input is nested.

- [ ] **Step 1: Write the failing tests:**

```python
def test_from_raster_copies_everything(self, tmp_path): ...
    # 2-band uint16 tile, nodata 0, descriptions ("a","b"), unit "m", tags {"X": "1"}, band tag {"Y": "2"}:
    # from_raster(p).open() equals the file in data, profile fields, descriptions, units, tags
def test_stack_two_single_band_files(self, tmp_path): ...  # count == 2, band 2 == file 2
def test_stack_selects_bands(self, tmp_path): ...  # bands=[[2], None] on a 2-band + 1-band -> 2 bands
def test_stack_rejects_different_grids(self, tmp_path): ...  # match "use warp"
def test_stack_nests_vrt_inputs(self, tmp_path): ...
def test_band_names_length_mismatch_raises(self, tmp_path): ...
```

- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): from_raster and stack"`

---

### Task 8: `mosaic`: aligned tiles, any band count

**Files:**
- Modify: `src/freezebase/vrt/_builders.py`, `src/freezebase/vrt/__init__.py`
- Test: `tests/test_vrt_builders.py`

**Interfaces:**
- Produces: `mosaic(sources: Sequence[AnyPath | VRT], *, bounds: BoundingBox | tuple[float, float, float, float] | None = None, resolution: Literal["finest"] | tuple[float, float] | None = None, resampling: Resampling = Resampling.nearest, nodata: float | None = None, band_names: Sequence[str] | None = None, color_interp: Sequence[ColorInterp] | None = None) -> VRT`.
- This task implements `resolution=None`. Task 9 implements the other values.

Rules with `resolution=None`:
- Validation is today's (`_validate_mosaic_tiles` minus the single-band check): every tile has the reference's band count and dtypes.
- Band *i* reads band *i* of every source in order.
- Band nodata, description and colour interpretation come from the first source unless overridden.
- `nodata` sets the band NoDataValue and every source's `NODATA`.
- Sources are `ComplexSource` when the effective nodata is not `None`, otherwise `SimpleSource`.
- Integer `src_window` / `dst_window`, with `SourceProperties` filled in.

- [ ] **Step 1: Write the failing tests** (class `TestMosaicAligned`):

```python
def test_two_adjacent_single_band_tiles(self, tmp_path): ...  # width 12, values match each tile
def test_three_band_tiles_keep_every_band(self, tmp_path): ...  # count 3; descriptions from the first tile
def test_later_source_wins_where_valid(self, tmp_path): ...
    # two overlapping tiles, the second holds NaN in one pixel: the first tile's value shows there
def test_nodata_override_applies_to_sources(self, tmp_path): ...  # tiles nodata None, nodata=0.0 -> ComplexSource NODATA 0
def test_rejects_mismatched_band_count(self, tmp_path): ...  # match "band"
def test_rejects_mixed_nodata(self, tmp_path): ...  # Review Focus 5: None vs NaN -> match "NODATA"
def test_without_nodata_writes_simple_sources(self, tmp_path): ...
def test_vrt_inputs_are_nested(self, tmp_path): ...
def test_bounds_and_off_grid_errors_as_before(self, tmp_path): ...  # same messages as build_vrt_mosaic
```

- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): multi-band mosaic"`

---

### Task 9: `mosaic`: mixed resolutions and exact placement

**Files:**
- Modify: `src/freezebase/vrt/_builders.py`, `src/freezebase/vrt/_grid.py`
- Test: `tests/test_vrt_builders.py`

**Interfaces:**
- Consumes: `mosaic` from Task 8 and `RasterInfo` from Task 7.

Algorithm (spec, "Output grid anchor"):
1. **Target size `(rx, ry)`:** `"finest"` is `(min px, min py)` over all sources; a tuple is taken as is.
2. **Anchor and extent:**
   - with `bounds`, the anchor is `(left, top)`, and `_assert_extent_aligned(rx, ry, ...)` must pass;
   - otherwise, the anchor is the origin of the first source whose `px` and `py` equal the target within `_PIXEL_SIZE_TOL`, and the union extent grows outward onto the grid through that origin, with `_ALIGNMENT_TOL` slack so an exact edge does not add a pixel;
   - if no source has the target size, the anchor is the union's top-left corner, with width `ceil((right - left) / rx - _ALIGNMENT_TOL)` and the height computed the same way.
3. **Per source:** `col = (s.left - left) / rx`, `row = (top - s.top) / ry`, `w = s.width * s.px / rx`, `h = s.height * s.py / ry`.
   - On the grid (same size, and `col` / `row` integers within `_ALIGNMENT_TOL`): integer windows and no `resampling`.
   - Otherwise: float `dst_window` (unrounded), and `resampling` set on the `Source`.
4. CRS, dtype, band count and nodata must still match.

If the pixel-centre test fails with fractional `DstRect`s, place each off-grid tile instead as a nested `warp(tile, like=<the tile's own footprint on the target grid>)`. Record which approach was used in the commit message.

- [ ] **Step 1: Write the failing tests** (class `TestMosaicMixedResolution`):

```python
def test_coarse_tile_is_placed_exactly(self, tmp_path):
    # fine: 10 m, 6x4, origin (500000, 5200000); coarse: 30 m, 4x2, origin (500063, 5200000) (3 m off-grid)
    m = mosaic([fine, coarse], resolution="finest")
    # reference: rasterio.warp.reproject(coarse, nearest) onto m's grid
    # assert np.array_equal(m.open().read(1)[mask], ref[mask]) where mask = coarse-only pixels
def test_finest_is_taken_per_axis(self, tmp_path): ...  # tiles 10x20 and 20x10 m -> resolution (10, 10)
def test_anchor_is_the_first_tile_at_target_size(self, tmp_path): ...  # transform.c == fine origin x after outward growth
def test_anchor_falls_back_to_union_corner(self, tmp_path): ...  # resolution=(7.0, 7.0), no tile at 7 m
def test_explicit_bounds_anchor_the_grid(self, tmp_path): ...
def test_coverage_is_not_lost(self, tmp_path): ...  # m.bounds contains every tile's bounds
def test_on_grid_tiles_are_not_resampled(self, tmp_path): ...  # Source.resampling is None for the fine tile
```

- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): mixed-resolution mosaics with exact placement"`

---

### Task 10: `derive`

**Files:**
- Modify: `src/freezebase/vrt/_builders.py`, `src/freezebase/vrt/__init__.py`
- Test: `tests/test_vrt_builders.py`

**Interfaces:**
- Produces: `derive(sources: AnyPath | VRT | Sequence[AnyPath | VRT], function: PixelFunction, *, dtype: str = "float32", nodata: float | None = nan, description: str = "", band: int | None = None) -> VRT`.

Rules:
- **Input count:** a `str`, `PathLike`, `UPath` or `VRT` counts as one input.
  - One input: one derived band per input band, each reading that band.
  - Several inputs: one band reading `band or 1` from each, in order.
- **Grids:** every input must share CRS, transform and size, otherwise `ValueError`.
- **`VRT` inputs are always nested inline**, never flattened. This is the guard against the silent-zeros case.
- **Nodata:**
  - each source carries its input band's nodata as `ComplexSource` `NODATA`, or is a `SimpleSource` when that nodata is `None`;
  - `args` gains `propagateNoData="true"` unless the caller's function already sets it;
  - band nodata is `nodata`, and `transfer_type` defaults to `dtype`.
- **Measured on GDAL 3.12.4:** with this setup, `dB` of a `0` sentinel and `diff`/`div` of `-9999` give NaN, and `sum` needs `propagateNoData` to do the same.

- [ ] **Step 1: Write the failing tests** (class `TestDerive`):

```python
def test_db_over_an_in_memory_mosaic_is_not_zero(self, tmp_path):
    # Previously a dB band over a two-tile band returned 0 everywhere.
    m = mosaic([tile_100, tile_1000])
    a = derive(m, PixelFunction.db()).open().read(1)
    assert np.allclose(a, [[20.0] * 6 + [30.0] * 6] * 4)
def test_db_of_nodata_zero_is_nan(self, tmp_path): ...     # uint16 nodata 0 -> NaN, not -inf
def test_diff_of_integer_nodata_is_nan(self, tmp_path): ...  # int16 -9999 in either input -> NaN
def test_sum_propagates_nodata(self, tmp_path): ...
def test_one_input_derives_every_band(self, tmp_path): ...  # 2-band input -> 2 derived bands
def test_several_inputs_read_the_given_band(self, tmp_path): ...  # band=2
def test_rejects_different_grids(self, tmp_path): ...
def test_source_without_nodata_is_simple(self, tmp_path): ...  # Review Focus 5
def test_caller_propagate_nodata_is_kept(self, tmp_path): ...  # args {"propagateNoData": "false"} stays
```

- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): derive bands with GDAL pixel functions"`

---

### Task 11: `warp`: grids, tolerance and ballpark

**Files:**
- Create: `src/freezebase/vrt/_warp.py`
- Modify: `src/freezebase/vrt/__init__.py`
- Test: `tests/test_vrt_warp.py`

**Interfaces:**
- Consumes: `from_xml` (Task 5), `raster_info` and `_snap_out` (Task 7).
- Produces: `warp(source: AnyPath | VRT, *, crs: CRS | str | None = None, src_crs: CRS | str | None = None, resolution: float | tuple[float, float] | None = None, bounds: BoundingBox | tuple[float, float, float, float] | None = None, like: AnyPath | VRT | None = None, resampling: Resampling = Resampling.average, snap: bool = True, dtype: str | None = None, nodata: float | None = None, tolerance: float | Literal["auto"] = "auto", allow_ballpark: bool = False) -> VRT`.

Rules:
- **Target CRS:** `crs` is required unless `like` is given; `like` supplies CRS, transform and size exactly.
- **`resolution`:**
  - a float means `(r, r)`;
  - a tuple gives per-axis sizes;
  - `None` uses `rasterio.warp.calculate_default_transform(src_crs, crs, w, h, *src.bounds)`.
- **`bounds` / `snap`** behave as in today's `create_warped_vrt`, per axis.
- **Non-positive resolution** raises as today.
- **`dtype` / `nodata`** behave as today: `src_nodata` is passed when `nodata` is given. There is no implicit promotion; the docstring states it, citing the `uint16` 1.5 → 2 measurement.
- **Opening the source:** a path inside `_env_for_path`, a `VRT` inside `vrt.env()`. freezebase does not touch PROJ's search path.
- **Serialization:** call `WarpedVRT(src, src_crs=..., crs=..., transform=..., width=..., height=..., resampling=..., tolerance=<resolved>, **extras)`, where `extras` holds the dtype/nodata keywords, plus `ALLOW_BALLPARK="NO"` unless `allow_ballpark`. Copy it with `rasterio.shutil.copy(vrt, memfile.name, driver="VRT")` into `MemoryFile(ext=".vrt")`, then call `from_xml(..., base=None)`.
  - Warp a `VRT` source through `rasterio.open(source.to_xml())`. GDAL embeds it as escaped XML, and `from_xml` turns that into a nested `VRT`.
  - Warp a path source by its GDAL path. `from_xml` maps the absolute path back to the `UPath`, and the source's storage options are reapplied.
- **`tolerance="auto"`** resolves to `0.125`, or to `0.0` when a vertical shift applies (Task 12).
- **`allow_ballpark=False`** passes `ALLOW_BALLPARK="NO"`; `True` passes nothing.
- **Errors:** a GDAL "Cannot find coordinate operations" error is re-raised as `ValueError`. Its message says no exact transformation is available, and suggests `python -m pyproj sync` for grids or `allow_ballpark=True`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_reprojects_onto_the_requested_grid(self, tmp_path): ...  # port of TestWarpedVrt's test
def test_anisotropic_resolution(self, tmp_path): ...  # resolution=(10, 20) -> transform.a == 10, e == -20
def test_natural_grid_matches_calculate_default_transform(self, tmp_path): ...
def test_like_copies_the_grid_exactly(self, tmp_path): ...
def test_matches_rasterio_reproject(self, tmp_path): ...  # nearest; array_equal against reproject
def test_warp_of_an_in_memory_mosaic_round_trips(self, tmp_path):
    w = warp(mosaic([a, b]), crs="EPSG:4326", resolution=1e-4)
    assert isinstance(w.warp.source, VRT)
    assert read(w.write(tmp_path / "w.vrt")) == w
def test_written_warp_keeps_its_options(self, tmp_path): ...
    # ALLOW_BALLPARK NO in the written XML; tolerance auto -> MaxError 0.125 present
def test_tolerance_number_overrides_auto(self, tmp_path): ...  # tolerance=0 -> no ApproxTransformer
def test_integer_average_is_not_promoted(self, tmp_path): ...  # uint16 [[1,2],[1,2]] -> dtype uint16, value 2
def test_ballpark_is_refused(self, tmp_path, monkeypatch): ...
    # inside rasterio.Env(PROJ_DATA=<rasterio's own proj_data only>, PROJ_NETWORK="OFF"), so no grid is reachable;
    # warp(dem in EPSG:2056+5728, crs="EPSG:4979") raises ValueError match "pyproj sync"
def test_allow_ballpark_opts_out(self, tmp_path, monkeypatch): ...  # same setup, allow_ballpark=True -> no error
```

`test_ballpark_is_refused` passes `src_crs` with a vertical part and depends on Task 12's inference. Write it in this task, mark it `xfail(strict=True, reason="vertical inference lands in Task 12")`, and remove the mark in Task 12.

- [ ] **Steps 2–4:** fail → implement → pass.
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): warp onto explicit, natural or reference grids"`

---

### Task 12: `warp`: vertical shifts

**Files:**
- Modify: `src/freezebase/vrt/_warp.py`, `.github/workflows/ci.yml` (sync the grids)
- Test: `tests/test_vrt_warp.py`

**Interfaces:**
- Produces: `_warp._has_vertical(crs: CRS) -> bool`, true for:
  - a pyproj compound CRS with a vertical sub-CRS;
  - a vertical CRS;
  - a geographic CRS with three axes.

Rules:
- **When both `src_crs` (or the source's CRS) and `crs` have a vertical part,** pass `APPLY_VERTICAL_SHIFT="YES"`, and `"auto"` tolerance resolves to `0.0`.
- **When the source has one and the target does not,** raise `ValueError("... vertical datum ... pass a 3D or compound crs, or a 2D src_crs to keep heights unconverted")`.
- **CI:** in the test job and the lowest-direct job, before `pytest`, add a step `uv run python -m pyproj sync --source-id ch_swisstopo` (about 2 MB).

Test setup in `tests/conftest.py`, not the library:
- `GRIDS = Path(pyproj.datadir.get_user_data_dir())`;
- a session fixture `proj_grids`:
  - skips when `GRIDS / "ch_swisstopo_chgeo2004_ETRS89_LN02.tif"` or `GRIDS / "ch_swisstopo_chgeo2004_ETRS89_LHN95.tif"` is missing;
  - otherwise builds a directory in `tmp_path_factory` holding symlinks to every file of `rasterio.env.PROJDataFinder().search()` and of `GRIDS`, and yields it;
- tests that need grids enter `rasterio.Env(PROJ_DATA=str(proj_grids))` around both building and reading the warp. Measured: this works even after PROJ has been used in the process. The DEM fixture is zeros in `EPSG:2056+5728` with origin `(2620000, 1160000)`, 100 m pixels, 300 × 300.

- [ ] **Step 1: Write the failing tests:**

```python
def _pyproj_shift(src, dst, xs, ys): ...  # Transformer(only_best=True, allow_ballpark=False), z=0
def test_compound_to_3d_gives_ellipsoidal_heights(self, proj_grids, dem):
    # crs="EPSG:4979": values within 0.02 m of _pyproj_shift("EPSG:2056+5728", "EPSG:4979", ...) at the
    # source pixel centres (evaluate at the source pixel each output pixel reads, resampling=nearest);
    # range roughly 51-54 m
def test_compound_to_compound(self, proj_grids, dem): ...  # crs="EPSG:2056+5729", within 0.02 m of pyproj (0.15-0.54 m)
def test_3d_to_compound(self, proj_grids, tmp_path): ...  # EPSG:4979 zeros -> "EPSG:2056+5728": negative undulation, within 0.02 m
def test_vertical_source_into_2d_target_raises(self, dem):
    with pytest.raises(ValueError, match="vertical"):
        warp(dem, crs="EPSG:4326")
def test_2d_src_crs_keeps_heights(self, dem): ...  # src_crs="EPSG:2056", crs="EPSG:4326" -> all 0.0
def test_auto_tolerance_is_exact_with_a_shift(self, dem): ...  # no ApproxTransformer in to_xml()
def test_written_shift_reopens_with_the_shift(self, proj_grids, dem, tmp_path): ...  # read(write()).open() equals warp.open()
```

Remove the `xfail` on `test_ballpark_is_refused` from Task 11.

- [ ] **Steps 2–4:** fail → implement → pass, with the grids synced locally (`uv run python -m pyproj sync --source-id ch_swisstopo`).
- [ ] **Step 5: Commit** — `git commit -m "feat(vrt): vertical datum shifts inferred from the CRSs"`

---

### Task 13: Legacy functions as wrappers

**Files:**
- Modify: `src/freezebase/vrt/_legacy.py`, `tests/test_vrt.py`

**Interfaces:**
- Consumes: `mosaic`, `warp`, `derive`, `stack`, `Band` / `Source` (`lut`), `VRT.write`.
- Produces: unchanged signatures, except that `create_warped_vrt` gains `resolution: float | tuple[float, float] | None`.

| Function | Implementation |
|---|---|
| `build_vrt_mosaic(files, dst, *, bounds)` | `mosaic(files, bounds=bounds).write(dst)` |
| `create_warped_vrt(src, dst, *, crs, resolution, bounds, resampling, snap, dtype, nodata)` | `warp(...).write(dst, relative=False)` |
| `create_decibel_vrt(src, dst, *, from_intensity)` | `derive(src, PixelFunction.db(10 if from_intensity else 20), description="dB").write(dst)` |
| `create_rgb_vrt` | `stack([vv, vh], band_names=..., color_interp=(red, green))` plus a third band, appended from `derive([vv, vh], PixelFunction.diff() if decibel_scale else PixelFunction.div())`; same validation messages |
| `create_rgba_vrt` | `Band`s with `lut` sources, built directly; same alpha behaviour |

Delete the now-unused private helpers from `_legacy.py`; they moved to `_grid.py` in Task 7.

- [ ] **Step 1: Update `tests/test_vrt.py`.** Change only the tests the spec changes:
  - `test_rejects_multiband_reference_tile` and `test_rejects_multiband_non_reference_tile` become `test_accepts_multiband_tiles`: `count == 2` and both bands mosaicked.
  - `test_tile_outside_vrt_directory_is_referenced_absolutely` becomes `test_tile_outside_vrt_directory_is_referenced_relatively`, expecting `[("a.tif", "1"), ("other/b.tif", "1")]`.
  - In `test_relative_source_outside_vrt_directory_resolves_absolute`, assert the VRT reopens from another cwd; it no longer expects an absolute path.
  - Add `test_decibel_of_nonnan_nodata_is_nodata`: a uint16 source with nodata 0 gives NaN where the source is 0.
  - Point `_nodata_equal` imports at `freezebase.vrt._grid`.
  - Every other test is unchanged and must pass.
- [ ] **Step 2: Run; expect the changed tests to fail.**
- [ ] **Step 3: Rewrite the wrappers.**
- [ ] **Step 4: Run `tests/test_vrt.py` and `tests/test_raster.py`; expect PASS.**
- [ ] **Step 5: CHANGELOG and commit.**
  - **Changed:** `build_vrt_mosaic` accepts multi-band tiles; same-store sources are written relative to the VRT even outside its directory; `create_warped_vrt` takes `(x, y)` or `None` resolution.
  - **Changed:** `create_decibel_vrt` writes nodata, not `-inf`, where a source with non-NaN NODATA holds that value.

  ```bash
  git commit -am "refactor(vrt): rebuild the existing VRT functions on the model"
  ```

---

### Task 14: Pull request 1 wrap-up: VRT integration tests, docs, release notes

**Files:**
- Modify: `tests/test_s3_integration.py`, `great-docs.yml`, `CHANGELOG.md`
- Create: `user_guide/04-vrts.qmd`

- [ ] **Step 1: Write the integration tests** (class `TestVrtOnS3`, `pytest.mark.integration`):

```python
def test_vrt_with_dotdot_sources_reopens(self, bucket_root, tmp_path): ...
    # tiles uploaded to <prefix>/tiles/, mosaic written to <prefix>/vrts/m.vrt (relative "../tiles/...");
    # read(...).open() reads both tiles
def test_env_and_open_on_s3_sources(self, bucket_root, tmp_path): ...
def test_source_files_follow_a_local_vrt_to_s3(self, bucket_root, tmp_path): ...
    # local warped.vrt over an S3 mosaic VRT: source_files(storage_options=bucket_root.storage_options) are the S3 tiles
```

- [ ] **Step 2: Run against RustFS** (AGENTS.md command); expect PASS.
- [ ] **Step 3: Write the docs.**
  - **`great-docs.yml`:** rename the "VRT generation" section's `desc` to "Inspectable VRTs: build, edit, read and write without the GDAL Python bindings". List `vrt.VRT`, `vrt.Band`, `vrt.Source`, `vrt.PixelFunction`, `vrt.mosaic`, `vrt.warp`, `vrt.stack`, `vrt.derive`, `vrt.from_raster`, `vrt.read`, followed by the legacy functions. Add `raster.http_env` and `raster.GDAL_HTTP_OPTIONS` under Raster I/O.
  - **`user_guide/04-vrts.qmd`** has these sections:
    - the model;
    - the four verbs;
    - chaining in memory vs on disk;
    - path policy;
    - credentials and threads (enter `env()` in the main thread);
    - vertical shifts: inferred from the CRSs, and ballpark refused. Making grids reachable is the caller's setup: `python -m pyproj sync --source-id <id>`, then either `PROJ_NETWORK=ON` set before the process starts, or `rasterio.Env(PROJ_DATA=<dir>)` with a directory holding rasterio's PROJ data and the grids.

    Code cells are non-executing. Never name the private downstream repository.
- [ ] **Step 4: Build the docs.** Run `uv sync --all-extras --group docs && uv run great-docs build`; expect no errors.
- [ ] **Step 5: Lower-bound check**

```bash
uv sync --all-extras --resolution lowest-direct --python 3.12
uv run --no-sync pytest -q
git checkout uv.lock && uv sync --all-extras
git diff --stat uv.lock   # must be empty
```

- [ ] **Step 6: CHANGELOG pass.**
  - **Added:** the model and builders, `read`, `env` / `open` / `source_files`, and vertical shifts in `warp`.
  - Check that every Breaking entry has a before/after.
  - `grep -i glacier CHANGELOG.md user_guide/04-vrts.qmd` must print nothing.
- [ ] **Step 7: Coverage**

  ```bash
  uv run pytest -q --cov=freezebase --cov-branch --cov-report=term-missing
  ```

  New modules must be ≥ 90% covered.
- [ ] **Step 8: Commit** — `git commit -am "docs(vrt): user guide, API reference and changelog"`
- [ ] **Step 9: Stop.** Do not push or open the pull request. The controller asks the user first.

---

### Task 15: `write_cog` streams a `VRT` or raster path

**Files:**
- Modify: `src/freezebase/raster.py` (`write_cog` and its overloads)
- Test: `tests/test_raster.py`

**Interfaces:**
- Produces: `write_cog(data: np.ndarray | VRT | AnyPath, dst_file: AnyPath, profile: dict[str, Any] | None = None, *, ...)`. The keywords and `checksum` overloads are unchanged; `profile` is required only for an ndarray (`ValueError` otherwise).

Rules for `VRT` and path inputs (a path goes through `vrt.from_raster`):
- **Metadata:** work on a `copy.deepcopy` of the VRT. `band_names`, `color_interp`, `units`, `tags` and `band_tags` are applied to the model before copying, with the existing per-band length check.
- **`profile["nodata"]` remaps:**
  - every source of every band gets `nodata` set to the band's current nodata, which makes it a `ComplexSource`;
  - then `band.nodata` becomes the new value.
  - Skip when the two are `_nodata_equal`.
- **`profile["dtype"]` sets `band.dtype`.** When the target is an integer dtype and the effective nodata is NaN or doesn't fit `np.iinfo(dtype)`, raise `ValueError("casting to {dtype} needs a nodata value that fits ...")`.
- **Creation options:** taken from `profile` the way the ndarray path takes them (`COG_PROFILE | {k: profile[k] for k in _COG_CREATION_OPTIONS}`, with the LERC predictor rule).
- **Copy:** `rasterio.shutil.copy(vrt.to_xml(), dst, driver="COG", **options)` inside `vrt.env()`.
  - Local: a temp name beside the destination, then `replace`.
  - S3: encode to a temporary directory, then `_upload`.
  - `checksum=True`: hash the encoded file with `file_sha256`, never the whole buffer in memory.

- [ ] **Step 1: Write the failing tests** (class `TestWriteCogFromVrt`):

```python
def test_streams_a_mosaic(self, tmp_path): ...  # result is a valid COG (rio_cogeo not needed: check ds.profile["driver"]=="GTiff" and
                                               # ds.tags(ns="IMAGE_STRUCTURE")["LAYOUT"] == "COG"), data equals the mosaic
def test_accepts_a_raster_path(self, tmp_path): ...
def test_keeps_dtype_and_nodata_by_default(self, tmp_path): ...  # regression guard from the spec
def test_metadata_lands_in_the_header(self, tmp_path): ...  # band_names, tags; LAYOUT still COG
def test_nodata_change_remaps_pixels(self, tmp_path):
    # source uint16 nodata 0 holding [0, 5]; profile={"nodata": 65535} -> pixels [65535, 5], ds.nodata == 65535
def test_cast_with_nan_nodata_needs_an_integer_nodata(self, tmp_path):
    with pytest.raises(ValueError, match="nodata"):
        write_cog(float_vrt_with_nan_nodata, tmp_path / "o.tif", {"dtype": "uint8"})
def test_checksum_matches_the_file(self, tmp_path): ...
def test_s3_destination_uploads_once(self, tmp_path, s3_uploads): ...
def test_ndarray_still_requires_a_profile(self, tmp_path): ...
```

- [ ] **Steps 2–4:** fail → implement → pass. The whole `tests/test_raster.py` must pass unchanged.
- [ ] **Step 5: CHANGELOG** (**Added:** `write_cog` accepts a `VRT` or a raster path and streams it; a `nodata` change remaps pixels; a cast refuses an unrepresentable nodata), then:

```bash
git commit -am "feat(raster): write_cog streams VRTs and raster files"
```

---

### Task 16: `rewrite_tiff` on the model; remove `merge_tiffs`

**Files:**
- Modify: `src/freezebase/raster.py`, `tests/test_raster.py`, `tests/test_s3_integration.py`, `great-docs.yml`
- Test: `tests/test_raster.py`

**Interfaces:**
- Produces: `rewrite_tiff(src_file: AnyPath | VRT, dst_file, profile=None, band_names=None, color_interp=None, *, tags=None, band_tags=None, units=None, move=False) -> None`.
  - **Behaviour:** unchanged for paths, including in-place, `move`, staging, S3 single-put upload, and ignoring `dtype` / `nodata` in `profile`.
  - **Implementation:**
    - `from_raster(src)` for a path;
    - `tags` and `band_tags` are **merged into** the existing tags, as today;
    - `band_names`, `units` and `color_interp` replace them;
    - one staged `rasterio.shutil.copy` from `vrt.to_xml()` to the driver in `profile`.
    - `_rewrite_via_memory`, `_rewrite_via_tempfile` and `_inject_band_metadata` are deleted if nothing else uses them. `merge_tiffs` was the other user of `_inject_band_metadata`.
  - **A `VRT` source** with `move=True` raises `ValueError("move needs a source file")`.
  - **Removed:** `merge_tiffs`, the `from rasterio.merge import merge` import, its tests in `tests/test_raster.py`, `raster.merge_tiffs` in `great-docs.yml`, and `test_merge_to_s3` / `test_merge_a_local_and_an_s3_source` in `tests/test_s3_integration.py`.
  - **Replacement tests in the same file:** `test_write_cog_from_a_mosaic_to_s3` and `test_mosaic_of_a_local_and_an_s3_source`, using `write_cog(mosaic([...]), s3_key)` with the same assertions.

- [ ] **Step 1: Write the failing tests:**

```python
def test_rewrite_a_vrt_to_gtiff(self, tmp_path): ...  # rewrite_tiff(mosaic([a, b]), dst, {"driver": "GTiff"}) -> width 12
def test_rewrite_vrt_with_move_raises(self, tmp_path): ...
def test_rewrite_merges_tags(self, tmp_path): ...  # source tag A kept, new tag B added
def test_merge_tiffs_is_gone(self):
    assert not hasattr(raster, "merge_tiffs")
```

- [ ] **Step 2: Run; expect failures.**
- [ ] **Step 3: Implement and remove.** Every existing `rewrite_tiff` test passes unchanged.
- [ ] **Step 4: Run the whole suite; expect PASS.**
- [ ] **Step 5: CHANGELOG** under **Removed**:

  ```
  **Breaking:** `merge_tiffs`. Before: `merge_tiffs([a, b], "out.tif")`.
  After: `write_cog(vrt.mosaic([a, b]), "out.tif")` for a COG, or
  `rewrite_tiff(vrt.mosaic([a, b]), "out.tif", profile)` for a GTiff;
  `method="first"` is `vrt.mosaic(reversed(files))`; `min`/`max`/`sum`/`count`
  have no replacement.
  ```

  Under **Added:** "`rewrite_tiff` accepts a `VRT`." Then:

  ```bash
  git commit -am "feat(raster)!: rewrite_tiff takes a VRT; remove merge_tiffs"
  ```

---

### Task 17: Pull request 2 wrap-up: raster integration tests, docs, release notes

**Files:**
- Modify: `tests/test_s3_integration.py`, `great-docs.yml`, `CHANGELOG.md`, `user_guide/04-vrts.qmd`, `user_guide/02-rasters-on-s3.qmd` (if it mentions `merge_tiffs`)

- [ ] **Step 1: Integration test.** Add `test_write_cog_from_an_s3_mosaic(self, bucket_root, tmp_path)` to `TestVrtOnS3`: `write_cog(mosaic(<two S3 tiles>), <S3 key>)` reads back with width 12.
- [ ] **Step 2: Run against RustFS;** expect PASS, including Task 16's replacement merge tests.
- [ ] **Step 3: Docs.**
  - Remove `raster.merge_tiffs` from `great-docs.yml` if Task 16 hasn't already.
  - Add a "Writing a VRT out" section to `user_guide/04-vrts.qmd`: `write_cog(vrt)`, `rewrite_tiff(vrt)`, nodata remap, and the cast refusal.
  - `grep -rn merge_tiffs user_guide README.md` must print nothing.
- [ ] **Step 4: Build the docs;** expect no errors.
- [ ] **Step 5: Lower-bound check** (same commands as Task 14 Step 5, with `git checkout uv.lock` after).
- [ ] **Step 6: CHANGELOG pass.** Every Breaking entry has a before/after. `grep -i glacier CHANGELOG.md` prints nothing.
- [ ] **Step 7: Coverage** ≥ 90% on the changed lines of `raster.py`.
- [ ] **Step 8: Commit** — `git commit -am "docs(raster): writing VRTs out"`
- [ ] **Step 9: Stop.** Do not push or open the pull request.
