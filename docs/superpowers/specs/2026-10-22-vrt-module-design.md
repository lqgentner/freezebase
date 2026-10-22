# `freezebase.vrt`: a general VRT model

Status: design, approved for planning on 2026-10-22.

## Why

`freezebase.vrt` today is five single-purpose functions (`build_vrt_mosaic`,
`create_warped_vrt`, `create_decibel_vrt`, `create_rgb_vrt`, `create_rgba_vrt`), each writing
VRT XML by hand for one fixed layout. Downstream code keeps hitting the edges and writing its
own XML:

- a multi-band mosaic writer (about 120 lines) duplicating `build_vrt_mosaic`, which rejects
  `count > 1`;
- warps onto anisotropic grids, warps onto the grid GDAL would choose, and mosaics of tiles
  with different native resolutions (products whose longitude spacing coarsens with
  latitude);
- `glace-catalog`'s `tagged_vrt`, which re-parses a VRT to add metadata and a band
  description, and a `relativize_vrt` that rewrites source paths relative to the VRT;
- `glace-catalog`'s `sources_env` / `chain_env`, which carry a `sources_at` path beside a VRT
  chain because a local VRT does not say that its sources need S3 credentials;
- public HTTPS COGs, which every builder rejects because `_to_vsi_uri` and `_env_for_path`
  accept only local and `s3` paths;
- streaming a large VRT into a COG, which `write_cog(ndarray)` cannot do.

The goal is one inspectable Python model of a VRT, built by a few domain verbs, written and
read back losslessly, and used underneath `freezebase.raster`'s rewrite and COG
functions. The osgeo bindings stay out. rasterio is a core dependency: reuse it wherever it
already does the job (`WarpedVRT`, `rasterio.open` of VRT XML, `rasterio.shutil.copy`), and
implement only what it lacks.

## Goals

1. A `VRT` value object: grid, bands, sources, pixel functions, metadata, optional warp.
2. Builders returning a `VRT`: `mosaic`, `warp`, `stack`, `derive`, `from_raster`.
3. `read` / `write` / `to_xml` with lossless round trips for everything freezebase writes, and
   for the VRTs rasterio's `WarpedVRT` and GDAL's `gdalbuildvrt` write.
4. In-memory composition: a chain (mosaic, then warp, then dB, then COG) runs without writing
   intermediate files.
5. Source inspection and credentials: list a VRT's underlying files, and enter the matching
   rasterio environment for local, S3 and public HTTPS sources.
6. `write_cog` and `rewrite_tiff` rebuilt on the model, and `merge_tiffs` removed; `write_cog` streams a VRT
   or raster file into a COG.
7. The existing five `vrt` functions kept as thin wrappers.

## Non-goals

- Python pixel functions (`PixelFunctionLanguage="Python"`). Deliberately excluded: that work
  belongs in Python, and a VRT carrying code is a risk once VRTs come back from a store.
- Mirroring all of GDAL's VRT schema. Out: overviews, mask bands, `KernelFilteredSource`,
  multidimensional VRTs, processed datasets, `vrt://` connection strings, mixing several
  credentials in one VRT (GDAL holds one set of S3 credentials per environment).
- Mixed-CRS mosaics in one call. The caller warps each CRS group first (`group_tiffs_by_crs`
  stays as it is).
- Vertical shifts other than through `warp` (no grid arithmetic of our own), PROJ grid syncing,
  and other raster work not about VRTs.

## Anti-goals: what would make this a failure even if it works

- **Silent wrong pixels.** Measured on GDAL 3.12.4: a `dB` derived band over a two-tile mosaic
  band returns `0` everywhere, with no error. A coarse tile placed by rounding to the fine grid
  is misregistered by up to half a pixel. Every such case must either be handled correctly or
  raise.
- **A changed product without a trace.** dtype and NODATA of an output never change unless
  the caller asked for it.
- **Lossy editing.** `read` then `write` must not silently drop elements it does not model.
- **Re-implementing rasterio.** No hand-written warp: `warp` wraps `WarpedVRT`.

## Facts this design rests on

Verified on 2026-10-22 against rasterio 1.5.1 / GDAL 3.12.4 unless noted:

- `rasterio.open(xml_string)` opens VRT XML directly; nothing on disk needed.
- GDAL ≥ 3.11 documents an inline `<VRTDataset>` element in place of `<SourceFilename>`. It
  works, including inside a derived band. rasterio 1.5.0 (the lower bound) bundles GDAL 3.12.1.
- `WarpedVRT` over an in-memory VRT works. Written out via `rasterio.shutil.copy(...,
  driver="VRT")`, GDAL embeds the source as escaped XML in `<SourceDataset>`, and the file
  reopens. So nesting happens even without designing for it.
- A warped VRT's root is `<VRTDataset subClass="VRTWarpedDataset">` with the source in
  `<GDALWarpOptions>`; its bands have no sources.
- In a `VRTDerivedRasterBand`, each source is a separate input to the pixel function; sources
  are not mosaicked first.
- `min`, `max` and `mean` pixel functions over partly overlapping tiles with NaN nodata give
  correct mosaics with `propagateNoData="false"`. `sum` returns `0`, not nodata, where every
  input is nodata. `count` was not tested.
- Fractional `SrcRect` / `DstRect` are accepted.
- Relative `../` source paths resolve on local disk. Not yet verified inside `/vsis3/`.
- `WarpedVRT(..., APPLY_VERTICAL_SHIFT="YES", ALLOW_BALLPARK="NO", tolerance=0)` with a
  compound or 3D source CRS applies the geoid shift, and raises when the grid is missing. See
  `warp` for the measurements.
- GDAL 3.12 does not open sources on `rasterio.open` of a VRT, with or without
  `<SourceProperties>`.
- An in-memory dB VRT over an S3 tile from a profile-configured, custom-endpoint bucket reads
  inside `s3_env`.

## The model

Mutable dataclasses, so editing is attribute assignment (`v.bands[0].description = "x"`).
Paths are always absolute `UPath`s in memory; relative paths exist only in written XML.

```python
type SourceRef = UPath | VRT

@dataclass
class PixelFunction:
    name: str                              # a GDAL built-in: "dB", "div", "diff", "scale", "expression", ...
    args: dict[str, str | int | float] = ...  # -> <PixelFunctionArguments .../>
    transfer_type: str | None = None       # numpy dtype name -> <SourceTransferType>

@dataclass
class Source:
    path: SourceRef                        # a file, or a nested VRT serialized inline
    band: int = 1
    src_window: Window | None = None       # rasterio Window; float offsets/sizes allowed
    dst_window: Window | None = None
    nodata: float | None = None            # makes it a ComplexSource
    lut: list[tuple[float, float]] | None = None   # ComplexSource <LUT>
    resampling: Resampling | None = None
    properties: SourceProperties | None = None     # width, height, dtype, block shape

@dataclass
class Band:
    dtype: str                             # numpy name; mapped to GDAL names on write
    sources: list[Source]
    nodata: float | None = None
    description: str = ""
    color_interp: ColorInterp = ColorInterp.gray
    unit: str | None = None
    metadata: dict[str, str] = ...
    pixel_function: PixelFunction | None = None   # makes it a VRTDerivedRasterBand

@dataclass
class Warp:
    source: SourceRef
    options_xml: str                       # <GDALWarpOptions> kept verbatim for round trips

@dataclass
class VRT:
    crs: CRS
    transform: Affine
    width: int
    height: int
    bands: list[Band]
    metadata: dict[str, str] = ...
    warp: Warp | None = None               # set => VRTWarpedDataset; bands then carry no sources

    sources: list[SourceRef]               # property; direct sources in document order, no I/O
    def source_files(self, *, follow: bool = True) -> list[UPath]: ...
    def to_xml(self, *, relative_to: UPath | None = None) -> str: ...
    def write(self, path: AnyPath, *, relative: bool = True) -> UPath: ...
    def open(self, **kwargs) -> ContextManager[DatasetReader]: ...
    def env(self) -> ContextManager[None]: ...
```

Field names are indicative; the builder may adjust them. The shape is not.

- **Simple vs complex source.** Written as `SimpleSource` unless `nodata`, `lut` or another
  `ComplexSource` option is set.
- **`SourceProperties`.** Filled by the builders, which read the headers anyway, and written,
  as `gdalbuildvrt` does.
- **`sources` and `source_files`.** `sources` lists direct sources in document order (order is
  meaning: in a mosaic a later source wins). Duplicates are removed and it does no I/O.
  `source_files(follow=True)` returns the leaf files: it recurses into nested VRTs and the
  warp source, and reads referenced `.vrt` files to follow them too. A `glace-catalog` chain's
  final mosaic points at local warped VRTs whose sources are S3 tiles two levels down.

## Builders

Every builder takes `AnyPath | VRT` inputs. A path is referenced as a file; a `VRT` is nested
inline. All return a `VRT`; nothing is written.

### `mosaic(sources, *, bounds=None, resolution=None, resampling=Resampling.nearest, nodata=None, band_names=None, color_interp=None)`

- Any band count. All sources share CRS, band count and dtype. Band *i* of the mosaic reads
  band *i* of each source. Band descriptions, colour interpretation and NODATA default to the
  first source's.
- Source order is draw order: later sources win where they have data.
- `bounds`: fixed extent, which must lie on the grid; defaults to the union.
- `resolution`:
  - `None` requires one common pixel size, and every origin on one grid. This is today's
    behaviour and validation, unchanged.
  - `"finest"` takes the smallest pixel size per axis.
  - `(x, y)` sets the size explicitly.
- With a non-`None` `resolution`, any source not exactly on the output grid, whether coarser,
  finer or off-grid at the same size, is resampled with `resampling`. Placement must be
  georeferenced exactly. Neither rounding to the grid (the half-pixel misregistration) nor
  stretching a tile to whole pixels is acceptable.
- **Output grid anchor** (*my call*):
  - with `bounds`, its top-left corner, and the extent must be whole pixels;
  - otherwise the origin of the first source whose pixel size equals the target, with the
    union extent grown outward onto that grid so no coverage is lost;
  - if no source has the target size, the union's top-left corner, as `gdalbuildvrt` does.

  This keeps natively fine tiles unresampled and makes the grid deterministic for a given
  source list.
- `nodata` overrides the band NODATA and is applied to every source, so overlaps blend on
  valid pixels.
- Validation as today: shared CRS, no rotation or shear, origins on the grid within tolerance.
  Errors name the offending file.

### `warp(source, *, crs=None, src_crs=None, resolution=None, bounds=None, like=None, resampling=Resampling.average, snap=True, dtype=None, nodata=None, tolerance="auto", allow_ballpark=False)`

- Wraps `rasterio.vrt.WarpedVRT`, serializes it, and parses the result into a `VRT` with
  `warp` set.
- `resolution`: a float, an `(x, y)` tuple for anisotropic grids, or `None` for the grid
  `rasterio.warp.calculate_default_transform` derives.
- `like`: a path or `VRT` whose CRS, transform and size are copied exactly. Use it to warp onto
  a reference raster's grid.
- `bounds`, `snap`, `dtype` and `nodata` behave as in today's `create_warped_vrt`.
- **No implicit promotion** (*my call*). Measured: an averaging warp of a `uint16` band stays
  `uint16` and rounds the mean (1.5 becomes 2). The caller passes `dtype="float32"` to keep the
  fraction, with `nodata` when the old sentinel should not stay as a valid-looking `0.0`. The
  docstring says so.
- `src_crs` overrides the source's CRS, for files whose header lacks the vertical datum (or is
  wrong).
- **Vertical shifts are inferred from the CRSs** (decided). When both `src_crs` (or the source's
  CRS) and `crs` carry a vertical reference, `warp` sets `APPLY_VERTICAL_SHIFT=YES`. A vertical
  reference means a compound CRS with a vertical part, or a 3D geographic CRS (ellipsoidal
  heights). PROJ then picks the operation, including any Helmert shift between ellipsoids. There
  is no flag. rasterio never does this by itself; the `gdalwarp` CLI does. Measured on a DEM of
  zeros:
  - `2056+5728 → 4979` gives 51.31–53.56 m;
  - `2056+5728 → 4326+3855` gives −0.24 to 0.57 m;
  - `4979 → 2056+5728` gives −53.6 to −51.2 m;
  - with exact transforms, agreement with pyproj is within 14 mm.
- **A vertical source into a 2D target raises** (decided). The heights would stay as they were
  while the output CRS no longer says which datum they refer to. The caller passes a 3D or
  compound `crs`, or a 2D `src_crs` to state that heights are not to be converted.
- **`tolerance="auto"`** (decided) is GDAL's ⅛-pixel approximation for horizontal-only warps,
  and exact (`0`) whenever a vertical shift applies. The approximation's error is capped in
  pixels, not metres: it cost 0.42 m vertically in the measurement above. A number overrides
  `"auto"`. Measured cost of exact transforms: 36 Mpx UTM → WGS84 took 8.7 s instead of 1.9 s,
  and a warped VRT pays it on every read.
- **No ballpark operations by default** (decided). `ALLOW_BALLPARK=NO` is passed as a warp
  option, so PROJ raises "Cannot find coordinate operations" when it lacks a grid or a datum
  link. Otherwise it silently treats the datums as identical: a zero vertical shift, or a
  horizontal error of tens to hundreds of metres. Ordinary warps (UTM → WGS84, LV95 → WGS84) are
  unaffected. `allow_ballpark=True` opts out. Both settings must survive into the serialized
  warp options, so a written VRT reopens with the same behaviour.
- **Grids are the caller's setup** (decided). rasterio's bundled PROJ searches only its own
  `proj_data` directory, not pyproj's user data directory where synced grids land. freezebase
  does not work around that packaging. With ballpark refused, a missing grid is a loud error,
  never a zero shift. The user guide documents the setup: set `PROJ_NETWORK=ON` before the
  process starts, or pass `PROJ_DATA` pointing at a directory holding rasterio's PROJ data and
  the grids. Tests do this in a `conftest.py` fixture, not in the library.

### `stack(sources, *, bands=None, band_names=None, color_interp=None)`

Sources on one identical grid (CRS, transform, size; otherwise raise and point at `warp`)
become successive bands. By default every band of every source is used; `bands` selects
source bands per input.

### `derive(sources, function, *, dtype="float32", nodata=nan, description="", band=None)`

- One source: one derived band per input band, each reading that band.
- Several sources (e.g. `div` of VV and VH): one derived band reading `band` (default 1) from
  each, in order. All inputs must share a grid.
- **Multi-source input bands are nested, never flattened.** An input `VRT` whose band has more
  than one source becomes an inline nested source, so the pixel function sees one mosaicked
  array per input. This is the guard against the silent-zeros case above. Whether single-source
  inputs are inlined or flattened is the builder's call.
- `transfer_type` defaults to `dtype` so integer sources are not truncated before `log10`.
- **Input NODATA is honoured** (*my call*). Each source is a `ComplexSource` carrying its
  input's NODATA, with `propagateNoData="true"` where the function supports it, so a nodata
  input yields output nodata. Otherwise `dB` of a `0` sentinel gives `-inf`, and `diff` of
  `-9999` gives a plausible-looking value.

### `from_raster(path)`

A 1:1 VRT over one raster file: same grid, bands, descriptions, colour interpretation, units,
NODATA and metadata. Used to edit metadata or cast before copying (`write_cog`,
`rewrite_tiff`).

### Pixel functions

`PixelFunction(name, args)` passes any GDAL built-in through as written. Typed constructors
cover the common ones, at least `dB` (with `fact`), `div`, `diff`, `scale`, `min`, `max` and
`mean`. Which others get one is the builder's call.

## Reading and writing

- `write(path, relative=True) -> UPath` returns the path written, so a step can be referenced
  as a file afterwards:

  ```python
  zone = vrt.mosaic(zone_files).write(dir / f"{key}_{z}.vrt")
  web = vrt.warp(zone, crs=grid.crs, resolution=grid.resolution, bounds=ext).write(...)
  ```

  Local writes are atomic. S3 writes upload the bytes in one put, as `_write_vrt` does today.
- **Path policy on write.**
  - Same store as the VRT (local, or the same S3 bucket): relative to the VRT, `../`
    included.
  - Different store: absolute GDAL path (`/vsis3/...`, `/vsicurl/https://...`).
  - `relative=False` writes every path absolute.
  - Inside an inline nested VRT, paths are absolute unless tests show GDAL resolves them
    against the outer file.
- **Read.** `read(path) -> VRT` parses:
  - `/vsis3/` back into `s3://` `UPath`s that inherit the reading path's storage options (same
    profile and endpoint);
  - S3 credentials (*my call*): `read(path, *, storage_options=None)`. When given, the options
    apply to every S3 source, overriding inheritance. This covers a local VRT over S3 tiles on a
    profile-configured endpoint, `glace-catalog`'s chain. Without either, S3 sources get bare
    `s3://` paths, which means the default AWS credential chain. `source_files(follow=True)`
    passes the options on to every `.vrt` it reads;
  - relative paths, resolved against the VRT's location;
  - inline `<VRTDataset>` sources, and escaped XML in `<SourceFilename>` / `<SourceDataset>`,
    which GDAL writes for warps of in-memory VRTs.
- **Unknown elements raise** `ValueError` naming the element, rather than being dropped. The
  exception is `GDALWarpOptions`, kept verbatim. Everything freezebase, `WarpedVRT` and
  `gdalbuildvrt` emit for supported features must read.
- Parse with the standard library (`xml.etree`); Python ≥ 3.12 ships an expat with
  entity-expansion protection. No `defusedxml` dependency.

## Environments and credentials

- `VRT.env()` picks the rasterio environment from `source_files()`:
  - all local: a plain `rasterio.Env`;
  - S3 sources sharing one set of storage options: `s3_env`;
  - public HTTPS: an HTTP environment (below);
  - local mixed with exactly one of the others: that one;
  - S3 sources with different credentials or endpoints: `ValueError`.
- `VRT.open()` is `env()` plus `rasterio.open(self.to_xml())`. Paths are absolute, because an
  unwritten VRT has no base to be relative to.
- Callers using thread pools enter `env()` themselves in the main thread. Credentials set inside
  a worker thread do not reach GDAL's own reader threads. Document this on `env()`.
- **HTTPS (in scope).** `_to_vsi_uri` maps `http(s)://` to `/vsicurl/...`. `_env_for_path`
  gains an `http`/`https` branch: a plain `Env`, no credentials, with timeouts, retries,
  `GDAL_DISABLE_READDIR_ON_OPEN=EMPTY_DIR` and a VSI cache. These options are exposed publicly
  (e.g. `GDAL_HTTP_OPTIONS` and `http_env()`) so callers can reuse them; their location is the
  builder's call.

## `freezebase.raster` on top of the model

- **`write_cog(data, dst, profile=None, *, ...)`.** `data` widens to `np.ndarray | VRT |
  AnyPath`.
  - A `VRT` or a path (via `from_raster`) is streamed block by block with
    `rasterio.shutil.copy(..., driver="COG")` inside `env()`.
  - `band_names`, `tags`, `band_tags`, `units` and `color_interp` are applied to a copy of the
    VRT *before* the copy, so they land in the COG header without breaking its layout.
  - For VRT and path inputs, `profile` carries COG creation options. `dtype` / `nodata` in it
    are applied to the bands, never inherited silently.
  - **A nodata change remaps pixels** (decided). Pixels at the source's NODATA are written as
    the new value: a `ComplexSource` with the old `NODATA` under a band with the new
    `NoDataValue`. Unlike `gdal_translate -a_nodata`, the header alone never changes.
  - **A cast never invents data.** Casting to an integer dtype from a source whose NODATA is NaN,
    or doesn't fit the target dtype, needs an explicit `nodata` that fits, or the call raises.
  - The ndarray path, `checksum`, the S3 single-put upload and atomic local writes keep their
    current behaviour. With `checksum=True`, a VRT input is encoded to a temporary file and
    hashed from it, never held whole in memory.
- **`merge_tiffs` is removed** (**Breaking**, decided). No known downstream code calls it. The
  replacement is a mosaic copied out:

  ```python
  # before
  merge_tiffs([a, b], "merged.tif")
  # after: a COG, or a GTiff via rewrite_tiff
  write_cog(vrt.mosaic([a, b]), "merged.tif")
  rewrite_tiff(vrt.mosaic([a, b]), "merged.tif", profile)
  ```

  `method="first"` is `mosaic(reversed(files))`; `"last"` is the default order. `min` / `max` /
  `sum` / `count` have no replacement. A `combine=` option on `mosaic` (a pixel function over
  the tiles, as verified above) can be added when someone needs it. Its tests in
  `test_raster.py`, the S3 merge tests in `test_s3_integration.py` and the `great-docs.yml`
  entry go with it. The S3 tests are replaced by the same cases on `write_cog(vrt)` and
  `rewrite_tiff(vrt)`.
- **`rewrite_tiff`.** `src_file` widens to `AnyPath | VRT` (*my call*), so a mosaic can be written
  as a plain GTiff. `move=True` with a `VRT` raises. It becomes `from_raster` (for a path), then
  metadata edits and profile overrides, then a staged copy. If possible it collapses `_rewrite_via_memory` / `_rewrite_via_tempfile` into one
  path. Signature and behaviour unchanged: `s1bursts` calls it, including in-place and `move`.
  It keeps ignoring `dtype` and `nodata` in `profile`, as it does today (*my call*). `s1bursts`
  passes its full profile, and applying them would start casting its products unasked.

## The existing `vrt` functions

All stay importable from `freezebase.vrt`, reimplemented as wrappers:

| Function | Becomes | Behaviour change |
|---|---|---|
| `build_vrt_mosaic(files, dst, *, bounds)` | `mosaic(files, bounds=bounds).write(dst)` | Accepts multi-band tiles (widening). Same-store sources not beside the VRT are now written relative rather than absolute. **Changed**, not breaking: the VRT reads the same. |
| `create_warped_vrt(...)` | `warp(...).write(dst, relative=False)` | `resolution` widens to `float \| tuple \| None`. Absolute source path kept, as documented. |
| `create_decibel_vrt` | `derive(src, PixelFunction.db(fact)).write(dst)` | A source with non-NaN NODATA now gives nodata, not `-inf` (**Changed**) |
| `create_rgb_vrt` | `stack` + `derive` | none (no known callers; kept because it is cheap) |
| `create_rgba_vrt` | bands with `lut` sources | none |

Existing tests for these functions must keep passing, except where an assertion pins the old
absolute-path XML that the table above changes.

## Constraints

- AGENTS.md applies: Python 3.12+, Pyrefly strict, Ruff, numpy docstrings, offline unit tests,
  90% patch coverage, CHANGELOG under `## Unreleased` with **Breaking:** entries showing
  before/after. This is a minor release.
- No new runtime dependency.
- Inline nesting needs GDAL ≥ 3.11. rasterio's wheels satisfy it from the lower bound; a
  source-built rasterio against an older GDAL must get a clear error when a nested VRT is
  written or opened, not a GDAL parse failure.
- `freezebase.vrt` stays the import path. Turning `vrt.py` into a package is fine.

## Testing

- Offline fixtures: tiny GeoTIFFs written in `tmp_path`.
- **Round trips.** Build → `to_xml` → `read` → equal, for every builder, nested chains and
  warps. `read` of `WarpedVRT` output and of `gdalbuildvrt`-style XML (hand-written fixtures,
  since there is no GDAL CLI).
- **Pixel correctness against rasterio references.**
  - `mosaic` vs `rasterio.merge`.
  - `warp` vs `rasterio.warp.reproject`.
  - Mixed-resolution `mosaic` vs per-tile `reproject` onto the fine grid. This is the
    misregistration guard: compare pixel centres, not just shapes.
- **Regression guards.**
  - `derive(dB)` over an in-memory two-tile mosaic gives dB values, not zeros.
  - `write_cog(vrt)` keeps dtype and NODATA unless the profile changes them.
- **Vertical shifts.** Fixtures use small synthetic DEMs and the PROJ grids. Tests that need
  a grid are skipped when it is unavailable offline, as integration tests are without RustFS.
  - Inferred shift for compound → 3D, compound → compound and 3D → compound, against pyproj
    within 2 cm.
  - A vertical source into a 2D target raises.
  - A missing grid raises (point PROJ at an empty data directory with the network off).
  - `tolerance="auto"` resolves to `0` exactly when a shift applies.
  - A written and re-read warped VRT keeps `APPLY_VERTICAL_SHIFT` and `ALLOW_BALLPARK`.
- **Integration (RustFS).**
  - A VRT written to S3 with relative and `../` sources reopens.
  - `env()` and `open()` on S3 sources.
  - `source_files(follow=True)` across a local VRT pointing at S3 tiles.
- Lower-bound check (`--resolution lowest-direct --python 3.12`) passes.

## Documentation

A user-guide page on VRTs: the model, the four verbs, chaining in memory vs on disk, the path
policy, credentials and threads. API reference entries for the new public names.

## Left to the builder

- Module layout inside `freezebase.vrt`, and private helper structure.
- Exact field and helper names, within the shape above.
- Inlining vs flattening single-source inputs to `derive`.
- How mixed-resolution placement is achieved (fractional `DstRect`, or nested per-tile warps),
  provided the pixel-centre test passes.
- Which typed `PixelFunction` constructors beyond the listed ones exist.
- Where `GDAL_HTTP_OPTIONS` / `http_env` live.

## Delivery (decided)

Two pull requests:

1. **The VRT core:** the model, builders, `warp` with vertical shifts, the legacy wrappers, and
   the HTTPS paths that `VRT.env()` needs.
2. **`freezebase.raster` on the model:** `write_cog` and `rewrite_tiff` on VRTs, and removing
   `merge_tiffs`. Stacked on the first.

## Downstream follow-ups (not part of this project)

- `glace-catalog`: replace `tagged_vrt` with `read` + edit + `write`, and `sources_env` /
  `chain_env` with `VRT.env()`. Optionally build the chain in memory.
- Other downstream code: drop its own multi-band VRT writer, mixed-resolution mosaic, natural-
  grid helper and relative-path rewriting in favour of `mosaic`, `warp`, and `write`'s path
  policy. Replace `rasterio.shutil.copy` COG writes with `write_cog(vrt)`.
  Replace explicit geoid-undulation code with `warp` between the compound and the 3D or
  compound target CRS.
